"""
lm-evaluation-harness 기반 Hybrid-CDF Equal-Mass Chunking 평가 스크립트.

eval_equal_mass.py 구조를 따르되, generate_until()에서
hybrid_cdf_chunking 기반 생성 함수를 사용한다.

핵심 파라미터:
  --model_args ...,lam=0.5,...
  λ=1.0 → 순수 attention CDF
  λ=0.0 → 완전 균등 분할
"""

import accelerate
import torch
import re
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer, AutoConfig

from model.modeling_llada import LLaDAModelLM
from gsm8k_hybrid_cdf_eval import (
    generate_hybrid_cdf,
    generate_fixed_block,
    generate_block_argmax1,
    get_depth_adaptive_rollout,
    get_baseline_rollout,
    hybrid_cdf_chunking,
    anchor_partition,
    add_gumbel_noise,
    get_num_transfer_tokens,
    select_transfer_index_threshold,
    select_transfer_index_topk,
)


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _parse_manual_block_sizes(value) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, list):
        out = [int(x) for x in value]
        return out if out else None
    s = str(value).strip()
    if not s or s.lower() == "none":
        return None
    parts = [x.strip() for x in re.split(r"[,|:;\s]+", s) if x.strip()]
    if not parts:
        return None
    return [int(x) for x in parts]


def _compute_promptlen_block_schedule(
    gen_length: int,
    prompt_len_chars: int,
) -> Tuple[List[int], float]:
    """
    block_0 = clip(54 - 0.009 * prompt_len_chars, 43, 57)
    Remaining 7 blocks share (gen_length - block_0) as evenly as possible.
    Remainder is assigned from block_1 to block_7.
    """
    raw_block0 = 54.0 - 0.009 * float(prompt_len_chars)
    block0 = int(round(raw_block0))
    block0 = max(43, min(57, block0))

    remain = int(gen_length) - block0
    if remain <= 0:
        raise ValueError(
            f"Invalid remain length: gen_length={gen_length}, block0={block0}"
        )

    tail_blocks = 7
    base = remain // tail_blocks
    remainder = remain % tail_blocks
    tail = [base + (1 if i < remainder else 0) for i in range(tail_blocks)]

    sizes = [block0] + tail
    if sum(sizes) != int(gen_length):
        raise ValueError(
            f"Prompt-length scheduler sum mismatch: sum={sum(sizes)} gen_length={gen_length}"
        )
    if any(s <= 0 for s in sizes):
        raise ValueError(f"Prompt-length scheduler produced non-positive sizes: {sizes}")
    return sizes, raw_block0


@register_model("llada_hybrid_cdf")
class LLaDAHybridCDFHarness(LM):
    def __init__(
        self,
        model_path="",
        mask_id=126336,
        max_length=4096,
        batch_size=1,
        mc_num=128,
        is_check_greedy=False,
        # generation params
        gen_length=256,
        steps_per_block=32,
        # hybrid-cdf params
        strategy="hybrid_cdf_sigmoid",
        num_blocks=8,
        lam=0.5,
        # fixed-block params (baseline comparison)
        block_length=32,
        # common
        temperature=0.0,
        threshold=0.9,
        inverse=False,
        control_mode="none",
        control_min_size=28,
        control_max_size=32,
        scheduler_seed=42,
        seed=42,
        first_block_size=-1,
        manual_block_sizes="",
        high_score_top_k=None,
        gold_prefix_blocks=0,
        gold_source="dataset_answer",
        cap_alpha=1.0,
        cap_b_min=8,
        cap_max_iter=50,
        # smoothed inverse-cdf params
        smoothing_window=32,
        # anchor partition params
        anchor_mode="off",
        anchor_size=64,
        anchor_fraction=0.5,
        anchor_pos_type="center",
        anchor_min_block_size=26,
        anchor_all_right=False,
        device="cuda",
        save_dir=None,
        show_speed=False,
        verbose=False,
        save_trace=False,
        apply_stop=False,
        **kwargs,
    ):
        super().__init__()

        self.seed = int(seed)
        set_seed(self.seed)

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None

        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({"device_map": {"": f"{self.accelerator.device}"}})

        config = AutoConfig.from_pretrained(model_path)
        config.flash_attention = True
        self.model = LLaDAModelLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            config=config,
            **model_kwargs,
        )
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f"{self.accelerator.device}")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model = self.model.to(device)

        self.mask_id = int(mask_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        self.mc_num = int(mc_num)
        self.batch_size = int(batch_size)
        assert self.mc_num % self.batch_size == 0
        self.sampling_eps = 0.0
        self.max_length = int(max_length)
        self.is_check_greedy = (
            is_check_greedy
            if isinstance(is_check_greedy, bool)
            else str(is_check_greedy).lower() == "true"
        )

        self.gen_length = int(gen_length)
        self.steps_per_block = int(steps_per_block)

        self.strategy = str(strategy)
        self.num_blocks = int(num_blocks)
        self.lam = float(lam)
        self.block_length = int(block_length)

        self.temperature = float(temperature)
        self.threshold = (
            float(threshold)
            if threshold is not None and str(threshold).lower() != "none"
            else None
        )
        self.inverse = (
            inverse
            if isinstance(inverse, bool)
            else str(inverse).lower() == "true"
        )
        self.control_mode = str(control_mode).lower()
        valid_control = {
            "none",
            "balanced_random",
            "inverse_permuted",
            "inverse_head_rescaled_tail",
        }
        if self.control_mode not in valid_control:
            raise ValueError(
                f"Unknown control_mode '{self.control_mode}'. Valid: {sorted(valid_control)}"
            )
        self.control_min_size = int(control_min_size)
        self.control_max_size = int(control_max_size)
        self.scheduler_seed = int(scheduler_seed)
        self.first_block_size = int(first_block_size)

        self.manual_block_sizes = _parse_manual_block_sizes(manual_block_sizes)
        self.high_score_top_k = (
            int(high_score_top_k)
            if high_score_top_k is not None and str(high_score_top_k).lower() != "none"
            else None
        )
        self.gold_prefix_blocks = int(gold_prefix_blocks)
        if self.gold_prefix_blocks < 0:
            raise ValueError(f"gold_prefix_blocks must be >= 0, got {self.gold_prefix_blocks}")
        self.gold_source = str(gold_source).lower()
        valid_gold_sources = {"dataset_answer"}
        if self.gold_source not in valid_gold_sources:
            raise ValueError(
                f"Unknown gold_source '{self.gold_source}'. Valid: {sorted(valid_gold_sources)}"
            )
        self.cap_alpha = float(cap_alpha)
        self.cap_b_min = int(cap_b_min)
        self.cap_max_iter = int(cap_max_iter)
        self.smoothing_window = int(smoothing_window)

        # Anchor partition
        self.anchor_mode = str(anchor_mode).lower()
        valid_anchor_modes = {"off", "score", "random", "fixed", "uniform"}
        if self.anchor_mode not in valid_anchor_modes:
            raise ValueError(
                f"Unknown anchor_mode '{self.anchor_mode}'. Valid: {sorted(valid_anchor_modes)}"
            )
        self.anchor_size = int(anchor_size)
        self.anchor_fraction = float(anchor_fraction)
        self.anchor_pos_type = str(anchor_pos_type).lower()
        if self.anchor_pos_type not in {"center", "start"}:
            raise ValueError(f"anchor_pos_type must be 'center' or 'start', got '{self.anchor_pos_type}'")
        self.anchor_min_block_size = int(anchor_min_block_size)
        self.anchor_all_right = (
            anchor_all_right
            if isinstance(anchor_all_right, bool)
            else str(anchor_all_right).lower() == "true"
        )

        if self.strategy == "manual_blocks":
            if self.manual_block_sizes is None:
                raise ValueError(
                    "strategy='manual_blocks' requires manual_block_sizes (e.g. '48,40,34,30,28,26,26,24')"
                )
            if any(int(x) <= 0 for x in self.manual_block_sizes):
                raise ValueError("manual_block_sizes must contain only positive integers")
            if sum(self.manual_block_sizes) != self.gen_length:
                raise ValueError(
                    f"sum(manual_block_sizes)={sum(self.manual_block_sizes)} must equal gen_length={self.gen_length}"
                )

        if self.strategy.startswith("cap_context"):
            if self.cap_alpha < 0:
                raise ValueError(f"cap_alpha must be >= 0, got {self.cap_alpha}")
            if self.cap_b_min <= 0:
                raise ValueError(f"cap_b_min must be > 0, got {self.cap_b_min}")
            if self.cap_b_min * self.num_blocks > self.gen_length:
                raise ValueError(
                    "cap_context requires cap_b_min * num_blocks <= gen_length, "
                    f"got {self.cap_b_min} * {self.num_blocks} > {self.gen_length}"
                )

        self.is_instruct = "instruct" in model_path.lower()
        self.save_dir = save_dir
        self.show_speed = (
            show_speed
            if isinstance(show_speed, bool)
            else str(show_speed).lower() == "true"
        )
        self.verbose = (
            verbose
            if isinstance(verbose, bool)
            else str(verbose).lower() == "true"
        )
        self.save_trace = (
            save_trace
            if isinstance(save_trace, bool)
            else str(save_trace).lower() == "true"
        )
        self.apply_stop = (
            apply_stop
            if isinstance(apply_stop, bool)
            else str(apply_stop).lower() == "true"
        )

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape
        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)
        x = torch.round(
            torch.linspace(
                float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device
            )
        ).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)
        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat(
            [torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask],
            dim=1,
        )
        noisy_batch = torch.where(is_mask, self.mask_id, batch)
        return noisy_batch, is_mask

    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                context_enc = [self.tokenizer.bos_token_id]
            else:
                context_enc = self.tokenizer(context)["input_ids"]
            continuation_enc = self.tokenizer(continuation)["input_ids"]
            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def _loglikelihood_tokens(self, requests):
        out = []
        for _, context_enc, continuation_enc in tqdm(requests, desc="loglikelihood"):
            context_enc_tensor = torch.tensor(
                context_enc, dtype=torch.long, device=self.device
            ).unsqueeze(0)
            whole_enc = torch.tensor(
                context_enc + continuation_enc, dtype=torch.long, device=self.device
            ).unsqueeze(0)
            prompt_index = torch.zeros(whole_enc.shape[1], device=self.device)
            prompt_index[: context_enc_tensor.shape[1]] = 1

            mc_num = self.mc_num
            ll = 0
            for _ in range(mc_num // self.batch_size):
                batch = whole_enc.repeat(self.batch_size, 1)
                noisy_batch, is_mask = self._forward_process(batch, prompt_index)
                with torch.no_grad():
                    outputs = self.model(noisy_batch)
                    logits = outputs.logits
                token_probs = torch.softmax(logits, dim=-1)
                target_probs = token_probs.gather(
                    dim=-1, index=whole_enc.repeat(self.batch_size, 1).unsqueeze(-1)
                ).squeeze(-1)
                target_probs = target_probs.clamp(min=1e-10)
                log_probs = target_probs.log()
                ll += (log_probs * is_mask).sum(dim=-1).mean().item()

            ll = ll / (mc_num // self.batch_size)

            if self.is_check_greedy:
                prefix = context_enc
                target = continuation_enc
                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)
                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
            else:
                out.append((ll, False))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests):
        output = []
        num_tokens = 0
        num_nfe = 0
        processed_count = 0
        trace_path = None

        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            rank = self.rank
            save_path = os.path.join(self.save_dir, f"rank_{rank}.jsonl")
            if self.save_trace:
                trace_path = os.path.join(self.save_dir, f"rank_{rank}_trace_blocks.jsonl")
            print(f"save_path: {save_path}")
            if os.path.exists(save_path):
                print(f"load from {save_path}")
                with open(save_path, "r", encoding="utf-8") as f:
                    output = [json.loads(line) for line in f]
                    processed_count = len(output)
                print(f"processed_count: {processed_count}")

        batched_requests = [[]]
        for i, req in enumerate(tqdm(requests, desc="Batching...")):
            if i < processed_count:
                continue
            batched_requests[-1].append(req)
            if len(batched_requests[-1]) == self.batch_size:
                batched_requests.append([])
        if len(batched_requests[-1]) == 0:
            batched_requests.pop()

        start_time = time.time()
        sample_index_cursor = processed_count

        for batch in tqdm(batched_requests, desc="Generating..."):
            batched_input_ids = []
            batched_prompt_len_chars = []
            max_len = 0
            pad_len = []
            for req in batch:
                question = req.args[0]
                batched_prompt_len_chars.append(len(question))
                is_humaneval = (
                    hasattr(req, "doc")
                    and req.doc is not None
                    and "task_id" in req.doc
                    and str(req.doc["task_id"]).lower().startswith("humaneval")
                )
                use_chat_template = self.is_instruct

                if use_chat_template:
                    m = [{"role": "user", "content": question}]
                    user_input = self.tokenizer.apply_chat_template(
                        m, add_generation_prompt=True, tokenize=False
                    )
                    input_ids = self.tokenizer(user_input)["input_ids"]
                else:
                    user_input = question
                    input_ids = self.tokenizer(user_input)["input_ids"]
                batched_input_ids.append(input_ids)
                max_len = max(max_len, len(input_ids))
                pad_len.append(max_len - len(input_ids))

            batched_input_ids = [
                torch.cat(
                    [
                        torch.full(
                            (1, max_len - len(ids)),
                            self.tokenizer.pad_token_id,
                            dtype=torch.long,
                            device=self.device,
                        ),
                        torch.tensor(
                            ids, dtype=torch.long, device=self.device
                        ).unsqueeze(0),
                    ],
                    dim=1,
                )
                for ids in batched_input_ids
            ]
            batched_input_ids = torch.cat(batched_input_ids, dim=0).to(self.device)

            stop_tokens = batch[0].args[1]["until"]
            input_ids = batched_input_ids

            assert input_ids.shape[0] == 1, (
                "Hybrid-CDF generation currently supports batch_size=1 only."
            )

            manual_block_sizes_for_call = self.manual_block_sizes
            prompt_len_chars_for_call = None
            if self.strategy == "promptlen_char_block0_scheduler":
                prompt_len_chars_for_call = int(batched_prompt_len_chars[0])
                dynamic_sizes, _ = _compute_promptlen_block_schedule(
                    gen_length=self.gen_length,
                    prompt_len_chars=prompt_len_chars_for_call,
                )
                manual_block_sizes_for_call = dynamic_sizes

            gold_prefix_tokens = None
            gold_prefix_tokens_requested = 0
            gold_prefix_text = None
            if self.strategy == "fixed_block" and self.gold_prefix_blocks > 0:
                gold_prefix_tokens_requested = self.gold_prefix_blocks * self.block_length
                doc = getattr(batch[0], "doc", None)
                if not isinstance(doc, dict) or "answer" not in doc:
                    raise ValueError(
                        "gold_prefix_blocks>0 requires req.doc['answer'] to be available"
                    )
                gold_prefix_text = str(doc["answer"])
                gold_prefix_ids = self.tokenizer(
                    gold_prefix_text,
                    add_special_tokens=False,
                )["input_ids"]
                gold_prefix_tokens = torch.tensor(
                    gold_prefix_ids[:gold_prefix_tokens_requested],
                    dtype=torch.long,
                    device=self.device,
                )

            if self.strategy == "fixed_block":
                generated_answer, nfe, info = generate_fixed_block(
                    model=self.model,
                    prompt=input_ids,
                    gen_length=self.gen_length,
                    mask_id=self.mask_id,
                    block_length=self.block_length,
                    steps_per_block=self.steps_per_block,
                    temperature=self.temperature,
                    threshold=self.threshold,
                    gold_prefix_tokens=gold_prefix_tokens,
                )
                info["gold_prefix_blocks"] = int(self.gold_prefix_blocks)
                info["gold_source"] = self.gold_source if self.gold_prefix_blocks > 0 else None
                info["gold_prefix_tokens_requested"] = int(gold_prefix_tokens_requested)
                info["gold_prefix_text"] = gold_prefix_text
            elif self.strategy == "block_argmax1":
                generated_answer, nfe, info = generate_block_argmax1(
                    model=self.model,
                    prompt=input_ids,
                    gen_length=self.gen_length,
                    mask_id=self.mask_id,
                    block_length=self.block_length,
                    temperature=self.temperature,
                )
            elif self.strategy == "anchor":
                # ── Anchor Partition Strategy ──
                # Determine anchor_pos based on anchor_mode
                anchor_info = {
                    "anchor_mode": self.anchor_mode,
                    "anchor_size": self.anchor_size,
                    "anchor_pos_type": self.anchor_pos_type,
                    "anchor_all_right": self.anchor_all_right,
                }

                if self.anchor_mode == "uniform":
                    # Control baseline: uniform blocks, no anchor
                    base = self.gen_length // self.num_blocks
                    rem = self.gen_length % self.num_blocks
                    uniform_sizes = [base + (1 if i < rem else 0) for i in range(self.num_blocks)]
                    anchor_blocks_manual = uniform_sizes
                    anchor_info["anchor_pos"] = None
                    anchor_info["anchor_start"] = None
                    anchor_info["anchor_end"] = None
                elif self.anchor_mode == "score":
                    # anchor_pos determined by step-0 rollout — handled inside generate_hybrid_cdf
                    # Pass anchor params so it can compute after rollout
                    anchor_blocks_manual = None  # will be computed inside generate
                elif self.anchor_mode == "random":
                    rng = random.Random(int(self.scheduler_seed) + int(sample_index_cursor))
                    half = self.anchor_size // 2
                    lo = half
                    hi = self.gen_length - self.anchor_size + half
                    if lo > hi:
                        lo = hi = self.gen_length // 2
                    anchor_pos = rng.randint(lo, hi)
                    blocks = anchor_partition(
                        gen_length=self.gen_length,
                        num_blocks=self.num_blocks,
                        anchor_size=self.anchor_size,
                        anchor_pos=anchor_pos,
                        min_block_size=self.anchor_min_block_size,
                        pos_type=self.anchor_pos_type,
                        all_right=self.anchor_all_right,
                    )
                    anchor_blocks_manual = [e - s for s, e in blocks]
                    anchor_info["anchor_pos"] = anchor_pos
                    # Find the anchor block (the largest one)
                    max_size = max(e - s for s, e in blocks)
                    for s, e in blocks:
                        if e - s == max_size:
                            anchor_info["anchor_start"] = s
                            anchor_info["anchor_end"] = e
                            break
                elif self.anchor_mode == "fixed":
                    anchor_pos = int(self.gen_length * self.anchor_fraction)
                    blocks = anchor_partition(
                        gen_length=self.gen_length,
                        num_blocks=self.num_blocks,
                        anchor_size=self.anchor_size,
                        anchor_pos=anchor_pos,
                        min_block_size=self.anchor_min_block_size,
                        pos_type=self.anchor_pos_type,
                        all_right=self.anchor_all_right,
                    )
                    anchor_blocks_manual = [e - s for s, e in blocks]
                    anchor_info["anchor_pos"] = anchor_pos
                    max_size = max(e - s for s, e in blocks)
                    for s, e in blocks:
                        if e - s == max_size:
                            anchor_info["anchor_start"] = s
                            anchor_info["anchor_end"] = e
                            break
                else:
                    raise ValueError(f"Unsupported anchor_mode '{self.anchor_mode}' for strategy='anchor'")

                if self.anchor_mode == "score":
                    # For score mode, we need to run generate_hybrid_cdf with rollout,
                    # then use the rollout scores to determine anchor_pos.
                    # We pass anchor params via a new code path in generate_hybrid_cdf.
                    generated_answer, nfe, info = generate_hybrid_cdf(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        prompt=input_ids,
                        gen_length=self.gen_length,
                        mask_id=self.mask_id,
                        num_blocks=self.num_blocks,
                        steps_per_block=self.steps_per_block,
                        lam=0.0,
                        temperature=self.temperature,
                        threshold=self.threshold,
                        rollout_mode="sigmoid",
                        inverse=False,
                        control_mode="none",
                        scheduler_seed=self.scheduler_seed,
                        first_block_size=-1,
                        manual_block_sizes=None,
                        sample_index=sample_index_cursor,
                        verbose=self.verbose,
                        strategy="anchor_score",
                        cap_alpha=self.cap_alpha,
                        cap_b_min=self.cap_b_min,
                        cap_max_iter=self.cap_max_iter,
                        anchor_size=self.anchor_size,
                        anchor_min_block_size=self.anchor_min_block_size,
                        anchor_pos_type=self.anchor_pos_type,
                        anchor_all_right=self.anchor_all_right,
                    )
                    info.update(anchor_info)
                    # anchor_pos was computed inside generate; extract from block_boundaries
                    if info.get("block_boundaries"):
                        sizes = [e - s for s, e in info["block_boundaries"]]
                        max_idx = sizes.index(max(sizes))
                        ab = info["block_boundaries"][max_idx]
                        info["anchor_start"] = ab[0]
                        info["anchor_end"] = ab[1]
                        info["anchor_pos"] = (ab[0] + ab[1]) // 2
                else:
                    # For random/fixed/uniform: blocks are pre-computed, pass as manual
                    generated_answer, nfe, info = generate_hybrid_cdf(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        prompt=input_ids,
                        gen_length=self.gen_length,
                        mask_id=self.mask_id,
                        num_blocks=self.num_blocks,
                        steps_per_block=self.steps_per_block,
                        lam=0.0,
                        temperature=self.temperature,
                        threshold=self.threshold,
                        rollout_mode="sigmoid",
                        inverse=False,
                        control_mode="none",
                        scheduler_seed=self.scheduler_seed,
                        first_block_size=-1,
                        manual_block_sizes=anchor_blocks_manual,
                        sample_index=sample_index_cursor,
                        verbose=self.verbose,
                        strategy="manual_blocks" if anchor_blocks_manual else "hybrid_cdf_sigmoid",
                        cap_alpha=self.cap_alpha,
                        cap_b_min=self.cap_b_min,
                        cap_max_iter=self.cap_max_iter,
                    )
                    info.update(anchor_info)
            else:
                if self.strategy.startswith("smoothed_inverse_cdf_"):
                    rollout_mode = self.strategy.replace("smoothed_inverse_cdf_", "")
                elif self.strategy.startswith("hybrid_cdf_"):
                    rollout_mode = self.strategy.replace("hybrid_cdf_", "")
                elif self.strategy.startswith("lowest_score_boundary_"):
                    rollout_mode = self.strategy.replace("lowest_score_boundary_", "")
                elif self.strategy.startswith("high_score_boundary_before_"):
                    rollout_mode = self.strategy.replace("high_score_boundary_before_", "")
                elif self.strategy.startswith("high_score_boundary_after_"):
                    rollout_mode = self.strategy.replace("high_score_boundary_after_", "")
                else:
                    rollout_mode = "sigmoid"
                generated_answer, nfe, info = generate_hybrid_cdf(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=input_ids,
                    gen_length=self.gen_length,
                    mask_id=self.mask_id,
                    num_blocks=self.num_blocks,
                    steps_per_block=self.steps_per_block,
                    lam=self.lam,
                    temperature=self.temperature,
                    threshold=self.threshold,
                    rollout_mode=rollout_mode,
                    inverse=self.inverse,
                    control_mode=self.control_mode,
                    control_min_size=self.control_min_size,
                    control_max_size=self.control_max_size,
                    scheduler_seed=self.scheduler_seed,
                    first_block_size=self.first_block_size,
                    manual_block_sizes=manual_block_sizes_for_call,
                    high_score_top_k=self.high_score_top_k,
                    prompt_len_chars=prompt_len_chars_for_call,
                    sample_index=sample_index_cursor,
                    verbose=self.verbose,
                    strategy=self.strategy,
                    cap_alpha=self.cap_alpha,
                    cap_b_min=self.cap_b_min,
                    cap_max_iter=self.cap_max_iter,
                    smoothing_window=self.smoothing_window,
                )

            batched_generated_answer = []
            trace_rows = []
            for i in range(len(generated_answer)):
                generated_answer_i = self.tokenizer.decode(
                    generated_answer[i][input_ids.shape[1]:],
                    skip_special_tokens=False,
                )
                if self.apply_stop:
                    for stop_seq in stop_tokens:
                        if stop_seq in generated_answer_i:
                            generated_answer_i = generated_answer_i.split(stop_seq)[0]

                generated_answer_ids = torch.tensor(
                    self.tokenizer(generated_answer_i)["input_ids"]
                )
                if self.show_speed:
                    num_tokens += (generated_answer_ids != 126081).sum()
                    num_nfe += nfe

                generated_answer_i = self.tokenizer.decode(
                    generated_answer_ids, skip_special_tokens=True
                )
                batched_generated_answer.append(generated_answer_i)
                if self.save_trace:
                    trace_rows.append(
                        {
                            "sample_index": int(sample_index_cursor + i),
                            "scheduler_seed": int(self.scheduler_seed),
                            "control_mode": self.control_mode,
                            "strategy": self.strategy,
                            "nfe": int(nfe),
                            "generated_text": generated_answer_i,
                            "trace": info,
                        }
                    )

            output.extend(batched_generated_answer)
            sample_index_cursor += len(batched_generated_answer)

            if self.save_dir is not None:
                with open(save_path, "a", encoding="utf-8") as f:
                    for ans in batched_generated_answer:
                        f.write(json.dumps(ans, ensure_ascii=False) + "\n")
                if trace_path is not None and trace_rows:
                    with open(trace_path, "a", encoding="utf-8") as f:
                        for row in trace_rows:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if self.verbose:
                for i in range(len(batched_generated_answer)):
                    print("=" * 20)
                    print("answer: ", batched_generated_answer[i])
                    print("nfe: ", nfe)
                    if len(output) > 0:
                        print("avg nfe: ", num_nfe / len(output))
                    print("=" * 20, end="\n\n")

        end_time = time.time()
        if self.show_speed:
            print(f"Total number of tokens generated: {num_tokens}")
            print(f"Total time taken: {end_time - start_time} seconds")
            if end_time - start_time > 0:
                print(f"Tokens per second: {num_tokens / (end_time - start_time)}")
            print(f"Total NFE is {num_nfe}")

            if self.save_dir is not None:
                elapsed = end_time - start_time
                tps = num_tokens / elapsed if elapsed > 0 else None
                speed_stats = {
                    "num_tokens": int(num_tokens),
                    "elapsed_sec": float(elapsed),
                    "tokens_per_sec": float(tps) if tps is not None else None,
                    "total_nfe": int(num_nfe),
                    "avg_nfe_per_sample": float(num_nfe / max(len(output), 1)),
                    "strategy": self.strategy,
                    "lam": self.lam,
                    "inverse": self.inverse,
                    "control_mode": self.control_mode,
                    "control_min_size": self.control_min_size,
                    "control_max_size": self.control_max_size,
                    "scheduler_seed": self.scheduler_seed,
                    "seed": self.seed,
                    "first_block_size": self.first_block_size,
                    "manual_block_sizes": self.manual_block_sizes,
                    "high_score_top_k": self.high_score_top_k,
                    "gold_prefix_blocks": self.gold_prefix_blocks,
                    "gold_source": self.gold_source,
                    "cap_alpha": self.cap_alpha,
                    "cap_b_min": self.cap_b_min,
                    "cap_max_iter": self.cap_max_iter,
                    "gen_length": self.gen_length,
                    "num_blocks": self.num_blocks,
                    "steps_per_block": self.steps_per_block,
                    "apply_stop": self.apply_stop,
                }
                os.makedirs(self.save_dir, exist_ok=True)
                speed_path = os.path.join(
                    self.save_dir, f"rank_{self.rank}_speed.json"
                )
                with open(speed_path, "w", encoding="utf-8") as f:
                    json.dump(speed_stats, f, ensure_ascii=False, indent=2)
                print(f"[Saved] speed stats → {speed_path}")

        return output


if __name__ == "__main__":
    cli_evaluate()
