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
    get_depth_adaptive_rollout,
    get_baseline_rollout,
    hybrid_cdf_chunking,
    add_gumbel_noise,
    get_num_transfer_tokens,
    select_transfer_index_threshold,
    select_transfer_index_topk,
)


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        device="cuda",
        save_dir=None,
        show_speed=False,
        verbose=False,
        **kwargs,
    ):
        super().__init__()

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
        valid_control = {"none", "balanced_random", "inverse_permuted"}
        if self.control_mode not in valid_control:
            raise ValueError(
                f"Unknown control_mode '{self.control_mode}'. Valid: {sorted(valid_control)}"
            )
        self.control_min_size = int(control_min_size)
        self.control_max_size = int(control_max_size)
        self.scheduler_seed = int(scheduler_seed)

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

        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            rank = self.rank
            save_path = os.path.join(self.save_dir, f"rank_{rank}.jsonl")
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
            max_len = 0
            pad_len = []
            for req in batch:
                question = req.args[0]
                is_humaneval = (
                    hasattr(req, "doc")
                    and req.doc is not None
                    and "task_id" in req.doc
                    and str(req.doc["task_id"]).lower().startswith("humaneval")
                )
                use_chat_template = self.is_instruct and not is_humaneval

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
                )
            else:
                rollout_mode = (
                    self.strategy.replace("hybrid_cdf_", "")
                    if self.strategy.startswith("hybrid_cdf_")
                    else "sigmoid"
                )
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
                    sample_index=sample_index_cursor,
                    verbose=self.verbose,
                )

            batched_generated_answer = []
            for i in range(len(generated_answer)):
                generated_answer_i = self.tokenizer.decode(
                    generated_answer[i][input_ids.shape[1]:],
                    skip_special_tokens=False,
                )
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

            output.extend(batched_generated_answer)
            sample_index_cursor += len(batched_generated_answer)

            if self.save_dir is not None:
                with open(save_path, "a", encoding="utf-8") as f:
                    for ans in batched_generated_answer:
                        f.write(json.dumps(ans, ensure_ascii=False) + "\n")

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
                    "gen_length": self.gen_length,
                    "num_blocks": self.num_blocks,
                    "steps_per_block": self.steps_per_block,
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
