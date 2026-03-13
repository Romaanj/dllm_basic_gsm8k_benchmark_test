import torch
import torch.nn.functional as F
import time
import re
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from model.modeling_llada import LLaDAModelLM

# [Utility: GSM8K 정답 추출기는 기존과 동일]
def extract_answer(text):
    match = re.search(r"####\s*(-?\d+)", text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"-?\d+", text)
    return numbers[-1] if numbers else ""

# -------------------------------------------------------------------------
# 1. 핵심 로직: Confidence-Gap 기반 Anchor Selection
# -------------------------------------------------------------------------
def get_gap_anchors(logits, masked_pos, ratio=0.1, max_same_token=2):
    """
    신뢰도 격차(Confidence-Gap)만을 사용하여 앵커를 선정합니다.
    """
    device = logits.device
    probs = F.softmax(logits[0].float(), dim=-1)
    
    # 각 위치에서 상위 2개의 확률값 추출
    top_probs, top_indices = torch.topk(probs, k=2, dim=-1)
    
    # Gap 계산: P(top1) - P(top2)
    gap = top_probs[:, 0] - top_probs[:, 1]
    
    # 앵커 개수 결정 (기존과 동일한 ratio 적용)
    k_adaptive = max(1, int(len(masked_pos) * ratio))
    
    # 마스킹된 위치의 Gap 점수 정렬
    masked_gap_scores = gap[masked_pos]
    sorted_indices = torch.argsort(masked_gap_scores, descending=True)
    
    preds = top_indices[:, 0] # Top-1 토큰들
    
    selected_pos, token_counts = [], {}
    for idx in sorted_indices:
        pos = masked_pos[idx]
        t_id = preds[pos].item()
        
        # 중복 토큰 제약 (Diversity Filter 유지)
        count = token_counts.get(t_id, 0)
        if count < max_same_token:
            selected_pos.append(pos)
            token_counts[t_id] = count + 1
        
        if len(selected_pos) >= k_adaptive:
            break
            
    return torch.tensor(selected_pos, device=device)

# -------------------------------------------------------------------------
# 2. 메인 추론 엔진: Gap-based Adaptive Island Decoding
# -------------------------------------------------------------------------
@torch.no_grad()
def island_inference_gap(model, tokenizer, question, ratio=0.1, threshold=0.9):
    device = model.device
    m = [{"role": "user", "content": question}]
    prompt_f = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt_f, return_tensors="pt")['input_ids'].to(device)
    prompt_len = input_ids.shape[1]
    
    x = torch.full((1, prompt_len + 128), 126336, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids.clone()
    
    intervened = False
    entropy_history = []
    steps = 0

    for s in range(128):
        steps = s
        mask_index = (x == 126336)
        if not mask_index.any(): break
        
        logits = model(x).logits
        probs = F.softmax(logits[0].float(), dim=-1)
        conf, preds = torch.max(probs, dim=-1)
        
        # Adaptive Timing을 위한 Entropy Gradient는 그대로 유지 (동일 조건 비교를 위해)
        avg_mask_entropy = (-torch.sum(probs * torch.log(probs + 1e-9), dim=-1))[mask_index[0]].mean().item()
        entropy_history.append(avg_mask_entropy)
        
        if not intervened and s > 5:
            grad = entropy_history[-2] - entropy_history[-1] if len(entropy_history) > 1 else 0
            if grad > 0.05:
                masked_pos = torch.where(mask_index[0])[0]
                # 여기서 Gap-based 앵커 선정 로직 호출
                commit_idx = get_gap_anchors(logits, masked_pos, ratio=ratio)
                x[0, commit_idx] = preds[commit_idx]
                intervened = True

        mask_index = (x == 126336)
        if mask_index.any():
            current_conf = torch.where(mask_index[0], conf, torch.tensor(-1e9, device=device))
            if not intervened:
                x[0, torch.argmax(current_conf)] = preds[torch.argmax(current_conf)]
            else:
                t_mask = (current_conf > threshold)
                if t_mask.any(): x[0, t_mask] = preds[t_mask]
                else: x[0, torch.argmax(current_conf)] = preds[torch.argmax(current_conf)]

    return tokenizer.decode(x[0, prompt_len:], skip_special_tokens=True), steps

# -------------------------------------------------------------------------
# 3. 벤치마크 실행부
# -------------------------------------------------------------------------
def run_benchmark(model, tokenizer, num_samples=100):
    ds = load_dataset("gsm8k", "main", split=f"test[:{num_samples}]")
    stats = {"correct": 0, "total_tokens": 0, "total_time": 0.0, "steps": []}

    print(f"\n>>> Running Gap-based Anchor Benchmark ({num_samples} samples)")
    for data in tqdm(ds, desc="Processing"):
        start = time.time()
        output, steps = island_inference_gap(model, tokenizer, data['question'])
        elapsed = time.time() - start
        
        gold = extract_answer(data['answer'])
        pred = extract_answer(output)
        
        if gold == pred: stats["correct"] += 1
        stats["total_time"] += elapsed
        stats["total_tokens"] += len(tokenizer.encode(output))
        stats["steps"].append(steps)

    print(f"\n[Gap-based Result] Acc: {stats['correct']/num_samples*100:.2f}% | TPS: {stats['total_tokens']/stats['total_time']:.2f} | Avg Steps: {np.mean(stats['steps']):.1f}")

if __name__ == "__main__":
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    
    run_benchmark(model, tokenizer, num_samples=500)