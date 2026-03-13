import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


def ids_to_human_readable_tokens(tokenizer, ids):
    """
    토큰 ID 리스트를 사람이 읽을 수 있는 문자열 리스트로 변환.
    BPE/SentencePiece의 특수 prefix(Ġ, ▁ 등)를 제거하고 공백으로 정리.
    
    Args:
        tokenizer: Tokenizer 객체
        ids: 토큰 ID 리스트, tensor, 또는 numpy array
    """
    # tensor나 numpy array면 리스트로 변환, 이미 리스트면 그대로 사용
    if hasattr(ids, 'tolist'):
        ids_list = ids.tolist()
    elif isinstance(ids, np.ndarray):
        ids_list = ids.tolist()
    else:
        ids_list = ids
    
    human = []
    for tid in ids_list:
        # decode를 시도 (단일 토큰)
        s = tokenizer.decode([tid], skip_special_tokens=True)
        # decode 결과가 빈 문자열이거나 이상하면 raw token을 가져와서 정리
        if not s.strip() or len(s) == 0:
            raw = tokenizer.convert_ids_to_tokens([tid])[0]
            # LLaMA 계열에서 공백/단어 경계를 나타내는 기호들 정리
            for marker in ["▁", "Ġ"]:
                raw = raw.replace(marker, " ")
            s = raw.strip()
        # 특수 토큰이나 이상한 문자는 그대로 두되, 공백 정리
        s = s.replace("Ċ", "\n").replace("Ć", "").strip()
        if not s:
            s = f"<{tid}>"  # 완전히 빈 경우 token id 표시
        human.append(s)
    return human


def get_newly_unmasked_per_step(history, prompt_len, mask_id):
    """
    각 step에서 새로 unmask된 토큰 위치와 단어를 계산.
    
    반환:
        step_unmasked: List[Dict]
            각 원소는 {"step": int, "positions": List[int], "tokens": List[str], "text": str}
            positions는 response 구간 기준 인덱스 (0부터 시작)
    """
    step_unmasked = []
    
    if len(history) == 0:
        return step_unmasked
    
    prev_mask = None
    for t, h in enumerate(history):
        x_t = h["x"][0]  # (L,)
        curr_mask = (x_t == mask_id).numpy()  # (L,)
        resp_mask = curr_mask[prompt_len:]  # (L_resp,)
        
        if prev_mask is not None:
            prev_resp_mask = prev_mask[prompt_len:]  # (L_resp,)
            # 이전 step에서 mask였고, 현재 step에서 unmask된 위치
            newly_unmasked = prev_resp_mask & (~resp_mask)
            positions = np.where(newly_unmasked)[0].tolist()
            
            if len(positions) > 0:
                # 해당 위치의 토큰 ID들
                resp_ids = x_t[prompt_len:].numpy()
                token_ids = [int(resp_ids[pos]) for pos in positions]
                # 사람이 읽을 수 있는 형태로 변환
                tokens = ids_to_human_readable_tokens(
                    h.get("tokenizer", None), token_ids
                ) if "tokenizer" in h else [f"<{tid}>" for tid in token_ids]
                # 연속된 토큰들을 합쳐서 텍스트로 만들기
                text = " ".join(tokens).strip()
                
                step_unmasked.append({
                    "step": t,
                    "positions": positions,
                    "token_ids": token_ids,
                    "tokens": tokens,
                    "text": text,
                })
        
        prev_mask = curr_mask
    
    return step_unmasked


def build_mask_heatmap(history, prompt_len, mask_id):
    """
    history: generate_with_tracking 에서 반환된 history 리스트
    prompt_len: 프롬프트 길이 (응답 구간만 시각화하기 위함)
    mask_id: [MASK] 토큰 id

    반환:
        heatmap: (T, L_resp) float32
            - 0: 아직 mask 상태 (보라색)
            - 1: 이미 unmask 된 상태 (노란색)
    
    주의: Diffusion process는 denoising 과정입니다.
    - Step 0 (아래): 대부분 mask 상태 (보라색)
    - Step T-1 (위): 대부분 unmask 상태 (노란색)
    """
    if len(history) == 0:
        raise ValueError("history가 비어 있습니다.")

    T = len(history)
    x0 = history[0]["x"]  # (B, L)
    B, L = x0.shape
    assert B == 1, "현재 시각화 코드는 batch_size=1만 가정합니다."

    resp_len = L - prompt_len
    heatmap = np.zeros((T, resp_len), dtype=np.float32)

    for t, h in enumerate(history):
        x_t = h["x"][0]  # (L,)
        mask_t = (x_t == mask_id).numpy()
        # 응답 구간만 사용, mask=False(=unmask) -> 1.0, mask=True -> 0.0
        heatmap[t] = (~mask_t[prompt_len:]).astype(np.float32)

    return heatmap


def plot_mask_heatmap_with_tokens(
    history,
    first_unmask_step,
    unmasked_token_id,
    tokenizer,
    prompt_len,
    mask_id,
    figsize=None,
    title="Denoising Progress",
    save_path=None,
):
    if len(history) == 0:
        raise ValueError("history가 비어 있습니다.")

    # 1. 데이터 준비
    heatmap = build_mask_heatmap(history, prompt_len, mask_id)
    T, L_resp = heatmap.shape

    # 최종 토큰 디코딩
    final_x = history[-1]["x"][0]
    final_resp_ids = final_x[prompt_len:]
    final_resp_tokens = []
    
    for tid in final_resp_ids.tolist():
        decoded = tokenizer.decode([tid], skip_special_tokens=True)
        if not decoded.strip():
            raw = tokenizer.convert_ids_to_tokens([tid])[0]
            token_str = raw.replace("▁", " ").replace("Ġ", " ").replace("Ċ", "\n").replace("Ć", "").strip()
            if not token_str: token_str = f"<{tid}>"
        else:
            token_str = decoded.strip()
        final_resp_tokens.append(token_str)

    fus = first_unmask_step[0, prompt_len : prompt_len + L_resp].numpy()

    # 2. Figure 크기 동적 계산
    if figsize is None:
        width = max(20, L_resp * 0.3) 
        height = 12
        figsize = (width, height)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1.5], width_ratios=[4, 1], hspace=0.1, wspace=0.1)

    ax_main = fig.add_subplot(gs[0, 0])
    ax_step = fig.add_subplot(gs[1, 0], sharex=ax_main)
    
    # --- 메인 히트맵 ---
    im = ax_main.imshow(heatmap, aspect="auto", origin="lower", interpolation="nearest", cmap="viridis")
    ax_main.set_ylabel("Diffusion Step")
    ax_main.set_title(f"{title} (Length: {L_resp})")
    ax_main.grid(False)
    
    # --- 하단: First Unmask Step 시각화 ---
    fus_for_plot = fus.copy().astype(float)
    fus_for_plot[fus_for_plot < 0] = np.nan
    
    im2 = ax_step.imshow(
        fus_for_plot[None, :], 
        aspect="auto", 
        origin="lower", 
        interpolation="nearest", 
        cmap="plasma"
    )
    
    ax_step.set_yticks([])
    ax_step.set_ylabel("First Unmask\nStep Index", rotation=0, labelpad=30, va='center')
    
    # --- 핵심: X축 토큰 텍스트 ---
    ax_step.set_xticks(np.arange(L_resp))
    ax_step.set_xticklabels([]) 

    for pos in range(L_resp):
        token_str = final_resp_tokens[pos]
        if len(token_str) > 15: token_str = token_str[:13] + ".."
        
        # [수정됨] pe.withStroke 객체 사용
        ax_step.text(
            pos, 0.0,
            token_str,
            rotation=90,
            ha='center',
            va='center',
            fontsize=10,
            fontfamily='monospace',
            fontweight='bold',
            color='white',
            path_effects=[pe.withStroke(linewidth=3, foreground="black")]  # 여기가 핵심 수정 사항!
        )

    # 컬러바 추가
    cbar = fig.colorbar(im, ax=ax_main, fraction=0.02, pad=0.01)
    cbar.set_label("Unmasked(1) / Masked(0)")
    
    cbar2 = fig.colorbar(im2, ax=ax_step, fraction=0.02, pad=0.01)
    cbar2.set_label("Step Index")

    # [수정됨] tight_layout 경고 방지를 위해 제거 (savefig의 bbox_inches='tight'가 대신 처리)
    # plt.tight_layout() 
    
    if save_path:
        # bbox_inches='tight'가 여백을 자동으로 잘라줍니다.
        plt.savefig(save_path, dpi=150, bbox_inches='tight') 
        print(f"Saved visualization to {save_path}")
    
    return fig


def debug_print_step_tokens(history, tokenizer, prompt_len, target_step):
    """
    특정 diffusion step에서 응답 구간 토큰을 사람이 읽을 수 있게 프린트.
    - 괄호, 등호, 숫자 등 패턴을 육안으로 확인할 때 사용.
    """
    if target_step < 0 or target_step >= len(history):
        raise ValueError(f"target_step {target_step} 가 history 길이 {len(history)} 를 벗어났습니다.")

    x_t = history[target_step]["x"][0]  # (L,)
    resp_ids = x_t[prompt_len:]
    resp_tokens = ids_to_human_readable_tokens(tokenizer, resp_ids)

    print(f"[Step {target_step}] response tokens:")
    print(" | ".join(resp_tokens))
    # 전체를 decode해서 한 줄로도 보여주기
    full_text = tokenizer.decode(resp_ids.tolist(), skip_special_tokens=True)
    print(f"[Step {target_step}] full text: {full_text}")

