"""
lm-evaluation-harness entrypoint for smoothed inverse-CDF decoding.

Moving average smoothing을 적용한 뒤 inverse CDF equal-mass partition을 수행.
  1. scores에 window-size moving average → peak 영향 확산
  2. 1/(smoothed + ε) → high attention 구간에 큰 블록 배정
  3. CDF equal-mass partition

Fixed parameters:
  - strategy = smoothed_inverse_cdf_sigmoid
  - lam = 1.0, inverse = True
  - control_mode = none
"""

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from eval_hybrid_cdf import LLaDAHybridCDFHarness


@register_model("llada_smoothed_inverse_cdf")
class LLaDASmoothedInverseCDFHarness(LLaDAHybridCDFHarness):
    def __init__(
        self,
        model_path="",
        mask_id=126336,
        max_length=4096,
        batch_size=1,
        mc_num=128,
        is_check_greedy=False,
        gen_length=256,
        steps_per_block=32,
        num_blocks=8,
        threshold=0.9,
        temperature=0.0,
        block_length=32,
        smoothing_window=32,
        seed=42,
        device="cuda",
        save_dir=None,
        show_speed=False,
        verbose=False,
        save_trace=False,
        apply_stop=False,
        **kwargs,
    ):
        # Drop advanced scheduler knobs if provided by mistake.
        kwargs.pop("strategy", None)
        kwargs.pop("lam", None)
        kwargs.pop("inverse", None)
        kwargs.pop("control_mode", None)
        kwargs.pop("control_min_size", None)
        kwargs.pop("control_max_size", None)
        kwargs.pop("scheduler_seed", None)
        kwargs.pop("seed", None)
        kwargs.pop("first_block_size", None)
        kwargs.pop("manual_block_sizes", None)
        kwargs.pop("high_score_top_k", None)
        kwargs.pop("cap_alpha", None)
        kwargs.pop("cap_b_min", None)
        kwargs.pop("cap_max_iter", None)

        super().__init__(
            model_path=model_path,
            mask_id=mask_id,
            max_length=max_length,
            batch_size=batch_size,
            mc_num=mc_num,
            is_check_greedy=is_check_greedy,
            gen_length=gen_length,
            steps_per_block=steps_per_block,
            strategy="smoothed_inverse_cdf_sigmoid",
            num_blocks=num_blocks,
            lam=1.0,
            block_length=block_length,
            temperature=temperature,
            threshold=threshold,
            inverse=True,
            control_mode="none",
            control_min_size=28,
            control_max_size=32,
            scheduler_seed=42,
            seed=seed,
            first_block_size=-1,
            manual_block_sizes="",
            high_score_top_k=None,
            smoothing_window=smoothing_window,
            cap_alpha=1.0,
            cap_b_min=8,
            cap_max_iter=50,
            device=device,
            save_dir=save_dir,
            show_speed=show_speed,
            verbose=verbose,
            save_trace=save_trace,
            apply_stop=apply_stop,
            **kwargs,
        )


if __name__ == "__main__":
    cli_evaluate()
