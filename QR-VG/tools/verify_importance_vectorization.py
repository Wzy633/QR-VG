import os
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.text_guided_pruning import TextGuidedPruning


def run_once(module: TextGuidedPruning, vision: torch.Tensor, text: torch.Tensor, level: int, legacy: bool):
    if legacy:
        os.environ["PRVG_IMPORTANCE_LEGACY_LOOP"] = "1"
    else:
        os.environ["PRVG_IMPORTANCE_LEGACY_LOOP"] = "0"
    with torch.inference_mode():
        out = module.compute_importance_scores(vision, text, level=level)
    return out


def bench(module: TextGuidedPruning, vision: torch.Tensor, text: torch.Tensor, level: int, legacy: bool, iters: int):
    for _ in range(10):
        _ = run_once(module, vision, text, level, legacy)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = run_once(module, vision, text, level, legacy)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    device = "cuda"
    torch.manual_seed(0)

    B = int(os.environ.get("B", "8"))
    N = int(os.environ.get("N", "13294"))
    L = int(os.environ.get("L", "20"))
    C = int(os.environ.get("C", "256"))
    level = int(os.environ.get("LEVEL", "2"))
    iters = int(os.environ.get("ITERS", "50"))

    m = TextGuidedPruning(d_model=C).to(device).eval()
    vision = torch.randn(N, B, C, device=device)
    text = torch.randn(L, B, C, device=device)

    out_new = run_once(m, vision, text, level, legacy=False)
    out_old = run_once(m, vision, text, level, legacy=True)

    diff = (out_new - out_old).abs()
    print("shape:", tuple(out_new.shape))
    print("max_abs_diff:", float(diff.max().item()))
    print("mean_abs_diff:", float(diff.mean().item()))
    print("cos_sim:", float(torch.nn.functional.cosine_similarity(out_new.flatten(), out_old.flatten(), dim=0).item()))

    t_new = bench(m, vision, text, level, legacy=False, iters=iters)
    t_old = bench(m, vision, text, level, legacy=True, iters=iters)
    print(f"new_ms: {t_new:.3f}   legacy_ms: {t_old:.3f}   speedup: {t_old / max(t_new, 1e-9):.2f}x")


if __name__ == "__main__":
    main()

