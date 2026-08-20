"""이미 학습된 baseline/finetuned YOLO26 가중치로 real_lowq_restored만 재평가.

14 를 통째로 재실행(재학습 포함)할 필요가 없다 — 14가 만든 가중치를 그대로
불러와 새로 완성된 복원 이미지셋에 대해서만 평가를 다시 돈다. 14 실행 시점에는
복원이 6장만 끝난 상태라 그 결과가 무효했다(§ 진행 로그).

출력: reports/eval_yolo26.json / .md / fig12 를 올바른 값으로 갱신.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import paths, viz  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "t14", Path(__file__).resolve().parent / "14_train_eval_yolo26.py")
t14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t14)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

BEST_WEIGHTS = paths.ROOT / "runs" / "finetune_bear_yolo26" / "weights" / "best.pt"


def main() -> int:
    n_restored = len(list((paths.INTERIM / "real_lowq_restored" / "images").glob("*.jpg")))
    print(f"[check] real_lowq_restored 이미지 {n_restored}장")
    if n_restored < 100:
        raise SystemExit("복원이 아직 완료되지 않은 것 같습니다")

    yml = t14.build_real_lowq_yaml("real_lowq_restored")

    results = json.loads((paths.REPORTS / "eval_yolo26.json").read_text(encoding="utf-8"))

    baseline = YOLO(t14.BASE_MODEL)
    results["baseline"]["real_lowq_restored"] = t14.eval_bear(
        baseline, yml, "baseline/real_lowq_restored")

    ft = YOLO(str(BEST_WEIGHTS))
    results["finetuned"]["real_lowq_restored"] = t14.eval_bear(
        ft, yml, "finetuned/real_lowq_restored")

    (paths.REPORTS / "eval_yolo26.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    sets = ["real_lowq", "real_lowq_restored", "real_val"]
    lines = ["# YOLO26 — baseline vs 파인튜닝 (진짜 저화질 실사진 평가)", "",
            "| 세트 | 지표 | baseline | finetuned | Δ |", "|---|---|---|---|---|"]
    for k in sets:
        for metric in ("map50", "recall", "precision"):
            b, f_ = results["baseline"][k][metric], results["finetuned"][k][metric]
            lines.append(f"| {k} (bear) | {metric} | {b:.3f} | {f_:.3f} | {f_-b:+.3f} |")
    for metric in ("map50", "map5095"):
        b = results["baseline"]["reg_holdout"][metric]
        f_ = results["finetuned"]["reg_holdout"][metric]
        lines.append(f"| reg_holdout (79클래스) | {metric} | {b:.3f} | {f_:.3f} | {f_-b:+.3f} |")
    (paths.REPORTS / "eval_yolo26.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    viz.use_theme()
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    labels = {"real_lowq": "real_lowq\n(진짜 저화질)",
             "real_lowq_restored": "real_lowq\n+복원전처리",
             "real_val": "real_val\n(원본화질)"}
    x = np.arange(len(sets))
    w = 0.32
    b_vals = [results["baseline"][k]["map50"] for k in sets]
    f_vals = [results["finetuned"][k]["map50"] for k in sets]
    ax.bar(x - w / 2, b_vals, w, color=viz.BLUE, label="baseline (사전학습 YOLO26)")
    ax.bar(x + w / 2, f_vals, w, color=viz.CRITICAL, label="finetuned (+생성데이터 clean/deg)")
    for xi, v in zip(x - w / 2, b_vals):
        ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8.5, color=viz.INK)
    for xi, v in zip(x + w / 2, f_vals):
        ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8.5, color=viz.INK, fontweight="bold")
    ax.set_xticks(x, [labels[k] for k in sets])
    ax.set_ylabel("bear AP@0.5")
    ax.set_title("YOLO26 — 진짜 저화질 실사진 기준 baseline 대비 파인튜닝", loc="left", pad=10)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, frameon=False)
    viz.finish(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_fig = paths.FIGURES / "fig12_yolo26_results.png"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"[figure] {out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
