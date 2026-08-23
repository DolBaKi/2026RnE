"""7차 — 생성 데이터(cctv_synth_v2)로 학습한 YOLO26 vs baseline, 진짜 저화질로 검증.

이번엔 진짜로 bear와 같은 실험이다: 학습 데이터가 전부 **생성모델이 만든
이미지**다(6차는 COCO 실사진을 그대로 썼다 — 그래서 부적절했다). 평가는 세
갈래로 한다.

  real_lowq_cctv           진짜 저화질 실사진 (핵심 조건, 6차와 동일 세트 재사용)
  real_lowq_cctv_restored  위에 복원 전처리 적용 (6차와 동일 세트 재사용)
  cctv_val_real            6차의 cctv_synth(v1, 실사진 기반) val 분할 — "생성
                          데이터로만 학습했는데 실사진 clean 이미지에도
                          일반화되는가"를 보는 부가 점검. 학습에 전혀 안 쓰임.

출력
  runs/finetune_cctv_gen/…
  reports/eval_cctv_gen.json / .md
  reports/figures/fig19_cctv_gen_results.png
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

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

TARGET_IDS = [0, 1, 2, 3, 5, 24, 26, 28]
BASE_MODEL = "yolo26n.pt"
RUNS = paths.ROOT / "runs"
SYNTH_V2 = paths.DATASET / "cctv_synth_v2"
SEED = 20260827


def eval_targets(model: YOLO, data_yaml: Path, tag: str) -> dict:
    m = model.val(data=str(data_yaml), classes=TARGET_IDS, split="val", verbose=False, plots=False)
    return {"tag": tag, "map50": float(m.box.map50), "map5095": float(m.box.map),
            "precision": float(m.box.mp), "recall": float(m.box.mr)}


def main() -> int:
    for req in (SYNTH_V2 / "manifest.jsonl", paths.INTERIM / "real_lowq_cctv" / "manifest.json"):
        if not req.exists():
            raise SystemExit(f"먼저 앞 단계 스크립트 실행: {req} 없음")

    real_lowq_yaml = paths.DATASET / "eval_sets" / "real_lowq_cctv" / "dataset.yaml"
    real_lowq_restored_yaml = paths.DATASET / "eval_sets" / "real_lowq_cctv_restored" / "dataset.yaml"
    cctv_val_real_yaml = paths.DATASET / "cctv_synth" / "dataset.yaml"  # 6차 산출물 재사용(실사진)
    has_restored = real_lowq_restored_yaml.exists()

    print(f"[baseline] {BASE_MODEL} 평가")
    baseline = YOLO(BASE_MODEL)
    results = {"baseline": {
        "real_lowq_cctv": eval_targets(baseline, real_lowq_yaml, "baseline/real_lowq_cctv"),
        "cctv_val_real": eval_targets(baseline, cctv_val_real_yaml, "baseline/cctv_val_real"),
    }}
    if has_restored:
        results["baseline"]["real_lowq_cctv_restored"] = eval_targets(
            baseline, real_lowq_restored_yaml, "baseline/real_lowq_cctv_restored")

    print("[finetune] YOLO26n, 생성데이터(cctv_synth_v2) clean+deg 학습 …")
    ft = YOLO(BASE_MODEL)
    ft.train(
        data=str(SYNTH_V2 / "dataset.yaml"), epochs=40, imgsz=512, batch=16,
        optimizer="AdamW", lr0=0.001, freeze=10, patience=15, seed=SEED,
        workers=2, project=str(RUNS), name="finetune_cctv_gen", exist_ok=True,
        verbose=False, plots=True,
    )

    print("[finetune] 평가 …")
    results["finetuned"] = {
        "real_lowq_cctv": eval_targets(ft, real_lowq_yaml, "finetuned/real_lowq_cctv"),
        "cctv_val_real": eval_targets(ft, cctv_val_real_yaml, "finetuned/cctv_val_real"),
    }
    if has_restored:
        results["finetuned"]["real_lowq_cctv_restored"] = eval_targets(
            ft, real_lowq_restored_yaml, "finetuned/real_lowq_cctv_restored")

    out_json = paths.REPORTS / "eval_cctv_gen.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {out_json}")

    sets = ["real_lowq_cctv"] + (["real_lowq_cctv_restored"] if has_restored else []) + ["cctv_val_real"]
    lines = ["# CCTV 생성데이터 파일럿 — baseline vs 파인튜닝(생성데이터로 학습)", "",
            "| 세트 | 지표 | baseline | finetuned | Δ |", "|---|---|---|---|---|"]
    for k in sets:
        for metric in ("map50", "recall", "precision"):
            b, f_ = results["baseline"][k][metric], results["finetuned"][k][metric]
            lines.append(f"| {k} | {metric} | {b:.3f} | {f_:.3f} | {f_-b:+.3f} |")
    (paths.REPORTS / "eval_cctv_gen.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    viz.use_theme()
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    labels = {"real_lowq_cctv": "real_lowq_cctv\n(진짜 저화질)",
             "real_lowq_cctv_restored": "real_lowq_cctv\n+복원전처리",
             "cctv_val_real": "cctv_val_real\n(실사진 clean, 학습에 안 씀)"}
    x = np.arange(len(sets))
    w = 0.32
    b_vals = [results["baseline"][k]["map50"] for k in sets]
    f_vals = [results["finetuned"][k]["map50"] for k in sets]
    ax.bar(x - w / 2, b_vals, w, color=viz.BLUE, label="baseline (사전학습 YOLO26)")
    ax.bar(x + w / 2, f_vals, w, color=viz.CRITICAL, label="finetuned (생성데이터 clean/deg)")
    for xi, v in zip(x - w / 2, b_vals):
        ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8.5, color=viz.INK)
    for xi, v in zip(x + w / 2, f_vals):
        ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8.5, color=viz.INK, fontweight="bold")
    ax.set_xticks(x, [labels[k] for k in sets])
    ax.set_ylabel("8클래스 평균 mAP@0.5")
    ax.set_title("CCTV 생성데이터 파일럿 — 진짜 저화질 기준 baseline 대비 파인튜닝", loc="left", pad=10)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right")
    viz.finish(ax)
    out_fig = paths.FIGURES / "fig19_cctv_gen_results.png"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"[figure] {out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
