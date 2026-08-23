"""6차 CCTV 파일럿 — baseline YOLO26 vs clean+deg 파인튜닝, 8클래스 동시 평가.

bear(1~5차)와 다른 점은 딱 하나다 — 대상이 희소 클래스 1개가 아니라 흔한
클래스 8개(사람+동행 소지품/이동수단)라 재앙적 망각을 걱정할 필요가 없다
(다른 클래스를 밀어낼 이유가 없다. 애초에 밀어낼 "다른 클래스 전용 데이터"가
학습 배치에 없다). 그래서 reg_fit 같은 정칙화 표본을 섞지 않는다 — 이 8개
클래스 자체가 이미 COCO 최빈 클래스들이라 사전학습 가중치가 원래 이 클래스들에
강하고, 우리는 "화질이 나빠져도 그 강함을 유지하는가"만 본다.

평가 3세트 (전부 홀드아웃, cctv_synth 학습에 등장한 적 없음)
  real_lowq_cctv           원래부터 저화질/블러인 실제 COCO 사진 (핵심 조건)
  real_lowq_cctv_restored  위에 NAFNet+Real-ESRGAN 복원 적용 (5차와 동일 절차로 재검증)
  cctv_val                 cctv_synth의 clean val 분할 (참고용, 학습 분포 내부 성능)

출력
  runs/finetune_cctv/…
  reports/eval_cctv.json / eval_cctv.md
  reports/figures/fig15_cctv_results.png
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import paths, viz  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

TARGET_IDS = [0, 1, 2, 3, 5, 24, 26, 28]  # person,bicycle,car,motorcycle,bus,backpack,handbag,suitcase
BASE_MODEL = "yolo26n.pt"
RUNS = paths.ROOT / "runs"
SYNTH = paths.DATASET / "cctv_synth"
SEED = 20260824


def coco_names() -> dict[int, str]:
    import ultralytics.cfg as C

    p = Path(C.__file__).parent / "datasets" / "coco.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))["names"]


def build_eval_yaml(name: str, img_dir: Path, lbl_dir: Path) -> Path:
    d = paths.DATASET / "eval_sets" / name
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(parents=True, exist_ok=True)
    for p in img_dir.glob("*.jpg"):
        dst = d / "images" / p.name
        if not dst.exists():
            shutil.copy(p, dst)
    for p in lbl_dir.glob("*.txt"):
        dst = d / "labels" / p.name
        if not dst.exists():
            shutil.copy(p, dst)
    yml = {"path": str(d).replace("\\", "/"), "train": "images", "val": "images",
           "names": coco_names()}
    (d / "dataset.yaml").write_text(yaml.safe_dump(yml, allow_unicode=True, sort_keys=False),
                                    encoding="utf-8")
    return d / "dataset.yaml"


def eval_targets(model: YOLO, data_yaml: Path, tag: str) -> dict:
    m = model.val(data=str(data_yaml), classes=TARGET_IDS, split="val", verbose=False, plots=False)
    return {"tag": tag, "map50": float(m.box.map50), "map5095": float(m.box.map),
            "precision": float(m.box.mp), "recall": float(m.box.mr)}


def main() -> int:
    for req in (SYNTH / "manifest.jsonl", paths.INTERIM / "real_lowq_cctv" / "manifest.json"):
        if not req.exists():
            raise SystemExit(f"먼저 앞 단계 스크립트 실행: {req} 없음")

    print("[build] 평가셋 구성 …")
    real_lowq_yaml = build_eval_yaml("real_lowq_cctv",
                                     paths.INTERIM / "real_lowq_cctv" / "images",
                                     paths.INTERIM / "real_lowq_cctv" / "labels")
    has_restored = (paths.INTERIM / "real_lowq_cctv_restored" / "images").exists() and \
        len(list((paths.INTERIM / "real_lowq_cctv_restored" / "images").glob("*.jpg"))) > 0
    if has_restored:
        real_lowq_restored_yaml = build_eval_yaml(
            "real_lowq_cctv_restored", paths.INTERIM / "real_lowq_cctv_restored" / "images",
            paths.INTERIM / "real_lowq_cctv" / "labels")  # 복원은 이미지만 바뀌고 라벨(bbox)은 동일
    cctv_val_yaml = SYNTH / "dataset.yaml"  # val split 그대로 평가용으로도 사용

    print(f"[baseline] {BASE_MODEL} 평가")
    baseline = YOLO(BASE_MODEL)
    results = {"baseline": {
        "real_lowq_cctv": eval_targets(baseline, real_lowq_yaml, "baseline/real_lowq_cctv"),
        "cctv_val": eval_targets(baseline, cctv_val_yaml, "baseline/cctv_val"),
    }}
    if has_restored:
        results["baseline"]["real_lowq_cctv_restored"] = eval_targets(
            baseline, real_lowq_restored_yaml, "baseline/real_lowq_cctv_restored")

    print("[finetune] YOLO26n, clean+deg 혼합 학습 …")
    ft = YOLO(BASE_MODEL)
    ft.train(
        data=str(SYNTH / "dataset.yaml"), epochs=40, imgsz=512, batch=16,
        optimizer="AdamW", lr0=0.001, freeze=10, patience=15, seed=SEED,
        workers=2, project=str(RUNS), name="finetune_cctv", exist_ok=True,
        verbose=False, plots=True,
    )

    print("[finetune] 평가 …")
    results["finetuned"] = {
        "real_lowq_cctv": eval_targets(ft, real_lowq_yaml, "finetuned/real_lowq_cctv"),
        "cctv_val": eval_targets(ft, cctv_val_yaml, "finetuned/cctv_val"),
    }
    if has_restored:
        results["finetuned"]["real_lowq_cctv_restored"] = eval_targets(
            ft, real_lowq_restored_yaml, "finetuned/real_lowq_cctv_restored")

    out_json = paths.REPORTS / "eval_cctv.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {out_json}")

    sets = ["real_lowq_cctv"] + (["real_lowq_cctv_restored"] if has_restored else []) + ["cctv_val"]
    lines = ["# CCTV 파일럿 — baseline vs 파인튜닝 (8클래스 person/vehicle/carrying-object)", "",
            "| 세트 | 지표 | baseline | finetuned | Δ |", "|---|---|---|---|---|"]
    for k in sets:
        for metric in ("map50", "recall", "precision"):
            b, f_ = results["baseline"][k][metric], results["finetuned"][k][metric]
            lines.append(f"| {k} | {metric} | {b:.3f} | {f_:.3f} | {f_-b:+.3f} |")
    (paths.REPORTS / "eval_cctv.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    viz.use_theme()
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    labels = {"real_lowq_cctv": "real_lowq_cctv\n(진짜 저화질)",
             "real_lowq_cctv_restored": "real_lowq_cctv\n+복원전처리",
             "cctv_val": "cctv_val\n(clean, 학습분포 내)"}
    x = np.arange(len(sets))
    w = 0.32
    b_vals = [results["baseline"][k]["map50"] for k in sets]
    f_vals = [results["finetuned"][k]["map50"] for k in sets]
    ax.bar(x - w / 2, b_vals, w, color=viz.BLUE, label="baseline (사전학습 YOLO26)")
    ax.bar(x + w / 2, f_vals, w, color=viz.CRITICAL, label="finetuned (+clean/deg)")
    for xi, v in zip(x - w / 2, b_vals):
        ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8.5, color=viz.INK)
    for xi, v in zip(x + w / 2, f_vals):
        ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8.5, color=viz.INK, fontweight="bold")
    ax.set_xticks(x, [labels[k] for k in sets])
    ax.set_ylabel("8클래스 평균 mAP@0.5")
    ax.set_title("CCTV 파일럿 — 진짜 저화질 기준 baseline 대비 파인튜닝", loc="left", pad=10)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right")
    viz.finish(ax)
    out_fig = paths.FIGURES / "fig15_cctv_results.png"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"[figure] {out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
