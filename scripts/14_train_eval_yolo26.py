"""YOLO26 로 재구성한 최종 파이프라인 검증.

방법론(3.1~3.3) 반영 사항
  - 검출 모델: YOLO26n (COCO 사전학습)
  - 학습 데이터: bear_synth_v2의 clean + Albumentations 열화 쌍을 모두 사용
    (class→COCO index 21 재라벨) + reg_fit(다른 79클래스, 재앙적 망각 점검용,
    scripts/08과 동일 분할 재사용)
  - 평가: 학습에 전혀 등장하지 않는 4세트, 전부 홀드아웃
      real_lowq            원래부터 저화질/블러인 실제 COCO 사진 (핵심 평가 조건)
      real_lowq_restored   위에 NAFNet+Real-ESRGAN 복원을 적용한 버전
                          (3.2 — 파이프라인에 포함되는 보조 전처리 단계의 효과 확인)
      real_val             원본 화질 실사진 (참고용 — 화질 저하가 없을 때 성능 유지 확인)
      reg_holdout          다른 79클래스 (재앙적 망각 점검)

출력
  runs/finetune_bear_yolo26/…
  reports/eval_yolo26.json
  reports/eval_yolo26.md
  reports/figures/fig12_yolo26_results.png
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

BEAR_IDX = 21
BASE_MODEL = "yolo26n.pt"
RUNS = paths.ROOT / "runs"
TRAIN_ROOT = paths.DATASET / "train_mix_v3"
SYNTH_V2 = paths.DATASET / "bear_synth_v2"
REG_FIT_DIR = paths.DATASET / "eval_sets" / "reg_fit"
SEED = 20260823


def coco_names() -> dict[int, str]:
    import ultralytics.cfg as C

    p = Path(C.__file__).parent / "datasets" / "coco.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))["names"]


def relabel_dir(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for lp in src.glob("*.txt"):
        lines = []
        for line in lp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            parts[0] = str(BEAR_IDX)
            lines.append(" ".join(parts))
        (dst / lp.name).write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    yml_path = d / "dataset.yaml"
    yml_path.write_text(yaml.safe_dump(yml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return yml_path


def build_real_lowq_yaml(name: str) -> Path:
    """real_lowq / real_lowq_restored 라벨(class 0)을 COCO 인덱스 21로 재라벨."""
    src_root = paths.INTERIM / name
    tmp_lbl = paths.INTERIM / f"{name}_coco_idx_labels"
    relabel_dir(src_root / "labels", tmp_lbl)
    return build_eval_yaml(name, src_root / "images", tmp_lbl)


def build_real_val_yaml() -> Path:
    src_root = paths.INTERIM / "real_bear" / "real_val"
    tmp_lbl = paths.INTERIM / "real_bear" / "real_val_coco_idx_labels_v2"
    relabel_dir(src_root / "labels", tmp_lbl)
    return build_eval_yaml("real_val_v2", paths.RAW / "coco" / "images" / "real_val", tmp_lbl)


def build_train_mix() -> Path:
    """bear_synth_v2(clean+deg, class→21) + reg_fit(원 라벨)."""
    for split in ("train", "val"):
        (TRAIN_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (TRAIN_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        src_img = SYNTH_V2 / "images" / split
        src_lbl = SYNTH_V2 / "labels" / split
        lbl_reidx = paths.INTERIM / "generated" / f"bear_synth_v2_{split}_coco_idx"
        relabel_dir(src_lbl, lbl_reidx)
        for p in src_img.glob("*.jpg"):
            shutil.copy(p, TRAIN_ROOT / "images" / split / p.name)
        for p in lbl_reidx.glob("*.txt"):
            shutil.copy(p, TRAIN_ROOT / "labels" / split / p.name)

    n_bear_train = len(list((TRAIN_ROOT / "images" / "train").glob("*.jpg")))
    for p in (REG_FIT_DIR / "images").glob("*.jpg"):
        shutil.copy(p, TRAIN_ROOT / "images" / "train" / p.name)
    for p in (REG_FIT_DIR / "labels").glob("*.txt"):
        shutil.copy(p, TRAIN_ROOT / "labels" / "train" / p.name)
    n_reg = len(list((REG_FIT_DIR / "images").glob("*.jpg")))
    print(f"[mix] bear_synth_v2(clean+deg) train {n_bear_train}장 + reg_fit {n_reg}장")

    yml = {"path": str(TRAIN_ROOT).replace("\\", "/"),
           "train": "images/train", "val": "images/val", "names": coco_names()}
    yml_path = TRAIN_ROOT / "dataset.yaml"
    yml_path.write_text(yaml.safe_dump(yml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return yml_path


def eval_bear(model: YOLO, data_yaml: Path, tag: str) -> dict:
    m = model.val(data=str(data_yaml), classes=[BEAR_IDX], split="val", verbose=False, plots=False)
    return {"tag": tag, "map50": float(m.box.map50), "map5095": float(m.box.map),
            "precision": float(m.box.mp), "recall": float(m.box.mr)}


def eval_reg(model: YOLO, data_yaml: Path, tag: str) -> dict:
    m = model.val(data=str(data_yaml), split="val", verbose=False, plots=False)
    return {"tag": tag, "map50": float(m.box.map50), "map5095": float(m.box.map)}


def main() -> int:
    for req in (SYNTH_V2 / "manifest.jsonl", REG_FIT_DIR, paths.INTERIM / "real_lowq" / "manifest.json"):
        if not Path(req).exists():
            raise SystemExit(f"먼저 앞 단계 스크립트 실행: {req} 없음")

    print("[build] 평가셋 구성 …")
    real_lowq_yaml = build_real_lowq_yaml("real_lowq")
    has_restored = (paths.INTERIM / "real_lowq_restored" / "images").exists() and \
        len(list((paths.INTERIM / "real_lowq_restored" / "images").glob("*.jpg"))) > 0
    if has_restored:
        real_lowq_restored_yaml = build_real_lowq_yaml("real_lowq_restored")
    real_val_yaml = build_real_val_yaml()
    reg_holdout_yaml = paths.DATASET / "eval_sets" / "reg_holdout" / "dataset.yaml"

    print("[build] 학습 혼합셋 구성 …")
    train_yaml = build_train_mix()

    print(f"[baseline] {BASE_MODEL} 사전학습 가중치로 평가")
    baseline = YOLO(BASE_MODEL)
    results = {"baseline": {
        "real_lowq": eval_bear(baseline, real_lowq_yaml, "baseline/real_lowq"),
        "real_val": eval_bear(baseline, real_val_yaml, "baseline/real_val"),
        "reg_holdout": eval_reg(baseline, reg_holdout_yaml, "baseline/reg_holdout"),
    }}
    if has_restored:
        results["baseline"]["real_lowq_restored"] = eval_bear(
            baseline, real_lowq_restored_yaml, "baseline/real_lowq_restored")

    print("[finetune] YOLO26n, clean+deg 혼합 학습 …")
    ft = YOLO(BASE_MODEL)
    ft.train(
        data=str(train_yaml), epochs=40, imgsz=512, batch=16,
        optimizer="AdamW", lr0=0.0008, freeze=10, patience=15, seed=SEED,
        workers=2, project=str(RUNS), name="finetune_bear_yolo26", exist_ok=True,
        verbose=False, plots=True,
    )

    print("[finetune] 평가 …")
    results["finetuned"] = {
        "real_lowq": eval_bear(ft, real_lowq_yaml, "finetuned/real_lowq"),
        "real_val": eval_bear(ft, real_val_yaml, "finetuned/real_val"),
        "reg_holdout": eval_reg(ft, reg_holdout_yaml, "finetuned/reg_holdout"),
    }
    if has_restored:
        results["finetuned"]["real_lowq_restored"] = eval_bear(
            ft, real_lowq_restored_yaml, "finetuned/real_lowq_restored")

    out_json = paths.REPORTS / "eval_yolo26.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {out_json}")

    sets = ["real_lowq"] + (["real_lowq_restored"] if has_restored else []) + ["real_val"]
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
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
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
    ax.legend(loc="upper right")
    viz.finish(ax)
    out_fig = paths.FIGURES / "fig12_yolo26_results.png"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"[figure] {out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
