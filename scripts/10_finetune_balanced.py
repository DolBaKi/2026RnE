"""1차 파인튜닝(scripts/08)에서 나온 재앙적 망각 트레이드오프에 대한 대응 실험.

08의 결과: bear_synth(540) : reg_fit(245) ≈ 2.2:1 로 학습 배치가 곰에 치우쳐
①원본 화질 곰 검출 -0.402, ②다른 79클래스 -0.254 라는 대가를 냈다
(docs/03_results.md §3.3). 여기서 검증하는 가설은 단 하나 —
**정칙화 표본을 오버샘플링해 두 클래스 그룹의 비율을 1:1에 가깝게 맞추면
망각이 줄어드는가?** 나머지 하이퍼파라미터(freeze, lr, epoch, seed)는 08과
동일하게 고정해 이 변수 하나만 분리해서 본다.

이미 만든 평가셋(real_val/real_val_lq/reg_holdout)과 reg_fit 분할을 그대로
재사용한다 — 08과 이 스크립트가 완전히 같은 홀드아웃으로 비교돼야 하기 때문에
REG_HOLDOUT_FRAC/SEED를 08과 동일하게 유지한다.

출력
  reports/eval_v1_vs_v2.json
  reports/eval_results.md (갱신 — v1/v2 비교로 재작성)
  reports/figures/fig08_v1_vs_v2.png
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
BASE_MODEL = "yolov8n.pt"
RUNS = paths.ROOT / "runs"
TRAIN_ROOT = paths.DATASET / "train_mix_v2"
REG_DIR = paths.INTERIM / "reg_sample"
REG_HOLDOUT_FRAC = 0.30  # 08과 동일 — 같은 홀드아웃을 재사용하기 위함
SEED = 20260823          # 08과 동일 시드로 같은 reg_fit/reg_holdout 분할 재현


def coco_names() -> dict[int, str]:
    import ultralytics.cfg as C

    p = Path(C.__file__).parent / "datasets" / "coco.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))["names"]


def relabel_bear_synth_to_coco_index(src: Path, dst: Path) -> None:
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


def existing_eval_yaml(name: str) -> Path:
    p = paths.DATASET / "eval_sets" / name / "dataset.yaml"
    if not p.exists():
        raise SystemExit(f"먼저 scripts/08_train_eval_yolo.py 를 한 번 실행하세요: {p} 없음")
    return p


def build_balanced_train_mix(reg_fit_dir: Path) -> Path:
    """bear_synth(540) : reg_fit 이 1:1에 가깝도록 reg_fit 을 오버샘플링(반복 복사)."""
    for split in ("train", "val"):
        (TRAIN_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (TRAIN_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    bear_img = paths.DATASET / "bear_synth" / "images" / "train"
    bear_lbl_src = paths.DATASET / "bear_synth" / "labels" / "train"
    bear_lbl_reidx = paths.INTERIM / "generated" / "bear_synth_labels_coco_idx"
    relabel_bear_synth_to_coco_index(bear_lbl_src, bear_lbl_reidx)
    n_bear = 0
    for p in bear_img.glob("*.jpg"):
        shutil.copy(p, TRAIN_ROOT / "images" / "train" / p.name)
        n_bear += 1
    for p in bear_lbl_reidx.glob("*.txt"):
        shutil.copy(p, TRAIN_ROOT / "labels" / "train" / p.name)

    reg_images = sorted((reg_fit_dir / "images").glob("*.jpg"))
    n_reg = len(reg_images)
    factor = max(1, round(n_bear / max(n_reg, 1)))
    for k in range(factor):
        for p in reg_images:
            stem = p.stem
            lp = reg_fit_dir / "labels" / f"{stem}.txt"
            shutil.copy(p, TRAIN_ROOT / "images" / "train" / f"dup{k}_{p.name}")
            if lp.exists():
                shutil.copy(lp, TRAIN_ROOT / "labels" / "train" / f"dup{k}_{stem}.txt")
    print(f"[balance] bear_synth {n_bear}장 : reg_fit {n_reg}장 × {factor}회 반복 "
         f"= {n_reg * factor}장  (비율 {n_bear}:{n_reg*factor})")

    bear_val_img = paths.DATASET / "bear_synth" / "images" / "val"
    bear_val_lbl_src = paths.DATASET / "bear_synth" / "labels" / "val"
    bear_val_lbl_reidx = paths.INTERIM / "generated" / "bear_synth_val_labels_coco_idx"
    relabel_bear_synth_to_coco_index(bear_val_lbl_src, bear_val_lbl_reidx)
    for p in bear_val_img.glob("*.jpg"):
        shutil.copy(p, TRAIN_ROOT / "images" / "val" / p.name)
    for p in bear_val_lbl_reidx.glob("*.txt"):
        shutil.copy(p, TRAIN_ROOT / "labels" / "val" / p.name)

    yml = {"path": str(TRAIN_ROOT).replace("\\", "/"),
           "train": "images/train", "val": "images/val", "names": coco_names()}
    yml_path = TRAIN_ROOT / "dataset.yaml"
    yml_path.write_text(yaml.safe_dump(yml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return yml_path


def eval_model(model: YOLO, data_yaml: Path, tag: str) -> dict:
    m = model.val(data=str(data_yaml), classes=[BEAR_IDX], split="val",
                 verbose=False, plots=False)
    return {"tag": tag, "map50": float(m.box.map50), "map5095": float(m.box.map),
            "precision": float(m.box.mp), "recall": float(m.box.mr)}


def eval_reg(model: YOLO, data_yaml: Path, tag: str) -> dict:
    m = model.val(data=str(data_yaml), split="val", verbose=False, plots=False)
    return {"tag": tag, "map50": float(m.box.map50), "map5095": float(m.box.map)}


def main() -> int:
    real_val_yaml = existing_eval_yaml("real_val")
    real_val_lq_yaml = existing_eval_yaml("real_val_lq")
    reg_holdout_yaml = existing_eval_yaml("reg_holdout")
    reg_fit_dir = paths.DATASET / "eval_sets" / "reg_fit"
    if not reg_fit_dir.exists():
        raise SystemExit("먼저 scripts/08_train_eval_yolo.py 를 한 번 실행하세요 (reg_fit 없음)")

    prior = paths.REPORTS / "eval_baseline_vs_finetuned.json"
    if not prior.exists():
        raise SystemExit(f"08 실행 결과가 없습니다: {prior}")
    v1 = json.loads(prior.read_text(encoding="utf-8"))

    print("[build] 균형 학습 혼합셋 구성 …")
    train_yaml = build_balanced_train_mix(reg_fit_dir)

    print("[finetune v2] 동일 하이퍼파라미터, reg_fit 오버샘플링만 다르게 …")
    ft = YOLO(BASE_MODEL)
    ft.train(
        data=str(train_yaml), epochs=40, imgsz=512, batch=16,
        optimizer="AdamW", lr0=0.0008, freeze=10, patience=15, seed=SEED,
        workers=2, project=str(RUNS), name="finetune_bear_v2", exist_ok=True,
        verbose=False, plots=True,
    )

    print("[finetune v2] 평가 …")
    v2 = {
        "real_val": eval_model(ft, real_val_yaml, "finetuned_v2/real_val"),
        "real_val_lq": eval_model(ft, real_val_lq_yaml, "finetuned_v2/real_val_lq"),
        "reg_holdout": eval_reg(ft, reg_holdout_yaml, "finetuned_v2/reg_holdout"),
    }

    combined = {"baseline": v1["baseline"], "finetuned_v1": v1["finetuned"], "finetuned_v2": v2}
    out_json = paths.REPORTS / "eval_v1_vs_v2.json"
    out_json.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {out_json}")

    lines = ["# baseline vs 파인튜닝 v1(불균형) vs v2(정칙화 오버샘플링) 평가 결과", "",
            "| 세트 | 지표 | baseline | v1 | v2 | v2-v1 |", "|---|---|---|---|---|---|"]
    for k in ("real_val", "real_val_lq"):
        for metric in ("map50", "recall", "precision"):
            b = combined["baseline"][k][metric]
            v1m = combined["finetuned_v1"][k][metric]
            v2m = combined["finetuned_v2"][k][metric]
            lines.append(f"| {k} (bear) | {metric} | {b:.3f} | {v1m:.3f} | {v2m:.3f} | {v2m-v1m:+.3f} |")
    for metric in ("map50", "map5095"):
        b = combined["baseline"]["reg_holdout"][metric]
        v1m = combined["finetuned_v1"]["reg_holdout"][metric]
        v2m = combined["finetuned_v2"]["reg_holdout"][metric]
        lines.append(f"| reg_holdout (79클래스) | {metric} | {b:.3f} | {v1m:.3f} | {v2m:.3f} | {v2m-v1m:+.3f} |")
    (paths.REPORTS / "eval_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    viz.use_theme()
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    groups = ["real_val\n(원본화질)", "real_val_lq\n(열화)", "reg_holdout\n(79클래스, mAP50)"]
    b_vals = [combined["baseline"]["real_val"]["map50"], combined["baseline"]["real_val_lq"]["map50"],
             combined["baseline"]["reg_holdout"]["map50"]]
    v1_vals = [combined["finetuned_v1"]["real_val"]["map50"], combined["finetuned_v1"]["real_val_lq"]["map50"],
              combined["finetuned_v1"]["reg_holdout"]["map50"]]
    v2_vals = [combined["finetuned_v2"]["real_val"]["map50"], combined["finetuned_v2"]["real_val_lq"]["map50"],
              combined["finetuned_v2"]["reg_holdout"]["map50"]]
    x = np.arange(len(groups))
    w = 0.26
    ax.bar(x - w, b_vals, w, color=viz.BLUE, label="baseline")
    ax.bar(x, v1_vals, w, color=viz.ORANGE, label="v1 (불균형 2.2:1)")
    ax.bar(x + w, v2_vals, w, color=viz.CRITICAL, label="v2 (정칙화 오버샘플링 1:1)")
    for xi, vals in ((x - w, b_vals), (x, v1_vals), (x + w, v2_vals)):
        for xii, v in zip(xi, vals):
            ax.annotate(f"{v:.2f}", (xii, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7.5, color=viz.INK)
    ax.set_xticks(x, groups)
    ax.set_ylabel("mAP@0.5")
    ax.set_title("정칙화 오버샘플링이 재앙적 망각을 줄이는가", loc="left", pad=10)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right")
    viz.finish(ax)
    out_fig = paths.FIGURES / "fig08_v1_vs_v2.png"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"[figure] {out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
