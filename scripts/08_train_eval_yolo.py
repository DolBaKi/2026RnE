"""사전학습 YOLO(baseline) vs 생성 데이터로 파인튜닝한 YOLO(augmented) 비교.

핵심 질문: 생성한 희소 클래스 데이터가 실제로 저화질 곰 검출을 개선하는가?
COCO 80클래스 라벨 공간을 그대로 쓴다 (bear = index 21) — 그래야 파인튜닝이
"bear 하나만 아는 모델"이 아니라 "79개 클래스는 유지하면서 bear가 느는 모델"이
되고, 재앙적 망각도 같은 잣대로 잴 수 있다.

평가 3세트, 전부 학습에 쓰이지 않은 홀드아웃
  real_val      COCO val2017 실제 bear 이미지 49장 (원본 화질)
  real_val_lq   위와 동일 이미지에 열화 파이프라인 적용 (연구의 핵심 타깃 조건)
  reg_holdout   regularization sample 중 학습에 넣지 않은 나머지 → 다른 79클래스
               성능이 파인튜닝 후에도 유지되는지 확인

학습 데이터 = bear_synth(생성, class 21로 재라벨) + reg_sample 앞부분(다중 클래스,
원 라벨 유지). 저수준 특징을 보존하기 위해 백본 앞단을 freeze하고 헤드 위주로
가볍게, 낮은 lr로 파인튜닝한다(재앙적 망각 완화).

출력
  runs/finetune/…                 (ultralytics 학습 산출물)
  reports/eval_baseline_vs_finetuned.json
  reports/eval_results.md
  reports/figures/fig07_ap_comparison.png
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
TRAIN_ROOT = paths.DATASET / "train_mix"
REG_DIR = paths.INTERIM / "reg_sample"
REG_HOLDOUT_FRAC = 0.30
SEED = 20260823


def coco_names() -> dict[int, str]:
    import ultralytics.cfg as C

    p = Path(C.__file__).parent / "datasets" / "coco.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))["names"]


def relabel_bear_synth_to_coco_index(src: Path, dst: Path) -> None:
    """bear_synth 라벨은 class 0(단일 클래스) → COCO 인덱스 21로 바꿔 복사."""
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
    # ultralytics check_det_dataset 은 train/val 키가 둘 다 있어야 한다.
    # 평가 전용 세트라 둘 다 같은 이미지 폴더를 가리키게 하고 val 로만 평가한다.
    yml = {"path": str(d).replace("\\", "/"), "train": "images", "val": "images",
           "names": coco_names()}
    yml_path = d / "dataset.yaml"
    yml_path.write_text(yaml.safe_dump(yml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return yml_path


def relabel_real_bear_to_coco_index(name: str) -> Path:
    """real_val / real_val_lq 라벨(class 0)을 COCO 인덱스 21로 바꿔 평가셋 구성."""
    src_root = paths.INTERIM / "real_bear" / name
    img_root = paths.RAW / "coco" / "images" / "real_val" if name == "real_val" else src_root / "images"
    tmp_lbl = paths.INTERIM / "real_bear" / f"{name}_coco_idx_labels"
    relabel_bear_synth_to_coco_index(src_root / "labels", tmp_lbl)
    return build_eval_yaml(name, img_root, tmp_lbl)


def build_reg_holdout_yaml() -> tuple[Path, Path]:
    """reg_sample 을 fit용 앞부분 / holdout 뒷부분으로 고정 분할."""
    manifest = json.loads((REG_DIR / "manifest.json").read_text(encoding="utf-8"))
    recs = [r for r in manifest["images"] if (REG_DIR / "images" / r["file_name"]).exists()]
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(recs))
    n_hold = int(len(recs) * REG_HOLDOUT_FRAC)
    hold_recs = [recs[i] for i in idx[:n_hold]]
    fit_recs = [recs[i] for i in idx[n_hold:]]

    for tag, subset in (("reg_fit", fit_recs), ("reg_holdout", hold_recs)):
        d = paths.DATASET / "eval_sets" / tag
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "labels").mkdir(parents=True, exist_ok=True)
        for r in subset:
            stem = Path(r["file_name"]).stem
            src_img = REG_DIR / "images" / r["file_name"]
            src_lbl = REG_DIR / "labels" / f"{stem}.txt"
            if src_img.exists():
                shutil.copy(src_img, d / "images" / r["file_name"])
            if src_lbl.exists():
                shutil.copy(src_lbl, d / "labels" / f"{stem}.txt")
        yml = {"path": str(d).replace("\\", "/"), "train": "images", "val": "images",
               "names": coco_names()}
        (d / "dataset.yaml").write_text(yaml.safe_dump(yml, allow_unicode=True, sort_keys=False),
                                        encoding="utf-8")
    return (paths.DATASET / "eval_sets" / "reg_fit" / "dataset.yaml",
            paths.DATASET / "eval_sets" / "reg_holdout" / "dataset.yaml")


def build_train_mix_yaml(reg_fit_yaml: Path) -> Path:
    """bear_synth(class→21 재라벨) + reg_fit 을 한 학습셋으로 합친다.

    ultralytics의 img2label_paths()는 경로에 os.sep+'images'+os.sep 부분 문자열이
    있어야 labels 디렉터리를 찾는다 (images_val 처럼 붙여 쓰면 인식 못해 라벨 0으로
    처리됨 — 실제로 처음 겪은 버그). 그래서 images/train, images/val 처럼 표준
    YOLO 중첩 구조를 그대로 따른다.
    """
    for split in ("train", "val"):
        (TRAIN_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (TRAIN_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    bear_img = paths.DATASET / "bear_synth" / "images" / "train"
    bear_lbl_src = paths.DATASET / "bear_synth" / "labels" / "train"
    bear_lbl_reidx = paths.INTERIM / "generated" / "bear_synth_labels_coco_idx"
    relabel_bear_synth_to_coco_index(bear_lbl_src, bear_lbl_reidx)
    for p in bear_img.glob("*.jpg"):
        shutil.copy(p, TRAIN_ROOT / "images" / "train" / p.name)
    for p in bear_lbl_reidx.glob("*.txt"):
        shutil.copy(p, TRAIN_ROOT / "labels" / "train" / p.name)

    reg_fit_dir = reg_fit_yaml.parent
    for p in (reg_fit_dir / "images").glob("*.jpg"):
        shutil.copy(p, TRAIN_ROOT / "images" / "train" / p.name)
    for p in (reg_fit_dir / "labels").glob("*.txt"):
        shutil.copy(p, TRAIN_ROOT / "labels" / "train" / p.name)

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
    for req in (paths.DATASET / "bear_synth" / "manifest.jsonl", REG_DIR / "manifest.json"):
        if not req.exists():
            raise SystemExit(f"먼저 앞 단계 스크립트를 실행하세요: {req} 없음")

    print("[build] 평가셋 3종 구성 …")
    real_val_yaml = relabel_real_bear_to_coco_index("real_val")
    real_val_lq_yaml = relabel_real_bear_to_coco_index("real_val_lq")
    reg_fit_yaml, reg_holdout_yaml = build_reg_holdout_yaml()

    print("[build] 학습 혼합셋 구성 …")
    train_yaml = build_train_mix_yaml(reg_fit_yaml)

    print(f"[baseline] {BASE_MODEL} 사전학습 가중치로 평가")
    baseline = YOLO(BASE_MODEL)
    results = {
        "baseline": {
            "real_val": eval_model(baseline, real_val_yaml, "baseline/real_val"),
            "real_val_lq": eval_model(baseline, real_val_lq_yaml, "baseline/real_val_lq"),
            "reg_holdout": eval_reg(baseline, reg_holdout_yaml, "baseline/reg_holdout"),
        }
    }

    print("[finetune] 백본 동결 + 저lr 파인튜닝 …")
    ft = YOLO(BASE_MODEL)
    ft.train(
        data=str(train_yaml), epochs=40, imgsz=512, batch=16,
        optimizer="AdamW", lr0=0.0008, freeze=10, patience=15, seed=SEED,
        workers=2,  # Windows: worker 수가 높으면 여러 epoch 뒤 WinError 1450(리소스 고갈)로 죽는 경우가 있었음
        project=str(RUNS), name="finetune_bear", exist_ok=True, verbose=False, plots=True,
    )

    print("[finetune] 평가 …")
    results["finetuned"] = {
        "real_val": eval_model(ft, real_val_yaml, "finetuned/real_val"),
        "real_val_lq": eval_model(ft, real_val_lq_yaml, "finetuned/real_val_lq"),
        "reg_holdout": eval_reg(ft, reg_holdout_yaml, "finetuned/reg_holdout"),
    }

    out_json = paths.REPORTS / "eval_baseline_vs_finetuned.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {out_json}")

    lines = ["# baseline vs 파인튜닝 평가 결과", "",
            "| 세트 | 지표 | baseline | finetuned | Δ |", "|---|---|---|---|---|"]
    for k in ("real_val", "real_val_lq"):
        for metric in ("map50", "recall", "precision"):
            b, f_ = results["baseline"][k][metric], results["finetuned"][k][metric]
            lines.append(f"| {k} (bear) | {metric} | {b:.3f} | {f_:.3f} | {f_-b:+.3f} |")
    for metric in ("map50", "map5095"):
        b = results["baseline"]["reg_holdout"][metric]
        f_ = results["finetuned"]["reg_holdout"][metric]
        lines.append(f"| reg_holdout (79클래스 전체) | {metric} | {b:.3f} | {f_:.3f} | {f_-b:+.3f} |")
    (paths.REPORTS / "eval_results.md").write_text("\n".join(lines), encoding="utf-8")

    viz.use_theme()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    sets = ["real_val", "real_val_lq"]
    b_vals = [results["baseline"][s]["map50"] for s in sets]
    f_vals = [results["finetuned"][s]["map50"] for s in sets]
    x = np.arange(len(sets))
    w = 0.32
    ax.bar(x - w / 2, b_vals, w, color=viz.BLUE, label="baseline (사전학습)")
    ax.bar(x + w / 2, f_vals, w, color=viz.CRITICAL, label="finetuned (+생성데이터)")
    for xi, v in zip(x - w / 2, b_vals):
        ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8.5, color=viz.INK)
    for xi, v in zip(x + w / 2, f_vals):
        ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8.5, color=viz.INK, fontweight="bold")
    ax.set_xticks(x, ["real_val\n(원본 화질)", "real_val_lq\n(열화 적용)"])
    ax.set_ylabel("bear AP@0.5")
    ax.set_title("사전학습 vs 파인튜닝 — 실데이터 곰 검출 성능", loc="left", pad=10)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right")
    viz.finish(ax)
    out_fig = paths.FIGURES / "fig07_ap_comparison.png"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"[figure] {out_fig}")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
