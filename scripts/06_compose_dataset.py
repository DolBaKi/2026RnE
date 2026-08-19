"""곰 전경 × 배경 합성 → 열화 적용 → YOLO 데이터셋 조립.

bbox는 검출기 추론이나 마스크 임계값 재계산이 아니라 **합성에 실제로 쓴 좌표
그 자체**이므로 라벨 오차가 없다. 목표 크기 분포는 configs/pipeline.yaml 의
target_distribution.size_bins (small 0.60 / medium 0.30 / large 0.10) 를 따른다 —
COCO real_val 이 large 편중(53/71)인 것과 반대로, 결핍 구간(작고 저화질인 곰)을
의도적으로 다수로 만든다.

night_ir 배경에는 합성 전체를 그레이스케일로 바꿔 색 불일치(붙여넣기 티)를 없앤다.

출력
  data/dataset/bear_synth/images/{train,val}/*.jpg
  data/dataset/bear_synth/labels/{train,val}/*.txt   (YOLO, class 0 = bear)
  data/dataset/bear_synth/dataset.yaml
  data/dataset/bear_synth/manifest.jsonl
  reports/generated_dataset_stats.md
  reports/figures/fig06_synth_contact_sheet.png
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import compose, degrade, paths, viz  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

FG_DIR = paths.INTERIM / "generated" / "bear_fg"
FG_MANIFEST = paths.INTERIM / "generated" / "bear_fg_manifest.jsonl"
BG_DIR = paths.INTERIM / "generated" / "backgrounds"
BG_MANIFEST = paths.INTERIM / "generated" / "backgrounds_manifest.jsonl"

OUT = paths.DATASET / "bear_synth"
CANVAS = 512
SEED = 20260821
TOTAL = 600
VAL_FRAC = 0.10           # 학습 모니터링용 내부 분할. real_val(실데이터)과 무관
MAX_REUSE_FG = 6
MAX_REUSE_BG = 5

# COCO area 정의 그대로 (px^2). 512 캔버스에서 side 기준 정의.
SIZE_BIN_AREA = {
    "small": (14 ** 2, 32 ** 2),
    "medium": (32 ** 2, 96 ** 2),
    "large": (96 ** 2, 210 ** 2),
}


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def pick_size_bin(rng: random.Random, dist: dict[str, float]) -> str:
    r = rng.random()
    acc = 0.0
    for k, v in dist.items():
        acc += v
        if r <= acc:
            return k
    return list(dist)[-1]


def compose_one(fg_rec: dict, bg_rec: dict, size_bin: str, rng: random.Random) -> tuple[np.ndarray, tuple, dict]:
    fg_rgba = cv2.imread(str(FG_DIR / fg_rec["file"]), cv2.IMREAD_UNCHANGED)
    fg_bgr, alpha = fg_rgba[..., :3], fg_rgba[..., 3].astype(np.float32) / 255.0
    bg = cv2.imread(str(BG_DIR / bg_rec["file"]))
    bg = cv2.resize(bg, (CANVAS, CANVAS))

    lo, hi = SIZE_BIN_AREA[size_bin]
    target_area = rng.uniform(lo, hi)
    src_area = fg_bgr.shape[0] * fg_bgr.shape[1]
    scale = (target_area / src_area) ** 0.5
    tw = max(4, int(fg_bgr.shape[1] * scale))
    th = max(4, int(fg_bgr.shape[0] * scale))
    tw, th = min(tw, CANVAS - 2), min(th, CANVAS - 2)

    r_bgr, r_a = compose.resize_rgba(fg_bgr, alpha, tw, th)

    x = rng.randint(0, CANVAS - tw)
    y_min = int(CANVAS * 0.30)
    y = rng.randint(min(y_min, CANVAS - th), CANVAS - th)

    bg_patch = bg[y:y + th, x:x + tw]
    r_bgr = compose.match_brightness(r_bgr, r_a, bg_patch)

    composited = compose.paste(bg, r_bgr, r_a, x, y)
    if bg_rec["time_of_day"] == "night_ir":
        gray = cv2.cvtColor(composited, cv2.COLOR_BGR2GRAY)
        composited = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    meta = {"size_bin": size_bin, "target_area": round(target_area, 1),
            "scale": round(scale, 4), "bbox_xywh": [x, y, tw, th]}
    return composited, (x, y, tw, th), meta


def write_yolo_label(path: Path, bbox_xywh: tuple, canvas: int) -> None:
    x, y, w, h = bbox_xywh
    cx, cy = (x + w / 2) / canvas, (y + h / 2) / canvas
    path.write_text(f"0 {cx:.6f} {cy:.6f} {w / canvas:.6f} {h / canvas:.6f}\n", encoding="utf-8")


def main() -> int:
    if not FG_MANIFEST.exists() or not BG_MANIFEST.exists():
        raise SystemExit("먼저 scripts/04, 05 를 실행하세요 (전경/배경 생성)")

    fgs = load_jsonl(FG_MANIFEST)
    bgs = load_jsonl(BG_MANIFEST)
    print(f"[load] 전경 {len(fgs)}개 / 배경 {len(bgs)}개")

    rng = random.Random(SEED)
    with open(paths.CONFIGS / "pipeline.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dist = cfg["target_distribution"]["size_bins"]

    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    fg_use = {r["idx"]: 0 for r in fgs}
    bg_use = {r["idx"]: 0 for r in bgs}
    manifest = []
    n_val = int(TOTAL * VAL_FRAC)

    for i in range(TOTAL):
        split = "val" if i < n_val else "train"
        size_bin = pick_size_bin(rng, dist)

        fg_pool = [r for r in fgs if fg_use[r["idx"]] < MAX_REUSE_FG]
        bg_pool = [r for r in bgs if bg_use[r["idx"]] < MAX_REUSE_BG]
        fg_rec = rng.choice(fg_pool)
        bg_rec = rng.choice(bg_pool)
        fg_use[fg_rec["idx"]] += 1
        bg_use[bg_rec["idx"]] += 1

        composited, bbox_xywh, meta = compose_one(fg_rec, bg_rec, size_bin, rng)
        degraded, dparams = degrade.degrade_image(composited, rng)

        name = f"synth_{i:04d}"
        cv2.imwrite(str(OUT / "images" / split / f"{name}.jpg"), degraded,
                   [cv2.IMWRITE_JPEG_QUALITY, 92])
        write_yolo_label(OUT / "labels" / split / f"{name}.txt", bbox_xywh, CANVAS)

        manifest.append({
            "name": name, "split": split, "fg_idx": fg_rec["idx"], "bg_idx": bg_rec["idx"],
            "species": fg_rec["species"], "scene": bg_rec["scene"],
            "time_of_day": bg_rec["time_of_day"], **meta, "degrade": dparams.to_dict(),
        })
        if (i + 1) % 100 == 0:
            print(f"\r  {i+1}/{TOTAL}", end="")
    print()

    with open(OUT / "manifest.jsonl", "w", encoding="utf-8") as f:
        for rec in manifest:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    dataset_yaml = {
        "path": str(OUT).replace("\\", "/"),
        "train": "images/train", "val": "images/val",
        "names": {0: "bear"},
    }
    (OUT / "dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False),
                                      encoding="utf-8")

    achieved = {k: sum(1 for r in manifest if r["size_bin"] == k) for k in dist}
    print(f"[분포] 목표={dist}  실제={achieved}")
    print(f"[재사용] 전경 최대 {max(fg_use.values())}회 / 배경 최대 {max(bg_use.values())}회")

    lines = [
        "# 합성 데이터셋 생성 통계", "",
        f"- 총 {TOTAL}장 (train {TOTAL - n_val} / val {n_val})",
        f"- 전경 풀 {len(fgs)}개, 배경 풀 {len(bgs)}개 "
        f"(전경 재사용 최대 {max(fg_use.values())}회, 배경 최대 {max(bg_use.values())}회)",
        f"- 목표 크기 분포 {dist}", f"- 실제 크기 분포 {achieved}",
        "", "## real_val(실데이터) 대비", "",
        "| 세트 | small | medium | large |",
        "|---|---|---|---|",
        f"| real_val (실사진) | 5 (7%) | 13 (18%) | 53 (75%) |",
        f"| bear_synth (생성) | {achieved.get('small',0)} "
        f"({100*achieved.get('small',0)/TOTAL:.0f}%) | {achieved.get('medium',0)} "
        f"({100*achieved.get('medium',0)/TOTAL:.0f}%) | {achieved.get('large',0)} "
        f"({100*achieved.get('large',0)/TOTAL:.0f}%) |",
    ]
    (paths.REPORTS / "generated_dataset_stats.md").write_text("\n".join(lines), encoding="utf-8")

    # 접촉지: bbox 를 그려 육안 검수
    viz.use_theme()
    sample = manifest[n_val:][:: max(1, (TOTAL - n_val) // 12)][:12]
    fig, axes = plt.subplots(3, 4, figsize=(14, 10.5))
    for ax, rec in zip(axes.flatten(), sample):
        img = cv2.cvtColor(cv2.imread(str(OUT / "images" / "train" / f"{rec['name']}.jpg")),
                           cv2.COLOR_BGR2RGB)
        x, y, w, h = rec["bbox_xywh"]
        ax.imshow(img)
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor=viz.CRITICAL, lw=2))
        ax.set_title(f"{rec['size_bin']} / {rec['time_of_day']}", fontsize=8, color=viz.INK_2)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.flatten()[len(sample):]:
        ax.axis("off")
    fig.suptitle("합성 데이터셋 샘플 — bbox 오버레이 육안 검수", x=0.02, ha="left",
                fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_fig = paths.FIGURES / "fig06_synth_contact_sheet.png"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"[figure] {out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
