"""7차 — 생성된 전경(8클래스)을 CCTV풍 배경에 합성 → clean+deg 쌍 데이터셋.

bear의 06(compose_dataset)과 같은 원리: bbox는 검출기 추론이 아니라 합성에
실제로 쓴 좌표 그 자체다. 다른 점은 한 이미지에 사람+동행 소지품처럼 여러
클래스가 동시에 나오는 "장면(scenario)"을 구성한다는 것 — 실제 CCTV에서
사람이 가방을 들거나 자전거를 타는 것과 같은 공동발생을 반영한다.

크기 처리: 각 클래스의 "기준 높이"(캔버스 대비 비율, 중간 거리 기준)를 정해
두고, 장면 전체에 공통으로 곱하는 거리 배율(작게/보통/크게)을 뽑는다 — 사람과
그 사람이 든 가방이 같은 거리 배율을 공유해야 상대적 크기가 어색하지 않다.

출력
  data/dataset/cctv_synth_v2/images/{train,val}/*.jpg  (clean+deg 쌍)
  data/dataset/cctv_synth_v2/labels/{train,val}/*.txt
  data/dataset/cctv_synth_v2/manifest.jsonl
  reports/figures/fig18_cctv_synth_contact_sheet.png
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import albu_degrade, compose, paths, viz  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

FG_DIR = paths.INTERIM / "generated" / "cctv_fg"
FG_MANIFEST = paths.INTERIM / "generated" / "cctv_fg_manifest.jsonl"
BG_DIR = paths.INTERIM / "generated" / "cctv_bg"
BG_MANIFEST = paths.INTERIM / "generated" / "cctv_bg_manifest.jsonl"

OUT = paths.DATASET / "cctv_synth_v2"
CANVAS = 512
SEED = 20260827
N_TRAIN = 380
N_VAL = 40
MAX_REUSE_FG = 6
MAX_REUSE_BG = 5

# 캔버스 높이 대비 "중간 거리" 기준 비율 (사람 기준으로 다른 클래스를 상대 보정)
REF_HEIGHT_FRAC = {
    "person": 0.34, "bicycle": 0.22, "motorcycle": 0.20, "car": 0.16, "bus": 0.30,
    "backpack": 0.11, "handbag": 0.07, "suitcase": 0.09,
}
DIST_SCALE = {"small": 0.55, "medium": 1.0, "large": 1.7}
DIST_WEIGHTS = {"small": 0.55, "medium": 0.32, "large": 0.13}

# (장면 구성 클래스 목록, 가중치) — 실제 CCTV에서 그럴듯한 공동발생 조합
SCENARIOS = [
    (["person"], 0.24),
    (["person", "backpack"], 0.12),
    (["person", "handbag"], 0.09),
    (["person", "suitcase"], 0.07),
    (["person", "bicycle"], 0.10),
    (["car"], 0.13),
    (["car", "person"], 0.07),
    (["motorcycle"], 0.05),
    (["motorcycle", "person"], 0.03),
    (["bus"], 0.05),
    (["bicycle"], 0.05),
]


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def coco_names() -> dict[int, str]:
    import ultralytics.cfg as C

    p = Path(C.__file__).parent / "datasets" / "coco.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))["names"]


def pick_weighted(rng: random.Random, items_weights: list[tuple]) -> object:
    items, weights = zip(*items_weights)
    return rng.choices(items, weights=weights, k=1)[0]


def compose_scene(scenario: list[str], fg_by_class: dict[str, list[dict]],
                  fg_use: dict[tuple, int], bg_rec: dict, rng: random.Random) -> tuple:
    bg = cv2.resize(cv2.imread(str(BG_DIR / bg_rec["file"])), (CANVAS, CANVAS))
    dist = pick_weighted(rng, list(DIST_WEIGHTS.items()))
    scale_mult = DIST_SCALE[dist] * rng.uniform(0.85, 1.15)

    placed = []
    base_x = rng.randint(int(CANVAS * 0.15), int(CANVAS * 0.75))
    base_y_frac = rng.uniform(0.45, 0.85)  # 지면 위치(공유 기준선)

    for i, cls_name in enumerate(scenario):
        pool = [r for r in fg_by_class[cls_name] if fg_use[(cls_name, r["idx"])] < MAX_REUSE_FG]
        if not pool:
            pool = fg_by_class[cls_name]
        rec = rng.choice(pool)
        fg_use[(cls_name, rec["idx"])] += 1

        rgba = cv2.imread(str(FG_DIR / rec["file"]), cv2.IMREAD_UNCHANGED)
        fg_bgr, alpha = rgba[..., :3], rgba[..., 3].astype(np.float32) / 255.0
        target_h = max(6, int(CANVAS * REF_HEIGHT_FRAC[cls_name] * scale_mult))
        target_w = max(6, int(fg_bgr.shape[1] * (target_h / fg_bgr.shape[0])))
        target_w, target_h = min(target_w, CANVAS - 2), min(target_h, CANVAS - 2)
        r_bgr, r_a = compose.resize_rgba(fg_bgr, alpha, target_w, target_h)

        if i == 0:
            x = min(max(base_x, 0), CANVAS - target_w)
            y = min(max(int(CANVAS * base_y_frac) - target_h, 0), CANVAS - target_h)
        else:
            # 동행 물체: 주 피사체 옆, 같은 지면선 근처에 배치
            dx = rng.randint(-target_w - 10, target_w + 10)
            x = min(max(base_x + dx, 0), CANVAS - target_w)
            y_ground = int(CANVAS * base_y_frac) + rng.randint(-8, 8)
            y = min(max(y_ground - target_h, 0), CANVAS - target_h)

        bg_patch = bg[y:y + target_h, x:x + target_w]
        r_bgr = compose.match_brightness(r_bgr, r_a, bg_patch)
        bg = compose.paste(bg, r_bgr, r_a, x, y)
        placed.append({"class_name": cls_name, "yolo_idx": rec["yolo_idx"],
                       "bbox_xywh": [x, y, target_w, target_h]})

    if bg_rec["time_of_day"] == "night_ir":
        gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        bg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return bg, placed, dist


def write_yolo(placed: list[dict], canvas: int, path: Path) -> None:
    lines = []
    for p in placed:
        x, y, w, h = p["bbox_xywh"]
        cx, cy = (x + w / 2) / canvas, (y + h / 2) / canvas
        lines.append(f"{p['yolo_idx']} {cx:.6f} {cy:.6f} {w/canvas:.6f} {h/canvas:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if not FG_MANIFEST.exists() or not BG_MANIFEST.exists():
        raise SystemExit("먼저 scripts/24, 25 실행 (전경/배경 생성)")

    fgs = load_jsonl(FG_MANIFEST)
    bgs = load_jsonl(BG_MANIFEST)
    fg_by_class: dict[str, list[dict]] = {}
    for r in fgs:
        fg_by_class.setdefault(r["class_name"], []).append(r)
    print(f"[load] 전경 {len(fgs)}개 ({ {k: len(v) for k, v in fg_by_class.items()} })  배경 {len(bgs)}개")

    rng = random.Random(SEED)
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    fg_use = {(c, r["idx"]): 0 for c, recs in fg_by_class.items() for r in recs}
    bg_use = {r["idx"]: 0 for r in bgs}
    manifest = []
    total = N_TRAIN + N_VAL
    dist_counts = {"small": 0, "medium": 0, "large": 0}

    pipe = albu_degrade.build_pipeline()

    for i in range(total):
        split = "val" if i < N_VAL else "train"
        scenario = pick_weighted(rng, SCENARIOS)
        bg_pool = [r for r in bgs if bg_use[r["idx"]] < MAX_REUSE_BG] or bgs
        bg_rec = rng.choice(bg_pool)
        bg_use[bg_rec["idx"]] += 1

        composited, placed, dist = compose_scene(scenario, fg_by_class, fg_use, bg_rec, rng)
        dist_counts[dist] += 1
        deg = albu_degrade.degrade_bgr(composited, pipe)

        name = f"synth_{i:04d}"
        cv2.imwrite(str(OUT / "images" / split / f"{name}_clean.jpg"), composited,
                   [cv2.IMWRITE_JPEG_QUALITY, 95])
        write_yolo(placed, CANVAS, OUT / "labels" / split / f"{name}_clean.txt")
        cv2.imwrite(str(OUT / "images" / split / f"{name}_deg.jpg"), deg,
                   [cv2.IMWRITE_JPEG_QUALITY, 90])
        write_yolo(placed, CANVAS, OUT / "labels" / split / f"{name}_deg.txt")

        manifest.append({"name": name, "split": split, "scenario": scenario, "dist": dist,
                         "time_of_day": bg_rec["time_of_day"],
                         "classes": [p["class_name"] for p in placed]})
        if (i + 1) % 50 == 0:
            print(f"\r  {i+1}/{total}", end="")
    print()

    (OUT / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in manifest), encoding="utf-8")
    names = coco_names()
    dataset_yaml = {"path": str(OUT).replace("\\", "/"),
                    "train": "images/train", "val": "images/val", "names": names}
    (OUT / "dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False),
                                      encoding="utf-8")

    print(f"[done] {total}쌍(clean+deg={total*2}장)  거리분포={dist_counts}")
    print(f"[재사용] 전경 최대 {max(fg_use.values())}회 / 배경 최대 {max(bg_use.values())}회")

    # 접촉지: bbox 오버레이
    viz.use_theme()
    sample = manifest[N_VAL:][:: max(1, (total - N_VAL) // 12)][:12]
    fig, axes = plt.subplots(3, 4, figsize=(14, 10.5))
    for ax, rec in zip(axes.flatten(), sample):
        img = cv2.cvtColor(cv2.imread(str(OUT / "images" / "train" / f"{rec['name']}_clean.jpg")),
                           cv2.COLOR_BGR2RGB)
        lbl_path = OUT / "labels" / "train" / f"{rec['name']}_clean.txt"
        ax.imshow(img)
        for line in lbl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cls_idx, cx, cy, w, h = map(float, line.split())
            x, y = (cx - w / 2) * CANVAS, (cy - h / 2) * CANVAS
            ax.add_patch(plt.Rectangle((x, y), w * CANVAS, h * CANVAS, fill=False,
                                       edgecolor=viz.CRITICAL, lw=1.6))
        ax.set_title(f"{'+'.join(rec['classes'])} / {rec['dist']}/{rec['time_of_day']}",
                    fontsize=8, color=viz.INK_2)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.flatten()[len(sample):]:
        ax.axis("off")
    fig.suptitle("CCTV 합성 데이터셋 샘플 — bbox 오버레이 육안 검수", x=0.02, ha="left",
                fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_fig = paths.FIGURES / "fig18_cctv_synth_contact_sheet.png"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"[figure] {out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
