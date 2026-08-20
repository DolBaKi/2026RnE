"""06의 개정판 — 원본(clean) + Albumentations 열화(degraded) 쌍을 함께 저장.

방법론 3.1: "학습 단계에서는 원본 이미지뿐 아니라 블러링 및 다운스케일링이 적용된
이미지를 함께 사용" — 06은 열화본만 저장했으므로 이 요구를 충족하지 못했다.
같은 전경/배경 풀·같은 시드를 그대로 재사용해 06과 동일한 합성 결과(bbox 포함)를
얻고, 여기에 clean 사본을 추가로 저장 + 열화를 degrade.py 대신
src/rareobj/albu_degrade.py(Albumentations)로 교체한다.

bbox는 열화 여부와 무관하게 동일하다 — 열화가 리사이즈 왕복만 하고 캔버스
크기를 바꾸지 않기 때문에, clean/degraded 두 파일이 라벨 하나를 공유한다.

출력
  data/dataset/bear_synth_v2/images/{train,val}/synth_XXXX_{clean,deg}.jpg
  data/dataset/bear_synth_v2/labels/{train,val}/synth_XXXX_{clean,deg}.txt
  data/dataset/bear_synth_v2/dataset.yaml
  data/dataset/bear_synth_v2/manifest.jsonl
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import albu_degrade, compose, paths  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# 06_compose_dataset.py 와 동일 — 같은 시드로 같은 합성을 재현하기 위함
FG_DIR = paths.INTERIM / "generated" / "bear_fg"
FG_MANIFEST = paths.INTERIM / "generated" / "bear_fg_manifest.jsonl"
BG_DIR = paths.INTERIM / "generated" / "backgrounds"
BG_MANIFEST = paths.INTERIM / "generated" / "backgrounds_manifest.jsonl"

OUT = paths.DATASET / "bear_synth_v2"
CANVAS = 512
SEED = 20260821
TOTAL = 600
VAL_FRAC = 0.10
MAX_REUSE_FG = 6
MAX_REUSE_BG = 5

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


def compose_one(fg_rec: dict, bg_rec: dict, size_bin: str, rng: random.Random):
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
    albu_rng = np.random.default_rng(SEED)
    with open(paths.CONFIGS / "pipeline.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dist = cfg["target_distribution"]["size_bins"]
    albu_pipe = albu_degrade.build_pipeline()

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
        degraded = albu_degrade.degrade_bgr(composited, albu_pipe)

        name = f"synth_{i:04d}"
        for tag, img in (("clean", composited), ("deg", degraded)):
            cv2.imwrite(str(OUT / "images" / split / f"{name}_{tag}.jpg"), img,
                       [cv2.IMWRITE_JPEG_QUALITY, 95])
            write_yolo_label(OUT / "labels" / split / f"{name}_{tag}.txt", bbox_xywh, CANVAS)

        manifest.append({
            "name": name, "split": split, "fg_idx": fg_rec["idx"], "bg_idx": bg_rec["idx"],
            "species": fg_rec["species"], "scene": bg_rec["scene"],
            "time_of_day": bg_rec["time_of_day"], **meta,
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

    n_train_files = len(list((OUT / "images" / "train").glob("*.jpg")))
    n_val_files = len(list((OUT / "images" / "val").glob("*.jpg")))
    print(f"[done] train {n_train_files}장(clean+deg) / val {n_val_files}장(clean+deg)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
