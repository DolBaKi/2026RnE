"""7차 — CCTV 시점 배경 장면 생성 (bear의 05와 동일 구조).

거리·주차장·건물 입구 등 감시카메라가 실제로 설치되는 장소를, 감시카메라
특유의 높은 각도·넓은 화각으로 생성한다. 프롬프트에 사람/차량을 넣지 않지만
우연히 등장할 수 있어 CLIP으로 안전장치를 건다.

출력
  data/interim/generated/cctv_bg/{idx:04d}.jpg
  data/interim/generated/cctv_bg_manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import genmodels, paths  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

OUT_DIR = paths.INTERIM / "generated" / "cctv_bg"
MANIFEST = paths.INTERIM / "generated" / "cctv_bg_manifest.jsonl"
CANVAS = 512

SCENES = [
    ("empty city street, daytime", "day"),
    ("parking lot, daytime, overcast", "day"),
    ("building entrance and sidewalk, daytime", "day"),
    ("subway station platform, empty", "day"),
    ("shopping street, daytime", "day"),
    ("alleyway between buildings, daytime", "day"),
    ("bus stop and road, daytime", "day"),
    ("apartment complex courtyard, daytime", "day"),
    ("street corner with crosswalk, dusk", "dusk"),
    ("parking garage interior, fluorescent lighting", "day"),
]
NIGHT_SCENES = [
    "empty city street at night, infrared security camera view, grainy black and white",
    "parking lot at night, infrared cctv camera, monochrome, high contrast",
    "building entrance at night, night vision security camera footage",
    "alleyway at night, infrared surveillance camera, grainy monochrome",
]
CAMERA_STYLE = ("shot from an elevated security camera, wide angle lens, high angle "
               "surveillance view, empty scene, no people, no vehicles")
NEG_PROMPT = ("person, people, human, car, vehicle, bicycle, cartoon, illustration, "
             "painting, low quality, watermark, text, logo, blurry")
CLIP_LABELS = ["a photo containing a person or vehicle", "an empty scene with no person or vehicle"]
LEAK_MAX = 0.35


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-attempts", type=int, default=170)
    ap.add_argument("--seed", type=int, default=20260826)
    args = ap.parse_args()

    paths.ensure()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[load] SDXL-Turbo …")
    pipe = genmodels.load_sdxl_turbo(device)
    print("[load] CLIP …")
    clip = genmodels.ClipScorer(device)

    rng = np.random.default_rng(args.seed)
    manifest = []
    stats = {"attempt": 0, "leak": 0, "kept": 0}

    for i in range(args.n_attempts):
        stats["attempt"] += 1
        is_night = rng.random() < 0.30
        if is_night:
            desc = NIGHT_SCENES[rng.integers(0, len(NIGHT_SCENES))]
            tod = "night_ir"
            prompt = f"{desc}, {CAMERA_STYLE}"
        else:
            desc, tod = SCENES[rng.integers(0, len(SCENES))]
            prompt = f"{desc}, {CAMERA_STYLE}"

        seed = int(rng.integers(0, 2**31 - 1))
        gen = torch.Generator(device=device).manual_seed(seed)
        image = pipe(prompt=prompt, negative_prompt=NEG_PROMPT, num_inference_steps=2,
                    guidance_scale=0.0, width=CANVAS, height=CANVAS, generator=gen).images[0]

        probs = clip.classify(image, CLIP_LABELS)
        if probs["a photo containing a person or vehicle"] > LEAK_MAX:
            stats["leak"] += 1
            continue

        idx = stats["kept"]
        out_path = OUT_DIR / f"{idx:04d}.jpg"
        image.save(out_path, quality=95)
        manifest.append({"idx": idx, "file": out_path.name, "prompt": prompt, "seed": seed,
                        "scene": desc, "time_of_day": tod})
        stats["kept"] += 1
        if stats["kept"] % 10 == 0:
            print(f"\r  kept {stats['kept']}  (attempt {i+1}/{args.n_attempts})", end="")

    print(f"\n[stats] {stats}")
    with open(MANIFEST, "w", encoding="utf-8") as f:
        for rec in manifest:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    n_night = sum(1 for r in manifest if r["time_of_day"] == "night_ir")
    print(f"[write] {MANIFEST}  ({len(manifest)}개, day/dusk={len(manifest)-n_night} night_ir={n_night})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
