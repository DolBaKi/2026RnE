"""SDXL-Turbo로 야생동물 활동/실외 감시 도메인의 빈 배경 장면 생성.

동물이 프롬프트에 없어도 우연히 등장할 수 있어, CLIP으로 "이미 곰이 있는지"
안전장치를 건다(합성 단계에서 곰을 두 번 넣는 상황 방지).

출력
  data/interim/generated/backgrounds/{idx:04d}.jpg
  data/interim/generated/backgrounds_manifest.jsonl
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

OUT_DIR = paths.INTERIM / "generated" / "backgrounds"
MANIFEST = paths.INTERIM / "generated" / "backgrounds_manifest.jsonl"
CANVAS = 512

SCENES = [
    ("dense pine forest trail, dappled sunlight", "day"),
    ("open grassland field at dusk", "dusk"),
    ("rocky mountain slope with sparse shrubs, morning light", "day"),
    ("riverbank with pebbles and reeds, foggy morning", "day"),
    ("snowy alpine meadow, overcast sky", "day"),
    ("swamp wetland with tall reeds, midday", "day"),
    ("dirt forest path with fallen leaves, autumn", "day"),
    ("dense underbrush and ferns, deep forest shade", "day"),
    ("lake shoreline with driftwood, golden hour", "dusk"),
    ("mountain forest clearing, light rain, overcast", "day"),
]
NIGHT_SCENES = [
    "dense forest trail at night, infrared trail camera view, grainy black and white",
    "grassland field at night, infrared motion camera, monochrome, glowing eyes reflection",
    "riverside at night, night vision trail camera footage, black and white grain",
    "forest clearing at night, infrared wildlife camera, high contrast monochrome",
]

CAMERA_STYLE = ("shot on a remote trail camera, motion-activated wildlife camera, "
               "wide angle lens, natural outdoor scene, empty scene, no animals, no people")
NEG_PROMPT = ("animal, bear, wildlife, person, human, cartoon, illustration, painting, "
             "low quality, watermark, text, logo, blurry")

CLIP_LABELS = ["a photo of a bear", "an empty outdoor scene with no animal"]
BEAR_LEAK_MAX = 0.35


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-attempts", type=int, default=180)
    ap.add_argument("--seed", type=int, default=20260820)
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
    stats = {"attempt": 0, "bear_leak": 0, "kept": 0}

    for i in range(args.n_attempts):
        stats["attempt"] += 1
        is_night = rng.random() < 0.30
        if is_night:
            desc = rng.choice(NIGHT_SCENES)
            tod = "night_ir"
            prompt = f"{desc}, {CAMERA_STYLE}"
        else:
            desc, tod = SCENES[rng.integers(0, len(SCENES))]
            prompt = f"a {desc}, {CAMERA_STYLE}"

        seed = int(rng.integers(0, 2**31 - 1))
        gen = torch.Generator(device=device).manual_seed(seed)
        image = pipe(
            prompt=prompt, negative_prompt=NEG_PROMPT,
            num_inference_steps=2, guidance_scale=0.0,
            width=CANVAS, height=CANVAS, generator=gen,
        ).images[0]

        probs = clip.classify(image, CLIP_LABELS)
        if probs["a photo of a bear"] > BEAR_LEAK_MAX:
            stats["bear_leak"] += 1
            continue

        idx = stats["kept"]
        out_path = OUT_DIR / f"{idx:04d}.jpg"
        image.save(out_path, quality=95)
        manifest.append({
            "idx": idx, "file": out_path.name, "prompt": prompt, "seed": seed,
            "scene": desc, "time_of_day": tod,
            "clip_bear_leak": round(probs["a photo of a bear"], 4),
        })
        stats["kept"] += 1
        if stats["kept"] % 10 == 0:
            print(f"\r  kept {stats['kept']}  (attempt {i+1}/{args.n_attempts})", end="")

    print(f"\n[stats] {stats}")
    with open(MANIFEST, "w", encoding="utf-8") as f:
        for rec in manifest:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[write] {MANIFEST}  ({len(manifest)}개)")

    n_night = sum(1 for r in manifest if r["time_of_day"] == "night_ir")
    print(f"[분포] day/dusk={len(manifest)-n_night}  night_ir={n_night}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
