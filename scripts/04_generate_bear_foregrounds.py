"""SDXL-Turbo로 곰 전경(foreground) 생성 → GrabCut 분리 → CLIP 품질 게이트 → 중복 제거.

작은 목표 스케일로 바로 생성하지 않는 이유는 src/rareobj/compose.py 상단 설명 참고.
여기서는 "선명하고 배경이 단순한 곰"을 큰 스케일로 뽑아 정확한 컷아웃을 만드는
것까지만 담당한다. 배경 합성·리사이즈·열화는 06 스크립트가 한다.

품질 게이트 (자동 라벨링 파이프라인의 신뢰도 확보 장치)
  G1 GrabCut 분할 성공 (전경 비율 3~92%)
  G2 CLIP 판별에서 "a photo of a bear" 확률이 다른 후보 라벨보다 높고 margin 이상
  G3 근접 중복 제거 (average hash 해밍 거리 < 6 이면 중복으로 간주)

출력
  data/interim/generated/bear_fg/{idx:04d}.png   (RGBA)
  data/interim/generated/bear_fg_manifest.jsonl
  reports/figures/fig05_bear_fg_contact_sheet.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import compose, genmodels, paths, viz  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

OUT_DIR = paths.INTERIM / "generated" / "bear_fg"
MANIFEST = paths.INTERIM / "generated" / "bear_fg_manifest.jsonl"
CANVAS = 512

SPECIES = ["brown bear", "black bear", "grizzly bear", "polar bear",
           "sun bear", "sloth bear", "asiatic black bear"]
POSE = ["standing on all fours", "walking", "sitting upright", "standing on hind legs",
        "foraging with head down", "turning its head", "mid-stride walking"]
BACKDROP = ["plain light gray studio background", "seamless white background",
            "softly blurred neutral bokeh background", "plain beige studio backdrop"]

NEG_PROMPT = ("cartoon, drawing, illustration, painting, anime, toy, plush, low quality, "
              "blurry, cropped, watermark, text, logo, extra limbs, deformed, multiple animals")

CLIP_LABELS = ["a photo of a bear", "a photo of a dog", "a photo of a person",
               "an empty background with no animal", "a photo of a cat"]
CLIP_MARGIN = 0.20
HASH_DIST_MIN = 6


def build_prompt(rng: np.random.Generator) -> tuple[str, dict]:
    sp = rng.choice(SPECIES)
    po = rng.choice(POSE)
    bd = rng.choice(BACKDROP)
    prompt = (f"a full-body photo of a single {sp} {po}, {bd}, "
             f"sharp focus, wildlife photography, high detail")
    return prompt, {"species": sp, "pose": po, "backdrop": bd}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-attempts", type=int, default=220,
                    help="생성 시도 횟수 (품질 게이트 통과율 감안해 목표보다 크게)")
    ap.add_argument("--seed", type=int, default=20260819)
    args = ap.parse_args()

    paths.ensure()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[load] SDXL-Turbo …")
    pipe = genmodels.load_sdxl_turbo(device)
    print("[load] CLIP …")
    clip = genmodels.ClipScorer(device)

    rng = np.random.default_rng(args.seed)
    kept_hashes: list[int] = []
    stats = {"attempt": 0, "grabcut_fail": 0, "clip_fail": 0, "dup_fail": 0, "kept": 0}
    manifest = []

    for i in range(args.n_attempts):
        stats["attempt"] += 1
        prompt, tags = build_prompt(rng)
        seed = int(rng.integers(0, 2**31 - 1))
        gen = torch.Generator(device=device).manual_seed(seed)

        image = pipe(
            prompt=prompt, negative_prompt=NEG_PROMPT,
            num_inference_steps=2, guidance_scale=0.0,
            width=CANVAS, height=CANVAS, generator=gen,
        ).images[0]
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        mask = compose.grabcut_segment(bgr)
        if mask is None:
            stats["grabcut_fail"] += 1
            continue

        crop_bgr, alpha = compose.crop_to_mask(bgr, mask)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)

        probs = clip.classify(crop_pil, CLIP_LABELS)
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        top_label, top_p = ranked[0]
        second_p = ranked[1][1]
        if top_label != "a photo of a bear" or (top_p - second_p) < CLIP_MARGIN:
            stats["clip_fail"] += 1
            continue

        h = genmodels.average_hash(crop_pil)
        if any(genmodels.hamming(h, kh) < HASH_DIST_MIN for kh in kept_hashes):
            stats["dup_fail"] += 1
            continue
        kept_hashes.append(h)

        idx = stats["kept"]
        rgba = np.dstack([crop_bgr, (alpha * 255).astype(np.uint8)])
        out_path = OUT_DIR / f"{idx:04d}.png"
        cv2.imwrite(str(out_path), rgba)

        manifest.append({
            "idx": idx, "file": out_path.name, "prompt": prompt, "seed": seed,
            **tags, "clip_probs": {k: round(v, 4) for k, v in probs.items()},
            "mask_area_frac": float(mask.mean() / 255.0),
            "crop_w": crop_bgr.shape[1], "crop_h": crop_bgr.shape[0],
            "ahash": h,
        })
        stats["kept"] += 1
        if stats["kept"] % 10 == 0:
            print(f"\r  kept {stats['kept']}  (attempt {i+1}/{args.n_attempts})  {stats}", end="")

    print(f"\n[stats] {stats}")

    with open(MANIFEST, "w", encoding="utf-8") as f:
        for rec in manifest:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[write] {MANIFEST}  ({len(manifest)}개)")

    # 접촉지 (품질 육안 확인용)
    viz.use_theme()
    sample = manifest[:: max(1, len(manifest) // 12)][:12]
    cols = 4
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6))
    for ax, rec in zip(np.atleast_1d(axes).flatten(), sample):
        rgba = cv2.imread(str(OUT_DIR / rec["file"]), cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(rgba[..., :3], cv2.COLOR_BGR2RGB)
        checker = np.indices(rgb.shape[:2]).sum(axis=0) % 2 * 40 + 200
        bgc = np.dstack([checker] * 3).astype(np.uint8)
        a = (rgba[..., 3:4].astype(np.float32) / 255.0)
        disp = (rgb * a + bgc * (1 - a)).astype(np.uint8)
        ax.imshow(disp)
        ax.set_title(f"{rec['species']}\nclip={rec['clip_probs']['a photo of a bear']:.2f}",
                    fontsize=7.5, color=viz.INK_2)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in np.atleast_1d(axes).flatten()[len(sample):]:
        ax.axis("off")
    fig.suptitle("생성된 곰 전경 컷아웃 (품질 게이트 통과 샘플)", x=0.02, ha="left",
                fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_fig = paths.FIGURES / "fig05_bear_fg_contact_sheet.png"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"[figure] {out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
