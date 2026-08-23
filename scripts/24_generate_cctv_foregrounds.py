"""7차 — CCTV 8클래스 전경 생성 (bear 프로젝트의 04와 동일 구조).

6차에서 놓쳤던 것: "person/car/bag 클래스 자체는 COCO에 흔하다"는 사실과
"실제 CCTV처럼 높은 각도·저화질로 찍힌 이 클래스들의 사진은 부족하다"는
사실은 별개다. bear가 "작고 저화질인 곰"이 부족했던 것과 완전히 같은
구조 — 그래서 부족한 조건(CCTV 시점)을 생성모델로 직접 만든다.

방식은 bear(scripts/04)와 동일: 스튜디오풍 배경으로 각 클래스를 선명하게
생성 → GrabCut 분리 → CLIP 품질 게이트 → 근접중복 제거. 가방류(backpack/
handbag/suitcase)는 사람이 착용한 상태가 아니라 **독립된 물체**로 생성한다
— PETS 데이터셋의 "방치된 수하물" 프레이밍과 같은 구성이라 실제 감시 상황과
자연스럽게 맞고, bear처럼 "전경 하나 = 클래스 하나 = bbox 하나"를 유지할 수
있어 라벨이 정확하다.

출력
  data/interim/generated/cctv_fg/{class_name}/{idx:04d}.png   (RGBA)
  data/interim/generated/cctv_fg_manifest.jsonl
  reports/figures/fig17_cctv_fg_contact_sheet.png
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

OUT_DIR = paths.INTERIM / "generated" / "cctv_fg"
MANIFEST = paths.INTERIM / "generated" / "cctv_fg_manifest.jsonl"
CANVAS = 512
N_PER_CLASS = 45

# (클래스명, yolo idx, 프롬프트 묘사, CLIP 판별용 라벨)
CLASSES = [
    ("person", 0, "a person standing, full body, casual clothing", "a photo of a person"),
    ("bicycle", 1, "a bicycle, side view", "a photo of a bicycle"),
    ("car", 2, "a complete car, entire vehicle fully visible in frame, not cropped, "
              "small scale within the frame, side view", "a photo of a car"),
    ("motorcycle", 3, "a motorcycle, side view", "a photo of a motorcycle"),
    ("bus", 5, "a city bus, side view", "a photo of a bus"),
    ("backpack", 24, "a backpack alone, no hands, no person, standing upright on the ground",
     "a photo of a backpack"),
    ("handbag", 26, "a handbag alone, no hands, no person, sitting on the ground",
     "a photo of a handbag"),
    ("suitcase", 28, "a suitcase alone, no hands, no person, sitting on the ground, closed",
     "a photo of a suitcase"),
]
BACKDROP = ["plain light gray studio background", "seamless white background",
           "plain beige studio backdrop", "plain pale blue studio backdrop"]
# "흐린 보케 배경"은 뺐다 — 밝은 보케 얼룩을 GrabCut이 전경으로 잘못 잡아
# 원본체와 분리된 조각을 붙여버리는 실패가 1차 시도에서 person/car/motorcycle/
# handbag/suitcase 전반에 반복됐다(§docs/06_cctv_generative.md). 완전한
# 단색/무지 배경만 남겨 그 실패 경로 자체를 없앤다.
NEG_PROMPT = ("cartoon, drawing, illustration, painting, low quality, blurry, cropped, "
             "watermark, text, logo, extra limbs, deformed, multiple objects, collage")
CLIP_MARGIN = 0.15
HASH_DIST_MIN = 6
DISTRACTOR_LABELS = ["an empty background with no object"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=N_PER_CLASS)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--classes", type=str, default=None,
                    help="쉼표구분 클래스명만 (재)생성. 미지정 시 전체 8개.")
    args = ap.parse_args()
    global CLASSES
    if args.classes:
        wanted = set(args.classes.split(","))
        CLASSES = [c for c in CLASSES if c[0] in wanted]

    paths.ensure()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[load] SDXL-Turbo …")
    pipe = genmodels.load_sdxl_turbo(device)
    print("[load] CLIP …")
    clip = genmodels.ClipScorer(device)

    all_labels = [c[3] for c in CLASSES] + DISTRACTOR_LABELS
    rng = np.random.default_rng(args.seed)
    manifest = []
    stats_all = {}

    for cls_name, yolo_idx, desc, clip_label in CLASSES:
        cls_dir = OUT_DIR / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        kept_hashes: list[int] = []
        stats = {"attempt": 0, "grabcut_fail": 0, "clip_fail": 0, "dup_fail": 0, "kept": 0}
        n_attempts = args.n_per_class

        while stats["kept"] < n_attempts and stats["attempt"] < n_attempts * 3:
            stats["attempt"] += 1
            backdrop = BACKDROP[rng.integers(0, len(BACKDROP))]
            prompt = f"{desc}, {backdrop}, sharp focus, high detail, professional photography"
            seed = int(rng.integers(0, 2**31 - 1))
            gen = torch.Generator(device=device).manual_seed(seed)

            image = pipe(prompt=prompt, negative_prompt=NEG_PROMPT, num_inference_steps=2,
                        guidance_scale=0.0, width=CANVAS, height=CANVAS, generator=gen).images[0]
            bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            mask = compose.grabcut_segment(bgr)
            if mask is None:
                stats["grabcut_fail"] += 1
                continue
            crop_bgr, alpha = compose.crop_to_mask(bgr, mask)
            crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

            probs = clip.classify(crop_pil, all_labels)
            ranked = sorted(probs.items(), key=lambda kv: -kv[1])
            top_label, top_p = ranked[0]
            second_p = ranked[1][1]
            if top_label != clip_label or (top_p - second_p) < CLIP_MARGIN:
                stats["clip_fail"] += 1
                continue

            h = genmodels.average_hash(crop_pil)
            if any(genmodels.hamming(h, kh) < HASH_DIST_MIN for kh in kept_hashes):
                stats["dup_fail"] += 1
                continue
            kept_hashes.append(h)

            idx = stats["kept"]
            rgba = np.dstack([crop_bgr, (alpha * 255).astype(np.uint8)])
            cv2.imwrite(str(cls_dir / f"{idx:04d}.png"), rgba)
            manifest.append({
                "class_name": cls_name, "yolo_idx": yolo_idx, "idx": idx,
                "file": f"{cls_name}/{idx:04d}.png", "prompt": prompt, "seed": seed,
                "clip_top_p": round(top_p, 4), "crop_w": crop_bgr.shape[1], "crop_h": crop_bgr.shape[0],
            })
            stats["kept"] += 1

        stats_all[cls_name] = stats
        print(f"[{cls_name}] {stats}")

    # --classes로 일부만 재생성한 경우, 손대지 않은 클래스의 기존 항목을 보존한다.
    if args.classes and MANIFEST.exists():
        regenerated = {c[0] for c in CLASSES}
        old = [json.loads(l) for l in MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
        manifest = [r for r in old if r["class_name"] not in regenerated] + manifest

    with open(MANIFEST, "w", encoding="utf-8") as f:
        for rec in manifest:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n[write] {MANIFEST}  (총 {len(manifest)}개)")
    print(f"[클래스별 수량] " + ", ".join(f"{c}={stats_all[c]['kept']}" for c, *_ in CLASSES))

    # 접촉지
    viz.use_theme()
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.4))
    for ax, (cls_name, *_rest) in zip(axes.flatten(), CLASSES):
        recs = [r for r in manifest if r["class_name"] == cls_name]
        if not recs:
            ax.axis("off")
            continue
        rec = recs[len(recs) // 2]
        rgba = cv2.imread(str(OUT_DIR / rec["file"]), cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(rgba[..., :3], cv2.COLOR_BGR2RGB)
        checker = np.indices(rgb.shape[:2]).sum(axis=0) % 2 * 40 + 200
        bgc = np.dstack([checker] * 3).astype(np.uint8)
        a = rgba[..., 3:4].astype(np.float32) / 255.0
        disp = (rgb * a + bgc * (1 - a)).astype(np.uint8)
        ax.imshow(disp)
        ax.set_title(f"{cls_name} (n={len(recs)})", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("CCTV 8클래스 생성 전경 샘플", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_fig = paths.FIGURES / "fig17_cctv_fg_contact_sheet.png"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"[figure] {out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
