"""8차(10명) — 10명 타깃 + 모르는 사람이 섞인 생성형 CCTV 인파 프레임.

인물 삽입 방식은 scripts/33(신원 조건부 인페인팅, 블러 일치, 신원 검증)을 그대로 쓴다.
프레임 하나에 타깃(10명 중 1~2명)과 "모르는 사람"(실존 인물이지만 10명이 아님)을 함께
그려 넣고, 나머지 생성 인물도 모르는 사람으로 남는다 — 실제 CCTV 검색처럼 대부분의
얼굴은 찾는 사람이 아니다.

  train  32 train 장면 중 N_TRAIN_SCENES 개 / 타깃 사진은 faces10 train 분할에서만,
         모르는 사람은 VGGFace2 나머지 470명(unknown_train)
  test   32 test 장면 전부 / 타깃 사진은 faces10 test_hq 분할(학습에 안 쓴 사진),
         모르는 사람은 eval_vgg 60명(학습에 한 번도 안 나온 사람)

라벨  YOLO 형식, cls 0~9 = 타깃, 10 = 모르는 사람(삽입한 실존 인물)
출력  data/dataset/faces10/cctv_frames/{train,test}/{images,labels,frames.jsonl}
      data/dataset/faces10/synth_cctv/<cls>/*.jpg + synth_manifest.csv  (train 프레임의 얼굴 크롭)
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rareobj import paths  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

c33 = importlib.import_module("33_compose_face_cctv_synth")
F10 = paths.DATASET / "faces10"
V1 = paths.DATASET / "faces_v1"
UNKNOWN = 10
N_TRAIN_SCENES = 600
MAX_TARGETS = {"train": 2, "test": 2}
N_UNKNOWN = 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-scenes", type=int, default=0)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--seed", type=int, default=20261001)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    split = json.loads((F10 / "split.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader(open(F10 / "manifest.csv", encoding="utf-8")))
    v1 = {r["path"]: r for r in csv.DictReader(open(V1 / "manifest.csv", encoding="utf-8"))}
    # IdentityBank 은 ROOT / r["path"] 로 읽으므로 절대경로를 넣는다
    tgt = {s: defaultdict(list) for s in ("train", "test")}
    for r in rows:
        if r["split"] == "train":
            tgt["train"][int(r["cls"])].append({"path": str(F10 / r["path"])})
        elif r["split"] == "test_hq":
            tgt["test"][int(r["cls"])].append({"path": str(F10 / r["path"])})
    unk = {s: defaultdict(list) for s in ("train", "test")}
    for p in split["unknown_train"]:
        unk["train"][v1[p]["gid"]].append({"path": str(V1 / p)})
    for p in split["unknown_test_hq"]:
        unk["test"][v1[p]["gid"]].append({"path": str(V1 / p)})

    models = c33.Models()
    banks = {(k, s): c33.IdentityBank(models, pool[s], rng) for k, pool in (("t", tgt), ("u", unk)) for s in pool}
    scenes = [json.loads(l) for l in open(c33.SCENE_MANIFEST, encoding="utf-8")]
    crop_rows = []
    stats = defaultdict(int)
    stats["cos_all"], stats["blur_sigma"] = [], []
    t0 = time.time()
    for sp in args.splits:
        sc = [s for s in scenes if s["split"] == sp]
        if sp == "train":
            sc = random.Random(1).sample(sc, min(N_TRAIN_SCENES, len(sc)))
        if args.max_scenes:
            sc = sc[:args.max_scenes]
        d = F10 / "cctv_frames" / sp
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "labels").mkdir(parents=True, exist_ok=True)
        fj = open(d / "frames.jsonl", "w", encoding="utf-8")
        use_t = defaultdict(int)
        unk_ids = list(unk[sp])
        for si, scene in enumerate(sc):
            base = cv2.imread(str(c33.SCENES / scene["file"]))
            canvas = cv2.resize(models.upscale4(base), (int(base.shape[1] * c33.F_RENDER),
                                                        int(base.shape[0] * c33.F_RENDER)), interpolation=cv2.INTER_AREA)
            slots = list(scene["slots"])
            rng.shuffle(slots)
            n_t = min(MAX_TARGETS[sp], max(1, len(slots) - N_UNKNOWN))
            plan = [("t", c) for c in sorted(range(10), key=lambda c: (use_t[c], rng.random()))[:n_t]]
            plan += [("u", rng.choice(unk_ids)) for _ in range(N_UNKNOWN)]
            faces, untouched = [], []
            for slot, (kind, who) in zip(slots, plan):
                ident = banks[(kind, sp)].get(who)
                res = None if ident is None else c33.insert_person(models, canvas, slot, ident, scene["night"], rng, stats)
                if res is None:
                    untouched.append(c33.unknown_box(slot))
                    continue
                box, cos = res
                cls = who if kind == "t" else UNKNOWN
                if kind == "t":
                    use_t[who] += 1
                faces.append({"cls": cls, "who": str(who), "box": [round(v, 1) for v in box], "id_cos": round(cos, 3)})
            untouched += [c33.unknown_box(s) for s in slots[len(plan):]]
            if not any(f["cls"] != UNKNOWN for f in faces):
                stats["frame_no_target"] += 1
                continue
            frame, p = c33.frame_degrade(canvas, rng)
            name = Path(scene["file"]).stem
            cv2.imwrite(str(d / "images" / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            Hh, Ww = frame.shape[:2]
            with open(d / "labels" / f"{name}.txt", "w", encoding="utf-8") as lf:
                for fc in faces:
                    b = fc["box"]
                    lf.write(f"{fc['cls']} {(b[0] + b[2]) / 2 / Ww:.6f} {(b[1] + b[3]) / 2 / Hh:.6f} "
                             f"{(b[2] - b[0]) / Ww:.6f} {(b[3] - b[1]) / Hh:.6f}\n")
            fj.write(json.dumps({"image": f"images/{name}.jpg", "scene": scene["file"], "size": [Ww, Hh],
                                 "night": scene["night"], "faces": faces, "untouched_frontal": untouched,
                                 "degrade": p.to_dict()}, ensure_ascii=False) + "\n")
            fj.flush()
            stats[f"{sp}_frames"] += 1
            stats[f"{sp}_targets"] += sum(f["cls"] != UNKNOWN for f in faces)
            stats[f"{sp}_unknowns"] += sum(f["cls"] == UNKNOWN for f in faces)
            if sp == "train":
                for k, fc in enumerate(faces):
                    rel = Path("synth_cctv") / f"{fc['cls']:02d}" / f"{name}_{k}.jpg"
                    (F10 / rel).parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(F10 / rel), c33.crop_box(frame, fc["box"]), [cv2.IMWRITE_JPEG_QUALITY, 95])
                    crop_rows.append({"path": rel.as_posix(), "cls": fc["cls"], "who": fc["who"],
                                      "frame": f"cctv_frames/train/images/{name}.jpg",
                                      "face_px": round(fc["box"][2] - fc["box"][0], 1), "id_cos": fc["id_cos"]})
            if si % 20 == 0:
                c = stats["cos_all"]
                summ = {k: v for k, v in stats.items() if not isinstance(v, list)}
                print(f"  [{sp}] scene {si}/{len(sc)}  {summ}  id_cos med "
                      f"{np.median(c) if c else float('nan'):.3f}  {time.time() - t0:.0f}s", flush=True)
        fj.close()
        if sp == "train":
            print(f"  target count per class: {dict(sorted(use_t.items()))}")

    if crop_rows:
        with open(F10 / "synth_manifest.csv", "w", newline="", encoding="utf-8") as fo:
            wr = csv.DictWriter(fo, fieldnames=list(crop_rows[0]))
            wr.writeheader()
            wr.writerows(crop_rows)
    c = np.array(stats["cos_all"])
    print(f"[done] {({k: v for k, v in stats.items() if not isinstance(v, list)})}  {time.time() - t0:.0f}s")
    if len(c):
        print(f"  id_cos q10 {np.quantile(c, .1):.3f}  med {np.median(c):.3f}  통과율 {(c >= c33.ID_COS_MIN).mean():.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
