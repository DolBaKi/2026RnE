"""9차 — 실제 CCTV 프레임의 얼굴 박스 안 얼굴을 inswapper_128 로 바꿔 학습 데이터를 늘린다.

입력  data/dataset/CCTV/pseudo_labels.jsonl (scripts/60)
규칙
  - 학습 카메라 프레임만 스왑한다. 연속 프레임이 거의 같아 TRAIN_STRIDE 장마다 하나씩 쓰고,
    원본 1장 + 스왑본 1장을 같은 라벨로 저장한다(스왑해도 박스는 그대로).
  - 바꿀 얼굴: SCRFD 점수 ≥ 0.6, 정면(scripts/60 is_frontal), 폭 ≥ 10px. 고개 숙인 얼굴에 스왑하면
    얼굴이 뭉개져 오히려 흔적이 된다(시험 프레임에서 확인).
  - 넣을 신원: faces_v1 학습 신원(VGGFace2 480명) 중 CLIP 남성 + faces10 남성 타깃. 이 영상은 남성
    예배 구역이라 여성 소스를 넣으면 수염이 사라지는 등 티가 났다. 타깃(cls 0~9)은 나중에 인식
    학습에 쓰도록 메타에 남긴다.
  - val/test 카메라는 스왑 없이 원본만 (검출기 평가용).
  - 원본과 스왑본을 같은 JPEG 품질로 저장 — 저장 형식 차이가 지름길이 되지 않게.

출력  data/dataset/CCTV_work/yolo/{images,labels}/{train,val,test}/   YOLO 형식(class 0 = face)
      data/dataset/CCTV_work/yolo/data.yaml, meta.jsonl
      data/dataset/CCTV_work/swap_sources.npz, previews/swap_zoom_*.jpg
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import genmodels, paths  # noqa: E402
from rareobj.cctv_faceswap import SCRFD, InSwapper  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

CCTV = paths.DATASET / "CCTV"
WORK = paths.DATASET / "CCTV_work"
YOLO = WORK / "yolo"
V1, F10 = paths.DATASET / "faces_v1", paths.DATASET / "faces10"
TRAIN_STRIDE, EVAL_STRIDE = 3, 5
LABEL_THR = 0.5
SWAP_THR, SWAP_MIN_W, SWAP_P, SWAP_MAX = 0.6, 10.0, 0.6, 25
TARGET_P = 0.25          # 바꾸는 얼굴 중 타깃(cls 0~9) 비율, 나머지는 모르는 사람(cls 10)
JPEG_Q = 95
UNKNOWN = 10


def build_sources(det: SCRFD, sw: InSwapper) -> dict:
    cache = WORK / "swap_sources.npz"
    if cache.exists():
        return dict(np.load(cache, allow_pickle=True))
    clip = genmodels.ClipScorer("cuda")
    labels = ["a photo of a man", "a photo of a woman"]

    def emb_of(paths_):
        es = []
        for p in paths_:
            im = cv2.copyMakeBorder(cv2.imread(str(p)), 80, 80, 80, 80, cv2.BORDER_CONSTANT)
            b, s, k = det.detect(im, scale=1.0, thr=0.5)
            if len(b):
                es.append(sw.embed(im, k[int(np.argmax(s))]))
        if not es:
            return None
        e = np.mean(es, 0)
        return e / np.linalg.norm(e)

    names, embs, cls = [], [], []
    rows = list(csv.DictReader(open(V1 / "manifest.csv", encoding="utf-8")))
    f10 = list(csv.DictReader(open(F10 / "manifest.csv", encoding="utf-8")))
    # faces10 타깃 10명은 faces_v1 학습 신원에서 뽑혔다 → 모르는 사람(cls 10) 풀에서 빼야 같은 사람이
    # 타깃이면서 모르는 사람으로 들어가지 않는다
    targets = {x["name"] for x in f10}
    ids = [r for r in csv.DictReader(open(V1 / "identities.csv", encoding="utf-8"))
           if r["role"] == "train" and r["name"] not in targets]
    for r in ids:
        imgs = [V1 / x["path"] for x in rows if x["gid"] == r["gid"]][:3]
        pm = clip.classify(Image.open(imgs[0]).convert("RGB"), labels)[labels[0]]
        if pm < 0.7:
            continue
        e = emb_of(imgs)
        if e is not None:
            names.append(r["name"]); embs.append(e); cls.append(UNKNOWN)
    for c in range(10):
        rr = [x for x in f10 if int(x["cls"]) == c and x["split"] == "train"]
        if rr[0]["gender"] != "M":
            continue
        e = emb_of([F10 / x["path"] for x in rr[:5]])
        if e is not None:
            names.append(rr[0]["name"]); embs.append(e); cls.append(c)
    out = {"names": np.array(names), "embs": np.stack(embs).astype(np.float32), "cls": np.array(cls)}
    np.savez(cache, **out)
    return out


def yolo_lines(rec) -> list[str]:
    W, H = rec["w"], rec["h"]
    out = []
    for (x0, y0, x1, y1), s in zip(rec["boxes"], rec["scores"]):
        if s < LABEL_THR:
            continue
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        out.append(f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
    return out


def zoom_sheet(pairs, path):
    tiles = []
    for a, o in pairs[:12]:
        z = lambda im: cv2.resize(im, (128, 128), interpolation=cv2.INTER_NEAREST)
        tiles.append(np.hstack([z(a), np.full((128, 4, 3), 255, np.uint8), z(o)]))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack([np.pad(t, ((4, 4), (8, 8), (0, 0)), constant_values=40) for t in tiles[i:i + 4]])
            for i in range(0, len(tiles), 4)]
    cv2.imwrite(str(path), np.vstack(rows))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=0, help="split 별 상한 (미리보기용)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    recs = [json.loads(l) for l in open(CCTV / "pseudo_labels.jsonl", encoding="utf-8")]
    det, sw = SCRFD(), InSwapper()
    src = build_sources(det, sw)
    print(f"스왑 소스 {len(src['names'])}명 (타깃 {int((src['cls'] < UNKNOWN).sum())})", flush=True)
    tgt_idx, unk_idx = np.where(src["cls"] < UNKNOWN)[0], np.where(src["cls"] == UNKNOWN)[0]
    (WORK / "previews").mkdir(parents=True, exist_ok=True)
    YOLO.mkdir(parents=True, exist_ok=True)
    meta = open(YOLO / "meta.jsonl", "w", encoding="utf-8")
    st = {"frames": 0, "swapped_faces": 0, "label_faces": 0}
    id_after, id_before = [], []
    t0 = time.time()
    for split in ("train", "val", "test"):
        (YOLO / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO / "labels" / split).mkdir(parents=True, exist_ok=True)
        rs = [r for r in recs if r["split"] == split]
        rs = rs[::TRAIN_STRIDE if split == "train" else EVAL_STRIDE]
        if args.max_frames:
            rs = rs[:args.max_frames]
        pairs = []
        for r in rs:
            img = cv2.imread(str(CCTV / "RawDataFrames" / r["file"]))
            stem = Path(r["file"]).stem
            lines = yolo_lines(r)
            variants = [(stem, img, [])]
            if split == "train":
                out, swaps = img.copy(), []
                cand = [i for i, (b, s, fr) in enumerate(zip(r["boxes"], r["scores"], r["frontal"]))
                        if s >= SWAP_THR and fr and b[2] - b[0] >= SWAP_MIN_W]
                rng.shuffle(cand)
                for i in [c for c in cand if rng.random() < SWAP_P][:SWAP_MAX]:
                    j = int(rng.choice(tgt_idx if len(tgt_idx) and rng.random() < TARGET_P else unk_idx))
                    kps = np.float32(r["kps"][i])
                    e0 = sw.embed(out, kps)
                    out, _ = sw.swap(out, kps, src["embs"][j])
                    e1 = sw.embed(out, kps)
                    cb, ca = float(e0 @ src["embs"][j]), float(e1 @ src["embs"][j])
                    id_before.append(cb); id_after.append(ca)
                    swaps.append({"face": i, "box": r["boxes"][i], "src": str(src["names"][j]),
                                  "cls": int(src["cls"][j]), "id_cos_before": round(cb, 3), "id_cos_after": round(ca, 3)})
                    if len(pairs) < 12 and rng.random() < 0.3:
                        x0, y0, x1, y1 = np.int32(r["boxes"][i]); m = int((x1 - x0) * 0.8)
                        sl = (slice(max(0, y0 - m), y1 + m), slice(max(0, x0 - m), x1 + m))
                        pairs.append((img[sl], out[sl]))
                if swaps:
                    variants.append((stem + "_sw", out, swaps))
                    st["swapped_faces"] += len(swaps)
            for name, im, swaps in variants:
                cv2.imwrite(str(YOLO / "images" / split / f"{name}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
                (YOLO / "labels" / split / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
                meta.write(json.dumps({"image": f"images/{split}/{name}.jpg", "split": split, "src_frame": r["file"],
                                       "cam": r["cam"], "clip": r["clip"], "n_faces": len(lines),
                                       "swaps": swaps}) + "\n")
                st["frames"] += 1
                st["label_faces"] += len(lines)
            if st["frames"] % 200 < len(variants):
                print(f"  [{split}] {st}  {time.time() - t0:.0f}s", flush=True)
        if pairs:
            zoom_sheet(pairs, WORK / "previews" / f"swap_zoom_{split}.jpg")
        print(f"[{split}] 프레임 {len(rs)}  누적 {st}", flush=True)
    meta.close()
    (YOLO / "data.yaml").write_text(f"path: {YOLO.as_posix()}\ntrain: images/train\nval: images/val\n"
                                    f"test: images/test\nnames:\n  0: face\n", encoding="utf-8")
    if id_after:
        print(f"신원 유사도(소스 대비) 스왑 전 med {np.median(id_before):.3f} → 후 med {np.median(id_after):.3f}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
