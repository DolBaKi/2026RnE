"""CCTV 영상/이미지에서 특정 인물 찾기 — 8차 최종 산출물.

찾을 사람의 사진(1장 이상)을 주면, CCTV 프레임 속 모든 얼굴을 검출해 그 사람과
비교하고, 임계값을 넘는 가장 비슷한 얼굴을 빨간 박스로 표시한다.

  python scripts/find_person.py --query 찾을사람1.jpg 찾을사람2.jpg \
      --input cctv_frame.jpg  (또는 폴더 / .mp4)  --weights runs/face/real_synth_s0.pt

임계값(--threshold)을 생략하면 scripts/34가 인파 테스트에서 잰 "다른 사람을 이 사람으로
착각하는 비율 1%" 지점(reports/eval_face.json → crowd_find.threshold@far1%)을 쓴다.

출력  <out>/<이름>.jpg (박스 표시), <out>/results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import paths  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

OUT_SIZE, MARGIN = 160, 14
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def crop_box(frame, box):
    H, W = frame.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    mx, my = MARGIN * bw / (OUT_SIZE - MARGIN), MARGIN * bh / (OUT_SIZE - MARGIN)
    x0, y0 = int(max(box[0] - mx / 2, 0)), int(max(box[1] - my / 2, 0))
    x1, y1 = int(min(box[2] + mx / 2, W)), int(min(box[3] + my / 2, H))
    return cv2.resize(frame[y0:y1, x0:x1], (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_CUBIC)


class Finder:
    def __init__(self, weights: str | None, device="cuda"):
        from facenet_pytorch import MTCNN, InceptionResnetV1

        self.device = device
        self.mtcnn = MTCNN(keep_all=True, min_face_size=14, device=device)
        self.net = InceptionResnetV1(pretrained="casia-webface").eval().to(device)
        if weights:
            self.net.load_state_dict(torch.load(weights, map_location=device))

    def faces(self, bgr, prob=0.90):
        boxes, probs = self.mtcnn.detect(Image.fromarray(bgr[:, :, ::-1]))
        if boxes is None:
            return []
        return [b for b, p in zip(boxes, probs) if p >= prob]

    @torch.no_grad()
    def embed(self, crops):
        x = torch.stack([torch.from_numpy((c[:, :, ::-1].astype(np.float32) - 127.5) / 128).permute(2, 0, 1)
                         for c in crops]).to(self.device)
        return F.normalize(self.net(x)).cpu().numpy()

    def template(self, query_paths):
        crops = []
        for p in query_paths:
            img = cv2.imread(str(p))
            fs = self.faces(img)
            if fs:
                b = max(fs, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))   # 사진 속 가장 큰 얼굴
                crops.append(crop_box(img, b))
            elif img is not None and img.shape[:2] == (OUT_SIZE, OUT_SIZE):
                crops.append(img)          # 이미 정렬된 160 크롭(faces10 등록 사진)
            else:
                print(f"[warn] 얼굴을 못 찾음: {p}")
        if not crops:
            raise SystemExit("찾을 사람 사진에서 얼굴을 하나도 찾지 못했습니다.")
        t = self.embed(crops).mean(0)
        return t / np.linalg.norm(t)

    def search(self, bgr, templates, thr):
        """templates: {이름: 벡터}. 얼굴마다 가장 비슷한 사람을 붙이고, 한 사람당 프레임에서
        가장 점수가 높은 얼굴 하나만 '찾음'으로 표시한다."""
        boxes = self.faces(bgr)
        if not boxes:
            return []
        names = list(templates)
        S = self.embed([crop_box(bgr, b) for b in boxes]) @ np.stack([templates[n] for n in names]).T
        best_face = {j: int(np.argmax(S[:, j])) for j in range(len(names))}
        hits = []
        for i, b in enumerate(boxes):
            j = int(np.argmax(S[i]))
            match = best_face[j] == i and S[i, j] >= thr
            hits.append({"box": [round(float(v), 1) for v in b], "who": names[j] if match else None,
                         "score": round(float(S[i, j]), 3), "match": bool(match)})
        return hits


COLORS = [(0, 0, 255), (0, 160, 255), (0, 220, 0), (255, 120, 0), (200, 0, 200),
          (0, 255, 255), (255, 0, 120), (120, 255, 0), (255, 255, 0), (80, 80, 255)]


def draw(bgr, hits, names):
    out = bgr.copy()
    t = max(2, out.shape[1] // 600)
    for h in sorted(hits, key=lambda h: h["match"]):
        x0, y0, x1, y1 = map(int, h["box"])
        col = COLORS[names.index(h["who"]) % len(COLORS)] if h["match"] else (160, 160, 160)
        cv2.rectangle(out, (x0, y0), (x1, y1), col, t if h["match"] else 1)
        if h["match"]:
            cv2.putText(out, f"{h['who']} {h['score']:.2f}", (x0, max(0, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5 * t, col, t)
    return out


def iter_frames(inp: Path, every: int):
    if inp.is_dir():
        for p in sorted(inp.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                yield p.stem, cv2.imread(str(p))
    elif inp.suffix.lower() in IMG_EXT:
        yield inp.stem, cv2.imread(str(inp))
    else:
        cap = cv2.VideoCapture(str(inp))
        k = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if k % every == 0:
                yield f"{inp.stem}_f{k:06d}", fr
            k += 1


def default_threshold(weights):
    if weights:
        arm, seed = Path(weights).stem.rsplit("_s", 1)
        key = f"{arm.replace('real_synth', 'real+synth')}/s{seed}"   # 가중치와 같은 시드의 임계값
    else:
        key = "baseline/s0"
    try:
        res = json.loads((paths.REPORTS / "eval_face10.json").read_text(encoding="utf-8"))
        r = res[key]
        thr = r["crowd"]["threshold@far1%"] if "crowd" in r else r["threshold@far1%"]
        print(f"[threshold] {thr:.3f}  ({key}, 모르는 사람 오인율 1% 지점)")
        return thr
    except (FileNotFoundError, StopIteration, KeyError):
        print("[threshold] 평가 결과가 없어 기본값 0.5 사용")
        return 0.5


def main() -> int:
    ap = argparse.ArgumentParser(description="CCTV 속 특정 인물 찾기")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--query", nargs="+", help="찾을 사람의 사진 (1장 이상)")
    g.add_argument("--targets10", action="store_true", help="faces10 등록 10명을 동시에 찾기")
    ap.add_argument("--input", required=True, help="CCTV 이미지 / 이미지 폴더 / 동영상")
    ap.add_argument("--weights", default=None, help="파인튜닝 가중치 (생략 시 CASIA 사전학습)")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--every", type=int, default=5, help="동영상일 때 N프레임마다 검사")
    ap.add_argument("--out", default=str(paths.ROOT / "runs" / "find_person"))
    args = ap.parse_args()

    thr = args.threshold if args.threshold is not None else default_threshold(args.weights)
    finder = Finder(args.weights)
    if args.targets10:
        f10 = paths.DATASET / "faces10"
        split = json.loads((f10 / "split.json").read_text(encoding="utf-8"))
        tg = list(csv.DictReader(open(f10 / "targets.csv", encoding="utf-8")))
        templates = {f"cls{int(t['cls'])}": finder.template(
            [f10 / p for p in split["enroll"] if p.startswith(f"images/{int(t['cls']):02d}/")]) for t in tg}
    else:
        templates = {"TARGET": finder.template(args.query)}
    names = list(templates)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for name, fr in iter_frames(Path(args.input), args.every):
        hits = finder.search(fr, templates, thr)
        m = [h for h in hits if h["match"]]
        cv2.imwrite(str(out / f"{name}.jpg"), draw(fr, hits, names))
        results.append({"frame": name, "n_faces": len(hits), "found": [h["who"] for h in m], "hits": m})
        print(f"  {name}: 얼굴 {len(hits)}개, " + (", ".join(f"{h['who']}({h['score']:.2f})" for h in m) if m else "찾는 사람 없음"))
    (out / "results.json").write_text(json.dumps({"threshold": thr, "results": results},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] {sum(bool(r['found']) for r in results)}/{len(results)} 프레임에서 발견 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
