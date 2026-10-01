"""8차: LFW + VGGFace2 → 신원(사람)별 클래스를 갖는 통합 얼굴 데이터셋.

주제(사용자 확정, 2026-10-01): 저화질 얼굴인식 — 특정 개인의 얼굴을 매칭해
신원을 특정(얼굴인식/재식별). 두 원본은 모두 "신원 폴더 = 한 사람" 구조라
바운딩박스 라벨이 없다. MTCNN으로 얼굴을 찾아 160x160 정렬 크롭으로 통일하고,
6,289명 전원에게 전역 클래스 ID(gid)를 부여한다.

역할(role) — 신원 단위로 나눠 학습/평가가 절대 섞이지 않게 한다
  train       VGGFace2 원본 train 신원 480명  → 파인튜닝 클래스 (cls 0..479)
  eval_vgg    VGGFace2 원본 test  신원  60명  → 1:N 식별 평가 (갤러리/프로브)
  eval_lfw    LFW 5,749명                     → 1:1 검증(pairs.csv) + 1:N 방해 갤러리

원본 해상도(얼굴 박스 크기 face_px)와 선명도(lap_var)를 manifest에 남긴다 —
scripts/31이 이걸로 "자연 발생 저화질" 프로브를 고른다(5차 교훈: 우리가 만든
열화가 아니라 원래부터 나쁜 사진으로 평가).

입력  data/raw/faces/{lfw,vggface2}  (data/dataset/*.zip 압축 해제본)
출력  data/dataset/faces_v1/images/<gid>/<n>.jpg
      data/dataset/faces_v1/manifest.csv
      data/dataset/faces_v1/identities.csv
"""

from __future__ import annotations

import csv
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import paths  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

SRC = paths.RAW / "faces"
OUT = paths.DATASET / "faces_v1"
OUT_SIZE = 160          # facenet-pytorch InceptionResnetV1 입력
MARGIN = 14             # facenet-pytorch LFW 평가 예제와 동일 (출력 160px 기준)
DET_SIDE = 320          # 검출은 축소본에서 (속도), 크롭은 원본에서 (화질)
TRAIN_CAP = 150         # train 신원당 최대 장수 — 480명 x 150 = 72k (계산 예산)
SEED = 0


def collect() -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED)
    idents, items = [], []

    def add_ident(name, source, role, cls):
        gid = len(idents)
        idents.append({"gid": gid, "name": name, "source": source, "role": role, "cls": cls})
        return gid

    vgg = SRC / "vggface2"
    train_ids = sorted(p.name for p in (vgg / "train").iterdir() if p.is_dir())
    for k, ident in enumerate(train_ids):
        gid = add_ident(ident, "vggface2", "train", k)
        files = sorted((vgg / "train" / ident).glob("*.jpg"))
        rng.shuffle(files)
        for f in sorted(files[:TRAIN_CAP]):
            items.append({"gid": gid, "src": f})
    for ident in sorted(p.name for p in (vgg / "val").iterdir() if p.is_dir()):
        gid = add_ident(ident, "vggface2", "eval_vgg", -1)
        for f in sorted((vgg / "val" / ident).glob("*.jpg")):
            items.append({"gid": gid, "src": f})
    lfw = SRC / "lfw" / "lfw-deepfunneled" / "lfw-deepfunneled"
    for d in sorted(p for p in lfw.iterdir() if p.is_dir()):
        gid = add_ident(d.name, "lfw", "eval_lfw", -1)
        for f in sorted(d.glob("*.jpg")):
            items.append({"gid": gid, "src": f})
    return idents, items


class Loader(Dataset):
    """원본 + 검출용 축소/패딩본(DET_SIDE 정사각)을 함께 돌려준다 — MTCNN 배치 처리용."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        bgr = cv2.imread(str(self.items[i]["src"]))
        h, w = bgr.shape[:2]
        s = DET_SIDE / max(h, w)
        small = cv2.resize(bgr, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((DET_SIDE, DET_SIDE, 3), np.uint8)
        canvas[: small.shape[0], : small.shape[1]] = small
        return i, bgr, canvas[:, :, ::-1].copy(), s


def collate(batch):
    return batch


def pick_box(boxes, probs, w, h):
    """중앙에 가깝고 큰 얼굴 — 두 원본 모두 대상 인물이 크롭 중앙에 있다."""
    if boxes is None:
        return None
    best, best_score = None, -1e9
    for b, p in zip(boxes, probs):
        if p is None or p < 0.90:
            continue
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        area = (b[2] - b[0]) * (b[3] - b[1])
        dist = np.hypot(cx - w / 2, cy - h / 2) / max(w, h)
        score = np.sqrt(area) / max(w, h) - dist
        if score > best_score:
            best, best_score = b, score
    return best


def crop_face(bgr, box):
    """facenet-pytorch extract_face와 같은 규칙: 박스에 출력 기준 MARGIN 만큼 여백."""
    h, w = bgr.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    mx = MARGIN * bw / (OUT_SIZE - MARGIN)
    my = MARGIN * bh / (OUT_SIZE - MARGIN)
    x0, y0 = int(max(box[0] - mx / 2, 0)), int(max(box[1] - my / 2, 0))
    x1, y1 = int(min(box[2] + mx / 2, w)), int(min(box[3] + my / 2, h))
    crop = bgr[y0:y1, x0:x1]
    interp = cv2.INTER_AREA if crop.shape[0] > OUT_SIZE else cv2.INTER_LINEAR
    return cv2.resize(crop, (OUT_SIZE, OUT_SIZE), interpolation=interp)


def main() -> int:
    from facenet_pytorch import MTCNN

    paths.ensure()
    idents, items = collect()
    gid2ident = {d["gid"]: d for d in idents}
    print(f"identities {len(idents)}  images {len(items)}")
    mtcnn = MTCNN(image_size=OUT_SIZE, margin=MARGIN, keep_all=True, device="cuda")
    (OUT / "images").mkdir(parents=True, exist_ok=True)

    dl = DataLoader(Loader(items), batch_size=64, num_workers=6, collate_fn=collate)
    rows, n_fail, t0 = [], 0, time.time()
    per_gid_count: dict[int, int] = {}
    for bi, batch in enumerate(dl):
        canv = np.stack([b[2] for b in batch])
        boxes, probs = mtcnn.detect(canv)
        for (i, bgr, _, s), bx, pr in zip(batch, boxes, probs):
            it = items[i]
            ident = gid2ident[it["gid"]]
            h, w = bgr.shape[:2]
            box = pick_box(None if bx is None else bx / s, pr, w, h)
            fallback = False
            if box is None:
                if ident["source"] != "lfw":
                    n_fail += 1
                    continue
                # LFW deepfunneled는 250x250에 얼굴이 중앙 정렬돼 있다 — 표준 관례대로
                # 검출 실패 시 중앙 크롭으로 대체해 6,000쌍 프로토콜을 온전히 유지한다.
                box = np.array([w * 0.3, h * 0.28, w * 0.7, h * 0.76])
                fallback = True
            face = crop_face(bgr, box)
            k = per_gid_count.get(it["gid"], 0)
            per_gid_count[it["gid"]] = k + 1
            rel = Path("images") / f"{it['gid']:05d}" / f"{k:04d}.jpg"
            (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(OUT / rel), face, [cv2.IMWRITE_JPEG_QUALITY, 95])
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            rows.append({
                "path": rel.as_posix(), "gid": it["gid"], "cls": ident["cls"],
                "name": ident["name"], "source": ident["source"], "role": ident["role"],
                "orig": it["src"].relative_to(SRC).as_posix(),
                "orig_w": w, "orig_h": h,
                "face_px": round(float(min(box[2] - box[0], box[3] - box[1])), 1),
                "lap_var": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
                "fallback": int(fallback),
            })
        if bi % 50 == 0:
            done = (bi + 1) * 64
            print(f"  {done}/{len(items)}  fail {n_fail}  {done / (time.time() - t0):.0f} img/s", flush=True)

    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with open(OUT / "identities.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["gid", "name", "source", "role", "cls", "n_images"])
        wr.writeheader()
        for d in idents:
            wr.writerow({**d, "n_images": per_gid_count.get(d["gid"], 0)})
    by_role: dict[str, int] = {}
    for r in rows:
        by_role[r["role"]] = by_role.get(r["role"], 0) + 1
    print(f"saved {len(rows)} faces, detection fail(dropped) {n_fail}, "
          f"lfw fallback {sum(r['fallback'] for r in rows)}  by role {by_role}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
