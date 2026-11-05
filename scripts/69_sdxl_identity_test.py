"""9차 — 다른 생성 모델로 만든 시험셋 (생성 모델 간 일반화 검증).

68 의 YOLO(11클래스)는 inswapper 로 넣은 얼굴로 학습했다. 같은 inswapper 로 만든 test 로는 "사람 A 를
배웠는지 / inswapper 가 그린 A 의 그림체를 배웠는지" 를 가를 수 없다(인식 모델에서 실제로 그림체를 배웠다:
스왑 test 88~98% → 저화질 실사진 11~49%). 그래서 test 프레임의 같은 자리에 **다른 생성 모델**로 넣는다.

  생성기     RealVisXL V5 인페인팅 + IP-Adapter FaceID-PlusV2 SDXL (8차 v4 의 insert_person 그대로)
  신원       타깃 = 학습·스왑에 안 쓴 사진(test_hq), 모르는 사람 = unknown_test
  자리       fewshot5d test 의 그 프레임에서 타깃·모르는 사람을 넣었던 자리(나이순 배정 그대로)
             — 원본 프레임(스왑 전)에 SDXL 로 다시 넣는다. 나머지 얼굴은 원래 얼굴.
  모르는 사람도 같은 SDXL 로 넣어 "SDXL 티 = 타깃" 이 되지 않게 한다.
  신원 검증  CASIA cos ≥ 0.40 (v4 와 같음, 생성 조건의 ArcFace 와 다른 모델), 실패하면 MAX_TRY 번까지 다시

출력  data/dataset/CCTV_work/fewshot5d_sdxl_test/{images,labels}/test/, meta.jsonl, previews/
"""

from __future__ import annotations

import importlib
import json
import random
import sys
import time
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

m51 = importlib.import_module("51_compose_face10_v4")
WORK = paths.DATASET / "CCTV_work"
SRC, OUT = WORK / "fewshot5d", WORK / "fewshot5d_sdxl_test"
RAW = {"cctv": paths.DATASET / "CCTV" / "RawDataFrames", "wider": paths.DATASET / "WIDER" / "WIDER_val" / "images"}
N_PER_SRC = 40
MAX_UNK = 2
MAX_TRY = 3
UNKNOWN = 10
# v4 의 머리 전체 마스크·strength 0.99 는 모자·두건까지 지우고 고개 숙인 사람을 정면 얼굴로 바꿔 붙인 티가 났다
# (첫 시트 확인 2026-11-03) → 눈썹~턱 타원만 다시 그리고, 모자·머리카락은 원래 것을 남긴다.
STRENGTH = 0.99                      # 0.75·0.85 는 신원이 거의 안 옮겨졌다(타깃 12 중 3~4 통과)


def face_mask(side, fs):
    """창 중심 = 눈 중심 + 0.15fs. 얼굴(눈썹 ~ 턱)은 눈 중심 −0.25fs ~ +0.65fs → 중심 c+0.05fs."""
    c = side / 2
    m = np.zeros((side, side), np.float32)
    cv2.ellipse(m, (int(c), int(c + 0.05 * fs)), (int(0.38 * fs), int(0.47 * fs)), 0, 0, 360, 1.0, -1)
    return m


def slot_of(f):
    k = np.float32(f["kps"])
    return {"leye": k[0].tolist(), "reye": k[1].tolist(), "eye_dist": float(np.linalg.norm(k[1] - k[0]))}


GENS = {   # 시험용 생성기 — sd15 = 8차 v3 의 SD1.5 인페인팅 + IP-Adapter FaceID-PlusV2 (학습에 안 쓴 제3 생성기)
    "sdxl": dict(out="fewshot5d_sdxl_test"),
    "sd15": dict(out="fewshot5d_sd15_test", side=512, shortcut=(1.2, 1.5), steps=12, max_unk=1),
}


def main() -> int:
    global OUT, MAX_UNK
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="sdxl", choices=list(GENS))
    g = GENS[ap.parse_args().gen]
    OUT = WORK / g["out"]
    if "side" in g:
        from rareobj.faceid_inpaint import FaceIDInpainter
        m51.FaceIDInpainterXL = FaceIDInpainter
        m51.SIDE, m51.SHORTCUT, m51.STEPS, MAX_UNK = g["side"], g["shortcut"], g["steps"], g["max_unk"]
    meta = [json.loads(l) for l in open(SRC / "meta.jsonl", encoding="utf-8")]
    ids = json.loads((WORK / "fewshot5" / "identities.json").read_text(encoding="utf-8"))
    test = {}
    for m in meta:
        if m["split"] == "test":
            test.setdefault(m["src"], []).append(m)
    pick = []
    for src, ms in sorted(test.items()):
        step = max(1, len(ms) // N_PER_SRC)
        pick += ms[::step][:N_PER_SRC]

    rng = random.Random(0)
    m = m51.Models()
    m51.head_mask = face_mask
    _inpaint = m.fi.inpaint
    m.fi.inpaint = lambda *a, **k: _inpaint(*a, **{**k, "strength": STRENGTH})
    tgt_bank = m51.IdentityBank(m, {t["name"]: t["test_hq"] for t in ids["targets"]}, rng)
    unk_bank = m51.IdentityBank(m, {u["name"]: u["imgs"] for u in ids["unknown_test"]}, rng)
    unk_names = [u["name"] for u in ids["unknown_test"]]
    prompt = m51.PROMPT[False].format(g="man")
    stats = {"no_face_after": 0, "face_off_frame": 0, "cos_all": [], "id_gate_reject": 0}
    for sub in ("images/test", "labels/test", "previews"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    # 끊겼다가 다시 돌리면 meta.jsonl 에 이미 있는 프레임은 건너뛴다
    have = {json.loads(l)["image"] for l in open(OUT / "meta.jsonl", encoding="utf-8")}         if (OUT / "meta.jsonl").exists() else set()
    fo = open(OUT / "meta.jsonl", "a", encoding="utf-8")
    t0, n_ok, n_fail = time.time(), {"tgt": 0, "unk": 0}, {"tgt": 0, "unk": 0}
    for k, mm in enumerate(pick):
        if f"images/test/{Path(mm['image']).stem}.jpg" in have:
            continue
        img = cv2.imread(str(RAW[mm["src"]] / mm["src_frame"]))
        H, W = img.shape[:2]
        canvas = img.copy()
        slots = [slot_of(f) for f in mm["swaps"]]
        ref = m51.scene_face_ref(img, slots)
        jobs = [f for f in mm["swaps"] if f["cls"] < UNKNOWN] + [f for f in mm["swaps"] if f["cls"] == UNKNOWN][:MAX_UNK]
        done = []
        for f in jobs:
            kind = "tgt" if f["cls"] < UNKNOWN else "unk"
            res = None
            for _ in range(MAX_TRY):
                ident = tgt_bank.get(f["who"]) if kind == "tgt" else unk_bank.get(rng.choice(unk_names))
                if ident is None:
                    continue
                res = m51.insert_person(m, canvas, slot_of(f), ident, prompt, ref, rng, stats)
                if res is not None:
                    break
            if res is None:
                n_fail[kind] += 1
                continue
            n_ok[kind] += 1
            done.append({"box": res["box"], "cls": f["cls"], "who": f["who"], "casia_cos": res["id_cos"],
                         "slot_box": f["box"]})
        # 라벨: 원래 라벨 박스에서 넣은 자리의 박스를 새로 그린 얼굴 박스로 바꾼다
        lines = (SRC / mm["image"].replace("images/", "labels/").replace(".jpg", ".txt")).read_text().split("\n")
        boxes = []
        for l in filter(None, lines):
            _, cx, cy, w, h = map(float, l.split())
            boxes.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
        for d in done:
            if boxes:
                o = [_iou(d["slot_box"], b) for b in boxes]
                if max(o) >= 0.3:
                    boxes[int(np.argmax(o))] = d["box"]
                    continue
            boxes.append(d["box"])
        name = Path(mm["image"]).stem
        cv2.imwrite(str(OUT / "images" / "test" / f"{name}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (OUT / "labels" / "test" / f"{name}.txt").write_text(
            "\n".join(f"0 {(b[0] + b[2]) / 2 / W:.6f} {(b[1] + b[3]) / 2 / H:.6f} {(b[2] - b[0]) / W:.6f} "
                      f"{(b[3] - b[1]) / H:.6f}" for b in boxes), encoding="utf-8")
        fo.write(json.dumps({"image": f"images/test/{name}.jpg", "split": "test", "src": mm["src"],
                             "src_frame": mm["src_frame"], "swaps": done}) + "\n")
        fo.flush()
        if k < 6:
            preview(img, canvas, done, OUT / "previews" / f"{name}.jpg")
        print(f"  {k + 1}/{len(pick)} {mm['src']} 넣음 {n_ok} 실패 {n_fail}  {time.time() - t0:.0f}s", flush=True)
    fo.close()
    c = np.array(stats["cos_all"])
    print(f"[done] 넣음 {n_ok} 실패 {n_fail}  CASIA cos 중앙 {np.median(c):.3f}  신원 탈락 {stats['id_gate_reject']}"
          f"  재검출 실패 {stats['no_face_after']}", flush=True)
    return 0


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def preview(orig, out, done, path, T=140):
    rows = []
    for d in done:
        x0, y0, x1, y1 = np.int32(d["box"])
        mg = int((x1 - x0) * 0.6)
        sl = (slice(max(0, y0 - mg), y1 + mg), slice(max(0, x0 - mg), x1 + mg))
        t = np.hstack([cv2.resize(orig[sl], (T, T)), cv2.resize(out[sl], (T, T))])
        cv2.putText(t, f"{'T' + str(d['cls']) if d['cls'] < UNKNOWN else 'unk'} {d['casia_cos']:.2f}", (4, 14), 0,
                    0.45, (0, 255, 255), 1)
        rows.append(t)
    if rows:
        cv2.imwrite(str(path), np.vstack(rows))


if __name__ == "__main__":
    raise SystemExit(main())
