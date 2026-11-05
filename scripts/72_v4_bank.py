"""9차 — 생성기 2: v4(RealVisXL 인페인팅 + IP-Adapter FaceID-PlusV2 SDXL) 은행, 같은 사진 5장.

70 의 inswapper 은행과 같은 생각 — 자리 × 신원 삽입을 한 번씩만 만들어 패치로 두고 변형은 조합만 바꾼다.
v4 는 삽입 1건 ~12초에 신원 검증(CASIA ≥ 0.40) 통과율이 낮아(시험셋 기준 타깃 ~30%) 은행을 작게 만든다.
  자리      fewshot5e 에서 타깃·모르는 사람이 들어간 자리. 타깃 자리는 inswapper 신원 cos 평균이 높은 순
  라운드    라운드마다 사진 5장 × 타깃 10명을 서로 다른 자리에 1번씩 + 모르는 사람 UNK_PER_ROUND 명.
            TIME_CAP 이 지나면 그 라운드까지 만든 것으로 멈춘다 (1라운드 = 타깃마다 사진당 1곳)
  설정      69 와 같음 — 눈썹~턱 타원만 다시 그림, strength 0.99, 프롬프트 "man"
변형
  fewshot5f_v4    사진마다 원본 + v4 변형 N_VAR 장
  fewshot5f_mix   사진마다 원본 + inswapper 변형 N_VAR/2 (fewshot5e v000~) + v4 변형 N_VAR/2
val/test 는 fewshot5d 그대로.
"""

from __future__ import annotations

import importlib
import json
import pickle
import random
import shutil
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
m69 = importlib.import_module("69_sdxl_identity_test")
WORK = paths.DATASET / "CCTV_work"
E, D = WORK / "fewshot5e", WORK / "fewshot5d"
BANK = WORK / "v4_bank.pkl"
OUTS = {"v4": WORK / "fewshot5f_v4", "mix": WORK / "fewshot5f_mix"}
RAW = {"cctv": m69.RAW["cctv"], "wider": paths.DATASET / "WIDER" / "WIDER_train" / "images"}  # 학습 사진
N_T, UNKNOWN = 10, 10
TIME_CAP = 100 * 60                  # 초 — 이 시간이 지나면 다음 라운드를 시작하지 않는다
MAX_ROUNDS, MAX_TRY, UNK_PER_ROUND = 3, 2, 8
N_VAR = 200


def diff_patch(img, out):
    d = np.any(out != img, axis=2)
    ys, xs = np.nonzero(d)
    if not len(ys):
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return int(y0), int(x0), out[y0:y1, x0:x1].copy(), d[y0:y1, x0:x1]


def paste(im, p):
    y0, x0, a, m = p
    im[y0:y0 + a.shape[0], x0:x0 + a.shape[1]][m] = a[m]


def photo_slots():
    """fewshot5e 학습 meta → 사진별 {face: (box, kps)}, 타깃 자리(품질 순), 모르는 사람 자리."""
    P = {}
    for l in open(E / "meta.jsonl", encoding="utf-8"):
        m = json.loads(l)
        if m["split"] != "train":
            continue
        p = P.setdefault(m["src_frame"], {"src": m["src"], "orig": None, "faces": {}, "q": {}, "unk": set()})
        if m["image"].endswith("_orig.jpg"):
            p["orig"] = m["image"]
        for f in m["swaps"]:
            p["faces"][f["face"]] = (f["box"], f["kps"])
            if f["cls"] < UNKNOWN:
                p["q"].setdefault(f["face"], []).append(f["id_cos"])
            else:
                p["unk"].add(f["face"])
    for p in P.values():
        p["tgt"] = sorted(p["q"], key=lambda i: -np.mean(p["q"][i]))
        p["unk"] = sorted(p["unk"])
    return P


def build_bank(P, ids):
    rng = random.Random(0)
    m = m51.Models()
    m51.head_mask = m69.face_mask
    _inpaint = m.fi.inpaint
    m.fi.inpaint = lambda *a, **k: _inpaint(*a, **{**k, "strength": m69.STRENGTH})
    tb = m51.IdentityBank(m, {t["name"]: t["swap_train"] + t["enroll"] for t in ids["targets"]}, rng)
    ub = m51.IdentityBank(m, {u["name"]: u["imgs"] for u in ids["unknown_train"]}, rng)
    unames = [u["name"] for u in ids["unknown_train"]]
    prompt = m51.PROMPT[False].format(g="man")
    stats = {"no_face_after": 0, "face_off_frame": 0, "cos_all": [], "id_gate_reject": 0}
    bank = {f: [] for f in P}            # [(face, cls, who, id_cos, patch)]
    t0 = time.time()
    for r in range(MAX_ROUNDS):
        if time.time() - t0 > TIME_CAP:
            break
        for f, p in P.items():
            img = cv2.imread(str(RAW[p["src"]] / f))
            ref = m51.scene_face_ref(img, [m69.slot_of({"kps": k}) for _, k in p["faces"].values()])
            have = {(b[0], b[1]) for b in bank[f]}
            top = p["tgt"][:max(2 * N_T, len(p["tgt"]) // 2)]
            used = set()
            jobs = [(t["cls"], t["name"]) for t in ids["targets"]]
            rng.shuffle(jobs)
            for c, name in jobs:
                ok = False
                for i in [i for i in top if i not in used and (i, c) not in have][:MAX_TRY]:
                    ident = tb.get(name)
                    if ident is None:
                        break
                    canvas = img.copy()
                    res = m51.insert_person(m, canvas, m69.slot_of({"kps": p["faces"][i][1]}), ident, prompt, ref,
                                            rng, stats)
                    if res is not None:
                        bank[f].append((i, c, name, res["id_cos"], res["box"], diff_patch(img, canvas)))
                        used.add(i)
                        ok = True
                        break
                if not ok:
                    pass
            for i in rng.sample(p["unk"], min(UNK_PER_ROUND, len(p["unk"]))):
                u = rng.choice(unames)
                ident = ub.get(u)
                if ident is None:
                    continue
                canvas = img.copy()
                res = m51.insert_person(m, canvas, m69.slot_of({"kps": p["faces"][i][1]}), ident, prompt, ref,
                                        rng, stats)
                if res is not None:
                    bank[f].append((i, UNKNOWN, u, res["id_cos"], res["box"], diff_patch(img, canvas)))
            nt = sum(b[1] < UNKNOWN for b in bank[f])
            print(f"  라운드 {r + 1}  {Path(f).name}  타깃 패치 {nt}  모르는 사람 {len(bank[f]) - nt}"
                  f"  {time.time() - t0:.0f}s", flush=True)
        with open(BANK, "wb") as fo:
            pickle.dump(bank, fo)
    print(f"[bank] 신원 탈락 {stats['id_gate_reject']}  재검출 실패 {stats['no_face_after']}", flush=True)
    return bank


def compose(P, bank):
    rng = np.random.default_rng(0)
    for o in OUTS.values():
        for sp in ("train", "val", "test"):
            (o / "images" / sp).mkdir(parents=True, exist_ok=True)
            (o / "labels" / sp).mkdir(parents=True, exist_ok=True)
    metas = {k: open(o / "meta.jsonl", "w", encoding="utf-8") for k, o in OUTS.items()}
    ein = {}
    for l in open(E / "meta.jsonl", encoding="utf-8"):
        m = json.loads(l)
        if m["split"] == "train":
            ein[m["image"]] = l if l.endswith("\n") else l + "\n"
    cnt = {k: 0 for k in OUTS}
    for f, p in P.items():
        img = cv2.imread(str(RAW[p["src"]] / f))
        H, W = img.shape[:2]
        orig_lab = (E / p["orig"].replace("images/", "labels/").replace(".jpg", ".txt")).read_text().split("\n")
        base = []
        for l in filter(None, orig_lab):
            _, cx, cy, w, h = map(float, l.split())
            base.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
        stem = Path(p["orig"]).stem[:-len("_orig")]
        for k, o in OUTS.items():                       # 원본 1장
            for a in (p["orig"], p["orig"].replace("images/", "labels/").replace(".jpg", ".txt")):
                shutil.copy(E / a, o / a)
            metas[k].write(ein[p["orig"]])
        tg = [b for b in bank[f] if b[1] < UNKNOWN]
        un = [b for b in bank[f] if b[1] == UNKNOWN]
        for v in range(N_VAR):
            out, used, faces, boxes = img.copy(), set(), [], [list(b) for b in base]
            for c in rng.permutation(N_T):
                cand = [b for b in tg if b[1] == c and b[0] not in used]
                if cand:
                    b = cand[int(rng.integers(len(cand)))]
                    used.add(b[0])
                    faces.append(b)
            for j in rng.permutation(len(un)):
                if len(faces) >= N_T + 10:
                    break
                if un[j][0] not in used:
                    used.add(un[j][0])
                    faces.append(un[j])
            sw = []
            for i, c, who, ic, box, pt in faces:
                if pt is not None:
                    paste(out, pt)
                o_ = [m69._iou(p["faces"][i][0], b) for b in boxes] if boxes else [0]
                if max(o_) >= 0.3:
                    boxes[int(np.argmax(o_))] = box
                else:
                    boxes.append(box)
                sw.append({"face": int(i), "box": box, "cls": int(c), "who": who, "id_cos": ic, "gen": "v4"})
            name = f"{stem}_g{v:03d}"
            lab = "\n".join(f"0 {(b[0] + b[2]) / 2 / W:.6f} {(b[1] + b[3]) / 2 / H:.6f} {(b[2] - b[0]) / W:.6f} "
                            f"{(b[3] - b[1]) / H:.6f}" for b in boxes)
            rec = json.dumps({"image": f"images/train/{name}.jpg", "split": "train", "src": p["src"],
                              "src_frame": f, "swaps": sw}) + "\n"
            targets = ["v4"] + (["mix"] if v < N_VAR // 2 else [])
            for k in targets:
                cv2.imwrite(str(OUTS[k] / "images" / "train" / f"{name}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
                (OUTS[k] / "labels" / "train" / f"{name}.txt").write_text(lab, encoding="utf-8")
                metas[k].write(rec)
                cnt[k] += 1
        for v in range(N_VAR // 2):                      # mix 의 나머지 절반 = inswapper
            a = f"images/train/{stem}_v{v:03d}.jpg"
            for x in (a, a.replace("images/", "labels/").replace(".jpg", ".txt")):
                shutil.copy(E / x, OUTS["mix"] / x)
            metas["mix"].write(ein[a])
            cnt["mix"] += 1
    for l in open(D / "meta.jsonl", encoding="utf-8"):
        m = json.loads(l)
        if m["split"] == "train":
            continue
        for k, o in OUTS.items():
            for a in (m["image"], m["image"].replace("images/", "labels/").replace(".jpg", ".txt")):
                if not (o / a).exists():
                    shutil.copy(D / a, o / a)
            metas[k].write(l if l.endswith("\n") else l + "\n")
    for fo in metas.values():
        fo.close()
    print("[compose] 학습 변형", cnt, flush=True)


def main() -> int:
    ids = json.loads((WORK / "fewshot5" / "identities.json").read_text(encoding="utf-8"))
    P = photo_slots()
    if "--compose-only" in sys.argv and BANK.exists():
        bank = pickle.load(open(BANK, "rb"))
    else:
        bank = build_bank(P, ids)
    compose(P, bank)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
