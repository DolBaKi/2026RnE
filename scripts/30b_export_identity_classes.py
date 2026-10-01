"""8차 — 사람별 클래스 색인 + 사진 적은 사람 제외.

faces_v1/images/<gid>/ 가 이미 "사람 1명 = 폴더 1개"(ImageFolder 형식)라 파일을 따로
복사하지 않고 색인(classes.csv)만 만든다. 처음엔 classes/ 아래로 복사했는데 D:가 exFAT
라 9만여 개의 작은 jpg가 클러스터 단위로 잡혀 13GB를 차지했다 — 중복이라 폐기.

사진이 MIN_IMAGES 장 미만인 사람은 클래스로 쓰기에 표본이 부족하므로 제외한다
(사용자 지시). LFW는 5,749명 대부분이 1~2장이라 거의 전부 빠지지만, LFW 표준 6,000쌍
검증과 1:N 방해 신원 용도로는 eval_protocol.json 에서 계속 쓴다(평가 프로토콜은 그대로).

출력  data/dataset/faces_v1/classes.csv
      (gid, cls, name, source, role, n_real, n_synth_cctv, folder, synth_folder)
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import paths  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

ROOT = paths.DATASET / "faces_v1"
MIN_IMAGES = 20


def main() -> int:
    rows = list(csv.DictReader(open(ROOT / "manifest.csv", encoding="utf-8")))
    by = defaultdict(list)
    for r in rows:
        by[int(r["gid"])].append(r)
    synth = defaultdict(int)
    sm = ROOT / "synth_cctv_manifest.csv"
    if sm.exists():
        for r in csv.DictReader(open(sm, encoding="utf-8")):
            synth[int(r["gid"])] += 1

    kept, dropped = [], defaultdict(int)
    for gid, rs in sorted(by.items()):
        if len(rs) < MIN_IMAGES:
            dropped[rs[0]["source"]] += 1
            continue
        r0 = rs[0]
        kept.append({"gid": gid, "cls": r0["cls"], "name": r0["name"], "source": r0["source"],
                     "role": r0["role"], "n_real": len(rs), "n_synth_cctv": synth.get(gid, 0),
                     "folder": f"images/{gid:05d}",
                     "synth_folder": f"synth_cctv/{gid:05d}" if gid in synth else ""})
    with open(ROOT / "classes.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(kept[0]))
        wr.writeheader()
        wr.writerows(kept)
    by_role = defaultdict(int)
    for k in kept:
        by_role[f"{k['source']}/{k['role']}"] += 1
    print(f"kept {len(kept)} people (>= {MIN_IMAGES} photos): {dict(by_role)}")
    print(f"dropped (< {MIN_IMAGES} photos): {dict(dropped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
