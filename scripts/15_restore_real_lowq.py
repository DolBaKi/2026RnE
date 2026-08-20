"""real_lowq(진짜 저화질 실사진) 148장에 NAFNet+Real-ESRGAN 복원을 적용.

3.2 방법론: 전처리는 별도 비교 대상이 아니라 파이프라인에 항상 포함되는 보조
단계다. 그래도 "복원을 거치면 검출이 달라지는가"는 파이프라인이 실제로 뭔가
하고 있다는 근거로서 부가적으로 확인해 둔다 — 원본과 복원본을 각각 검출기에
넣어보는 것으로 충분하다.

출력
  data/interim/real_lowq_restored/images/*.jpg  (labels 는 real_lowq 와 동일해 복사만)
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import paths, restore  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

SRC = paths.INTERIM / "real_lowq"
DST = paths.INTERIM / "real_lowq_restored"


def main() -> int:
    if not SRC.exists():
        raise SystemExit("먼저 scripts/12_build_real_lowquality_testset.py 실행")
    (DST / "images").mkdir(parents=True, exist_ok=True)
    (DST / "labels").mkdir(parents=True, exist_ok=True)
    for p in (SRC / "labels").glob("*.txt"):
        shutil.copy(p, DST / "labels" / p.name)

    r = restore.Restorer(
        paths.CHECKPOINTS / "nafnet" / "NAFNet-GoPro-width32.pth",
        paths.CHECKPOINTS / "realesrgan" / "RealESRGAN_x4plus.pth",
        device="cuda",
    )
    files = sorted((SRC / "images").glob("*.jpg"))
    t0 = time.time()
    for i, p in enumerate(files, 1):
        out_p = DST / "images" / p.name
        if out_p.exists():
            continue
        img = cv2.imread(str(p))
        out = r.restore(img)
        cv2.imwrite(str(out_p), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if i % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / i * (len(files) - i)
            print(f"\r  {i}/{len(files)}  경과 {elapsed/60:.1f}분  잔여 {eta/60:.1f}분", end="")
    print(f"\n[done] {len(files)}장 복원 완료 -> {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
