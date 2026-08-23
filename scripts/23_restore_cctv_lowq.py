"""real_lowq_cctv에 NAFNet+Real-ESRGAN 복원을 적용 (scripts/15와 동일 절차,
CCTV 파일럿 대상만 다르다). 5차에서 확인한 대로 Real-ESRGAN이 매끈하고
정보량 적은 영역에서 격자무늬를 만드는 경우가 있는데, 거리·군중 장면은 곰
사진보다 질감이 많아 그 실패가 얼마나 재현되는지도 부가적으로 관찰한다.

출력
  data/interim/real_lowq_cctv_restored/images/*.jpg
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

SRC = paths.INTERIM / "real_lowq_cctv"
DST = paths.INTERIM / "real_lowq_cctv_restored"


def main() -> int:
    if not SRC.exists():
        raise SystemExit("먼저 scripts/21_build_cctv_real_lowq.py 실행")
    (DST / "images").mkdir(parents=True, exist_ok=True)

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
