"""복원 파이프라인 체크포인트 다운로드 — NAFNet(디블러) + Real-ESRGAN(업스케일).

공식 배포가 Google Drive 위주라 스크립트로 바로 받기 어려운 경우가 많은데,
NAFNet은 HuggingFace 커뮤니티 미러(nyanko7/nafnet-models)에 원본 체크포인트가
그대로 올라와 있어 이걸 쓴다. 아키텍처 코드는 basicsr 패키지 설치 없이
src/rareobj/vendor/ 에 필요한 부분만 직접 옮겨왔다 (이유는 그 파일들 docstring).

출력
  data/raw/checkpoints/nafnet/NAFNet-GoPro-width32.pth
  data/raw/checkpoints/realesrgan/RealESRGAN_x4plus.pth
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import paths  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

FILES = [
    ("nafnet", "NAFNet-GoPro-width32.pth",
     "https://huggingface.co/nyanko7/nafnet-models/resolve/main/NAFNet-GoPro-width32.pth", 68671121),
    ("realesrgan", "RealESRGAN_x4plus.pth",
     "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth", 67040989),
]


def download(url: str, dst: Path, expected_size: int) -> None:
    if dst.exists() and dst.stat().st_size == expected_size:
        print(f"[skip] {dst.name} 이미 존재")
        return
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    print(f"[ok] {dst.name}  {len(data):,} B")


def main() -> int:
    paths.ensure()
    for sub, name, url, size in FILES:
        download(url, paths.CHECKPOINTS / sub / name, size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
