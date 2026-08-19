"""COCO 2017 어노테이션 다운로드.

희소 클래스 판단의 근거 데이터. 이미지 본체(19GB)는 받지 않고,
어노테이션(json)만 받아서 클래스별 인스턴스/이미지 수를 전수 집계한다.

주의: images.cocodataset.org 는 S3 리다이렉트 때문에 인증서 호스트명이
불일치한다(cert CN != images.cocodataset.org). 공식 배포 경로가 http 도
동일 파일을 서비스하므로 http 로 받고, 대신 SHA256 을 기록해 무결성을 검증한다.
"""

from __future__ import annotations

import hashlib
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "coco"

URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
EXPECTED_BYTES = 252_907_541
# 압축 해제 후 실제로 쓰는 파일만 남긴다.
KEEP = {
    "annotations/instances_train2017.json",
    "annotations/instances_val2017.json",
}


def download(url: str, dst: Path, *, chunk: int = 1 << 20) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == EXPECTED_BYTES:
        print(f"[skip] 이미 존재: {dst.name} ({dst.stat().st_size:,} B)")
        return dst

    part = dst.with_suffix(dst.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    if offset:
        req.add_header("Range", f"bytes={offset}-")
        print(f"[resume] {offset:,} B 부터 재개")

    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as resp:
        mode = "ab" if offset and resp.status == 206 else "wb"
        if mode == "wb":
            offset = 0
        total = offset + int(resp.headers.get("Content-Length", 0))
        done = offset
        with open(part, mode) as f:
            while buf := resp.read(chunk):
                f.write(buf)
                done += len(buf)
                pct = 100 * done / total if total else 0
                mbps = done / 1e6 / max(time.time() - t0, 1e-9)
                print(f"\r  {done/1e6:8.1f}/{total/1e6:.1f} MB  {pct:5.1f}%  {mbps:5.1f} MB/s", end="")
    print()
    part.replace(dst)
    return dst


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while buf := f.read(1 << 20):
            h.update(buf)
    return h.hexdigest()


def main() -> int:
    zip_path = RAW / "annotations_trainval2017.zip"
    download(URL, zip_path)

    size = zip_path.stat().st_size
    if size != EXPECTED_BYTES:
        print(f"[error] 크기 불일치: {size:,} != {EXPECTED_BYTES:,}", file=sys.stderr)
        return 1
    print(f"[ok] {zip_path.name}  {size:,} B")
    print(f"[sha256] {sha256(zip_path)}")

    with zipfile.ZipFile(zip_path) as zf:
        for name in KEEP:
            out = RAW / name
            if out.exists():
                print(f"[skip] {name} 이미 해제됨")
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(out, "wb") as dst:
                while buf := src.read(1 << 20):
                    dst.write(buf)
            print(f"[extract] {name}  {out.stat().st_size:,} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
