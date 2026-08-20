"""제출용 zip 패키징.

담는 것 — 전부 생성물이거나 우리가 만든 파생 자료 (재배포 문제 없음)
  data/dataset/bear_synth_v2/{images,labels,dataset.yaml,manifest.jsonl}
  (5차 방법론 반영판 — clean+Albumentations열화 쌍. 1~4차에서 쓴 bear_synth(v1,
  열화본만)는 docs/03_results.md 재현용으로 남겨두되 제출 데이터셋은 최신
  방법론을 반영한 v2로 낸다.)
  reports/*, docs/*, configs/pipeline.yaml

담지 않는 것
  실제 COCO 이미지(real_val/real_ref) — Flickr 개별 라이선스라 재배포 대상 아님.
  대신 manifest.json(URL) 으로 재구성 가능하도록 스크립트만 포함.
  모델 체크포인트 자체(수 GB) — HF/GitHub 링크로 누구나 재현 가능하므로 zip에
  무게를 싣지 않는다. runs/finetune 학습 결과(가중치 .pt)도 제외.
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import paths  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

OUT_ZIP = paths.ROOT / "submission" / "bear_synth_dataset.zip"
DOC_ZIP = paths.ROOT / "submission" / "pipeline_docs.zip"


def add_dir(zf: zipfile.ZipFile, src: Path, arc_prefix: str) -> int:
    n = 0
    for p in src.rglob("*"):
        if p.is_file():
            zf.write(p, f"{arc_prefix}/{p.relative_to(src).as_posix()}")
            n += 1
    return n


def main() -> int:
    OUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    bear_synth = paths.DATASET / "bear_synth_v2"
    if not bear_synth.exists():
        raise SystemExit(f"먼저 13_compose_dataset_v2.py 실행: {bear_synth} 없음")

    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        n = add_dir(zf, bear_synth, "bear_synth_v2")
        zf.write(paths.CONFIGS / "pipeline.yaml", "pipeline.yaml")
    size_mb = OUT_ZIP.stat().st_size / 1e6
    print(f"[dataset zip] {OUT_ZIP}  {n}개 파일  {size_mb:.1f} MB")

    with zipfile.ZipFile(DOC_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for d, prefix in ((paths.DOCS, "docs"), (paths.REPORTS, "reports")):
            if d.exists():
                add_dir(zf, d, prefix)
        zf.write(paths.ROOT / "README.md", "README.md")
        zf.write(paths.CONFIGS / "pipeline.yaml", "pipeline.yaml")
    print(f"[doc zip] {DOC_ZIP}  {DOC_ZIP.stat().st_size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
