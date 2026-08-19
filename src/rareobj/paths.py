"""프로젝트 경로 단일 정의. 모든 스크립트가 여기서 경로를 받아 쓴다."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
DATASET = DATA / "dataset"

COCO_ANN = RAW / "coco" / "annotations"

DOCS = ROOT / "docs"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
CONFIGS = ROOT / "configs"


def ensure() -> None:
    for p in (RAW, INTERIM, DATASET, REPORTS, FIGURES, DOCS, CONFIGS):
        p.mkdir(parents=True, exist_ok=True)
