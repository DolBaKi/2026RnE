# 저화질 영상의 희소 객체 인식 모델 개발

사전학습 YOLO(COCO 80클래스)가 데이터 부족으로 놓치는 **희소 객체**를,
생성 모델로 **저화질·소스케일 조건의 학습 데이터를 직접 만들어** 보완하는 연구.

**타깃 희소 클래스: `bear`** — COCO train2017 80클래스 중 아래 3중 조건을
동시에 만족하는 유일한 클래스. 선정 근거 전문은
[docs/01_class_distribution_analysis.md](docs/01_class_distribution_analysis.md).

| 기준 | 임계 | bear |
|---|---|---|
| S1 인스턴스 수 하위 사분위 | ≤ 4,328 | **1,294** (77위/80) |
| S2 small(area<32²) 인스턴스 절대 개수 하위 사분위 | ≤ 620 | **37** |
| S3 야생동물 활동 영상 도메인 | — | ✓ |

> COCO train2017 어노테이션 860,001개 중 "작은 곰"은 **37개**다.
> 같은 도메인의 `bird`(6,904개)의 **1/187**.
> 사전학습 YOLO는 곰을 작고 흐릿한 형태로 본 경험이 사실상 없다.

---

## 진행 상태

| 차수 | 내용 | 상태 |
|---|---|---|
| **1차** | 분포 전수 분석 → 희소 클래스 선정, 실데이터 기준셋 구축, 레포 골격 | ✅ 완료 |
| **2차** | 열화 모델 + 생성 파이프라인(t2i / inpainting) + 자동 라벨링 + 품질 게이트 | ⬜ |
| **3차** | 데이터셋 zip 패키징, 파이프라인 문서 완성, YOLO 재학습 검증 | ⬜ |

제출물(7/2 공지 기준)
1. 희소 객체 데이터셋 zip → 3차
2. 생성모델 기반 데이터 수집 과정 문서 (파이프라인, 모델 HF/GitHub 주소·체크포인트·파라미터) → 2·3차

## 구조

```
2026_RnE/
├─ configs/pipeline.yaml              파이프라인 단일 설정 (1차 값 확정)
├─ docs/
│  └─ 01_class_distribution_analysis.md   분포 분석 · 클래스 선정 근거
├─ reports/
│  ├─ coco_class_stats.{csv,md}       80클래스 전수 집계표
│  ├─ rare_class_selection.json       선정 기준·임계·통과 클래스
│  ├─ selection_console.txt           선정 과정 로그
│  └─ figures/*.png                   근거 figure 3종
├─ scripts/
│  ├─ 00_download_coco.py             어노테이션 다운로드 + SHA-256 검증
│  ├─ 01_analyze_coco_distribution.py 전수 집계 → 희소 클래스 선정
│  └─ 02_build_real_bear_sets.py      실 bear 이미지 → YOLO 기준셋
├─ src/rareobj/                       공용 모듈 (경로, figure 스타일)
└─ data/                              .gitignore 대상 (재현 가능하므로 미커밋)
```

## 재현

```bash
pip install -r requirements.txt
# torch 는 CUDA 빌드를 별도 인덱스에서:
#   pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126

python scripts/00_download_coco.py             # COCO 어노테이션 241 MB
python scripts/01_analyze_coco_distribution.py # 집계표 + figure + 선정 결과
python scripts/02_build_real_bear_sets.py      # 실 bear 이미지 449장 + YOLO 라벨
```

검증된 실행 환경: Windows 11 / Python 3.11.5 / torch 2.8.0+cu126 /
RTX 4060 Laptop 8 GB.

## 데이터 취급 원칙

- **실제 COCO 이미지는 레포·제출 zip 에 포함하지 않는다.** 449장 중 과반이 Flickr
  `NoDerivs` / `NonCommercial` 라이선스다. `manifest.json`(URL·라이선스)만 커밋하고
  스크립트로 재구성한다.
- `real_val`(49장/71박스)은 **홀드아웃 평가 전용**이며 학습·생성에 투입하지 않는다.
- 제출 zip 에는 생성 모델이 만든 이미지와 그 라벨만 넣는다.
