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

> **baseline 자체가 문제를 증명한다** — 사전학습 YOLO의 bear mAP@0.5는 원본 화질
> 0.904 → 동일 71박스에 열화만 적용하면 0.399로 반토막난다. 생성 데이터로
> 파인튜닝하면 그 저화질 조건에서 0.399→0.478(+0.079), recall 0.282→0.451로
> 회복한다 — 단, 원본 화질·다른 79클래스 성능은 대가로 낮아진다(§[03_results](docs/03_results.md)).
> 정칙화 표본을 1:1로 오버샘플링해 그 대가(재앙적 망각)를 줄여봤지만 격차의
> 8%만 좁혀졌다 — 표본 비율만으로는 해결되지 않는다는 것까지 확인했다.

---

## 진행 상태

| 차수 | 내용 | 상태 |
|---|---|---|
| **1차** | 분포 전수 분석 → 희소 클래스 선정, 실데이터 기준셋 구축, 레포 골격 | ✅ 완료 |
| **2차** | 열화 모델 + 생성 파이프라인(전경/배경 분리 합성) + 자동 라벨링 + 품질 게이트 | ✅ 완료 |
| **3차** | 데이터셋 zip 패키징, 파이프라인 문서 완성, YOLO 재학습 검증 | ✅ 완료 |
| **4차** | 재앙적 망각 완화 실험(정칙화 오버샘플링 1:1, v1 대비 비교) | ✅ 완료 |

제출물(7/2 공지 기준)
1. 희소 객체 데이터셋 zip → `submission/bear_synth_dataset.zip`
2. 생성모델 기반 데이터 수집 과정 문서 → [docs/02_generation_pipeline.md](docs/02_generation_pipeline.md),
   결과는 [docs/03_results.md](docs/03_results.md)

## 구조

```
2026_RnE/
├─ configs/pipeline.yaml              파이프라인 단일 설정 (전 과정 확정값)
├─ docs/
│  ├─ 01_class_distribution_analysis.md   분포 분석 · 클래스 선정 근거
│  ├─ 02_generation_pipeline.md            생성 파이프라인 · 사용 모델 · 파라미터
│  └─ 03_results.md                        데이터셋 통계 · baseline 대비 평가 결과
├─ reports/                            전수 집계표, 선정/평가 결과, figure 7종
├─ scripts/
│  ├─ 00_download_coco.py             어노테이션 다운로드 + SHA-256 검증
│  ├─ 01_analyze_coco_distribution.py 전수 집계 → 희소 클래스 선정
│  ├─ 02_build_real_bear_sets.py      실 bear 이미지 → YOLO 기준셋
│  ├─ 03_build_lq_eval_set.py         real_val 열화 → real_val_lq 평가셋
│  ├─ 04_generate_bear_foregrounds.py SDXL-Turbo 곰 전경 생성 + 품질 게이트
│  ├─ 05_generate_backgrounds.py      SDXL-Turbo 배경 장면 생성
│  ├─ 06_compose_dataset.py           합성 + 열화 → bear_synth YOLO 데이터셋
│  ├─ 07_build_regularization_sample.py 다중 클래스 표본 (재앙적 망각 점검용)
│  ├─ 08_train_eval_yolo.py           baseline vs 파인튜닝 평가
│  └─ 09_package_submission.py        제출 zip 패키징
├─ src/rareobj/                       공용 모듈 (경로, 열화, 합성, 생성모델 로더, figure 스타일)
├─ submission/                        패키징된 제출 zip (.gitignore 대상)
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
python scripts/03_build_lq_eval_set.py         # real_val_lq 평가셋
python scripts/04_generate_bear_foregrounds.py # 곰 전경 생성 (SDXL-Turbo, ~7GB 다운로드)
python scripts/05_generate_backgrounds.py      # 배경 장면 생성
python scripts/06_compose_dataset.py           # 합성 데이터셋 조립 (600장)
python scripts/07_build_regularization_sample.py
python scripts/08_train_eval_yolo.py           # baseline vs 파인튜닝 평가
python scripts/09_package_submission.py        # 제출 zip
```

검증된 실행 환경: Windows 11 / Python 3.11.5 / torch 2.8.0+cu126 /
RTX 4060 Laptop 8 GB.

## 데이터 취급 원칙

- **실제 COCO 이미지는 레포·제출 zip 에 포함하지 않는다.** 449장 중 과반이 Flickr
  `NoDerivs` / `NonCommercial` 라이선스다. `manifest.json`(URL·라이선스)만 커밋하고
  스크립트로 재구성한다.
- `real_val`(49장/71박스)은 **홀드아웃 평가 전용**이며 학습·생성에 투입하지 않는다.
- 제출 zip 에는 생성 모델이 만든 이미지와 그 라벨만 넣는다.
