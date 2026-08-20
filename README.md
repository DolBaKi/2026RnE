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
> 0.904 → 동일 71박스에 열화만 적용하면 0.399로 반토막난다.
>
> **하지만 "생성 데이터로 파인튜닝하면 저화질에서 좋아진다"는 초기 결론은
> 진짜 저화질 실사진(`real_lowq`)으로 검증하자 뒤집혔다** — baseline이 이미
> 0.893으로 강했고, 파인튜닝은 오히려 0.726으로 떨어뜨렸다. 우리 열화 함수로
> 흐리게 만든 `real_val_lq`에서만 개선(+0.079)이 보였던 건 실제 블러 강인성이
> 아니라 그 열화 함수의 특정 시그니처에 대한 과적합이었을 가능성이 높다 —
> 자세한 진단은 [docs/04_yolo26_pipeline.md](docs/04_yolo26_pipeline.md) §4.5.

---

## 진행 상태

| 차수 | 내용 | 상태 |
|---|---|---|
| **1차** | 분포 전수 분석 → 희소 클래스 선정, 실데이터 기준셋 구축, 레포 골격 | ✅ 완료 |
| **2차** | 열화 모델 + 생성 파이프라인(전경/배경 분리 합성) + 자동 라벨링 + 품질 게이트 | ✅ 완료 |
| **3차** | 데이터셋 zip 패키징, 파이프라인 문서 완성, YOLO 재학습 검증 | ✅ 완료 |
| **4차** | 재앙적 망각 완화 실험(정칙화 오버샘플링 1:1, v1 대비 비교) | ✅ 완료 |
| **5차** | 방법론 개정(YOLO26·Albumentations·clean+deg 학습) + 진짜 저화질 평가(`real_lowq`) + NAFNet/Real-ESRGAN 복원 파이프라인 | ✅ 완료 |

5차는 지도 조교님의 연구방법 3.1~3.3 스펙을 반영해 1~4차의 핵심 가정 하나를
재검증한 결과다 — 무엇이 왜 바뀌었는지는
[docs/04_yolo26_pipeline.md](docs/04_yolo26_pipeline.md) §4.0에 표로 정리했다.

제출물(7/2 공지 기준)
1. 희소 객체 데이터셋 zip → `submission/bear_synth_dataset.zip`
2. 생성모델 기반 데이터 수집 과정 문서 → [docs/02_generation_pipeline.md](docs/02_generation_pipeline.md) /
   [docs/04_yolo26_pipeline.md](docs/04_yolo26_pipeline.md),
   결과는 [docs/03_results.md](docs/03_results.md) / [docs/04_yolo26_pipeline.md](docs/04_yolo26_pipeline.md)

## 구조

```
2026_RnE/
├─ configs/pipeline.yaml              파이프라인 단일 설정 (전 과정 확정값)
├─ docs/
│  ├─ 01_class_distribution_analysis.md   분포 분석 · 클래스 선정 근거
│  ├─ 02_generation_pipeline.md            생성 파이프라인 · 사용 모델 · 파라미터
│  ├─ 03_results.md                        1~4차 결과 (YOLOv8n, real_val_lq 기준)
│  └─ 04_yolo26_pipeline.md                5차: YOLO26·진짜 저화질 평가·복원 파이프라인
├─ reports/                            전수 집계표, 선정/평가 결과, figure 13종
├─ scripts/
│  ├─ 00_download_coco.py             어노테이션 다운로드 + SHA-256 검증
│  ├─ 01_analyze_coco_distribution.py 전수 집계 → 희소 클래스 선정
│  ├─ 02_build_real_bear_sets.py      실 bear 이미지 → YOLO 기준셋
│  ├─ 03_build_lq_eval_set.py         real_val 열화 → real_val_lq 평가셋 (1~4차용)
│  ├─ 04_generate_bear_foregrounds.py SDXL-Turbo 곰 전경 생성 + 품질 게이트
│  ├─ 05_generate_backgrounds.py      SDXL-Turbo 배경 장면 생성
│  ├─ 06_compose_dataset.py           합성 + 열화 → bear_synth (1~4차용)
│  ├─ 07_build_regularization_sample.py 다중 클래스 표본 (재앙적 망각 점검용)
│  ├─ 08_train_eval_yolo.py           YOLOv8n baseline vs 파인튜닝 (1~4차)
│  ├─ 09_package_submission.py        제출 zip 패키징
│  ├─ 10_finetune_balanced.py         4차: 정칙화 오버샘플링 재실험
│  ├─ 11_download_restoration_models.py NAFNet+Real-ESRGAN 체크포인트
│  ├─ 12_build_real_lowquality_testset.py 진짜 저화질 실사진 평가셋(real_lowq)
│  ├─ 13_compose_dataset_v2.py        5차: clean+Albumentations열화 쌍 데이터셋
│  ├─ 14_train_eval_yolo26.py         5차: YOLO26 학습 + 평가
│  ├─ 15_restore_real_lowq.py         real_lowq에 복원 전처리 적용
│  └─ 16_eval_restored_only.py        복원본 재평가 (재학습 없이)
├─ src/rareobj/                       공용 모듈 (경로, 열화, 합성, 복원, 생성모델 로더, figure 스타일)
│  └─ vendor/                         NAFNet/RRDBNet 아키텍처 벤더링 (basicsr 미설치)
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
python scripts/08_train_eval_yolo.py           # 1~4차: YOLOv8n baseline vs 파인튜닝
python scripts/09_package_submission.py        # 제출 zip
python scripts/10_finetune_balanced.py         # 4차: 오버샘플링 재실험

# 5차 — 방법론 개정 (YOLO26 · Albumentations · 진짜 저화질 평가)
python scripts/11_download_restoration_models.py
python scripts/12_build_real_lowquality_testset.py
python scripts/13_compose_dataset_v2.py
python scripts/14_train_eval_yolo26.py
python scripts/15_restore_real_lowq.py
python scripts/16_eval_restored_only.py
```

검증된 실행 환경: Windows 11 / Python 3.11.5 / torch 2.8.0+cu126 /
RTX 4060 Laptop 8 GB.

## 데이터 취급 원칙

- **실제 COCO 이미지는 레포·제출 zip 에 포함하지 않는다.** 449장 중 과반이 Flickr
  `NoDerivs` / `NonCommercial` 라이선스다. `manifest.json`(URL·라이선스)만 커밋하고
  스크립트로 재구성한다.
- `real_val`(49장/71박스)은 **홀드아웃 평가 전용**이며 학습·생성에 투입하지 않는다.
- 제출 zip 에는 생성 모델이 만든 이미지와 그 라벨만 넣는다.
