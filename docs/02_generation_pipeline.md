# 2. 생성 모델 기반 희소 클래스 데이터 수집 파이프라인

> 7/2 공지 제출 항목 중 "생성모델 기반 데이터 수집 과정 문서"에 해당한다.
> 대상 클래스·선정 근거는 [01_class_distribution_analysis.md](01_class_distribution_analysis.md).

## 2.0 방식 전환 — inpainting 직접 시도 → 합성(compose) 방식

최초 계획(1차 문서 말미)은 "실외 배경 사진에 SD inpainting으로 곰을 직접 채워
넣고, 지정한 마스크 사각형을 그대로 bbox로 쓴다"였다. 이 방식을 버리고 지금의
2단계(전경 생성 → 배경에 합성) 방식으로 바꿨다.

**이유.** 이 연구가 필요로 하는 곰은 대부분 512 캔버스 기준 32px 이하(작은
구간)다. Stable Diffusion 계열은 8배 다운샘플된 latent 공간에서 동작하므로,
32px 마스크는 latent 상에서 4px 미만이다. 그 안에 디퓨전이 "곰처럼 보이는" 구조를
새로 그려 넣을 해상도 여유가 없다 — 실제로 시험 삼아 작은 마스크에 inpainting을
돌려보면 색이 뭉친 얼룩만 나온다. 반면 큰 스케일(전신이 프레임을 채우는 정도)로
생성하면 디퓨전이 원래 잘하는 영역이라 결과가 훨씬 선명하다.

그래서 "선명한 곰을 큰 스케일로 만드는 것"과 "그것을 작게 배치하는 것"을
분리했다. 후자는 생성이 아니라 리사이즈+합성이므로 디퓨전의 해상도 한계와
무관하다. 부수 효과로 **bbox가 검출기 추론이나 마스크 재계산 없이 합성에 쓴
좌표 그 자체**가 되어 자동 라벨링의 기하학적 오차가 원천적으로 없다.

## 2.1 파이프라인 개요

```
① 전경 생성 (SDXL-Turbo, t2i)        ② 배경 생성 (SDXL-Turbo, t2i)
   "곰, 단순 배경" 프롬프트               "빈 야외/트레일캠 장면" 프롬프트
        │                                      │
   GrabCut 분리 → RGBA 컷아웃                  │
        │                                      │
   CLIP 품질 게이트 (bear 확신도)              │
        │                                      │
   근접중복 제거 (average hash)                │
        └───────────────┬──────────────────────┘
                         ▼
              ③ 프로그램적 합성 (compose.py)
        목표 크기 분포에 맞춰 리사이즈 → 배경에 alpha 합성
        bbox = 합성에 쓴 좌표 그 자체 (라벨 오차 없음)
                         │
                         ▼
              ④ 열화 파이프라인 (degrade.py)
     모션블러 → 다운스케일 왕복 → 센서노이즈 → JPEG 재압축
     (real_val_lq 평가셋과 동일 모듈 — 학습/평가 조건 일치)
                         │
                         ▼
              YOLO 포맷 데이터셋 (bear_synth)
```

## 2.2 사용 모델

| 역할 | 모델 | 소스 | 정밀도 | 파라미터 |
|---|---|---|---|---|
| 전경(곰)·배경 t2i | SDXL-Turbo | https://huggingface.co/stabilityai/sdxl-turbo | fp16 | steps=2, guidance_scale=0.0, 512×512 |
| 품질 게이트 분류기 | CLIP ViT-B/32 | https://huggingface.co/openai/clip-vit-base-patch32 | fp16/fp32 | zero-shot 라벨 매칭 |
| 전경/배경 분리 | OpenCV GrabCut | (딥러닝 아님, 고전 CV) | — | iter=5, GC_INIT_WITH_RECT |
| 검출기(평가·재학습 기준) | YOLOv8n | https://github.com/ultralytics/ultralytics (`yolov8n.pt`, COCO 사전학습) | fp32 | — |

**라이선스 유의.** SDXL-Turbo는 Stability AI Non-Commercial Research Community
License다 — 본 데이터셋은 연구/과제 제출 목적이며 상업적 재배포를 하지 않는다.
CLIP·YOLOv8n은 각각 MIT / AGPL-3.0.

로더 구현: [src/rareobj/genmodels.py](../src/rareobj/genmodels.py) — `load_sdxl_turbo()`,
`ClipScorer`, `average_hash()`. 실행 파라미터의 단일 진실 원천은
[configs/pipeline.yaml](../configs/pipeline.yaml) 의 `generation` 블록이다.

## 2.3 ① 전경(곰) 생성

구현: [scripts/04_generate_bear_foregrounds.py](../scripts/04_generate_bear_foregrounds.py)

프롬프트를 종·자세·배경 세 축으로 조합해 다양성을 확보한다.

- 종: brown/black/grizzly/polar/sun/sloth/asiatic black bear (7종)
- 자세: 서기/걷기/앉기/두 발로 서기/먹이 찾기/고개 돌리기/보행 중 (7종)
- 배경: 단순 스튜디오 배경 4종 (GrabCut 분리 성공률을 높이기 위해 의도적으로 단순화)
- negative prompt: 만화/일러스트/저품질/워터마크/여러 마리/기형 배제

품질 게이트 3단

| 게이트 | 기준 | 목적 |
|---|---|---|
| G1 GrabCut 분할 | 전경 비율 3~92% | 분할 실패(전경이 안 잡히거나 프레임 전체를 덮음) 배제 |
| G2 CLIP 판별 | top-1 라벨이 "a photo of a bear" & 2위와 margin ≥ 0.20 | 곰이 아닌 것(사람/개/빈 배경 등)이 섞이는 것 방지 |
| G3 근접중복 제거 | average hash 해밍 거리 < 6 이면 중복 판정 | 시드 충돌로 거의 같은 이미지가 여러 번 들어가는 것 방지 |

## 2.4 ② 배경 생성

구현: [scripts/05_generate_backgrounds.py](../scripts/05_generate_backgrounds.py)

5/14 공유 자료의 야생동물 활동 영상 도메인에 맞춰 숲길·초원·강변·설원·습지 등
10가지 주간 장면과 4가지 야간 적외선(트레일캠) 장면을 프롬프트로 구성한다.
프롬프트에 동물을 넣지 않지만 우연히 등장할 수 있어 CLIP으로
"이미 곰처럼 보이는 것이 있는가"를 검사해 임계치 초과 시 폐기한다(합성 단계에서
곰이 중복 배치되는 것을 막기 위함).

## 2.5 ③ 합성 — 정확한 bbox의 근원

구현: [scripts/06_compose_dataset.py](../scripts/06_compose_dataset.py),
[src/rareobj/compose.py](../src/rareobj/compose.py)

1. `configs/pipeline.yaml`의 목표 크기 분포(`small 0.60 / medium 0.30 / large 0.10`)에서
   이번 합성의 크기 구간을 뽑는다. COCO 정의(`area<32²`/`32²~96²`/`>96²`)를 그대로 쓴다.
2. 목표 면적에 맞춰 전경 컷아웃을 등비율로 리사이즈한다(가로세로 개별 왜곡 없음).
3. x는 캔버스 내 임의 위치, y는 하단 70% 쪽에 가중치를 둬 "땅을 딛고 선" 자연스러운
   배치를 유도한다.
4. 붙여넣기 전 배경 국소 영역과 밝기를 맞춰(gain 0.7~1.35) 이질감을 줄인다.
5. alpha 합성. 배경이 야간 적외선 태그면 합성 결과 전체를 그레이스케일로
   변환해 색 불일치를 원천 차단한다(실제 IR 영상은 애초에 무채색이므로 사실적).
6. **bbox = 3에서 계산한 리사이즈 크기와 4의 배치 좌표, 그 자체.** 검출기 추론이나
   마스크 재이진화가 끼어들 자리가 없다.

## 2.6 ④ 열화 — 실제 촬영 조건 시뮬레이션

구현: [src/rareobj/degrade.py](../src/rareobj/degrade.py)

합성 이미지 전체(배경+붙여넣은 곰)에 4단계를 **이 순서 그대로** 적용한다. 순서를
바꾸면 안 된다 — 예를 들어 JPEG를 블러보다 먼저 하면 블록 아티팩트가 블러에
뭉개져 실제 저화질 영상의 결과와 달라진다.

| 단계 | 방법 | 강도 범위 |
|---|---|---|
| 모션 블러 | 방향성 선형 커널 | 길이 3~15px, 각도 0~180° |
| 다운스케일 왕복 | INTER_AREA 축소 → INTER_LINEAR 복원 | 원본의 18~45% |
| 센서 노이즈 | Poisson(신호 의존) + Gaussian(고정) | σ 2~10, poisson_scale 40~180 |
| JPEG 재압축 | 인코드/디코드 왕복 | quality 12~45 |

**같은 함수를 real_val에도 적용해 `real_val_lq`를 만들었다** (1차 문서 §1.5, §1.7).
학습 데이터와 평가 데이터가 같은 열화 분포를 공유해야 "저화질 조건에서
좋아졌는가"를 공정하게 잴 수 있기 때문이다.

## 2.7 자동 라벨링 신뢰성 요약

| 위험 | 이 파이프라인에서의 처리 |
|---|---|
| bbox 좌표 오차 | 없음 — 합성 좌표를 그대로 라벨로 씀 (검출 기반 pseudo-label 아님) |
| 곰이 아닌 것이 섞임 | CLIP 게이트(G2)로 생성 단계에서 차단 |
| 거의 같은 이미지 중복 | average hash 근접중복 제거(G3) |
| 배경에 곰이 이미 있어 중복 라벨 | 배경 생성 시 CLIP bear-leak 검사로 차단 |
| 붙여넣기 티(부자연스러움) | 밝기 매칭 + 야간 그레이스케일 통일 + 이후 열화로 경계 흐림 |

## 2.8 실행 순서 (재현)

```bash
python scripts/03_build_lq_eval_set.py           # real_val_lq 평가셋
python scripts/04_generate_bear_foregrounds.py   # 곰 전경 생성 + 품질 게이트
python scripts/05_generate_backgrounds.py        # 배경 장면 생성
python scripts/06_compose_dataset.py             # 합성 + 열화 + YOLO 라벨
python scripts/07_build_regularization_sample.py # 다중 클래스 표본 (재앙적 망각 점검용)
python scripts/08_train_eval_yolo.py             # baseline vs 파인튜닝 평가
python scripts/09_package_submission.py          # 제출 zip 패키징
```

## 2.9 결과

→ [03_results.md](03_results.md) 에 정리 (baseline vs 파인튜닝 AP 비교, 데이터셋 통계).
