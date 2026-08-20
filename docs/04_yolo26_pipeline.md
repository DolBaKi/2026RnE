# 4. 방법론 개정 — YOLO26 + Albumentations + 진짜 저화질 평가

> 지도 조교님이 준 연구방법 3.1~3.3 스펙을 반영해 1~4차 파이프라인을 다시 짠
> 결과다. 모델을 YOLO26으로, 열화 구현을 Albumentations로 바꾸고, 학습에
> 원본+열화 이미지를 함께 넣고, 평가를 "우리가 흐리게 만든 사진"이 아니라
> "원래부터 저화질/블러인 실제 사진"으로 바꿨다. 그 결과 이전 결과([03_results.md](03_results.md))의
> 핵심 결론 하나가 뒤집혔다 — §4.5에서 그대로 설명한다.

## 4.0 왜 다시 만들었나

| 항목 | 이전 (1~4차) | 방법론 3.1~3.3 반영 후 |
|---|---|---|
| 검출 모델 | YOLOv8n | **YOLO26n** |
| 열화 구현 | 직접 구현한 cv2 파이프라인(`degrade.py`) | **Albumentations**(`albu_degrade.py`) |
| 학습 데이터 | 열화본만 | **원본(clean) + 열화본을 함께** 사용 |
| 복원 전처리 | 검증 대상으로 취급(3차 계획) | **파이프라인에 항상 포함되는 보조 단계**(3.2) |
| 평가 데이터 | `real_val_lq` (깨끗한 실사진에 우리 열화를 입힌 것) | **`real_lowq`** (원래부터 저화질/블러인 실제 COCO 사진) |

## 4.1 검출 모델 — YOLO26

`ultralytics>=8.4.123` 부터 포함된 최신 모델. 사전학습 `yolo26n.pt` 는
COCO 80클래스, `bear` 인덱스는 기존과 동일하게 21이다(확인 완료).

## 4.2 학습 데이터 — clean + Albumentations 열화, 함께 사용

구현: [scripts/13_compose_dataset_v2.py](../scripts/13_compose_dataset_v2.py),
[src/rareobj/albu_degrade.py](../src/rareobj/albu_degrade.py)

2차에서 이미 만든 곰 전경(214개) · 배경(173개) 풀과 같은 시드를 그대로 재사용해
06과 동일한 합성(같은 bbox)을 재현하고, 이번에는 **원본(clean) 합성본과
Albumentations(`MotionBlur` + `Downscale`) 적용본을 둘 다 저장**한다. 열화가
bbox를 바꾸지 않으므로 두 파일이 라벨 하나를 공유한다.

- `bear_synth_v2`: 600쌍(=1200장), train 540쌍(1080장) / val 60쌍(120장)
- 학습 혼합셋 = `bear_synth_v2`(class→COCO index 21) + `reg_fit`(다른 79클래스,
  scripts/08과 동일한 분할 재사용 — 재앙적 망각 점검 조건을 이전과 비교 가능하게
  유지)

**Albumentations 버전 고정에 관한 실제 사고 기록.** `pip install albumentations`를
그냥 실행하면 최신 `albucore`(0.1.x+)가 `torch>=2.13`을 요구해서 pip 리졸버가
기존 CUDA 빌드 torch(2.8.0+cu126)를 CPU 전용 2.13으로 갈아치워 GPU 파이프라인
전체가 멈췄다. `albumentations==1.4.15` + `albucore==0.0.15`(하루 차이로 나온
서로 맞는 버전 쌍, torch 의존성 없음)를 `--no-deps`로 정확히 고정해서 해결했다
— [requirements.txt](../requirements.txt)에 사고 경위를 그대로 남겨 뒀다.

## 4.3 평가 데이터 — 진짜 저화질 실사진 (`real_lowq`)

구현: [scripts/12_build_real_lowquality_testset.py](../scripts/12_build_real_lowquality_testset.py)

기존 `real_val_lq`는 깨끗한 실사진에 **우리가 만든** 열화 함수를 입힌 것이라,
"우리 열화 함수의 특징을 인식하게 됐을 뿐"이라는 반박에서 자유롭지 않다. 그래서
COCO 실사진 449장(§1차 real_val+real_ref) 전체에서 **자연적으로 발생한**
저화질·블러 사진만 골랐다 — 어떤 인위적 가공도 거치지 않은, 촬영 조건 자체가
나빴던 사진들이다.

측정: bbox crop을 224px로 리사이즈 후 Laplacian 분산(낮을수록 흐림) + bbox
원본 픽셀 크기(작을수록 저해상도). 각 지표 하위 30% 이내인 인스턴스를 가진
이미지를 선택 → **605개 인스턴스 중 216개(148장)** 확보. 이 449장은 학습
파이프라인 어디에도 입력된 적이 없다(1차 문서 §1.5 — real_ref는 프롬프트 설계
참고용으로 사람이 눈으로만 봤을 뿐 모델에 들어간 적이 없다) — 학습-평가 누수
없이 그대로 평가셋으로 쓸 수 있다.

## 4.4 복원 전처리 — NAFNet(디블러) + Real-ESRGAN(업스케일)

구현: [src/rareobj/vendor/nafnet_arch.py](../src/rareobj/vendor/nafnet_arch.py),
[rrdbnet_arch.py](../src/rareobj/vendor/rrdbnet_arch.py),
[src/rareobj/restore.py](../src/rareobj/restore.py)

| 모델 | 체크포인트 | 설정 |
|---|---|---|
| NAFNet (디블러, GoPro) | https://huggingface.co/nyanko7/nafnet-models (`NAFNet-GoPro-width32.pth`) | width=32, enc=[1,1,1,28], mid=1, dec=[1,1,1,1], **TLC 적용**(train_size 256×256) |
| Real-ESRGAN (4x 업스케일) | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth | RRDBNet, num_block=23, scale=4 |

`basicsr`/`realesrgan` pip 패키지를 설치하지 않고 아키텍처 정의 파일만
직접 옮겨왔다(각 벤더 파일 docstring에 이유 기록) — `basicsr` 설치 시
`stringzilla`(Windows cp311 휠 부재)가 컴파일을 요구해 막히는 문제도 있었다.

**실제로 겪은 두 가지 불안정과 진단 과정.**

1. *NAFNet 수치 폭주.* 균일한 밝은 영역(암석, 하늘)이 넓은 사진에서 NAFNet
   출력이 [0,1]을 한참 벗어났다(-158~390). 공식 코드가 plain `NAFNet`이 아니라
   테스트 시 `NAFNetLocal`(TLC, Chu et al. arXiv:2112.04491)을 쓴다는 걸
   뒤늦게 확인했다 — SCA의 전역 평균풀링이 학습 크롭(256×256)과 테스트
   해상도(수백~수천 px) 사이의 통계 불일치를 일으키는 걸 국소 평균으로
   바로잡는 기법이다. 적용 후 폭주 폭은 줄었지만(-106~217) 완전히는 안
   없어졌다. NAFNet→Real-ESRGAN 사이에 `clamp(0,1)`을 추가로 넣었다.
2. *Real-ESRGAN 격자 아티팩트.* clamp를 넣고도 특정 사진(대부분 `real_lowq`가
   고른, 바로 그 "흐릿하고 디테일이 적은" 사진들)에서 규칙적인 격자무늬가
   나왔다. 정상 범위로 clamp된 입력을 Real-ESRGAN에 직접 넣어도 재현되는 걸
   확인해 **NAFNet의 문제가 아니라 Real-ESRGAN 자체가 스무스하고 정보량이
   적은 영역에서 고주파 디테일을 "환각"하다 만들어내는, GAN 기반 초해상도
   모델의 잘 알려진 실패 양상**으로 결론지었다. 코드로 고칠 버그가 아니라
   모델의 한계이므로 고치지 않고 결과에 그대로 반영했다 — 그리고 이게 §4.5의
   핵심 결과 하나를 설명한다.

![복원 실패 사례](figures/fig13_restore_failure_examples.png)

## 4.5 결과

학습: YOLO26n, 백본 앞 10개 층 동결, AdamW lr0=0.0008, 40 epoch,
`bear_synth_v2`(clean+deg, 1200장) + `reg_fit`(245장) 혼합.

| 세트 | 지표 | baseline | finetuned | Δ |
|---|---|---|---|---|
| **real_lowq** (진짜 저화질) | mAP@0.5 | **0.893** | **0.726** | **-0.167** |
| real_lowq | recall | 0.811 | 0.641 | -0.170 |
| real_lowq | precision | 0.912 | 0.846 | -0.066 |
| real_lowq + 복원전처리 | mAP@0.5 | 0.222 | 0.162 | -0.060 |
| real_lowq + 복원전처리 | recall | 0.206 | 0.154 | -0.051 |
| real_val (원본화질) | mAP@0.5 | 0.908 | 0.753 | -0.154 |
| reg_holdout (79클래스) | mAP@0.5 | 0.730 | 0.471 | -0.259 |

![YOLO26 결과](figures/fig12_yolo26_results.png)

### 핵심 발견 — 이전 결론이 뒤집혔다

1차~4차에서는 "생성 데이터로 파인튜닝하면 저화질 조건(`real_val_lq`)에서
mAP@0.5가 0.399→0.478로 개선된다"([03_results.md](03_results.md) §3.2)고
결론지었다. 그런데 **진짜 저화질 실사진(`real_lowq`)에서는 정반대다** —
baseline이 이미 0.893으로 강했고, 파인튜닝은 오히려 0.726으로 **떨어뜨렸다**.

두 결과를 나란히 놓으면:

| 평가 조건 | baseline | finetuned | 방향 |
|---|---|---|---|
| `real_val_lq` (우리 열화 함수로 흐리게 만든 실사진) | 0.399 | 0.478 | 개선 (+0.079) |
| `real_lowq` (원래부터 저화질인 실사진) | 0.893 | 0.726 | 악화 (-0.167) |

**가장 유력한 해석: 이전의 "개선"은 진짜 블러 강인성이 아니라 우리 자신의
열화 함수가 남기는 특정 시그니처(정확한 모션블러 커널, 정확한 다운스케일
비율, 정확한 JPEG 재압축 패턴)에 대한 과적합이었을 가능성이 높다.** `real_val_lq`
평가셋 자체가 학습 때 쓴 것과 물리적으로 매우 비슷한 열화를 실사진에 입힌
것이므로, 모델이 "그 열화 패턴이 보이면 곰이 있을 확률이 높다"는 얕은 규칙을
학습했어도 그 평가셋에서는 점수가 오른다. 반면 `real_lowq`는 실제 카메라
손떨림, 초점 실패, 압축 코덱, 저해상도 업로드 등 우리 모델이 전혀 흉내내지
않은 방식으로 저하된 사진이라, 그 얕은 규칙이 통하지 않고 오히려 clean 학습
분포에서 멀어진 대가만 남는다.

이건 정확히 3.3 방법론이 방지하려던 상황이다 — "학습에 쓴 인위적 열화가 아니라
실제 저화질 사진으로 검증"하지 않았다면 이 과적합을 발견하지 못한 채 "개선됐다"고
잘못 결론 내릴 뻔했다.

**복원 전처리는 도움이 되지 않았다.** baseline/finetuned 모두 복원 후 mAP@0.5가
0.89/0.73 → 0.22/0.16 으로 폭락했다 — §4.4에서 진단한 Real-ESRGAN의 격자
아티팩트 때문이다. `real_lowq`가 골라낸 사진들이 하필 이 실패 양상을 유발하는
"흐릿하고 정보량이 적은" 사진들이라 영향이 특히 컸다. 3.2가 명시한 대로 이건
중심 비교 대상이 아니라 보조 단계지만, 있는 그대로 정직하게 보고한다 — 현재
구성의 복원 전처리는 득보다 실이 크다.

## 4.6 종합 결론

| 질문 | 답 |
|---|---|
| 생성 데이터로 저화질 곰 검출이 실제로 좋아지는가? | **아니오** — 진짜 저화질 실사진 기준으로는 오히려 나빠졌다(-0.167) |
| 왜 이전엔 좋아진 것처럼 보였는가? | 평가셋(`real_val_lq`)이 학습 열화와 물리적으로 닮아 있어 과적합을 개선으로 오인했을 가능성 |
| 복원 전처리(NAFNet+Real-ESRGAN)는 도움이 되는가? | **아니오** — 현재 모델 조합은 Real-ESRGAN의 격자 아티팩트 때문에 오히려 크게 해롭다 |
| 재앙적 망각은? | 여전함(-0.259), 4차에서 시도한 오버샘플링으로도 크게 개선되지 않음 |

## 4.7 다음 시도

| 항목 | 제안 |
|---|---|
| 열화 다양성 부족 | 학습 열화에 다양한 블러 커널·압축 코덱·노이즈 모델을 더 섞어 특정 시그니처에 덜 치우치게 |
| 실제 열화 특성 반영 | `real_lowq`의 실제 블러/노이즈 특성을 역으로 분석해 학습 열화 파라미터 범위를 거기 맞추기 |
| 복원 모델 교체 | Real-ESRGAN 대신 격자 아티팩트가 덜한 초해상도 모델(예: SwinIR) 검토, 혹은 저화질 정도를 감지해 복원을 선택적으로만 적용 |
| 파인튜닝 안정성 | freeze 깊이·lr·epoch을 reg_holdout·real_lowq 양쪽을 기준으로 같이 조기종료하도록 재설계 |

## 4.8 실행 순서 (재현)

```bash
python scripts/12_build_real_lowquality_testset.py  # real_lowq 평가셋
python scripts/13_compose_dataset_v2.py             # clean+deg 학습 데이터
python scripts/11_download_restoration_models.py    # NAFNet+Real-ESRGAN 체크포인트
python scripts/14_train_eval_yolo26.py              # YOLO26 학습 + 1차 평가
python scripts/15_restore_real_lowq.py              # real_lowq 복원본 생성
python scripts/16_eval_restored_only.py             # 복원본 재평가 (14 재학습 없이)
```
