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
>
> **6차(CCTV 파일럿, 실사진+열화)에서 다른 도메인·다른 클래스로 독립
> 재현했다** — 사람·자전거·차량·가방 등 8개 흔한 클래스로 저화질 CCTV 탐지를
> 시도했더니, 여기서도 파인튜닝은 개선이 아니라 악화(-0.090)였고, 복원
> 전처리도 똑같이(오히려 더 심하게) 성능을 무너뜨렸다(0.513→0.054) —
> [docs/05_cctv_pipeline.md](docs/05_cctv_pipeline.md) §5.5.
>
> **7차에서 bear와 완전히 동일한 구조(생성모델로 부족한 조건을 직접 만들어
> 학습)로 CCTV를 다시 시도하자 훨씬 더 나쁜 결과가 나왔다** — 순수 생성
> 데이터(840장)로만 파인튜닝하니 진짜 저화질에서 -0.389, **학습에 전혀
> 쓰이지 않은 일반 실사진에서조차** -0.278 악화됐다. 이 프로젝트 전체의
> 결론: **생성형 데이터 증강은 사전학습이 "거의 모르는" 조건(bear)에서만
> 위험이 적고, 이미 강한 지식(흔한 클래스의 실사진 인식)을 소량의 순수
> 합성 데이터로 덮어쓰면 오히려 크게 해롭다** —
> [docs/06_cctv_generative.md](docs/06_cctv_generative.md) §6.5.
>
> **8차 — 주제를 "저화질 CCTV 인파 속 특정인 찾기(얼굴인식)"로 전환했다.** LFW+VGGFace2를
> 사람별 클래스로 통합하고, SDXL-Turbo 인파 장면에 **IP-Adapter FaceID 신원 조건부
> 인페인팅으로 특정 인물을 AI가 직접 그려 넣어** 10명 데이터셋을 만들었다. 실사진에 이
> AI 인파 데이터와 배경 군중 하드 네거티브를 섞어 파인튜닝하자 인파 속 찾기가
> 0.233 → 0.537(2.3배, 오인율 1% 기준)로 올랐고, 원래부터 저화질인 실사진도
> 0.767 → 0.932로 개선됐다. **AI 인파만으로 학습하면 실사진 인식이 무너져(0.97 → 0.46)
> 7차 결론이 재현됐다** — 합성은 실사진에 섞을 때만 쓸모가 있다 —
> [docs/07_face_crowd.md](docs/07_face_crowd.md).

---

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
│  ├─ 04_yolo26_pipeline.md                5차: YOLO26·진짜 저화질 평가·복원 파이프라인
│  ├─ 05_cctv_pipeline.md                  6차: CCTV 다중 클래스 파일럿(실사진+열화)
│  ├─ 06_cctv_generative.md                7차: CCTV 생성형 증강 파일럿·도메인 갭 확인
│  └─ 07_face_crowd.md                     8차: CCTV 인파 속 특정인 찾기(얼굴인식, 10명)
├─ reports/                            전수 집계표, 선정/평가 결과, figure 19종
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
│  ├─ 16_eval_restored_only.py        복원본 재평가 (재학습 없이)
│  ├─ 20_build_cctv_dataset.py        6차: CCTV 8클래스 clean+deg 데이터셋(실사진)
│  ├─ 21_build_cctv_real_lowq.py      6차/7차: 진짜 저화질 실사진 평가셋
│  ├─ 22_train_eval_cctv.py           6차: YOLO26 학습 + 평가
│  ├─ 23_restore_cctv_lowq.py         6차/7차: 복원 전처리 적용
│  ├─ 24_generate_cctv_foregrounds.py 7차: SDXL-Turbo 8클래스 전경 생성
│  ├─ 25_generate_cctv_backgrounds.py 7차: CCTV 시점 배경 생성
│  ├─ 26_compose_cctv_synth.py        7차: 다중클래스 장면 합성(생성데이터)
│  ├─ 27_train_eval_cctv_gen.py       7차: 생성데이터로 학습 + 평가
│  ├─ 30_build_face_dataset.py        8차: LFW+VGGFace2 → 사람별 정렬 얼굴 (6,289명)
│  ├─ 30b_export_identity_classes.py  8차: 클래스 색인 (20장 미만 제외)
│  ├─ 31_build_face_eval_sets.py      8차: 1:N 갤러리 + 자연 저해상도 프로브 + LFW pairs
│  ├─ 32_generate_face_cctv_scenes.py 8차: SDXL-Turbo 인파 장면 + YOLO26-pose 정면 슬롯
│  ├─ 33_compose_face_cctv_synth.py   8차: 신원 조건부 인페인팅 삽입 로직 (480명판)
│  ├─ 34_train_eval_face.py           8차: 학습/평가 공용 로직 (480명판)
│  ├─ 35_build_crowd_test_dets.py     8차: 인파 테스트 프레임 얼굴 검출·정답 매칭
│  ├─ 40_build_face10.py              8차: 10명 타깃 선정·정렬·분할
│  ├─ 41_compose_face10_cctv.py       8차: 10명 + 모르는 사람 AI 인파 프레임
│  ├─ 42_mine_crowd_negatives.py      8차: 배경 군중 하드 네거티브
│  ├─ 43_train_eval_face10.py         8차: baseline vs 파인튜닝 3종
│  ├─ 44_report_face10.py             8차: 결과표·그림
│  └─ find_person.py                  8차: CCTV 속 특정인 찾기 도구
├─ src/rareobj/                       공용 모듈 (경로, 열화, 합성, 복원, 생성모델 로더, FaceID 인페인팅, figure 스타일)
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

# 6차 — CCTV 다중 클래스 파일럿 (실사진+열화, 5차 결론 재검증)
python scripts/20_build_cctv_dataset.py
python scripts/21_build_cctv_real_lowq.py
python scripts/23_restore_cctv_lowq.py
python scripts/22_train_eval_cctv.py

# 7차 — CCTV 생성형 증강 파일럿 (bear와 동일 구조)
python scripts/24_generate_cctv_foregrounds.py
python scripts/25_generate_cctv_backgrounds.py
python scripts/26_compose_cctv_synth.py
python scripts/27_train_eval_cctv_gen.py

# 8차 — CCTV 인파 속 특정인 찾기 (얼굴인식, 10명) — 순서·소요시간은 docs/07 §7.8
pip install --no-deps facenet-pytorch peft
python scripts/30_build_face_dataset.py
python scripts/31_build_face_eval_sets.py
python scripts/30b_export_identity_classes.py
python scripts/32_generate_face_cctv_scenes.py
python scripts/40_build_face10.py
python scripts/41_compose_face10_cctv.py
python scripts/35_build_crowd_test_dets.py --data faces10
python scripts/42_mine_crowd_negatives.py
python scripts/43_train_eval_face10.py
python scripts/44_report_face10.py
python scripts/find_person.py --targets10 --input <cctv 이미지/폴더/동영상> --weights runs/face10/real_synth_s0.pt
```

검증된 실행 환경: Windows 11 / Python 3.11.5 / torch 2.8.0+cu126 (1~7차), 2.7.0+cu128 (8차) /
RTX 4060 Laptop 8 GB.

## 데이터 취급 원칙

- **실제 COCO 이미지는 레포·제출 zip 에 포함하지 않는다.** 449장 중 과반이 Flickr
  `NoDerivs` / `NonCommercial` 라이선스다. `manifest.json`(URL·라이선스)만 커밋하고
  스크립트로 재구성한다.
- `real_val`(49장/71박스)은 **홀드아웃 평가 전용**이며 학습·생성에 투입하지 않는다.
- 제출 zip 에는 생성 모델이 만든 이미지와 그 라벨만 넣는다.
- 8차 얼굴 데이터(LFW·VGGFace2 원본, 정렬본, 실존 인물을 그려 넣은 합성 프레임, 가중치)는
  레포에 올리지 않고 연구 목적 외로 배포하지 않는다.
