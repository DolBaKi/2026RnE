# CCTV 생성데이터 파일럿 — baseline vs 파인튜닝(생성데이터로 학습)

| 세트 | 지표 | baseline | finetuned | Δ |
|---|---|---|---|---|
| real_lowq_cctv | map50 | 0.513 | 0.124 | -0.389 |
| real_lowq_cctv | recall | 0.450 | 0.149 | -0.301 |
| real_lowq_cctv | precision | 0.657 | 0.339 | -0.318 |
| real_lowq_cctv_restored | map50 | 0.054 | 0.009 | -0.045 |
| real_lowq_cctv_restored | recall | 0.057 | 0.017 | -0.039 |
| real_lowq_cctv_restored | precision | 0.398 | 0.100 | -0.298 |
| cctv_val_real | map50 | 0.352 | 0.074 | -0.278 |
| cctv_val_real | recall | 0.324 | 0.111 | -0.214 |
| cctv_val_real | precision | 0.507 | 0.153 | -0.355 |