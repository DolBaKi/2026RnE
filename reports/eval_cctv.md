# CCTV 파일럿 — baseline vs 파인튜닝 (8클래스 person/vehicle/carrying-object)

| 세트 | 지표 | baseline | finetuned | Δ |
|---|---|---|---|---|
| real_lowq_cctv | map50 | 0.513 | 0.423 | -0.090 |
| real_lowq_cctv | recall | 0.450 | 0.405 | -0.045 |
| real_lowq_cctv | precision | 0.657 | 0.544 | -0.114 |
| real_lowq_cctv_restored | map50 | 0.054 | 0.046 | -0.008 |
| real_lowq_cctv_restored | recall | 0.057 | 0.052 | -0.005 |
| real_lowq_cctv_restored | precision | 0.398 | 0.417 | +0.019 |
| cctv_val | map50 | 0.352 | 0.340 | -0.012 |
| cctv_val | recall | 0.324 | 0.325 | +0.000 |
| cctv_val | precision | 0.507 | 0.521 | +0.014 |