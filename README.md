# tour-anomaly-benchmark
#### 가로형 — 시간대가 열로 펼쳐진 격자 데이터
python -m detection grid300m --pattern "data/제주특별자치도_시간대별_관광객_300그리드단위_유동인구_비율포함_20140731.csv" --out result

#### 단일 계열 — 그룹 없이 지표만 여럿
python -m detection arrivals --pattern "data/입도 관광객 추이(일별).csv" --out result