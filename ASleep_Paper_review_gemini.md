# Asleep Inc. 소리 기반 수면 AI 연구 포트폴리오

서울 기반 수면 기술 기업 Asleep Inc.(구 AsomBrain)는 **2021~2023년에 걸쳐 총 21편의 논문**을 발표하며, 소리 기반 수면 분석 플랫폼을 체계적으로 구축해왔다. 핵심 기술 파이프라인은 스마트폰 마이크 오디오 → Mel 스펙트로그램 → 딥러닝 → 수면 단계 분류, OSA 이벤트 감지, 코골이 모니터링으로 구성된다. 본 보고서는 각 논문의 주제, 모델 구조(Input/Output), 핵심 기여를 상세히 분석한다. 논문 2/6, 3/8, 14/17, 15/18, 19/20은 동일 연구의 저널/학회 초록 쌍이며, 해당 부분에서 명시하였다.

---

## 1. 기반 아키텍처: SoundSleepNet과 후속 모델 (논문 19–21)

**논문 21 — "Confidence-Based Framework Using Deep Learning for Automated Sleep Stage Scoring"** _Nature and Science of Sleep, 2021.12.24_

- **주제:** 자동 수면 단계 판정에서 잠재적 오류 예측을 감지하고 표적 검토를 가능하게 하는 신뢰도 추정 프레임워크 제안.
- **모델:** **TinySleepNet**(경량 CNN + 단방향 LSTM)을 5-class EEG 기반 수면 판정(W/N1/N2/N3/REM)에 적용. **SeqConfidNet**이라는 보조 신뢰도 모델이 에포크별 예측 신뢰도를 추정. 새로운 **Dropout Correct Rate (DCR)** 방법은 활성 드롭아웃 상태에서 다중 순전파를 수행하고, 정확한 예측 비율을 신뢰도 대리지표로 사용. **Input:** 단일 채널 EEG → **Output:** 30초 에포크당 수면 단계 라벨 + 신뢰도 점수.
- **핵심 기여:** 자동 수면 판정에 신뢰도 추정을 최초 적용. DCR은 오답 탐지에서 **AUROC 0.812** 달성. **최하위 20% 신뢰도 에포크만 재검토**하면 전체 정확도가 **76%→87%**로 향상되어, 임상 수면 판정에서 인간-AI 협업을 실현. SNUBH 702건, SHHS 2,804건 PSG로 검증.

**논문 19 — "End-to-End Sleep Staging Using Nocturnal Sounds from Microphone Chips for Mobile Devices"** _Nature and Science of Sleep, 2022.6.25_

- **주제:** 스마트폰 MEMS 마이크로 수집한 야간 소리에서 직접 수면 단계를 판정하는 최초의 엔드투엔드 딥러닝 모델 **SoundSleepNet** 제안.
- **모델:** 2단계 아키텍처. Phase 1: **특징 추출기**(CNN "Listener" + 양방향 LSTM + Transformer 인코더)를 개별 30초 Mel 스펙트로그램으로 사전학습. Phase 2: **다중 에포크 분류기**(Bi-LSTM + Transformer 인코더 + FC 레이어)를 **40개 연속 Mel 스펙트로그램** 시퀀스로 학습, **중간 20개 에포크**에 대해 4-class 수면 단계(Wake/Light/Deep/REM) 출력. 전처리에 적응적 스펙트럼 게이팅 잡음 억제 및 피치 시프팅 증강 적용. **Input:** Mel 스펙트로그램 시퀀스(40에포크) → **Output:** 4-class 수면 단계(20에포크).
- **핵심 기여:** 저품질 비접촉 스마트폰 오디오를 위한 최초의 Mel 스펙트로그램→수면 단계 엔드투엔드 모델. PSG-마이크 데이터(N=1,154)에서 **에포크별 정확도 70%**, 스마트폰 녹음(N=327)에서 **68%** 달성. 다양한 SNR 조건에서도 견고한 성능 유지. 이후 모든 Asleep 모델의 기반 아키텍처 확립.

**논문 20 — "Sleep staging using end-to-end deep learning model based on nocturnal sound for smartphones"** _Sleep 2022 (학회 초록), 2022.5.25_

- **주제:** 논문 19의 SoundSleepNet을 발표한 학회 초록.
- **모델:** 논문 19와 동일 — 2-서브모델 SoundSleepNet으로 4-class 수면 판정.
- **핵심 기여:** SLEEP 2022에서 수면의학 커뮤니티에 SoundSleepNet 소개. PSG 대비 **70% 일치도** 시연, 스마트폰 통합 가능성 강조.

---

## 2. 병원→가정 도메인 갭 극복 (논문 14–18)

**논문 18 — "Prediction of Sleep Stages Via Deep Learning Using Smartphone Audio Recordings in Home Environments"** _JMIR, 2023.6.1 (DOI: 10.2196/46216)_

- **주제:** 통제되지 않는 배경 소음으로 오디오 품질이 심각하게 저하되는 실제 가정 환경에서 작동하도록 SoundSleepNet을 확장한 **HomeSleepNet** 제안.
- **모델:** SoundSleepNet의 40→20 Mel 스펙트로그램 구조 유지. 학습 전략 3가지: (1) 812건 라벨 병원 PSG+오디오로 **지도 학습**; (2) 829건 비라벨 가정 스마트폰 녹음으로 적대적 도메인 판별기를 활용한 **비지도 도메인 적응**; (3) Freesound에서 수집한 **8,255개 가정 소음 샘플**을 랜덤 SNR(−10~10 dB)로 증강하고, Jensen-Shannon 발산 손실로 예측 불변성을 강제하는 **일관성 학습**. **Input:** 스마트폰 Mel 스펙트로그램 → **Output:** 에포크당 3-class(Wake/NREM/REM) 또는 4-class 수면 단계.
- **핵심 기여:** 소리 기반 수면 판정의 최초 가정환경 검증. 전이학습+일관성 학습 조합으로 가정 정확도 **69.2%(지도학습만) → 76.2%**(3-class) 향상, Macro F1 = 0.714. 45건 가정 PSG+스마트폰 세션으로 검증. 연령, 성별, BMI, AHI 등 모든 인구통계 하위그룹에서 일관된 성능.

**논문 17 — "Real-Time Detection of Sleep Apnea Based on Breathing Sounds and Prediction Reinforcement Using Home Noises"** _JMIR, 2023.2.22 (DOI: 10.2196/44818)_

- **주제:** 호흡 소리를 이용한 실시간 에포크별 OSA 이벤트 감지기 개발. 가정 환경 견고성을 위한 새로운 소음 증강 전략 제안.
- **모델:** SoundSleepNet을 수면무호흡 감지에 적용, 더 짧은 **14→10 아키텍처**(14개 입력 Mel 스펙트로그램 → 10개 중간 에포크 예측). 수면 판정보다 짧은 시간적 맥락을 반영. 특징 추출기가 무호흡 호흡 패턴을 식별하고, 다중 에포크 감지기가 이벤트 분류. **Output:** 에포크당 3-class(**무호흡 / 저호흡 / 정상**). 학습에 교차 엔트로피 손실 + **가정 소음 일관성 학습** 적용 — 9개 카테고리 22,500개 소음 샘플을 SNR −20~5 dB로 추가, MSE 일관성 손실로 깨끗한/노이즈 입력에 동일 예측 강제. 클래스 가중치(1.0/1.3/2.1)로 불균형 처리.
- **핵심 기여:** 호흡 소리만으로 에포크별 실시간 OSA 이벤트 감지를 최초 실현. **정확도 86%, Macro F1 = 0.75** (3-class). AHI ≥ 15 기준: 민감도 **0.85**, 특이도 **0.84**. AHI ≥ 5 기준: 민감도 **0.97**, AUC **0.93**. 다양한 SNR 조건에서 소음 견고성 검증.

**논문 16 — "Sound-Based Sleep Staging by Exploiting Real-World Unlabeled Data"** _ICLR 2023 Workshop (구두 발표), 2023.3.2_

- **주제:** 대량의 비라벨 가정 수면 녹음을 활용하는 새로운 반지도 학습 방법론으로 소리 기반 수면 판정 성능 향상.
- **모델:** **SleepFormer** 제안 — SoundSleepNet의 백본을 **MobileViTv2**(경량 비전 트랜스포머)로, 시퀀스 헤드를 **ViT**(Vision Transformer)로 업그레이드. 40-에포크 Mel 스펙트로그램 시퀀스로 4-class 판정. **2,574건 라벨 병원 데이터 + 2,731건 비라벨 가정 녹음**으로 학습. 2가지 새로운 반지도 손실: (1) **Sequential Consistency Loss (L_SC)** — 증강된 시퀀스 버전 간 쌍별 코사인 유사도 행렬을 거리 가중 마스킹으로 매칭, 시간적 구조 보존; (2) **Semi-Supervised Contrastive Learning (SSCL)** — 라벨 샘플을 신뢰 앵커로 사용한 대조 클러스터링, 비라벨 녹음의 OOD 데이터(빈 방, 2인 수면, 음악 재생 등) 처리.
- **핵심 기여:** SleepFormer + 반지도 학습으로 가정-PSG F1을 **0.5718(SoundSleepNet 기준) → 0.6804**로 +0.109 대폭 향상. Sequential Consistency Loss와 SSCL은 수면 데이터의 시계열 특성과 실세계 비라벨 녹음의 OOD 문제를 해결 — 표준 반지도 방법으로는 대응 불가능한 과제.

**논문 15 — "Sound-Based Sleep Staging at Home Using Smartphone via Deep Learning"** _Sleep 2023 (학회 초록), 2023.5.29_

- **주제:** HomeSleepNet(논문 18과 동일) 학회 발표 초록.
- **모델:** 논문 18과 동일 — 전이학습 + 일관성 학습 적용 SoundSleepNet.
- **핵심 기여:** SLEEP 2023에서 HomeSleepNet 결과 요약: 가정 3-class 정확도 76.2%, 병원 밖 스마트폰 수면 판정의 실현 가능성 입증.

**논문 14 — "Real-time acoustic apnea event detector by training a deep learning model with home noise added data"** _Sleep 2023 (학회 초록), 2023.5.29_

- **주제:** 논문 17의 OSA 이벤트 감지기 학회 초록.
- **모델:** 논문 17과 동일 — 14→10 아키텍처, 3-class OSA 이벤트 감지, 가정 소음 일관성 학습.
- **핵심 기여:** 동일 핵심 결과 발표: 정확도 86%, Macro F1 = 0.75의 실시간 무호흡/저호흡/정상 감지.

---

## 3. 멀티태스크 학습, 일주기 리듬 특징, 아키텍처 혁신 (논문 5, 7, 12)

**논문 7 — "Enhancing both Sleep Stage Classification and Obstructive Sleep Apnea Event Detection tasks with a unified sound-based multi-task model"** _World Sleep 2023, 2023.10.20_

- **주제:** 동일 오디오 입력에서 수면 단계 판정과 OSA 이벤트 감지를 동시에 수행하는 통합 멀티태스크 모델 제안.
- **모델:** **공유 특징 추출기**(호흡 소리 Mel 스펙트로그램 처리)가 **2개의 별도 Transformer 인코더 헤드**에 연결 — 하나는 4-class 수면 판정(Wake/Light/Deep/REM), 다른 하나는 3-class OSA 감지(무호흡/저호흡/정상). **지식 증류** 방식 학습: 2개의 단일태스크 "교사" 모델을 독립적으로 먼저 학습한 후, 멀티태스크 "학생" 모델을 PSG 라벨과 교사 생성 특징 모두로 학습. 데이터: 병원 PSG 2,048건.
- **핵심 기여:** 소리 기반 수면 판정 + OSA 감지 통합 멀티태스크 모델 최초 실현. Macro F1: **수면 판정 0.690, OSA 감지 0.635** — 개별 교사 모델을 소폭 상회하면서 **파라미터 ~50% 절감**. 지식 증류를 통해 학생이 전문화된 교사 지식의 혜택을 받음.

**논문 12 — "The utilization of circadian rhythm features to improve sound-based AI sleep staging"** _World Sleep 2023, 2023.10.20_

- **주제:** 수면 조절의 2-과정 모델에서 도출한 일주기 리듬 특징을 결합하여 소리 기반 수면 판정 성능 향상.
- **모델:** 기본 아키텍처는 교차 엔트로피 손실로 미세조정한 **MobileViTv2**. 입력에 Mel 스펙트로그램과 3가지 일주기 특징을 결합: (1) **멜라토닌 주기** — 코사인 함수 모델링; (2) **수면 압력** — 수면 시작 후 지수 감소 모델링; (3) **수면 경과 시간** — 경과 시간의 선형 함수. **Output:** 4-class 수면 판정. 학습: 병원 2,574건; 평가: 가정 스마트폰 128건.
- **핵심 기여:** 생리학적 일주기 지식과 딥러닝의 새로운 융합. 3가지 특징 모두 추가 시 가정 정확도 **67.0% → 70.4%**, Macro F1 **0.640 → 0.669** 향상. 절제 실험에서 각 특징이 점진적으로 기여하며, 수면 압력이 단일 최대 향상 제공. 단순한 일주기 생물학 수학적 모델이 소리 기반 수면 판정을 유의미하게 개선함을 입증.

**논문 5 — "SnoreFormer: Home snoring detection with deep neural networks"** _World Sleep 2023, 2023.10.20_

- **주제:** 가정 환경 스마트폰 녹음 오디오에서 코골이 이벤트를 감지하는 Transformer 기반 모델 SnoreFormer 개발.
- **모델:** 20분 오디오를 **30초 Mel 스펙트로그램 40에포크**로 분할 입력. **Self-attention Transformer** 아키텍처가 40-에포크 시퀀스의 시간적 의존성을 모델링, 에포크별 이진 코골이 분류(있음/없음) 출력. 학습: 병원 PSG-마이크 1,154건 + 병원 스마트폰 327건; 테스트: 가정 스마트폰 109건 포함.
- **핵심 기여:** 코골이 감지에 Transformer self-attention을 새롭게 적용, 20분 컨텍스트 윈도우의 시간적 코골이 패턴 포착. 임상 **82.9%**, 가정 **81.0%** 정확도 달성. 성별(남 81.5%, 여 85.1%), BMI, 연령 범주에서 견고. 이후 Nature and Science of Sleep(2025) 정규 논문으로 확장.

---

## 4. 임상 검증 연구 (논문 1, 2, 3/8, 6, 13)

**논문 1 — "In-Home Smartphone-Based Prediction of Obstructive Sleep Apnea in Conjunction With Level 2 Home Polysomnography"** _JAMA Otolaryngology, 2023.11.16_

- **주제:** Level 2 가정 PSG를 참조 표준으로 사용하여, 실제 가정 환경에서 스마트폰 기반 OSA 예측을 최초 검증.
- **모델:** 기개발 딥러닝 모델(논문 17/18 기반)이 스마트폰 마이크 녹음 호흡 소리(16 kHz, 단일 채널) 처리, AHI 5/15/30 이벤트/시간 기준 중증도 분류 예측. iOS와 Android 스마트폰 동시 테스트. **Output:** 4-class OSA 중증도(정상/경도/중등도/중증).
- **핵심 기여:** Level 2 PSG를 사용한 최초 가정환경 검증(기존 연구는 병원 PSG만 사용). 12개월간 **101명** 참여. iOS 결과: AHI 기준 5/15/30에서 민감도 **92.6%/90.9%/93.3%**; 특이도 **84.3%/94.4%/94.4%**; 정확도 **88.6%/93.3%/94.3%**. Android도 거의 동일한 성능. 양 플랫폼 모두 Cohen's κ **0.6~0.8 범위**, 실제 가정 조건에서 실질적 임상 일치도 입증.

**논문 2 — "Accuracy of 11 Wearable, Nearable, and Airable Consumer Sleep Trackers: Prospective Multicenter Validation Study"** _JMIR mHealth and uHealth, 2023.11.2_

- **주제:** 3개 기기 카테고리의 소비자 수면 트래커를 병원 PSG와 대조한 최대 규모 다기관 비교 검증.
- **모델:** 비교 검증 연구(모델 개발 아님). **11개 기기** 평가: 웨어러블 5종(Google Pixel Watch, Galaxy Watch 5, Fitbit Sense 2, Apple Watch 8, Oura Ring 3), 니어러블 3종(Withings Mat, Google Nest Hub 2, Amazon Halo Rise), 에어러블 3종(Asleep **SleepRoutine**, SleepScore, Pillow). 각 기기는 고유 센싱 모달리티(가속도계, PPG, 레이더, 오디오) 사용. **Output:** PSG와 에포크별 비교한 4-class 수면 단계 + 수면 지표.
- **핵심 기여:** **75명 참여, 349,114 에포크, 3,890시간 CST 데이터** 분석. Asleep의 SleepRoutine이 전체 11개 기기 중 **최고 Macro F1(0.6863)** 달성, 각성 및 REM 감지에서 우수. 소리 기반 비접촉 트래커를 지칭하는 **"에어러블(airable)"** 용어 도입. 소리 기반 앱이 웨어러블·니어러블 하드웨어 기기를 능가할 수 있음을 입증.

**논문 6 — "Comparative Analysis of 11 Consumer Sleep Trackers with Polysomnography"** _World Sleep 2023, 2023.10.20_

- **주제:** 논문 2와 동일 연구의 학회 초록.
- **모델/핵심 기여:** 논문 2와 동일 — SleepRoutine이 Macro F1 0.6863으로 전 기기 선두.

**논문 3 & 8 — "Enhancing Robustness of a Sound-Based AI Model for Automated Sleep Staging: Validating on Unseen Open Dataset"** _World Sleep 2023, 2023.10.20 (앙코르 발표로 2회 발표)_

- **주제:** 학습에 사용되지 않은 공개 데이터셋에서 Asleep 소리 기반 수면 판정 모델의 교차 데이터셋 일반화 테스트.
- **모델:** SoundSleepNet/SleepFormer 핵심 아키텍처로 Mel 스펙트로그램 기반 수면 단계 분류, 외부 공개 PSG-Audio 데이터셋에 재학습 없이 적용.
- **핵심 기여:** 도메인 일반화 입증 — 학습 데이터와 완전히 다른 소스, 녹음 조건, 인구 집단의 데이터에서도 의미 있는 성능 발휘. 임상적 신뢰성과 재현성 확립에 핵심적.

**논문 13 — "Evaluation of a Sound-Based Deep Learning Model with Polysomnography in a patient with Obstructive Sleep Apnea using Positive Airway Pressure Therapy"** _Chest 2023, 2023.10.10_

- **주제:** CPAP/PAP 치료 중인 OSA 환자에서 Asleep 소리 기반 모델의 수면 판정 및 OSA 평가 정확성을 검증하는 임상 사례 평가.
- **모델:** Asleep의 표준 소리 기반 딥러닝 파이프라인(오디오 → Mel 스펙트로그램 → DNN), PAP 기기 소음이 음향 환경을 근본적으로 변화시키는(기계 소음 도입, 호흡 소리 특성 변경) 녹음에 적용.
- **핵심 기여:** 핵심 임상 갭 해결: 비-PAP 오디오로 학습된 모델이 PAP 치료 환자에게 일반화 가능한지 검증. PAP 치료는 OSA의 표준 치료법이므로, 임상적으로 유용한 소리 기반 도구는 반드시 이 시나리오를 처리해야 함. 대규모 PAP 사용 인구에 대한 모델 임상 적용성 확장.

---

## 5. 수면 파트너 분리와 실세계 과제 (논문 9)

**논문 9 — "Enhancing Sleep Stage Prediction with Breathing Sound Separation in Home Environments with Sleep Partners"** _World Sleep 2023, 2023.10.20_

- **주제:** 2인이 침대를 공유할 때 혼합된 호흡 소리가 단일 피험자 분석을 방해하는 문제를 해결.
- **모델:** 2단계 파이프라인. Stage 1: **SepFormer**(Separation Transformer) — 최신 음성 분리 모델을 호흡 소리에 적응. 2인 혼합 오디오 입력 → 개별 호흡 트랙 분리 출력. Stage 2: 분리된 오디오에 Asleep 표준 수면 판정 모델 적용. 학습: 다양한 거리·진폭 비율로 **합성된 2인 녹음 1,000건**(병원 PSG 10건 + 가정 PSG 20건에서 생성).
- **핵심 기여:** 공유 수면 환경에서 호흡 소리 분리를 수면 판정에 최초 적용. 분리 적용 시 합성 데이터에서 Macro F1 **6.7% 향상**. 모든 진폭 비율에서 일관된 성능 향상. 이후 실제 2인 PSG 데이터(88명, 44쌍)로 검증한 2025년 Sleep Medicine 논문에서 4단계 분류 Macro F1 0.590 달성.

---

## 6. 인구 기반 설문 연구 (논문 10, 11)

**논문 10 — "Population-Based Estimates of Sleep Characteristics and Disruptors in the United States and South Korea"** _World Sleep 2023, 2023.10.20_

- **주제:** 미국과 한국 성인의 수면 시간, 질, 장애 요인을 비교하는 문화 간 역학 설문 조사.
- **모델:** AI/ML 모델 없음 — 전국 대표 표본(미국 1,009명, 한국 1,000명) 설문 연구. 인구통계 대표성을 위한 RIM 가중치 적용 통계 분석.
- **핵심 기여:** 양국 간 최초의 인구 수준 문화 간 수면 비교. 한국 성인은 주중 **43분 적게** 수면(6시간13분 vs 6시간56분), **25%가 수면의 질 나쁨** 보고(미국은 6%). 수면 장애 주요 원인: 가족/관계(미국) vs 직업/재정 걱정(한국). 한국의 심각한 수면 부족 문제를 부각.

**논문 11 — "Sleep Technology Use and Beliefs in the United States and South Korea"** _World Sleep 2023, 2023.10.20_

- **주제:** 논문 10의 동반 설문으로, 수면 기술 채택률, 소비자 신뢰도, 개인정보 보호 우려를 조사.
- **모델:** AI/ML 모델 없음 — 논문 10과 동일 설문 표본.
- **핵심 기여:** 수면 추적 기술 사용자는 미국 **~14%**, 한국 **~8%**에 불과하며, 광범위한 수면 부족에도 불구하고 채택률이 저조. 기술 채택과 수면 결과 간 **유의한 연관 없음**. 약 35~39%가 기술이 수면을 개선할 수 있다고 신뢰하지만, **57~61%가 개인정보 보호 우려** 표명. 기술 잠재력과 실제 채택 간 격차를 부각.

---

## 7. 수면 웰니스 중재 (논문 4)

**논문 4 — "Impact of preferred sleep induction sounds with guided imagery on sleep initiation and sleep quality"** _World Sleep 2023, 2023.10.20_

- **주제:** 개인화된 수면 유도 소리가 수면 개시 및 수면 구조를 개선하는지 검증하는 임상 중재 연구.
- **모델:** AI/ML 모델 없음 — PSG 기반 임상 연구. 불면증 참여자 27명(SOL >30분, ISI >8)을 선호 소리, 비선호 소리, 대조군으로 무작위 배정, 3일 연속 PSG 야간에 12가지 가이디드 이미저리 소리 옵션 제공.
- **핵심 기여:** 선호·비선호 소리 모두 대조군 대비 수면 개시 잠복기 개선(p<0.001, p=0.01). 그러나 **선호 소리 그룹만 깊은 수면(N3) 비율 유의하게 증가**(p=0.014). 소리 개인화가 주관적 선호뿐 아니라 객관적 수면 구조 개선에도 중요함을 입증.

---

## 21편 논문을 관통하는 기술적 발전 흐름

시간순으로 보면, 이 논문들은 점진적 엔지니어링 정교화의 일관된 이야기를 보여준다. 연구는 **EEG 기반 판정의 신뢰도 추정**(논문 21, 2021)에서 시작하여, 소리 기반 판정의 핵심 **SoundSleepNet 아키텍처**(논문 19–20, 2022)로 전환되었다. 2023년을 관통하며 팀은 실세계 배포의 모든 장애물을 체계적으로 공략했다: 일관성 학습을 통한 **가정 소음 견고성**(논문 17–18), 비라벨 가정 데이터의 **반지도 학습**(논문 16), **일주기 특징 융합**(논문 12), 판정+OSA 통합 **멀티태스크 학습**(논문 7), **수면 파트너 분리**(논문 9), **PAP 치료 호환성**(논문 13), **코골이 감지**(논문 5). 임상 검증은 병원 PSG에서 **Level 2 가정 PSG**(논문 1)로 격상되었고, 경쟁 벤치마킹에서 에어러블 접근법이 하드웨어 트래커를 능가함을 입증(논문 2/6). 설문 연구(논문 10–11)와 수면 유도 연구(논문 4)는 진단을 넘어 수면 웰니스까지 확장하는 비전을 완성한다. 전체를 관통하는 공통 기술 스레드는 **스마트폰 Mel 스펙트로그램 → Transformer/CNN-LSTM 아키텍처 → 에포크별 수면 예측** 파이프라인이며, 각 논문은 이 파이프라인을 비통제 가정 환경에서 임상적으로 실용화하기 위한 특정 혁신을 기여하고 있다.
