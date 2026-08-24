# When to Act — 행동 비용이 있는 환경에서 강화학습의 우위 조건 분석

> **EN summary** — In real control systems, every action has a cost (wear, energy, switching loss). This project sweeps an explicit action cost λ (reward shaped as `r' = r − λ·1[act]`) across Gymnasium environments (MountainCar, LunarLander, MinAtar) and asks: **at what cost level do agents that *learn when to act* (TempoRL-style action repetition, Lazy-MDP-style intervention) actually beat simple fixed rules** (act-every-k-steps, threshold triggers, no-op)? Results are reported with IQM and stratified bootstrap CIs over ≥10 seeds, following Agarwal et al. (NeurIPS 2021). Undergraduate thesis project, Dong-A University.

## 연구 질문

실제 기계 제어에서 행동은 공짜가 아니다 (마모, 전환 손실, 통신·연산 자원).
행동 1번마다 비용 λ를 물리면서 λ를 0부터 단계적으로 키우면 —

1. 학습 에이전트의 **행동 빈도**는 어떻게 변하는가?
2. **단순 고정 규칙 대비 성능 우위**는 어느 λ에서 사라지는가?

이를 통해 제어 문제에서 **"학습을 도입할 가치가 있는 조건"과 "단순 규칙이면 충분한 조건"** 을 구분하는 판단 근거를 제시한다.

## 방법

| 항목 | 내용 |
|---|---|
| 환경 | Gymnasium MountainCar-v0, LunarLander-v2, MinAtar — 보상에 행동 비용 항 추가 (`r' = r − λ·1[행동 실행]`) |
| 비교 대상 | (가) 표준 DQN (나) TempoRL 방식 행동 지속 길이 학습 (다) Lazy-MDP 방식 개입 상태 선택 (라) 고정 규칙: k스텝 주기 행동 · 임계값 규칙 · 무행동 |
| 실험 축 | 행동 비용 λ 격자 × 무작위 시드 10개 이상 |
| 지표 | 총보상, 행동 횟수, 학습 곡선, 규칙 기준선 대비 승률 |
| 통계 | 사분위평균(IQM) + 계층 부트스트랩 신뢰구간 (Agarwal et al. 2021, `rliable`) |
| 산출물 | λ-성능 지도, "학습이 규칙을 이기는 임계 비용" 표 |

## 저장소 구조

```
├─ src/
│  ├─ envs/          # 행동 비용 래퍼 (cost_wrapper.py)
│  ├─ agents/        # DQN, TempoRL, Lazy-MDP (구현 예정)
│  └─ baselines/     # 고정 규칙 기준선 (fixed_rules.py)
├─ experiments/      # 실험 config (yaml)
├─ results/          # 실험 로그·그림·보고서 (대용량 raw는 git 제외)
├─ docs/             # 실험 설계 · 로드맵 · 실험일지
└─ paper/            # 논문 원고
```

## 재현 방법

```bash
pip install swig                 # Windows에서 box2d(LunarLander) 빌드에 먼저 필요
pip install -r requirements.txt

# 고정 규칙 기준선 평가 (동작함)
python -m src.eval.run_fixed_rules --config experiments/configs/smoke_fixed_rules_mountaincar.yaml
python -m src.report.make_report --session results/session_2026-08-24.json

# 학습 에이전트 파일럿 (예정): python -m src.run --config experiments/configs/pilot_mountaincar.yaml
```

> **환경 버전 메모** — `LunarLander-v2`는 gymnasium 1.3.0에서 폐기되어 **v3**를 사용한다.
> torch는 CPU 빌드로 설치되었다(CUDA 없음). 자세한 세팅 기록은 `docs/실험일지.md`.

## 진행 상황

- [x] 계획서 제출 (2026-07-30)
- [x] 저장소·실험 설계 초안 (2026-08-24)
- [x] **1주차 — 배관 완성** (2026-08-24)
  - [x] 환경 세팅 — MountainCar · LunarLander-v3 · MinAtar 동작 확인 ([#1](../../issues/1))
  - [x] 비용 래퍼 + 자동 테스트 6종 통과 ([#2](../../issues/2))
  - [x] 고정 규칙 6종, 두 환경 × λ 6개 × 시드 10개 스윕 완료 ([#3](../../issues/3))
  - [x] DQN 베이스라인 — MountainCar에서 pump 규칙을 이기는 것 확인 (시드0 r' −110.0 vs 규칙 −119.3) ([#4](../../issues/4))
  - [x] 파일럿 2회 + 조건당 소요시간 측정 ([#5](../../issues/5))
- [x] **2주차 — 선행 연구 이식** (2026-08-24)
  - [x] TempoRL 방식 이식 (automl/TempoRL 공개 코드 대조) ([#6](../../issues/6))
  - [x] Lazy-MDP 방식 구현 (기본 정책 = 최고 고정 규칙) ([#7](../../issues/7))
- [ ] **3주차 — 본실험 + 분석** *(진행 중)*
  - [ ] MountainCar-v0 λ 6개 × 시드 10 × 3계열 = 180조건 ([#8](../../issues/8))
  - [ ] LunarLander-v3 λ 6개 × 시드 10 × 3계열 = 180조건 ([#9](../../issues/9))
  - [ ] λ-성능 지도 v1 + 임계 비용 λ* 표 ([#10](../../issues/10))
- [ ] 본실험 확장 (MinAtar, 비용 부과 방식 민감도) 및 통계 분석 마무리
- [ ] 논문 작성 · 발표 준비 (`paper/` — 3장 방법 초안 완료)

## 지금까지 확인된 것

숫자는 모두 `results/` 아래 로그 파일에서 인용했다.

**1. MountainCar에서 표준 ε-greedy 탐험은 목표에 닿지 못한다.**
무작위 정책의 200스텝 내 목표 도달률은 매 스텝 새 행동일 때 **0.0%**(500 에피소드),
같은 행동을 8스텝 유지하면 1.6%, 16스텝 유지하면 9.8%였다. 시간제한을 1000스텝으로 늘려도
매 스텝 무작위는 여전히 0.0%다 — 막힌 것은 시간이 아니라 탐험 방식이다.
이는 TempoRL(ICML 2021)이 주장한 "행동을 오래 유지하면 탐험이 좋아진다"의 재현이다.

**2. 행동에 값이 붙으면 학습은 '아무것도 안 하기'로 무너질 수 있다.**
초기 파일럿에서 표준 DQN은 λ=0.2만 돼도 세 시드 모두 에피소드당 행동 0회, 총보상 −200으로
수렴했다. 비용은 지금 당장 확실히 깎이는 반면 목표 보상은 멀고 드물게만 얻어지기 때문이다.

**3. λ 격자는 "무행동이 최고 규칙을 이기는 지점"에 맞춰 잡았다.**

| 환경 | 최고 고정 규칙 | 무행동 | 두 선이 만나는 λ | 실측 확인 |
|---|---|---|---|---|
| MountainCar-v0 | pump: r −119.8, 행동 119.2회 | −200 | 0.673 | λ=0.66에서 규칙 r' −198.1 |
| LunarLander-v3 | 임계값: r +32.7, 행동 120.7회 | −132.2 | 1.366 | λ=1.37에서 규칙 r' −130.6 |

**4. 두 환경의 역할이 다르다.** MountainCar는 표준 DQN이 탐험에서 불리한 환경,
LunarLander는 표준 DQN이 정상 학습되는 환경(10만 스텝에 r' +245.3)이다.
둘을 나란히 놓아야 "규칙에 진 이유가 비용 때문인가, 학습이 안 돼서인가"를 구분할 수 있다.

## 실험 인프라

```bash
# 조건 1개 학습 (체크포인트 저장·이어하기 지원)
python -m src.train.train_agent --config experiments/configs/main_mountaincar.yaml --agent temporl --lam 0.2 --seed 3

# 본실험 (조건 병렬 실행 + progress.json 기록, 그냥 다시 실행하면 이어서 함)
python -m src.train.runner --config experiments/configs/main_mountaincar.yaml

# 자가 감시 (프로세스 생존·진행 정체·로그 예외/NaN·디스크·학습 곡선 이상)
python -m src.monitor.watchdog

# 집계 → 그림 → HTML 보고서
python -m src.analysis.aggregate --env all
python -m src.analysis.plots --env all
python -m src.report.make_experiment_report --open
```

- 설정을 바꾸면 **설정 지문**이 달라져 예전 체크포인트를 자동 폐기한다 (결과 오염 방지)
- 조건이 3회 연속 실패하면 건너뛰고 나머지를 계속 진행하며, 사유를 실험일지에 남긴다
- 실행 순서는 시드 우선 — 중간에 멈춰도 λ 격자 전체가 채워져 그림을 그릴 수 있다

## 참고문헌

1. A. Biedenkapp, R. Rajan, F. Hutter, M. Lindauer, **"TempoRL: Learning When to Act"**, ICML 2021.
2. A. Jacq, J. Ferret, O. Pietquin, M. Geist, **"Lazy-MDPs: Towards Interpretable Reinforcement Learning by Learning When to Act"**, AAMAS 2022.
3. R. Agarwal, M. Schwarzer, P. S. Castro, A. Courville, M. G. Bellemare, **"Deep Reinforcement Learning at the Edge of the Statistical Precipice"**, NeurIPS 2021.

## 도구

실험 자동화(러너·감시 스크립트·집계 코드 작성)에 **Claude Code**를 활용했습니다. 실험 설계, 하이퍼파라미터 결정, 결과 해석은 저자가 직접 판단했으며 모든 수치는 `results/`의 로그 파일에서 인용합니다.

---
전혜성 · Dong-A University · 2026
