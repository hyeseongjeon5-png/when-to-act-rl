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
- [x] 저장소·실험 설계 (2026-08-24)
- [x] 1주차 — 비용 래퍼 + 자동 테스트 6종 + 고정 규칙 6종 ([#2](../../issues/2), [#3](../../issues/3))
- [x] 2주차 — TempoRL 이식(공개 코드 대조) + Lazy-MDP 구현 ([#6](../../issues/6), [#7](../../issues/7))
- [x] **3주차 — 본실험 완료 (510조건, 건너뜀 0건)** ([#8](../../issues/8), [#9](../../issues/9))
  - MountainCar-v0: λ 9개 × 시드 10 × 3계열 = 270조건 (약 13시간)
  - LunarLander-v3: λ 8개 × 시드 10 × 3계열 = 240조건 (10.25시간)
  - 고정 규칙: 1020조건
- [x] **λ-성능 지도 v1 + 임계 비용 λ\* 표** ([#10](../../issues/10))
- [ ] MinAtar 확장, 비용 부과 방식 민감도(결정 시점 vs 매 스텝)
- [ ] 논문 작성 (`paper/` — 3장 방법·4장 결과 초안 완료) · 발표 준비

## 결과 — 임계 비용 λ\*

시드 10개, IQM + 95% 계층 부트스트랩 신뢰구간 (Agarwal et al. 2021).
λ\* = 학습이 **그 λ에서 가장 센 고정 규칙**을 더 이상 이기지 못하게 되는 가장 작은 행동 비용.

| 환경 | 표준 DQN | TempoRL | Lazy-MDP | 해석 |
|---|---|---|---|---|
| MountainCar-v0 | **0** | **0** | **0** | λ=0에서도 규칙을 못 이긴다 |
| LunarLander-v3 | **2.0** | **2.0** | **2.0** | 넓은 비용 구간에서 학습이 이긴다 |

### MountainCar-v0 — 규칙을 쓰는 게 낫다

| 계열 | λ=0 | λ=0.01 | λ=0.02 | λ=0.05 | λ=0.1 |
|---|---|---|---|---|---|
| 표준 DQN | −160.4 (62%) | −197.5 (3%) | −200.4 (0%) | −200.0 | −200.0 |
| TempoRL | −141.4 (86%) | −192.8 (13%) | −200.2 (0%) | −200.0 | −200.0 |
| Lazy-MDP | −127.9 (94%) | −155.8 (64%) | −159.1 (59%) | −187.2 (22%) | −191.5 (13%) |
| **pump 규칙(기준)** | **−119.3 (100%)** | −120.5 | −121.7 | −125.3 | −131.3 |

괄호는 목표 도달률. **에피소드 보상의 1%에 해당하는 λ=0.01만으로도 표준 DQN의 도달률이
62% → 3%로 무너진다.** 비용은 즉각 확정 손해인 반면 목표 보상은 멀고 무작위 탐험으로는
닿을 수 없기 때문이다(측정 도달률 0.0%). 학습은 "아무것도 하지 않는" 정책으로 얼어붙는다.

![MountainCar λ-성능 지도](results/figures/MountainCar-v0_lambda_map_cost_return.png)

### LunarLander-v3 — 학습을 쓰는 게 낫다

| 계열 | λ=0 | λ=0.6 | λ=1.0 | λ=1.37 | λ=2.0 | λ=3.0 |
|---|---|---|---|---|---|---|
| 표준 DQN | 189 | 90 | −13 | −61 | −134 | −210 |
| TempoRL | 159 | 59 | −34 | −52 | −153 | −286 |
| Lazy-MDP | **210** | **103** | **37** | **−47** | −157 | −175 |
| 임계값 규칙(기준) | 36 | −37 | −86 | −131 | −207 | −328 |
| 최강 규칙 포락선 | 36 | −37 | −86 | −131 | **−131**(무행동) | **−131**(무행동) |

지정 기준선(임계값 규칙)만 보면 격자 끝까지 학습이 이긴다. 그러나 λ>1.366부터 **최강 규칙이
무행동으로 바뀌고**, λ=2.0에서 세 계열 모두 그 아래로 내려간다 — 이것이 실질적 λ\*다.

**비용이 오르면 학습은 실제로 행동을 아낀다**: DQN 340.8회(λ=0) → 63.4회(λ=3), 81% 감소.
고정 규칙은 121.3회에 묶여 있다.

![LunarLander λ-성능 지도](results/figures/LunarLander-v3_lambda_map_cost_return.png)

### 두 환경의 대조가 이 연구의 답이다

| | MountainCar-v0 | LunarLander-v3 |
|---|---|---|
| 보상 구조 | 희소 | 조밀 |
| 좋은 고정 규칙 | 있다 (100% 해결) | 약하다 |
| 무작위 탐험 목표 도달률 | 0.0% | 정상 |
| 임계 비용 λ\* | 0 | 2.0 |
| 판단 | **규칙을 쓰라** | **학습을 쓰라** |

**"행동에 비용이 붙을 때 학습을 도입할 가치가 있는가"의 답은 비용 λ 하나가 아니라
'보상이 조밀한가'와 '이미 좋은 규칙이 있는가'에 먼저 달려 있다.**

부수 결과: Lazy-MDP가 두 환경 모두에서 가장 강했다. 기본 정책에 위임할 수 있어
학습이 실패해도 규칙 수준의 밑바닥이 있고, 비용이 커지면 위임을 줄여 행동을 아낀다.

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
