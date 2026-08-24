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
pip install -r requirements.txt
# 파일럿 (예정): python -m src.run --config experiments/configs/pilot_mountaincar.yaml
```

## 진행 상황

- [x] 계획서 제출 (2026-07-30)
- [x] 저장소·실험 설계 초안 (2026-08-24)
- [ ] 1주차 — DQN 베이스라인 + 비용 래퍼 + 고정 규칙, MountainCar 파일럿
- [ ] 2주차 — TempoRL 재현(automl/TempoRL 기준) + Lazy-MDP 구현, λ 격자 파일럿
- [ ] 3주차 — 본실험 (λ 격자 × 시드 10+), IQM 분석, λ-성능 지도 v1
- [ ] 본실험 확장 (MinAtar 포함) 및 통계 분석 마무리
- [ ] 논문 작성 · 발표 준비

## 참고문헌

1. A. Biedenkapp, R. Rajan, F. Hutter, M. Lindauer, **"TempoRL: Learning When to Act"**, ICML 2021.
2. A. Jacq, J. Ferret, O. Pietquin, M. Geist, **"Lazy-MDPs: Towards Interpretable Reinforcement Learning by Learning When to Act"**, AAMAS 2022.
3. R. Agarwal, M. Schwarzer, P. S. Castro, A. Courville, M. G. Bellemare, **"Deep Reinforcement Learning at the Edge of the Statistical Precipice"**, NeurIPS 2021.

---
전혜성 · Dong-A University · 2026
