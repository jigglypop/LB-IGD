# AI Game Design Optimizer

게임 밸런스를 자동으로 최적화하는 Bilevel Optimization 시스템

## 개요

이 프로젝트는 **게임 설계 최적화(Game Design Optimization)**를 다룹니다. 일반적인 강화학습이 "주어진 게임에서 이기는 방법"을 학습한다면, 이 시스템은 **"어떤 단일 팩션도 지속적으로 우위를 점할 수 없는 게임 규칙"**을 찾습니다.

### 핵심 아이디어

```
설계 변수 x (맵, 유닛, 규칙)
    |
    v
플레이 레벨 MDP (강화학습으로 최적 정책 학습)
    |
    v
관측 통계 y(x) (승률, 교전 거리, 게임 길이 등)
    |
    v
손실 함수 L(y) (밸런스 평가)
    |
    v
설계 변수 업데이트 (Evolution Strategy)
```

이 구조는 **Bilevel Optimization**입니다:
- **Inner Loop (플레이 레벨)**: 고정된 게임 규칙에서 RL로 최적 정책 학습
- **Outer Loop (설계 레벨)**: 관측된 승률/통계를 기반으로 게임 규칙 조정

## 이론적 배경

### 왜 벨만 방정식/DP가 설계 문제에 적용되지 않는가?

벨만 방정식과 동적계획법(DP)은 **고정된 MDP** 위에서만 성립합니다. 설계 변수 $x$를 바꾸면 MDP 자체가 바뀌므로:

1. **상태 전이 구조 불변 조건 위반**: $x$가 바뀌면 전이확률 $P$도 바뀜
2. **마르코프 성질 보장 불가**: 설계 변경 이력이 미래에 영향
3. **재귀 분해 불가**: 설계 결정이 게임 전체 구조를 바꿈

### Evolution Strategy (ES)

설계 목적함수 $J(x)$는 RL 학습과 self-play를 포함하므로:
- 비미분 가능 (내부 루프가 black-box)
- 확률적 (시드, 초기화, 탐색 노이즈)
- 고비용 (매 평가마다 RL 학습 필요)

이런 함수에는 **Score Function Estimator (ES)**가 적합합니다:

$$\nabla_x J_\sigma(x) \approx \frac{1}{N\sigma} \sum_{i=1}^{N} J(x + \sigma\epsilon_i) \cdot \epsilon_i$$

## 프로젝트 구조

```
ai_designed/
    docs/
        bellman.md      # 제1장: 벨만 방정식과 설계 최적화의 분리
        blackbox.md     # 제2장: ES 유도와 교전 거리 분포 정합
        inverse.md      # 제3장: 역설계 (목표 메타 -> 팩션/맵)
        evaluation.md   # 제4장: self-play 기반 평가 프로토콜
    src/
        core/
            agent.py        # RL 정책 네트워크 (SimplePolicy, REINFORCE)
            designer.py     # ES 기반 설계 최적화 (DesignOptimizer)
            simulation.py   # 시뮬레이션 실행 및 통계 수집
        envs/
            simple_combat.py  # 격자 기반 전투 환경 (GridCombatEnv)
    main.py             # 실행 진입점
    pyproject.toml      # 의존성 관리 (uv)
```

## 환경: GridCombatEnv

체스보다 큰 격자 맵에서 2팩션이 전투하는 턴제 환경입니다.

### 유닛 타입 (6종)

| 타입 | 이동 패턴 | 특성 |
|------|----------|------|
| melee | 직선 (룩) | 근접 공격, 기본 유닛 |
| ranged | 전방향 (퀸) | 원거리 공격 |
| scout | L자 점프 (나이트) | 기동력 특화 |
| tank | 직선 | 고체력 |
| siege | 직선 | 초장거리 공격 |
| king | 전방향 | **잡히면 즉시 패배** |

### 게임 규칙

- **턴제**: 팩션이 교대로 행동, 턴당 1유닛만 행동 (체스/장기 방식)
- **선턴 공정성**: 평가 시 절반은 P0 선턴, 절반은 P1 선턴
- **승리 조건**: 상대 킹 잡기 또는 전멸
- **무승부 방지**: 일정 턴 동안 공격 없으면 조기 종료 + 페널티

### 비대칭 밸런스

- **P0**: 유닛 수 많음 (16개), 기동력 낮음
- **P1**: 유닛 수 적음 (10개), 기동력/화력 높음

## 설치 및 실행

### 요구사항

- Python 3.10+
- uv (패키지 관리자)

### 설치

```bash
# uv 설치 (없는 경우)
pip install uv

# 프로젝트 의존성 설치
uv sync
```

### 실행

```bash
# 가상환경 활성화 후 실행
.venv/Scripts/python main.py      # Windows
.venv/bin/python main.py          # Linux/Mac
```

### 설정

`main.py`에서 조정 가능한 파라미터:

```python
fast = True  # True: 빠른 테스트, False: 정밀 최적화

# fast=True
train_episodes = 12     # RL 학습 에피소드
eval_episodes = 8       # 평가 에피소드
n_samples = 4           # ES 샘플 수
outer_steps = 6         # 외부 최적화 스텝

# fast=False
train_episodes = 250
eval_episodes = 20
n_samples = 8
outer_steps = 20
```

## 최적화 파이프라인

### 1. 초기 설계

```python
initial_design = {
    "width": 10, "height": 10,
    "p0_melee_units": 6, "p0_ranged_units": 4, ...
    "p1_melee_move": 4.0, "p1_ranged_range": 5.0, ...
}
```

### 2. ES 최적화 루프

각 스텝에서:
1. 현재 설계 주변에서 랜덤 perturbation 생성
2. 각 후보 설계로 RL 학습 + self-play 평가
3. 승률이 0.5에 가까운 방향으로 그래디언트 추정
4. 설계 업데이트

### 3. 손실 함수

```python
def get_loss(stats):
    # 승률 편차 (0.5에서 벗어날수록 큰 페널티)
    win_diff = abs(stats["p0_win_rate"] - 0.5)
    
    # 무승부 페널티 (교전 없는 게임 방지)
    draw_penalty = 5.0 * stats["draw_rate"]
    
    # 교전 거리 정합 (목표 분포와의 차이)
    dist_loss = abs(stats["avg_distance"] - target_dist_mean)
    
    return 40.0 * win_diff + draw_penalty + dist_loss + ...
```

### 4. 출력 예시

```
[Step 1] Starting Optimization...
    Sample 1 (+): Loss=3.14 | P0=0.25 P1=0.75 Draw=0.00 Dist=2.64
    Sample 1 (-): Loss=12.84 | P0=0.00 P1=1.00 Draw=0.00 Dist=2.59
    Sample 2 (+): Loss=0.66 | P0=0.50 P1=0.50 Draw=0.00 Dist=2.94
    ...
Step  1 | Loss: 5.12 | Design: width: 10.00, height: 10.00, ...

Final P0 Win Rate: 0.35
Final P1 Win Rate: 0.65
Final Draw Rate: 0.00
Final Avg Distance: 3.53
```

## 최적화 가능한 파라미터

`DesignOptimizer.optimizable_keys`:

| 카테고리 | 파라미터 |
|----------|----------|
| 맵 크기 | width, height |
| 유닛 수 | p0/p1_melee/ranged/scout/tank/siege_units |
| 이동거리 | p0/p1_melee/ranged/scout_move |
| 사거리 | p0/p1_ranged/siege_range |
| 스탯 | p0/p1_melee/ranged_hp/damage, p1_scout/tank/siege_damage/hp |

킹 유닛은 각 팩션 1개로 고정되며 최적화 대상에서 제외됩니다.

## 퇴화 해(Degenerate Solution) 방지

최적화 과정에서 발생할 수 있는 퇴화 패턴과 대응책:

| 퇴화 패턴 | 원인 | 대응책 |
|-----------|------|--------|
| 100% 무승부 | 유닛들이 교전하지 않음 | 거리 쉐이핑 보상, 무교전 조기 종료 |
| 한쪽 100% 승리 | 유닛 수/스탯 불균형 | 양쪽 스탯 최적화, 최소 유닛 수 보장 |
| 유닛 0개 수렴 | ES가 유닛 제거 방향으로 수렴 | 모든 유닛 타입 최소 1개 강제 |

## 문서

이론적 배경과 수학적 정식화는 `docs/` 디렉토리를 참조하세요:

- **bellman.md**: 벨만 방정식의 전제와 설계 최적화가 다른 문제인 이유
- **blackbox.md**: ES(Score Function Estimator) 유도 과정
- **inverse.md**: 목표 메타에서 팩션/맵을 역으로 생성하는 문제
- **evaluation.md**: self-play 기반 평가 프로토콜과 성공 판정 기준

## 라이선스

MIT License

