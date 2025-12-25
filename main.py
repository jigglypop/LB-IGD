from src.core.designer import DesignOptimizer
import numpy as np
from src.core.simulation import run_simulation
import sys

def main():
    # Windows 터미널에서 한글 출력 깨짐 방지
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("AI Game Design System Started")
    
    # 1. 초기 설계 (불균형한 상태로 시작)
    # 목표: 체스(8x8)보다 큰 맵, 2팩션, 말 수 비대칭(한쪽은 체스보다 조금 많게, 한쪽은 더 적게)
    initial_design = {
        # fast일 때는 더 작은 맵/유닛으로 속도 확보
        "width": 10,
        "height": 10,

        # 유닛 수(타입 6종): P0=16, P1=10 (킹 1개씩 포함)
        # 킹은 자동으로 1개씩 추가됨
        "p0_melee_units": 6,
        "p0_ranged_units": 4,
        "p0_scout_units": 2,
        "p0_tank_units": 2,
        "p0_siege_units": 1,  # 6+4+2+2+1+1(king) = 16

        "p1_melee_units": 3,
        "p1_ranged_units": 2,
        "p1_scout_units": 2,
        "p1_tank_units": 1,
        "p1_siege_units": 1,  # 3+2+2+1+1+1(king) = 10

        # P0 스탯: 숫자가 많은 대신 기동력 낮음
        "p0_melee_move": 2.0,    # 직선 2칸
        "p0_melee_range": 1.0,
        "p0_melee_damage": 1.0,
        "p0_melee_hp": 3.0,

        "p0_ranged_move": 1.0,   # 전방향 1칸
        "p0_ranged_range": 3.0,
        "p0_ranged_damage": 1.0,
        "p0_ranged_hp": 2.0,

        "p0_scout_move": 1.0,    # L자 1회
        "p0_scout_range": 1.0,
        "p0_scout_damage": 1.0,
        "p0_scout_hp": 2.0,

        "p0_tank_move": 1.0,     # 직선 1칸
        "p0_tank_range": 1.0,
        "p0_tank_damage": 1.0,
        "p0_tank_hp": 5.0,

        "p0_siege_move": 1.0,    # 직선 1칸
        "p0_siege_range": 4.0,
        "p0_siege_damage": 1.2,
        "p0_siege_hp": 2.0,

        "p0_king_move": 1.0,     # 전방향 1칸
        "p0_king_range": 1.0,
        "p0_king_damage": 0.5,
        "p0_king_hp": 5.0,

        # P1 스탯: 숫자가 적은 대신 기동력/화력 높음
        "p1_melee_move": 4.0,    # 직선 4칸 (룩처럼)
        "p1_melee_range": 1.0,
        "p1_melee_damage": 1.5,
        "p1_melee_hp": 4.0,

        "p1_ranged_move": 3.0,   # 전방향 3칸 (퀸처럼)
        "p1_ranged_range": 5.0,  # 긴 사거리
        "p1_ranged_damage": 1.5,
        "p1_ranged_hp": 3.0,

        "p1_scout_move": 2.0,    # L자 2회 점프
        "p1_scout_range": 1.0,
        "p1_scout_damage": 1.5,
        "p1_scout_hp": 3.0,

        "p1_tank_move": 2.0,     # 직선 2칸
        "p1_tank_range": 1.0,
        "p1_tank_damage": 1.5,
        "p1_tank_hp": 8.0,

        "p1_siege_move": 1.0,    # 직선 1칸
        "p1_siege_range": 7.0,   # 초장거리
        "p1_siege_damage": 2.5,
        "p1_siege_hp": 2.0,

        "p1_king_move": 2.0,     # 전방향 2칸 (더 민첩)
        "p1_king_range": 1.0,
        "p1_king_damage": 0.5,
        "p1_king_hp": 6.0,       # 더 튼튼

        "max_steps": 80,
        # 교전 유도/퇴화 방지 튜닝
        # 턴제(교대 턴)에서는 같은 '30'이 동시행동 대비 실제 기회가 절반 수준이라 너무 빡빡합니다.
        "no_attack_limit": 60,
        "shaping_scale": 0.05,
    }
    
    # 2. 목표 설정
    # 목표: 승률 0.5 근처 + 교전 거리(공격 발생 거리)의 목표 평균
    target_dist_mean = 3.0
    
    # 속도 프리셋: 개발/디버그는 fast, 최종 검증만 크게
    fast = False
    train_episodes = 12 if fast else 250
    # ES에서 eval_episodes가 너무 작으면 승률이 (0, 0.33, 0.67, 1)로만 튀어
    # 균형(0.5) 최적화가 노이즈에 묻힙니다. fast에서도 최소한은 확보합니다.
    eval_episodes = 8 if fast else 20
    # n_samples=2(=pair 1개)는 ES 분산이 너무 커서 방향이 자주 틀어집니다.
    n_samples = 4 if fast else 8
    max_workers = 0  # 0이면 자동 (cpu//2, 최대 4)

    optimizer = DesignOptimizer(
        initial_design,
        target_dist_mean=target_dist_mean,
        sigma=0.2,
        lr=0.1,
        n_samples=n_samples,
        train_episodes=train_episodes,
        eval_episodes=eval_episodes,
        use_parallel=True,
        max_workers=max_workers,
        base_seed=42,
        verbose=True,
    )
    
    print(f"Initial Design: {initial_design}")
    print(f"Target Distance Mean: {target_dist_mean}")
    print("-" * 50)
    
    # 3. 최적화 루프 (Outer Loop)
    outer_steps = 6 if fast else 20
    for step in range(1, outer_steps + 1):
        print(f"\n[Step {step}] Starting Optimization...")
        loss, current_design = optimizer.step(step_index=step)
        
        # 보기 좋게 출력
        design_str = ", ".join([f"{k}: {v:.2f}" for k, v in current_design.items() if k in optimizer.optimizable_keys])
        print(f"Step {step:2d} | Loss: {loss:.4f} | Design: {design_str}")

    print("-" * 50)
        
    print("-" * 50)
    print("Optimization Finished.")
    print("Final Design:", current_design)
    
    # 최종 검증
    print("\nVerifying Final Design...")
    # 검증 시에는 학습을 충분히 시켜야 함
    # fast 모드에서는 최종 검증도 가볍게(기다림 방지)
    final_train = 80 if fast else 500
    final_eval = 20 if fast else 100
    final_stats = run_simulation(current_design, train_episodes=final_train, eval_episodes=final_eval)
    print(f"Final P0 Win Rate: {final_stats['p0_win_rate']:.2f}")
    print(f"Final P1 Win Rate: {final_stats['p1_win_rate']:.2f}")
    print(f"Final Draw Rate: {final_stats['draw_rate']:.2f}")
    print(f"Final Avg Distance: {final_stats['avg_distance']:.2f}")

if __name__ == "__main__":
    main()
