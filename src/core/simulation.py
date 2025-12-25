import torch
import numpy as np
from src.envs.simple_combat import GridCombatEnv
from src.core.agent import SimplePolicy, train_one_episode

def run_simulation(design_params: dict, train_episodes=50, eval_episodes=20, seed=42):
    """
    1. 설계 파라미터(x)로 환경 생성
    2. 내부 루프: 에이전트 학습 (train_episodes)
    3. 평가 루프: 통계 수집 (eval_episodes)
    4. 결과 반환: 승률, 거리 분포 등 (y(x))
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = GridCombatEnv(design_params, seed=seed)
    
    # 두 에이전트 초기화
    policies = [SimplePolicy(), SimplePolicy()]
    optimizers = [torch.optim.Adam(p.parameters(), lr=0.01) for p in policies]
    
    # 2. 내부 루프 (Inner Loop): 학습
    # 병렬 처리 시 tqdm 출력 간섭을 피하기 위해, 병렬 실행 중에는 개별 tqdm 끔
    # 상위 ProcessPoolExecutor에서 전체 진행률을 관리
    # 선턴을 반반으로 분배하여 학습 편향 제거
    for ep in range(train_episodes):
        first_turn = 0 if ep < train_episodes // 2 else 1
        train_one_episode(env, policies, optimizers, first_turn=first_turn)
        
    # 3. 평가 (Evaluation): 학습된 정책 동결 후 통계 수집
    # 선턴을 랜덤이 아닌 "반반"으로 명확히 분배
    win_counts = {0: 0, 1: 0, -1: 0}
    all_attack_distances = []
    
    with torch.no_grad():
        for ep in range(eval_episodes):
            # 절반은 P0 선턴, 절반은 P1 선턴
            first_turn = 0 if ep < eval_episodes // 2 else 1
            obs = env.reset(first_turn=first_turn)
            done = False
            while not done:
                obs0_units, obs1_units = obs
                side = int(getattr(env, "side_to_act", 0))
                if side == 0:
                    ui, a, _ = policies[0].get_turn_action(obs0_units)
                    obs, _, done, info = env.step((0, ui, a))
                else:
                    ui, a, _ = policies[1].get_turn_action(obs1_units)
                    obs, _, done, info = env.step((1, ui, a))
                
            win_counts[info["winner"]] += 1
            all_attack_distances.extend(info.get("attack_distances", []))
    
    # 디버깅 출력: 이번 시뮬레이션의 승률 분포 확인 (Verbose Mode)
    total = eval_episodes
    if train_episodes >= 50: # 학습이 어느 정도 진행된 경우만 출력
        p0_rate = win_counts[0] / total
        p1_rate = win_counts[1] / total
        draw_rate = win_counts[-1] / total
        avg_dist = np.mean(all_attack_distances) if all_attack_distances else 0.0
        # print(f"  [Sim Debug] P0:{p0_rate:.2f} P1:{p1_rate:.2f} Draw:{draw_rate:.2f} Dist:{avg_dist:.2f}")

    # 통계 요약
    total = eval_episodes
    p0_win_rate = win_counts[0] / total
    p1_win_rate = win_counts[1] / total
    if len(all_attack_distances) == 0:
        avg_distance = 0.0
        distance_std = 0.0
    else:
        avg_distance = float(np.mean(all_attack_distances))
        distance_std = float(np.std(all_attack_distances))
    
    stats = {
        "p0_win_rate": p0_win_rate,
        "p1_win_rate": p1_win_rate,
        "draw_rate": win_counts[-1] / total,
        "avg_distance": avg_distance,
        "distance_std": distance_std,
        "distance_samples": all_attack_distances
    }
    
    return stats

