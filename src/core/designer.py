import numpy as np
import torch
from typing import Dict, Any, Tuple
from src.core.simulation import run_simulation

def wasserstein_distance_1d(u_values, v_values):
    """
    1차원 Wasserstein 거리 (Earth Mover's Distance)
    Scipy의 wasserstein_distance와 동일 로직의 간단 구현
    """
    u_sorted = np.sort(u_values)
    v_sorted = np.sort(v_values)
    
    if len(u_sorted) == 0 or len(v_sorted) == 0:
        return 0.0

    # 샘플 수가 다르면 보간법 사용해야 하나, 여기서는 편의상 샘플링으로 맞춤
    min_len = min(len(u_sorted), len(v_sorted))
    if min_len <= 0:
        return 0.0
    # 균등 간격으로 샘플링하여 길이 맞춤
    u_indices = np.linspace(0, len(u_sorted)-1, min_len).astype(int)
    v_indices = np.linspace(0, len(v_sorted)-1, min_len).astype(int)
    
    u_resampled = u_sorted[u_indices]
    v_resampled = v_sorted[v_indices]
    
    return np.mean(np.abs(u_resampled - v_resampled))

from tqdm import tqdm
import concurrent.futures
import os

class DesignOptimizer:
    """
    blackbox.md 기반의 ES 최적화기
    목표: 승률 50:50 + 목표 교전 거리 분포
    """
    def __init__(
        self,
        initial_design: Dict[str, float],
        target_dist_mean: float,
        sigma: float = 0.1,
        lr: float = 0.1,
        n_samples: int = 4,
        train_episodes: int = 80,
        eval_episodes: int = 6,
        use_parallel: bool = True,
        max_workers: int = 0,
        base_seed: int = 42,
        verbose: bool = True,
    ):
        self.mean_design = initial_design.copy()
        self.target_dist_mean = target_dist_mean
        self.sigma = sigma # 탐색 노이즈 표준편차
        self.lr = lr       # 학습률
        self.n_samples = n_samples # ES 샘플 수 (짝수 권장)
        self.train_episodes = train_episodes
        self.eval_episodes = eval_episodes
        self.use_parallel = use_parallel
        self.base_seed = base_seed
        self.verbose = verbose

        if max_workers and max_workers > 0:
            self.max_workers = max_workers
        else:
            # 너무 과한 프로세스 생성 방지
            cpu = os.cpu_count() or 1
            self.max_workers = min(4, max(1, cpu // 2))

        # 총 유닛 수(팩션별) 목표: 초기 설계에서 고정
        def _sum_units(prefix: str) -> int:
            total = 0
            for name in ["melee", "ranged", "scout", "tank", "siege"]:
                k = f"{prefix}_{name}_units"
                if k in initial_design:
                    total += int(round(float(initial_design.get(k, 0))))
            return total

        self.target_units = {
            "p0": max(1, _sum_units("p0")),
            "p1": max(1, _sum_units("p1")),
        }
        
        # 최적화할 키 목록 (맵/말/사거리/이동거리 중심)
        self.optimizable_keys = [
            "width",
            "height",
            # 유닛 수 (킹은 고정 1개라 제외)
            "p0_melee_units",
            "p0_ranged_units",
            "p0_scout_units",
            "p0_tank_units",
            "p0_siege_units",
            "p1_melee_units",
            "p1_ranged_units",
            "p1_scout_units",
            "p1_tank_units",
            "p1_siege_units",
            # 이동거리/사거리 (핵심 밸런스 요소)
            "p0_melee_move",
            "p0_ranged_move",
            "p0_scout_move",
            "p0_ranged_range",
            "p1_melee_move",
            "p1_ranged_move",
            "p1_scout_move",
            "p1_ranged_range",
            "p1_siege_range",
            # 스탯
            "p0_melee_hp",
            "p0_ranged_hp",
            "p0_melee_damage",
            "p0_ranged_damage",
            "p1_melee_hp",
            "p1_ranged_hp",
            "p1_melee_damage",
            "p1_ranged_damage",
            "p1_scout_damage",
            "p1_tank_hp",
            "p1_siege_damage",
        ]

    def get_loss(self, stats: Dict) -> float:
        # 1. 승률 밸런스 손실: (p0_win - 0.5)^2
        # (evaluation.md: L_win)
        p0 = float(stats["p0_win_rate"])
        win_diff = (p0 - 0.5) ** 2
        # 극단(거의 0%/100%)으로 쏠리는 해를 추가로 벌점 (평가 에피소드가 적을 때도 효과적)
        blowout_penalty = 0.0
        if p0 <= 0.05 or p0 >= 0.95:
            blowout_penalty = 2.0
        
        # 2. 분포 정합 손실: Wasserstein 거리
        # 목표 분포: 정규분포 N(target_mean, 1.0)에서 샘플링한 것으로 가정
        dist_samples = stats.get("distance_samples", [])
        target_samples = np.random.normal(self.target_dist_mean, 1.0, size=len(dist_samples))
        # 음수 거리는 0으로 클리핑
        target_samples = np.clip(target_samples, 0, None)
        
        w2_dist = wasserstein_distance_1d(dist_samples, target_samples)

        # 2-b. 교전이 아예 없으면(거리 샘플 0개) 강한 페널티
        no_engagement_penalty = 0.0
        if len(dist_samples) == 0:
            no_engagement_penalty = 10.0
        
        # 3. 제약 조건 (퇴화 방지): 무승부를 강하게 페널티
        # (시간초과/무교전 draw로 수렴하는 경향을 끊기 위함)
        draw_penalty = float(stats["draw_rate"]) * 120.0
        
        # 총 손실
        # 승률 균형을 우선순위로 더 강하게 당김 (현재는 P0 완승 퇴화가 발생)
        total_loss = win_diff * 40.0 + blowout_penalty + w2_dist * 1.0 + draw_penalty + no_engagement_penalty
        return total_loss

    def _clamp_design(self, d: Dict[str, float]) -> Dict[str, float]:
        """
        ES는 연속값을 뱉으므로, 격자/유닛수/스탯에 대해 최소한의 정수화/클램프를 적용합니다.
        """
        out = dict(d)

        # 맵 크기 (체스보다 크게 유지)
        w = int(round(float(out.get("width", 12))))
        h = int(round(float(out.get("height", 12))))
        out["width"] = max(10, min(24, w))
        out["height"] = max(10, min(24, h))

        # 유닛 수(타입별): 너무 크면 속도 폭발, 너무 작으면 의미 상실
        p0_melee = int(round(float(out.get("p0_melee_units", 8))))
        p0_ranged = int(round(float(out.get("p0_ranged_units", 4))))
        p0_scout = int(round(float(out.get("p0_scout_units", 2))))
        p0_tank = int(round(float(out.get("p0_tank_units", 3))))
        p0_siege = int(round(float(out.get("p0_siege_units", 1))))

        p1_melee = int(round(float(out.get("p1_melee_units", 6))))
        p1_ranged = int(round(float(out.get("p1_ranged_units", 2))))
        p1_scout = int(round(float(out.get("p1_scout_units", 1))))
        p1_tank = int(round(float(out.get("p1_tank_units", 2))))
        p1_siege = int(round(float(out.get("p1_siege_units", 1))))

        # 모든 유닛 타입 최소 1개 이상 (0이 되면 밸런스 붕괴)
        out["p0_melee_units"] = max(1, min(10, p0_melee))
        out["p0_ranged_units"] = max(1, min(8, p0_ranged))
        out["p0_scout_units"] = max(1, min(6, p0_scout))
        out["p0_tank_units"] = max(1, min(6, p0_tank))
        out["p0_siege_units"] = max(1, min(4, p0_siege))
        out["p1_melee_units"] = max(1, min(6, p1_melee))
        out["p1_ranged_units"] = max(1, min(4, p1_ranged))
        out["p1_scout_units"] = max(1, min(4, p1_scout))
        out["p1_tank_units"] = max(1, min(4, p1_tank))
        out["p1_siege_units"] = max(1, min(4, p1_siege))

        # 총 유닛 수를 목표치로 맞추기 (P0=16, P1=10 고정)
        def fix_total(prefix: str):
            keys = [f"{prefix}_melee_units", f"{prefix}_ranged_units", f"{prefix}_scout_units", f"{prefix}_tank_units", f"{prefix}_siege_units"]
            # 모든 유닛 타입 최소 1개 이상
            if prefix == "p0":
                mins = {
                    f"{prefix}_melee_units": 1,
                    f"{prefix}_ranged_units": 1,
                    f"{prefix}_scout_units": 1,
                    f"{prefix}_tank_units": 1,
                    f"{prefix}_siege_units": 1,
                }
                maxs = {
                    f"{prefix}_melee_units": 10,
                    f"{prefix}_ranged_units": 8,
                    f"{prefix}_scout_units": 6,
                    f"{prefix}_tank_units": 6,
                    f"{prefix}_siege_units": 4,
                }
            else:
                mins = {
                    f"{prefix}_melee_units": 1,
                    f"{prefix}_ranged_units": 1,
                    f"{prefix}_scout_units": 1,
                    f"{prefix}_tank_units": 1,
                    f"{prefix}_siege_units": 1,
                }
                maxs = {
                    f"{prefix}_melee_units": 6,
                    f"{prefix}_ranged_units": 4,
                    f"{prefix}_scout_units": 4,
                    f"{prefix}_tank_units": 4,
                    f"{prefix}_siege_units": 4,
                }

            target = int(self.target_units[prefix])
            total = int(sum(int(out[k]) for k in keys))
            if total == target:
                return

            # 우선순위: melee -> ranged -> scout -> tank -> siege (기본 전투 성립을 위해 melee/ranged 유지)
            order = keys

            # 줄이기
            while total > target:
                changed = False
                for k in reversed(order):  # siege/tank/scout/ranged/melee 순으로 먼저 깎기
                    if int(out[k]) > int(mins[k]):
                        out[k] = int(out[k]) - 1
                        total -= 1
                        changed = True
                        if total == target:
                            return
                if not changed:
                    return

            # 늘리기
            while total < target:
                changed = False
                for k in order:  # melee/ranged/scout/tank/siege 순으로 채우기
                    if int(out[k]) < int(maxs[k]):
                        out[k] = int(out[k]) + 1
                        total += 1
                        changed = True
                        if total == target:
                            return
                if not changed:
                    return

        fix_total("p0")
        fix_total("p1")

        # 이동/사거리 (타입별 일부만 최적화 키로 사용)
        for k, lo, hi in [
            ("p0_melee_move", 1.0, 4.0),
            ("p1_melee_move", 1.0, 4.0),
            ("p0_ranged_range", 1.0, 6.0),
            ("p1_ranged_range", 1.0, 6.0),
        ]:
            out[k] = float(max(lo, min(hi, float(out.get(k, lo)))))

        # P1 파워 노브 (HP/데미지): 폭주/0으로 붕괴 방지
        for k, lo, hi in [
            ("p1_melee_hp", 1.0, 10.0),
            ("p1_ranged_hp", 1.0, 10.0),
            ("p1_melee_damage", 0.5, 2.5),
            ("p1_ranged_damage", 0.5, 2.5),
        ]:
            if k in out:
                out[k] = float(max(lo, min(hi, float(out.get(k, lo)))))

        # P0 파워 노브 (HP/데미지): P0가 너무 강한 쪽으로 고정되는 것을 막기 위해
        for k, lo, hi in [
            ("p0_melee_hp", 1.0, 10.0),
            ("p0_ranged_hp", 1.0, 10.0),
            ("p0_melee_damage", 0.5, 2.5),
            ("p0_ranged_damage", 0.5, 2.5),
        ]:
            if k in out:
                out[k] = float(max(lo, min(hi, float(out.get(k, lo)))))

        # 에피소드 길이(너무 길면 속도 폭발)
        out["max_steps"] = int(max(60, min(200, int(round(float(out.get("max_steps", 120)))))))

        return out

    def _eval_pair(self, design_pos: Dict[str, float], design_neg: Dict[str, float], seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        stats_pos = run_simulation(design_pos, train_episodes=self.train_episodes, eval_episodes=self.eval_episodes, seed=seed)
        stats_neg = run_simulation(design_neg, train_episodes=self.train_episodes, eval_episodes=self.eval_episodes, seed=seed)
        return stats_pos, stats_neg

    def step(self, step_index: int = 0):
        # ES (Score Function Estimator)
        # J_sigma(x) ~ E[J(x + sigma*epsilon)]
        
        gradients = {k: 0.0 for k in self.optimizable_keys}
        results = []
        
        # Antithetic Sampling (대칭 샘플링)
        # epsilon, -epsilon 쌍으로 샘플링
        pairs = []
        epsilons = []
        for i in range(self.n_samples // 2):
            epsilon = {k: float(np.random.randn()) for k in self.optimizable_keys}
            epsilons.append(epsilon)

            design_pos = self.mean_design.copy()
            design_neg = self.mean_design.copy()
            for k in self.optimizable_keys:
                design_pos[k] = max(0.1, float(design_pos[k]) + self.sigma * epsilon[k])
                design_neg[k] = max(0.1, float(design_neg[k]) - self.sigma * epsilon[k])

            # 정수/범위 클램프
            design_pos = self._clamp_design(design_pos)
            design_neg = self._clamp_design(design_neg)

            # CRN(공통 난수): 같은 pair는 같은 seed로 평가 (분산 감소)
            seed = int(self.base_seed + step_index * 1000 + i)
            pairs.append((i, design_pos, design_neg, seed))

        # tqdm: "후보 평가 완료 수" 기준으로 바로 움직이게 함 (pos/neg 2개씩이 아니라 pair 단위)
        pbar = tqdm(total=len(pairs), desc="ES Eval", leave=False)

        def _serial_eval():
            out = []
            for i, dpos, dneg, seed in pairs:
                stats_pos, stats_neg = self._eval_pair(dpos, dneg, seed)
                out.append((i, stats_pos, stats_neg))
                pbar.update(1)
            return out

        eval_results = []
        if self.use_parallel and self.max_workers > 1:
            try:
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as ex:
                    futures = {
                        ex.submit(run_simulation, dpos, self.train_episodes, self.eval_episodes, seed): (i, "pos")
                        for i, dpos, _, seed in pairs
                    }
                    futures.update({
                        ex.submit(run_simulation, dneg, self.train_episodes, self.eval_episodes, seed): (i, "neg")
                        for i, _, dneg, seed in pairs
                    })

                    tmp: Dict[int, Dict[str, Any]] = {}
                    done_pair = {i: 0 for i, _, _, _ in pairs}
                    for fut in concurrent.futures.as_completed(futures):
                        i, sign = futures[fut]
                        stats = fut.result()
                        if i not in tmp:
                            tmp[i] = {}
                        tmp[i][sign] = stats
                        done_pair[i] += 1
                        if done_pair[i] == 2:
                            eval_results.append((i, tmp[i]["pos"], tmp[i]["neg"]))
                            pbar.update(1)
            except Exception:
                eval_results = _serial_eval()
        else:
            eval_results = _serial_eval()

        pbar.close()
        eval_results.sort(key=lambda x: x[0])

        for i, stats_pos, stats_neg in eval_results:
            epsilon = epsilons[i]

            loss_pos = self.get_loss(stats_pos)
            loss_neg = self.get_loss(stats_neg)

            if self.verbose:
                print(
                    f"    Sample {i+1} (+): Loss={loss_pos:.4f} | P0={stats_pos['p0_win_rate']:.2f} "
                    f"P1={stats_pos['p1_win_rate']:.2f} Draw={stats_pos['draw_rate']:.2f} Dist={stats_pos['avg_distance']:.2f}"
                )
                print(
                    f"    Sample {i+1} (-): Loss={loss_neg:.4f} | P0={stats_neg['p0_win_rate']:.2f} "
                    f"P1={stats_neg['p1_win_rate']:.2f} Draw={stats_neg['draw_rate']:.2f} Dist={stats_neg['avg_distance']:.2f}"
                )

            diff = loss_pos - loss_neg
            for k in self.optimizable_keys:
                gradients[k] += diff * epsilon[k] / (2 * self.sigma)

            results.append((loss_pos, loss_neg))
            
        # 평균 그라디언트로 업데이트
        avg_grad = {k: v / (self.n_samples // 2) for k, v in gradients.items()}
        
        for k in self.optimizable_keys:
            self.mean_design[k] -= self.lr * avg_grad[k]
            # 범위 제약
            self.mean_design[k] = max(0.1, self.mean_design[k])

        # mean 자체도 클램프 (폭주/0으로 붕괴 방지)
        self.mean_design = self._clamp_design(self.mean_design)
            
        avg_loss = np.mean(results)
        return avg_loss, self.mean_design

