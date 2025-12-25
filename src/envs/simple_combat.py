import torch
import numpy as np
from typing import Dict, Tuple, List, Optional

class SimpleCombatEnv:
    """
    제1장~제4장 문서에 기반한 '설계 가능한' 1차원 전투 환경.
    설계 변수 x에 의해 환경의 파라미터(맵 크기, 유닛 스펙)가 결정됨.
    """
    def __init__(self, config: Dict[str, float]):
        self.config = config
        # 설계 변수로 제어될 파라미터들
        self.max_distance = config.get("map_size", 10.0)
        self.unit_specs = [
            {
                "range": config.get("p0_range", 2.0),
                "damage": config.get("p0_damage", 1.0),
                "speed": config.get("p0_speed", 1.0),
                "hp": config.get("p0_hp", 10.0),
            },
            {
                "range": config.get("p1_range", 2.0),
                "damage": config.get("p1_damage", 1.0),
                "speed": config.get("p1_speed", 1.0),
                "hp": config.get("p1_hp", 10.0),
            }
        ]
        self.max_steps = int(config.get("max_steps", 50))
        self.reset()

    def reset(self):
        # 상태: [거리, p0_hp, p1_hp]
        self.distance = self.max_distance
        self.hps = [self.unit_specs[0]["hp"], self.unit_specs[1]["hp"]]
        self.current_step = 0
        self.history = {
            "distances": [self.distance],
            "actions": []
        }
        return self._get_obs()

    def _get_obs(self):
        # 정규화된 관측값 반환
        return torch.tensor([
            self.distance / self.max_distance,
            self.hps[0] / self.unit_specs[0]["hp"],
            self.hps[1] / self.unit_specs[1]["hp"]
        ], dtype=torch.float32)

    def step(self, actions: List[int]) -> Tuple[torch.Tensor, List[float], bool, Dict]:
        """
        actions: [p0_action, p1_action]
        0: 대기, 1: 전진, 2: 후퇴, 3: 공격
        """
        rewards = [0.0, 0.0]
        
        # 1. 이동 처리 (동시 적용)
        moves = [0.0, 0.0]
        for pid in range(2):
            if actions[pid] == 1: # 전진
                moves[pid] = -self.unit_specs[pid]["speed"]
            elif actions[pid] == 2: # 후퇴
                moves[pid] = self.unit_specs[pid]["speed"]
        
        # 거리 갱신
        old_distance = self.distance
        delta_dist = 0.0
        if actions[0] == 1: delta_dist -= self.unit_specs[0]["speed"]
        if actions[0] == 2: delta_dist += self.unit_specs[0]["speed"]
        if actions[1] == 1: delta_dist -= self.unit_specs[1]["speed"]
        if actions[1] == 2: delta_dist += self.unit_specs[1]["speed"]
        
        self.distance = max(0.0, min(self.max_distance * 1.5, self.distance + delta_dist))
        
        # 거리 보상 (Shaping): 적에게 다가가면 +보상, 멀어지면 -보상 (교전 유도)
        # 단, 너무 가까우면(사거리 이내) 굳이 더 다가갈 필요는 없으므로 사거리 밖일 때만 적용
        dist_reward_scale = 0.05
        
        # p0 입장: 거리가 줄어들면 이득 (상대에게 접근)
        if self.distance > self.unit_specs[0]["range"]:
            if self.distance < old_distance: rewards[0] += dist_reward_scale
            elif self.distance > old_distance: rewards[0] -= dist_reward_scale
            
        # p1 입장: 거리가 줄어들면 이득
        if self.distance > self.unit_specs[1]["range"]:
            if self.distance < old_distance: rewards[1] += dist_reward_scale
            elif self.distance > old_distance: rewards[1] -= dist_reward_scale
        
        # 2. 공격 처리
        # 공격 가능 여부: 현재 거리가 사거리 이내일 것
        for pid in range(2):
            if actions[pid] == 3:
                opp_id = 1 - pid
                if self.distance <= self.unit_specs[pid]["range"]:
                    dmg = self.unit_specs[pid]["damage"]
                    self.hps[opp_id] -= dmg
                    rewards[pid] += 1.0 # 타격 보상
                    rewards[opp_id] -= 1.0 # 피격 페널티
                else:
                    # 헛스윙 페널티 (선택적)
                    rewards[pid] -= 0.1

        self.current_step += 1
        self.history["distances"].append(self.distance)
        self.history["actions"].append(actions)

        # 3. 종료 조건
        done = False
        winner = None # 0, 1, or None (draw)
        
        if self.hps[0] <= 0 or self.hps[1] <= 0:
            done = True
            if self.hps[0] > self.hps[1]:
                winner = 0
                rewards[0] += 5.0
                rewards[1] -= 5.0
            elif self.hps[1] > self.hps[0]:
                winner = 1
                rewards[1] += 5.0
                rewards[0] -= 5.0
            else:
                winner = -1 # 무승부 (동시 사망)
        
        elif self.current_step >= self.max_steps:
            done = True
            winner = -1 # 시간 초과 무승부

        info = {
            "winner": winner,
            "distances": self.history["distances"],
            "hps": self.hps
        }
        
        return self._get_obs(), rewards, done, info


class GridCombatEnv:
    """
    체스보다 큰 격자 맵에서 2팩션이 다수 유닛으로 교전하는 간단 환경.

    - 맵: width x height (기본 12x12 이상 권장)
    - 유닛: 팩션별 N개 (예: 한쪽은 체스(16)보다 조금 많게, 한쪽은 더 적게)
    - 행동(유닛별): 0~4 이동(정지/상/하/좌/우), 5 공격(사거리 내 가장 가까운 적 1개)
    - 이동거리(move_range): 1스텝에서 해당 방향으로 최대 move_range만큼 이동(빈 칸일 때만)
    - 공격거리(attack_range): 맨해튼 거리 기준
    - 킹: 각 팩션 1개씩, 잡히면 즉시 패배
    - 이동 패턴: 유닛 타입별로 다름 (직선/대각선/L자/전방향)
    """

    # 이동 방향 패턴 (dx, dy)
    # 직선(룩): 상하좌우
    MOVE_ORTHOGONAL = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    # 대각선(비숍): 대각 4방향
    MOVE_DIAGONAL = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    # 전방향(퀸/킹): 8방향
    MOVE_ALL = MOVE_ORTHOGONAL + MOVE_DIAGONAL
    # L자(나이트): 8가지 L자 점프
    MOVE_KNIGHT = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]

    # 유닛 타입별 이동 패턴
    TYPE_MOVE_PATTERNS = {
        "melee": MOVE_ORTHOGONAL,   # 룩처럼 직선
        "ranged": MOVE_ALL,         # 퀸처럼 전방향
        "scout": MOVE_KNIGHT,       # 나이트처럼 L자 점프
        "tank": MOVE_ORTHOGONAL,    # 직선만, 느림
        "siege": MOVE_ORTHOGONAL,   # 직선만, 느림
        "king": MOVE_ALL,           # 전방향, 1칸
    }

    TYPE_NAMES = ["melee", "ranged", "scout", "tank", "siege", "king"]  # 6종 (킹 추가)

    def __init__(self, config: Dict[str, float], seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)

        self.width = int(config.get("width", 12))
        self.height = int(config.get("height", 12))
        self.max_steps = int(config.get("max_steps", 120))

        # 유닛 타입 6종: 0..5 (melee/ranged/scout/tank/siege/king)
        # 킹은 항상 1개 고정, 나머지는 config에서 가져옴
        def get_type_counts(prefix: str, default_total: int) -> List[int]:
            provided = []
            any_provided = False
            for name in self.TYPE_NAMES:
                if name == "king":
                    # 킹은 항상 1개
                    provided.append(1)
                    continue
                v = config.get(f"{prefix}_{name}_units", None)
                if v is not None:
                    any_provided = True
                    provided.append(int(v))
                else:
                    provided.append(0)

            if any_provided:
                return provided

            # 기본 분배 (킹 제외한 총합)
            total = int(config.get(f"{prefix}_units", default_total)) - 1  # 킹 1개 제외
            ratios = np.array([0.45, 0.25, 0.10, 0.15, 0.05], dtype=np.float64)
            raw = np.floor(ratios * total).astype(int)
            while raw.sum() < total:
                raw[int(np.argmax(ratios))] += 1
            while raw.sum() > total:
                raw[int(np.argmax(raw))] -= 1
            result = raw.tolist()
            result.append(1)  # 킹 1개 추가
            return result

        p0_counts = get_type_counts("p0", 16)
        p1_counts = get_type_counts("p1", 10)

        self.type_counts = [p0_counts, p1_counts]
        self.n_units = [int(sum(p0_counts)), int(sum(p1_counts))]
        
        # 킹 인덱스 저장 (마지막 유닛이 킹)
        self.king_type_idx = self.TYPE_NAMES.index("king")

        # 타입별 스탯 (팩션별로 독립)
        # move: 이동 거리 (나이트는 1=L자 1회 점프)
        # range: 공격 사거리 (맨해튼)
        def type_specs(prefix: str):
            # 기본값(타입별) - config에 없으면 여기 기본이 적용됨
            defaults = {
                "melee": {"move": 3.0, "range": 1.0, "damage": 1.0, "hp": 3.0},   # 직선 3칸
                "ranged": {"move": 2.0, "range": 4.0, "damage": 1.0, "hp": 2.0},  # 전방향 2칸, 원거리 공격
                "scout": {"move": 1.0, "range": 1.0, "damage": 1.0, "hp": 2.0},   # L자 1회 점프
                "tank": {"move": 1.0, "range": 1.0, "damage": 1.0, "hp": 6.0},    # 직선 1칸, 고체력
                "siege": {"move": 1.0, "range": 6.0, "damage": 1.5, "hp": 2.0},   # 직선 1칸, 초장거리 공격
                "king": {"move": 1.0, "range": 1.0, "damage": 0.5, "hp": 5.0},    # 전방향 1칸, 잡히면 패배
            }

            specs = []
            for name in self.TYPE_NAMES:
                base = defaults[name]
                specs.append(
                    {
                        "move": float(config.get(f"{prefix}_{name}_move", base["move"])),
                        "range": float(config.get(f"{prefix}_{name}_range", base["range"])),
                        "damage": float(config.get(f"{prefix}_{name}_damage", base["damage"])),
                        "hp": float(config.get(f"{prefix}_{name}_hp", base["hp"])),
                    }
                )
            return specs

        self.typespecs = [
            type_specs("p0"),
            type_specs("p1"),
        ]

        self.reset()

    def reset(self, first_turn=None):
        """
        first_turn: None이면 랜덤, 0 또는 1이면 해당 진영 선턴
        """
        self.step_idx = 0
        # first_turn이 명시되면 그대로, 아니면 랜덤
        if first_turn is not None:
            self.side_to_act = int(first_turn)
        else:
            self.side_to_act = int(self.rng.integers(0, 2))

        # positions[f] = [(x,y), ...], hps[f] = [hp, ...]
        self.positions: List[List[Tuple[int, int]]] = [[], []]
        self.hps: List[List[float]] = [[], []]
        self.unit_types: List[List[int]] = [[], []]  # 타입 인덱스
        self.king_dead = [False, False]  # 킹 사망 플래그

        occupied = set()

        # 턴제(턴당 1유닛)에서는 가장자리 배치(좌 2열 vs 우 2열)가 교전까지 너무 오래 걸려
        # 무승부로 퇴화하기 쉽습니다. 맵은 크게 유지하되, 초반 교전이 가능하도록 중앙 근처에 배치합니다.
        def place_units(f: int):
            mid = int(self.width // 2)
            if f == 0:
                cols = [max(0, mid - 3), max(0, mid - 2)]
            else:
                cols = [min(self.width - 1, mid + 1), min(self.width - 1, mid + 2)]
            counts = self.type_counts[f]
            # 타입 배열 생성 후 섞기(배치 랜덤)
            types = []
            for t, c in enumerate(counts):
                types.extend([t] * int(c))
            self.rng.shuffle(types)
            count = len(types)
            attempts = 0
            while len(self.positions[f]) < count and attempts < max(1, count) * 200:
                x = int(self.rng.choice(cols))
                y = int(self.rng.integers(0, self.height))
                if (x, y) in occupied:
                    attempts += 1
                    continue
                occupied.add((x, y))
                self.positions[f].append((x, y))
                t = int(types[len(self.positions[f]) - 1])
                self.unit_types[f].append(t)
                self.hps[f].append(self.typespecs[f][t]["hp"])
                attempts += 1

        place_units(0)
        place_units(1)

        self.history = {
            "attack_distances": [],  # 교전 거리 표본(맨해튼)
        }
        self.no_attack_steps = 0
        self.any_attack = False
        self._prev_avg_d = None

        return self._get_obs()

    def _alive_indices(self, f: int) -> List[int]:
        return [i for i, hp in enumerate(self.hps[f]) if hp > 0]

    def _manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _nearest_enemy_all(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        각 팩션 유닛 i에 대해 '가장 가까운 적'의 (enemy_index, manhattan_distance)를 벡터화로 계산합니다.
        반환:
          nearest_idx[f]: shape (n_units_f,), dead/적없음은 -1
          nearest_dist[f]: shape (n_units_f,), dead/적없음은 0
        """
        nearest_idx = [np.full((self.n_units[0],), -1, dtype=np.int32), np.full((self.n_units[1],), -1, dtype=np.int32)]
        nearest_dist = [np.zeros((self.n_units[0],), dtype=np.int32), np.zeros((self.n_units[1],), dtype=np.int32)]

        for f in (0, 1):
            opp = 1 - f
            alive_self = np.array(self._alive_indices(f), dtype=np.int32)
            alive_opp = np.array(self._alive_indices(opp), dtype=np.int32)
            if alive_self.size == 0 or alive_opp.size == 0:
                continue

            self_pos = np.array([self.positions[f][i] for i in alive_self], dtype=np.int32)  # (A,2)
            opp_pos = np.array([self.positions[opp][j] for j in alive_opp], dtype=np.int32)  # (B,2)

            # manhattan distance matrix (A,B)
            dx = np.abs(self_pos[:, 0:1] - opp_pos[None, :, 0])
            dy = np.abs(self_pos[:, 1:2] - opp_pos[None, :, 1])
            dist = dx + dy

            argmin = dist.argmin(axis=1)
            dmin = dist[np.arange(dist.shape[0]), argmin]

            nearest_idx[f][alive_self] = alive_opp[argmin]
            nearest_dist[f][alive_self] = dmin.astype(np.int32)

        return nearest_idx, nearest_dist

    def _get_unit_obs(self, f: int, i: int) -> torch.Tensor:
        # dead 유닛은 0벡터
        if self.hps[f][i] <= 0:
            return torch.zeros(8, dtype=torch.float32)

        x, y = self.positions[f][i]
        hp = self.hps[f][i]
        hp_max = self.specs[f]["hp"]

        enemy_i, d = self._nearest_enemy(f, (x, y))
        if enemy_i is None:
            dx = 0.0
            dy = 0.0
            dist = 0.0
        else:
            ex, ey = self.positions[1 - f][enemy_i]
            dx = (ex - x) / max(1.0, float(self.width - 1))
            dy = (ey - y) / max(1.0, float(self.height - 1))
            dist = float(d) / max(1.0, float(self.width + self.height - 2))

        alive_self = len(self._alive_indices(f))
        alive_enemy = len(self._alive_indices(1 - f))

        return torch.tensor(
            [
                x / max(1.0, float(self.width - 1)),
                y / max(1.0, float(self.height - 1)),
                hp / max(1e-6, float(hp_max)),
                dx,
                dy,
                dist,
                alive_self / max(1.0, float(self.n_units[f])),
                alive_enemy / max(1.0, float(self.n_units[1 - f])),
            ],
            dtype=torch.float32,
        )

    def _get_obs(self):
        # 파이썬 루프 최소화: nearest를 한 번에 계산하고 obs를 구성
        nearest_idx, nearest_dist = self._nearest_enemy_all()

        obs_all = [[], []]
        for f in (0, 1):
            opp = 1 - f
            denom_x = max(1.0, float(self.width - 1))
            denom_y = max(1.0, float(self.height - 1))
            denom_d = max(1.0, float(self.width + self.height - 2))

            alive_self = self._alive_indices(f)
            alive_self_set = set(alive_self)

            for i in range(self.n_units[f]):
                if i not in alive_self_set:
                    obs_all[f].append(torch.zeros(12, dtype=torch.float32))
                    continue

                x, y = self.positions[f][i]
                hp = float(self.hps[f][i])
                t = int(self.unit_types[f][i])
                spec = self.typespecs[f][t]
                hp_max = float(spec["hp"])
                ei = int(nearest_idx[f][i])
                if ei < 0:
                    dxn = 0.0
                    dyn = 0.0
                    distn = 0.0
                    in_range = 0.0
                else:
                    ex, ey = self.positions[opp][ei]
                    dxn = (ex - x) / denom_x
                    dyn = (ey - y) / denom_y
                    distn = float(nearest_dist[f][i]) / denom_d
                    d_raw = int(nearest_dist[f][i])
                    in_range = 1.0 if d_raw <= int(max(1.0, round(spec["range"]))) else 0.0

                alive_self_cnt = len(alive_self)
                alive_enemy_cnt = len(self._alive_indices(opp))

                # 타입/스탯 피처(정규화)
                type_norm = float(t) / 4.0  # 0..1
                move_norm = float(spec["move"]) / 4.0
                range_norm = float(spec["range"]) / 6.0

                obs_all[f].append(
                    torch.tensor(
                        [
                            x / denom_x,
                            y / denom_y,
                            hp / max(1e-6, hp_max),
                            dxn,
                            dyn,
                            distn,
                            alive_self_cnt / max(1.0, float(self.n_units[f])),
                            alive_enemy_cnt / max(1.0, float(self.n_units[opp])),
                            type_norm,
                            move_norm,
                            range_norm,
                            in_range,
                        ],
                        dtype=torch.float32,
                    )
                )

        return obs_all

    def step(self, turn_action: Tuple[int, int, int]):
        """
        체스 룰: 교대 턴 + 턴당 1유닛만 행동.
        turn_action = (f, unit_index, action)
          - f: 현재 턴의 팩션(0 또는 1)이어야 함
          - unit_index: 해당 팩션 유닛 인덱스
          - action: 0~4 이동(정지/상/하/좌/우), 5 공격
        """
        rewards = [0.0, 0.0]

        f, i, a = int(turn_action[0]), int(turn_action[1]), int(turn_action[2])
        if f != int(getattr(self, "side_to_act", 0)):
            # 규칙 위반 입력은 정지로 처리
            f = int(getattr(self, "side_to_act", 0))
            i = 0
            a = 0

        # 현재 점유 맵 구성(교대턴이지만 이동 충돌 체크용)
        occupied = {}
        for ff in (0, 1):
            for ii in self._alive_indices(ff):
                occupied[self.positions[ff][ii]] = (ff, ii)

        # 1) 이동/정지: 현재 턴 유닛 1개만 처리
        # 유닛 타입별 이동 패턴 적용
        attacks_this_step = 0
        if i in self._alive_indices(f):
            t = int(self.unit_types[f][i])
            type_name = self.TYPE_NAMES[t]
            move_pattern = self.TYPE_MOVE_PATTERNS.get(type_name, self.MOVE_ORTHOGONAL)
            move_range = int(max(1.0, round(self.typespecs[f][t]["move"])))
            
            # action 0~7: 이동 방향 선택 (패턴에 따라 사용 가능한 방향 수 다름)
            # action >= 패턴 길이: 공격(5) 또는 무효
            if a < len(move_pattern):
                dx, dy = move_pattern[a]
                x, y = self.positions[f][i]
                
                # 나이트(L자)는 점프, 나머지는 슬라이딩
                if type_name == "scout":
                    # L자 점프: move_range만큼 반복 점프 가능
                    for _ in range(move_range):
                        nx = int(x + dx)
                        ny = int(y + dy)
                        if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in occupied:
                            del occupied[(x, y)]
                            occupied[(nx, ny)] = (f, i)
                            self.positions[f][i] = (nx, ny)
                            x, y = nx, ny
                        else:
                            break
                else:
                    # 직선/대각선 슬라이딩: move_range칸까지 이동
                    def try_move(dir_dx: int, dir_dy: int) -> bool:
                        nonlocal x, y
                        for step in range(move_range, 0, -1):
                            nx = int(max(0, min(self.width - 1, x + dir_dx * step)))
                            ny = int(max(0, min(self.height - 1, y + dir_dy * step)))
                            if (nx, ny) == (x, y):
                                continue
                            # 경로 상 장애물 체크 (슬라이딩이므로)
                            blocked = False
                            for s in range(1, step + 1):
                                check_x = int(x + dir_dx * s)
                                check_y = int(y + dir_dy * s)
                                if (check_x, check_y) in occupied and (check_x, check_y) != (nx, ny):
                                    blocked = True
                                    break
                            if blocked:
                                continue
                            if (nx, ny) in occupied:
                                continue
                            del occupied[(x, y)]
                            occupied[(nx, ny)] = (f, i)
                            self.positions[f][i] = (nx, ny)
                            return True
                        return False

                    moved = try_move(dx, dy)
                    if not moved:
                        # 다른 방향 시도
                        other_dirs = list(move_pattern)
                        self.rng.shuffle(other_dirs)
                        for odx, ody in other_dirs:
                            if (odx, ody) == (dx, dy):
                                continue
                            if try_move(odx, ody):
                                break

        # 2) 공격 처리: 현재 턴 유닛 1개만 처리
        # action이 이동 패턴 길이 이상이면 공격
        nearest_idx, nearest_dist = self._nearest_enemy_all()
        is_attack_action = (a >= len(move_pattern)) if i in self._alive_indices(f) else False
        if is_attack_action and i in self._alive_indices(f):
            t = int(self.unit_types[f][i])
            atk_range = int(max(1.0, round(self.typespecs[f][t]["range"])))
            dmg = float(self.typespecs[f][t]["damage"])
            enemy_i = int(nearest_idx[f][i])
            d = int(nearest_dist[f][i])
            if enemy_i >= 0 and d <= atk_range:
                def_f = 1 - f
                self.hps[def_f][enemy_i] -= dmg
                rewards[f] += dmg
                rewards[def_f] -= dmg
                self.history["attack_distances"].append(float(d))
                attacks_this_step = 1
                self.any_attack = True
                
                # 킹 사망 체크: 공격받은 유닛이 킹이고 HP <= 0이면 즉시 패배
                if self.unit_types[def_f][enemy_i] == self.king_type_idx and self.hps[def_f][enemy_i] <= 0:
                    self.king_dead[def_f] = True

        # 2-b) 거리 쉐이핑 보상(접근 유도): 살아있는 유닛들의 최근접 적 거리 평균을 줄이면 보상
        # 벡터화된 nearest_dist를 사용하므로 오버헤드가 거의 없음
        shaping_scale = float(self.config.get("shaping_scale", 0.02))
        for f in (0, 1):
            alive = self._alive_indices(f)
            if len(alive) == 0:
                continue
            # dead는 0, alive는 거리값이 들어있음
            avg_d = float(np.mean(nearest_dist[f][alive])) if len(alive) else 0.0
            if self._prev_avg_d is None:
                self._prev_avg_d = [avg_d, avg_d]
            prev = self._prev_avg_d
            # 거리 감소 -> +, 증가 -> -
            rewards[f] += shaping_scale * (prev[f] - avg_d)
            prev[f] = avg_d
            self._prev_avg_d = prev

        # 2-c) 무교전 조기 종료용 카운터
        if attacks_this_step == 0:
            self.no_attack_steps += 1
        else:
            self.no_attack_steps = 0

        self.step_idx += 1
        # 다음 턴으로
        self.side_to_act = 1 - int(getattr(self, "side_to_act", 0))

        done = False
        winner = None
        alive0 = len(self._alive_indices(0))
        alive1 = len(self._alive_indices(1))
        
        # 킹 사망 체크: 킹이 죽으면 즉시 패배 (체스처럼)
        if getattr(self, "king_dead", [False, False])[0]:
            done = True
            winner = 1
            rewards[1] += 20.0
            rewards[0] -= 20.0
        elif getattr(self, "king_dead", [False, False])[1]:
            done = True
            winner = 0
            rewards[0] += 20.0
            rewards[1] -= 20.0
        elif alive0 == 0 or alive1 == 0:
            done = True
            if alive0 > alive1:
                winner = 0
                rewards[0] += 10.0
                rewards[1] -= 10.0
            elif alive1 > alive0:
                winner = 1
                rewards[1] += 10.0
                rewards[0] -= 10.0
            else:
                winner = -1
        # 무교전이 일정 턴 지속되면 조기 종료 + 큰 페널티(퇴화 방지)
        no_attack_limit = int(self.config.get("no_attack_limit", 20))
        if not done and self.no_attack_steps >= no_attack_limit:
            done = True
            # 체스/장기류의 턴제에서는 "무교전"을 전력으로 판정하면
            # 유닛 수 비대칭이 곧바로 승률 편향으로 고정됩니다.
            # 무교전(공격 0회)이라면 draw로 처리하고, 디자이너 쪽 페널티로 퇴화를 끊습니다.
            if not self.any_attack:
                winner = -1
                rewards[0] -= 2.0
                rewards[1] -= 2.0
            else:
                # 교전이 있었으면 남은 전력(유닛 수/총 HP)으로 승부
                hp0 = float(sum(max(0.0, hp) for hp in self.hps[0]))
                hp1 = float(sum(max(0.0, hp) for hp in self.hps[1]))
                if alive0 > alive1 or (alive0 == alive1 and hp0 > hp1):
                    winner = 0
                    rewards[0] += 1.0
                    rewards[1] -= 1.0
                elif alive1 > alive0 or (alive0 == alive1 and hp1 > hp0):
                    winner = 1
                    rewards[1] += 1.0
                    rewards[0] -= 1.0
                else:
                    winner = -1
                    rewards[0] -= 2.0
                    rewards[1] -= 2.0

        elif self.step_idx >= self.max_steps:
            done = True
            # 시간초과도 무교전이면 draw (유닛 수 비대칭 고정 승패 방지)
            if not self.any_attack:
                winner = -1
                rewards[0] -= 1.0
                rewards[1] -= 1.0
            else:
                # 교전이 있었으면 남은 전력으로 승부
                if alive0 > alive1:
                    winner = 0
                    rewards[0] += 2.0
                    rewards[1] -= 2.0
                elif alive1 > alive0:
                    winner = 1
                    rewards[1] += 2.0
                    rewards[0] -= 2.0
                else:
                    hp0 = float(sum(max(0.0, hp) for hp in self.hps[0]))
                    hp1 = float(sum(max(0.0, hp) for hp in self.hps[1]))
                    if hp0 > hp1:
                        winner = 0
                        rewards[0] += 1.0
                        rewards[1] -= 1.0
                    elif hp1 > hp0:
                        winner = 1
                        rewards[1] += 1.0
                        rewards[0] -= 1.0
                    else:
                        winner = -1

        info = {
            "winner": winner,
            "attack_distances": list(self.history["attack_distances"]),
            "alive": (alive0, alive1),
            "no_attack_steps": self.no_attack_steps,
            "side_to_act": int(getattr(self, "side_to_act", 0)),
        }

        # 보상 스케일 정규화:
        # 유닛 수가 많은 팩션은 "같은 행동 품질"이라도 공격 기회/보상 합이 더 커져 학습이 유리해집니다.
        # 팩션별 초기 유닛 수로 나눠 학습 스케일을 맞춥니다. (평가 통계에는 영향 없음: 승률/거리만 사용)
        scale0 = 1.0 / float(max(1, int(self.n_units[0])))
        scale1 = 1.0 / float(max(1, int(self.n_units[1])))
        rewards[0] = float(rewards[0]) * scale0
        rewards[1] = float(rewards[1]) * scale1

        return self._get_obs(), rewards, done, info

