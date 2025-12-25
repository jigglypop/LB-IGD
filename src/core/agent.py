import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class SimplePolicy(nn.Module):
    """
    간단한 MLP 정책 네트워크.
    다유닛 격자 환경에서 유닛 관측 -> 행동 확률
    행동: 이동(유닛 타입별 방향 수) + 공격(1)
    최대 9개 action (8방향 이동 + 공격)
    """
    def __init__(self, input_dim=12, hidden_dim=64, output_dim=9):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # 교대 턴(체스 룰)에서 "어느 유닛이 행동할지" 선택을 위한 head
        self.fc_sel = nn.Linear(hidden_dim, 1)
        
    def forward_logits(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h)

    def forward_select_logit(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_sel(h).squeeze(-1)

    def forward(self, x):
        logits = self.forward_logits(x)
        return F.softmax(logits, dim=-1)

    def get_action(self, obs, action_mask=None, attack_bonus: float = 0.0):
        logits = self.forward_logits(obs)
        # in_range일 때 공격을 살짝 선호(완전 강제는 아님)
        if attack_bonus != 0.0:
            logits = logits.clone()
            logits[5] = logits[5] + float(attack_bonus)

        if action_mask is not None:
            mask = action_mask.to(dtype=torch.bool)
            masked_logits = logits.clone()
            masked_logits[~mask] = -1e9
            logits = masked_logits

        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def get_turn_action(self, unit_obs: List[torch.Tensor]):
        """
        체스 룰: 현재 턴에서 행동할 유닛 1개를 고르고, 그 유닛의 행동(0~5)을 선택합니다.
        반환: (unit_index, action, log_prob_total)
        """
        if len(unit_obs) == 0:
            return 0, 0, torch.tensor(0.0)

        # 살아있는 유닛만 선택 가능(HP=0이면 obs[2]=0)
        alive_mask = []
        for u in unit_obs:
            try:
                alive_mask.append(float(u[2].item()) > 0.0)
            except Exception:
                alive_mask.append(False)

        if not any(alive_mask):
            return 0, 0, torch.tensor(0.0)

        sel_logits = []
        for ok, u in zip(alive_mask, unit_obs):
            if not ok:
                sel_logits.append(torch.tensor(-1e9))
                continue
            sel_logits.append(self.forward_select_logit(u))
        sel_logits = torch.stack(sel_logits)
        sel_probs = F.softmax(sel_logits, dim=-1)
        sel_dist = torch.distributions.Categorical(sel_probs)
        ui = int(sel_dist.sample().item())
        logp_sel = sel_dist.log_prob(torch.tensor(ui))

        uobs = unit_obs[ui]
        in_range = 0.0
        try:
            in_range = float(uobs[11].item())
        except Exception:
            in_range = 0.0
        
        # 유닛 타입별 유효 action 수 (obs[10]에 type_norm 있음)
        # type_norm = type_idx / 5.0 -> type_idx = round(type_norm * 5)
        type_idx = 0
        try:
            type_norm = float(uobs[10].item())
            type_idx = int(round(type_norm * 5))
        except Exception:
            pass
        
        # 타입별 이동 방향 수: melee/tank/siege=4, ranged/king=8, scout=8
        move_dirs_count = [4, 8, 8, 4, 4, 8][type_idx] if type_idx < 6 else 4
        attack_action = move_dirs_count  # 공격 action은 이동 다음
        
        mask = torch.ones(9, dtype=torch.bool)
        # 유효하지 않은 이동 방향 마스킹
        for i in range(9):
            if i < move_dirs_count:
                mask[i] = True  # 이동 가능
            elif i == attack_action:
                mask[i] = in_range > 0.0  # 공격은 사거리 내일 때만
            else:
                mask[i] = False  # 유효하지 않은 action
        
        # 사거리 안이면 공격 우선
        if in_range > 0.0:
            # 이동 마스킹 (공격 강제)
            for i in range(move_dirs_count):
                mask[i] = False
            mask[attack_action] = True
        
        a, logp_a = self.get_action(uobs, action_mask=mask, attack_bonus=0.0)
        return ui, a, (logp_sel + logp_a)

def _actions_for_faction(policy: SimplePolicy, unit_obs: List[torch.Tensor]):
    actions = []
    log_probs = []
    for obs in unit_obs:
        # dead 유닛(HP=0)은 행동하지 않음: 정책 호출 자체를 줄여 속도 개선
        # obs[2] = hp_norm
        try:
            if float(obs[2].item()) <= 0.0:
                actions.append(0)
                continue
        except Exception:
            pass

        # obs[11] = in_range (0/1)
        in_range = 0.0
        try:
            in_range = float(obs[11].item())
        except Exception:
            in_range = 0.0

        # 공격은 사거리 안일 때만 허용 (out-of-range attack 낭비 제거)
        mask = torch.ones(9, dtype=torch.bool)
        # 기본적으로 action 8(공격)만 마스킹
        if in_range <= 0.0:
            mask[8] = False

        attack_bonus = 2.0 if in_range > 0.0 else 0.0
        a, lp = policy.get_action(obs, action_mask=mask, attack_bonus=attack_bonus)
        actions.append(a)
        log_probs.append(lp)
    return actions, log_probs


def train_one_episode(env, policies: List[SimplePolicy], optimizers: List[torch.optim.Optimizer], first_turn=None):
    """
    한 에피소드 진행 및 REINFORCE 업데이트 (공유 정책, faction 단위 보상)
    first_turn: None이면 랜덤, 0 또는 1이면 해당 진영 선턴
    """
    obs = env.reset(first_turn=first_turn)
    done = False
    
    # timestep마다 (턴에서의 logprob) 저장
    log_probs_sum = [[], []]
    rewards_episode = [[], []]  # faction reward
    
    while not done:
        obs0_units, obs1_units = obs
        side = int(getattr(env, "side_to_act", 0))
        if side == 0:
            ui, a, lp = policies[0].get_turn_action(obs0_units)
            next_obs, rewards, done, info = env.step((0, ui, a))
            log_probs_sum[0].append(lp)
            log_probs_sum[1].append(torch.tensor(0.0))
        else:
            ui, a, lp = policies[1].get_turn_action(obs1_units)
            next_obs, rewards, done, info = env.step((1, ui, a))
            log_probs_sum[0].append(torch.tensor(0.0))
            log_probs_sum[1].append(lp)
        rewards_episode[0].append(float(rewards[0]))
        rewards_episode[1].append(float(rewards[1]))

        obs = next_obs

    # 학습 (REINFORCE)
    for pid in range(2):
        R = 0
        loss = 0
        returns = []
        # Return 계산
        for r in reversed(rewards_episode[pid]):
            R = r + 0.95 * R # gamma = 0.95
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        if returns.numel() > 1:
            std = returns.std(unbiased=False)
            if std > 1e-8:
                returns = (returns - returns.mean()) / (std + 1e-8)
            
        for lp, ret in zip(log_probs_sum[pid], returns):
            loss -= lp * ret
            
        optimizers[pid].zero_grad()
        loss.backward()
        optimizers[pid].step()
        
    return info

