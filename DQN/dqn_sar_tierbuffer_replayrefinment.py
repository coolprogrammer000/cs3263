from __future__ import annotations

import random
from collections import Counter, deque, namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from astar import AStarPlanner, heuristic_landmark
from env import Action, SearchRescueEnv, State, Tile

# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# State Encoder
# ============================================================

class SARStateEncoder:
    def __init__(self, env: SearchRescueEnv):
        self.rows = env.rows
        self.cols = env.cols
        self.max_oxygen = env.max_oxygen

        self.victim_positions = sorted(env.victim_positions)
        self.exit_positions = sorted(env.exit_positions)

        self.door_positions = []
        self.debris_positions = []
        self.medkit_positions = []

        for r in range(env.rows):
            for c in range(env.cols):
                t = env.grid[r][c]
                if t in (Tile.DOOR, Tile.DOOR_L):
                    self.door_positions.append((r, c))
                elif t == Tile.DEBRIS:
                    self.debris_positions.append((r, c))
                elif t == Tile.MEDKIT:
                    self.medkit_positions.append((r, c))

        self.door_positions = sorted(self.door_positions)
        self.debris_positions = sorted(self.debris_positions)
        self.medkit_positions = sorted(self.medkit_positions)

        colors = set(env.door_colors.values()) | set(env.item_colors.values())
        self.key_names = sorted([f"key_{c}" for c in colors])

        self.tile_values = [
            Tile.EMPTY, Tile.WALL, Tile.DOOR, Tile.DOOR_L, Tile.DEBRIS,
            Tile.SMOKE, Tile.VICTIM, Tile.MEDKIT, Tile.KEY, Tile.CROWBAR, Tile.EXIT
        ]
        self.tile_to_idx = {int(t): i for i, t in enumerate(self.tile_values)}

        self.feature_dim = (
            2 +
            1 +
            2 + len(self.key_names) +
            len(self.door_positions) +
            len(self.debris_positions) +
            len(self.victim_positions) +
            len(self.victim_positions) +
            len(self.medkit_positions) +
            len(self.tile_values)
        )

    def encode(self, env: SearchRescueEnv, state: State) -> np.ndarray:
        feats: List[float] = []

        r, c = state.agent_pos
        feats.append(r / max(1, env.rows - 1))
        feats.append(c / max(1, env.cols - 1))
        feats.append(state.oxygen / max(1, env.max_oxygen))

        feats.append(1.0 if "crowbar" in state.inventory else 0.0)
        feats.append(1.0 if "medkit" in state.inventory else 0.0)
        for key_name in self.key_names:
            feats.append(1.0 if key_name in state.inventory else 0.0)

        door_open_set = set(state.doors_open)
        for pos in self.door_positions:
            feats.append(1.0 if pos in door_open_set else 0.0)

        debris_set = set(state.debris_cleared)
        for pos in self.debris_positions:
            feats.append(1.0 if pos in debris_set else 0.0)

        found_set = set(state.victims_found)
        for pos in self.victim_positions:
            feats.append(1.0 if pos in found_set else 0.0)

        rescued_set = set(state.victims_rescued)
        for pos in self.victim_positions:
            feats.append(1.0 if pos in rescued_set else 0.0)

        medkit_taken_set = set(state.medkits_taken)
        for pos in self.medkit_positions:
            feats.append(1.0 if pos in medkit_taken_set else 0.0)

        tile = int(env._effective_tile(r, c, state))
        one_hot = [0.0] * len(self.tile_values)
        if tile in self.tile_to_idx:
            one_hot[self.tile_to_idx[tile]] = 1.0
        feats.extend(one_hot)

        return np.asarray(feats, dtype=np.float32)


# ============================================================
# Replay Buffer (uniform sampling only)
# ============================================================

Transition = namedtuple(
    "Transition",
    ["state_vec", "action", "reward", "next_state_vec", "done", "legal_mask", "next_legal_mask"]
)

class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        if batch_size > len(self.buffer):
            raise ValueError(f"sample batch_size={batch_size} > buffer size={len(self.buffer)}")
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self) -> int:
        return len(self.buffer)


# ============================================================
# Q-Network
# ============================================================

class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Agent
# ============================================================

@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 128
    buffer_capacity: int = 100_000

    tier1_buffer_capacity: int = 15_000
    tier2_buffer_capacity: int = 12_000
    tier3_buffer_capacity: int = 10_000

    main_fraction: float = 0.50
    tier1_fraction: float = 0.15
    tier2_fraction: float = 0.15
    tier3_fraction: float = 0.20

    min_buffer_size: int = 2_000
    target_update_freq: int = 1000
    train_freq: int = 1
    max_steps_per_episode: int = 1000

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 30_000

    grad_clip_norm: float = 10.0
    double_dqn: bool = True

    # reward shaping
    use_reward_shaping: bool = True
    shape_pickup_useful_key: float = 4.0
    shape_pickup_crowbar: float = 3.0
    shape_pickup_medkit: float = 5.0
    shape_open_new_door: float = 3.0
    shape_clear_new_debris: float = 4.0
    shape_find_new_victim: float = 10.0
    shape_rescue_each_victim: float = 20.0
    shape_progress_scale: float = 0.3
    shape_no_progress_penalty: float = -0.2
    shape_episode_success_bonus: float = 20.0

    # transition priority weights used ONLY for tier assignment
    prio_base: float = 1.0
    prio_useful_key: float = 2.0
    prio_crowbar: float = 2.0
    prio_medkit: float = 1.0
    prio_open_door: float = 2.0
    prio_clear_debris: float = 2.0
    prio_find_victim: float = 4.0
    prio_rescue_victim: float = 6.0
    prio_success_bonus: float = 8.0
    prio_episode_score_scale: float = 0.3
    prio_position_bonus_scale: float = 1.0

    # thresholds for tier assignment
    tier1_priority_threshold: float = 4.0
    tier2_priority_threshold: float = 7.0
    tier3_priority_threshold: float = 10.0

    log_path: str = "dqn_train.log"


class DQNAgent:
    def __init__(self, env: SearchRescueEnv, device: str = "cpu", config: DQNConfig = DQNConfig()):
        self.env = env
        self.device = torch.device(device)
        self.config = config

        self.encoder = SARStateEncoder(env)

        self.num_actions = len(Action)
        self.online_net = DQN(self.encoder.feature_dim, self.num_actions).to(self.device)
        self.target_net = DQN(self.encoder.feature_dim, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config.lr)

        self.buffer = ReplayBuffer(config.buffer_capacity)
        self.tier1_buffer = ReplayBuffer(config.tier1_buffer_capacity)
        self.tier2_buffer = ReplayBuffer(config.tier2_buffer_capacity)
        self.tier3_buffer = ReplayBuffer(config.tier3_buffer_capacity)

        self.total_steps = 0

        self.position_counter = Counter()
        self.last_episode_position_counter = Counter()
        self.log_path = Path(config.log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(self, msg: str) -> None:
        print(msg)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def _format_visit_grid(self, counter: Counter) -> str:
        rows, cols = self.env.rows, self.env.cols
        grid = [[counter.get((r, c), 0) for c in range(cols)] for r in range(rows)]
        cell_width = max(3, max(len(str(v)) for row in grid for v in row))
        lines = []
        header = "r\\c ".ljust(5) + " ".join(f"{c:>{cell_width}}" for c in range(cols))
        lines.append(header)
        for r in range(rows):
            row_str = f"{r:<4} " + " ".join(f"{grid[r][c]:>{cell_width}}" for c in range(cols))
            lines.append(row_str)
        return "\n".join(lines)

    def print_final_visit_map(self) -> None:
        self._log("\n=== Final Visit Map (global) ===")
        self._log(self._format_visit_grid(self.position_counter))

    def _manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _current_goal_distance(self, state: State) -> float:
        pos = state.agent_pos
        unrescued_found = state.victims_found - state.victims_rescued
        if unrescued_found:
            return float(min(self._manhattan(pos, e) for e in self.env.exit_positions))

        unvisited_victims = [v for v in self.env.victim_positions if v not in state.victims_found]
        if unvisited_victims:
            return float(min(self._manhattan(pos, v) for v in unvisited_victims))

        return 0.0

    def prefill_with_landmark(
        self,
        num_episodes: int = 20,
        max_nodes: int = 300_000,
        use_shaped_reward: bool = True,
    ) -> None:
        """
        Use landmark A* to generate trajectories and prefill replay buffers
        before normal DQN training starts.
        """
        self._log(f"\n=== Prefilling replay with landmark A* for {num_episodes} episode(s) ===")

        successful_prefills = 0
        total_added = 0

        for ep in range(1, num_episodes + 1):
            # fresh env reset
            start_state = self.env.reset()

            planner = AStarPlanner(
                self.env,
                heuristic=heuristic_landmark,
                max_nodes=max_nodes,
            )
            result = planner.plan(initial_state=start_state)

            if not result.success or not result.actions:
                self._log(
                    f"Prefill episode {ep}: planner failed | "
                    f"expanded={result.nodes_expanded} | generated={result.nodes_generated}"
                )
                continue

            # replay the planned path and store transitions
            state = self.env.reset()
            episode_steps = []

            for action in result.actions:
                next_state, env_reward, done = self.env.step(action)

                reward = (
                    self.shape_reward(state, action, env_reward, next_state, done)
                    if use_shaped_reward
                    else env_reward
                )

                episode_steps.append({
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "done": done,
                })

                state = next_state
                if done:
                    break

            final_state = state
            episode_score = self._episode_quality_score(final_state)
            episode_len = len(episode_steps)

            for idx, step in enumerate(episode_steps):
                prio = self.compute_transition_priority(
                    state=step["state"],
                    action=step["action"],
                    next_state=step["next_state"],
                    done=step["done"],
                    episode_score=episode_score,
                    step_idx=idx,
                    episode_len=episode_len,
                )

                tr = self.build_transition(
                    step["state"],
                    step["action"],
                    step["reward"],
                    step["next_state"],
                    step["done"],
                )

                # always add to main buffer
                self.buffer.push(*tr)

                # also add to tier buffers based on transition priority
                tier = self._priority_to_tier(prio)
                if tier == 1:
                    self.tier1_buffer.push(*tr)
                elif tier == 2:
                    self.tier2_buffer.push(*tr)
                elif tier == 3:
                    self.tier3_buffer.push(*tr)

                total_added += 1

            successful_prefills += 1
            self._log(
                f"Prefill episode {ep}: success | "
                f"steps={len(episode_steps)} | "
                f"reward={result.total_reward:.1f} | "
                f"rescued={len(final_state.victims_rescued)}/{self.env.total_victims}"
            )

        self._log(
            f"=== Prefill done | successful_prefills={successful_prefills}/{num_episodes} | "
            f"transitions_added={total_added} | "
            f"main_buf={len(self.buffer)} | "
            f"tier1_buf={len(self.tier1_buffer)} | "
            f"tier2_buf={len(self.tier2_buffer)} | "
            f"tier3_buf={len(self.tier3_buffer)} ==="
        )

    def shape_reward(
        self,
        state: State,
        action: Action,
        env_reward: float,
        next_state: State,
        done: bool,
    ) -> float:
        if not self.config.use_reward_shaping:
            return env_reward

        shaped = env_reward
        cfg = self.config

        new_items = set(next_state.inventory) - set(state.inventory)
        useful_keys = {f"key_{color}" for color in self.env.door_colors.values()}
        useful_inventory_now = [
            item for item in next_state.inventory
            if item in useful_keys or item == "crowbar" or item == "medkit"
        ]
        num_useful_items_now = max(1, len(useful_inventory_now))

        if "crowbar" in new_items:
            shaped += cfg.shape_pickup_crowbar * num_useful_items_now
        if "medkit" in new_items:
            shaped += cfg.shape_pickup_medkit * num_useful_items_now
        for item in new_items:
            if item in useful_keys:
                shaped += cfg.shape_pickup_useful_key * num_useful_items_now

        new_doors = set(next_state.doors_open) - set(state.doors_open)
        if new_doors:
            shaped += cfg.shape_open_new_door * len(new_doors) * max(1, len(next_state.doors_open))

        new_debris = set(next_state.debris_cleared) - set(state.debris_cleared)
        if new_debris:
            shaped += cfg.shape_clear_new_debris * len(new_debris)

        new_found = set(next_state.victims_found) - set(state.victims_found)
        if new_found:
            total_found_now = len(next_state.victims_found)
            shaped += cfg.shape_find_new_victim * total_found_now * len(new_found)

        new_rescued = set(next_state.victims_rescued) - set(state.victims_rescued)
        if new_rescued:
            total_rescued_now = len(next_state.victims_rescued)
            shaped += cfg.shape_rescue_each_victim * total_rescued_now * len(new_rescued)

        d_before = self._current_goal_distance(state)
        d_after = self._current_goal_distance(next_state)
        delta = d_before - d_after
        if delta > 0:
            shaped += cfg.shape_progress_scale * delta
        elif delta < 0:
            shaped += cfg.shape_no_progress_penalty

        if done and next_state.success(self.env.total_victims):
            shaped += cfg.shape_episode_success_bonus

        return shaped

    def _episode_quality_score(self, final_state: State) -> float:
        ep_success = float(final_state.success(self.env.total_victims))
        ep_rescued = float(len(final_state.victims_rescued))
        ep_found = float(len(final_state.victims_found))

        useful_item_count = sum(
            1 for item in final_state.inventory
            if item == "crowbar" or item == "medkit" or item.startswith("key_")
        )
        opened_door_count = len(final_state.doors_open)
        cleared_debris_count = len(final_state.debris_cleared)

        return (
            8.0 * ep_success +
            4.0 * ep_rescued +
            2.0 * ep_found +
            1.0 * useful_item_count +
            1.0 * opened_door_count +
            1.0 * cleared_debris_count
        )

    def compute_transition_priority(
        self,
        state: State,
        action: Action,
        next_state: State,
        done: bool,
        episode_score: float = 0.0,
        step_idx: int = 0,
        episode_len: int = 1,
    ) -> float:
        cfg = self.config
        p = cfg.prio_base

        new_items = set(next_state.inventory) - set(state.inventory)
        useful_keys = {f"key_{color}" for color in self.env.door_colors.values()}

        if "crowbar" in new_items:
            p += cfg.prio_crowbar
        if "medkit" in new_items:
            p += cfg.prio_medkit
        for item in new_items:
            if item in useful_keys:
                p += cfg.prio_useful_key

        new_doors = set(next_state.doors_open) - set(state.doors_open)
        if new_doors:
            p += cfg.prio_open_door * len(new_doors)

        new_debris = set(next_state.debris_cleared) - set(state.debris_cleared)
        if new_debris:
            p += cfg.prio_clear_debris * len(new_debris)

        new_found = set(next_state.victims_found) - set(state.victims_found)
        if new_found:
            p += cfg.prio_find_victim * len(new_found)

        new_rescued = set(next_state.victims_rescued) - set(state.victims_rescued)
        if new_rescued:
            p += cfg.prio_rescue_victim * len(new_rescued)

        if done and next_state.success(self.env.total_victims):
            p += cfg.prio_success_bonus

        p += cfg.prio_episode_score_scale * episode_score

        frac = (step_idx + 1) / max(1, episode_len)
        p += cfg.prio_position_bonus_scale * frac

        return float(max(1e-6, p))

    def _priority_to_tier(self, priority: float) -> int:
        cfg = self.config
        if priority >= cfg.tier3_priority_threshold:
            return 3
        if priority >= cfg.tier2_priority_threshold:
            return 2
        if priority >= cfg.tier1_priority_threshold:
            return 1
        return 0

    def legal_action_mask(self, state: State) -> np.ndarray:
        mask = np.zeros(self.num_actions, dtype=np.float32)
        legal_actions = self.env.actions(state)
        for a in legal_actions:
            mask[int(a)] = 1.0
        return mask

    def epsilon(self) -> float:
        frac = min(1.0, self.total_steps / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + frac * (self.config.epsilon_end - self.config.epsilon_start)

    @torch.no_grad()
    def select_action(self, state: State, greedy: bool = False) -> Action:
        legal_actions = self.env.actions(state)
        if not legal_actions:
            return Action.MOVE_UP

        eps = 0.0 if greedy else self.epsilon()
        if random.random() < eps:
            return random.choice(legal_actions)

        state_vec = self.encoder.encode(self.env, state)
        x = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.online_net(x).squeeze(0).cpu().numpy()

        mask = self.legal_action_mask(state)
        q_values = np.where(mask > 0, q_values, -1e9)

        return Action(int(np.argmax(q_values)))

    def build_transition(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        done: bool,
    ) -> Transition:
        s_vec = self.encoder.encode(self.env, state)
        ns_vec = self.encoder.encode(self.env, next_state)
        legal_mask = self.legal_action_mask(state)
        next_legal_mask = self.legal_action_mask(next_state)

        return Transition(
            s_vec,
            int(action),
            float(reward),
            ns_vec,
            float(done),
            legal_mask,
            next_legal_mask,
        )

    def train_step(self) -> float | None:
        cfg = self.config
        if len(self.buffer) < cfg.min_buffer_size:
            return None

        tier1_bs = min(int(cfg.batch_size * cfg.tier1_fraction), len(self.tier1_buffer))
        tier2_bs = min(int(cfg.batch_size * cfg.tier2_fraction), len(self.tier2_buffer))
        tier3_bs = min(int(cfg.batch_size * cfg.tier3_fraction), len(self.tier3_buffer))

        main_bs = cfg.batch_size - tier1_bs - tier2_bs - tier3_bs

        if len(self.buffer) < main_bs:
            return None

        parts = []
        if main_bs > 0:
            parts.append(self.buffer.sample(main_bs))
        if tier1_bs > 0:
            parts.append(self.tier1_buffer.sample(tier1_bs))
        if tier2_bs > 0:
            parts.append(self.tier2_buffer.sample(tier2_bs))
        if tier3_bs > 0:
            parts.append(self.tier3_buffer.sample(tier3_bs))

        if not parts:
            return None

        batch = parts[0]
        for part in parts[1:]:
            batch = Transition(
                state_vec=batch.state_vec + part.state_vec,
                action=batch.action + part.action,
                reward=batch.reward + part.reward,
                next_state_vec=batch.next_state_vec + part.next_state_vec,
                done=batch.done + part.done,
                legal_mask=batch.legal_mask + part.legal_mask,
                next_legal_mask=batch.next_legal_mask + part.next_legal_mask,
            )

        state_vecs = torch.tensor(np.array(batch.state_vec), dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state_vecs = torch.tensor(np.array(batch.next_state_vec), dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_legal_masks = torch.tensor(np.array(batch.next_legal_mask), dtype=torch.float32, device=self.device)

        q_values = self.online_net(state_vecs).gather(1, actions)

        with torch.no_grad():
            if cfg.double_dqn:
                next_q_online = self.online_net(next_state_vecs)
                next_q_online = next_q_online.masked_fill(next_legal_masks == 0, -1e9)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)
                next_q_target = self.target_net(next_state_vecs).gather(1, next_actions)
            else:
                next_q_target_all = self.target_net(next_state_vecs)
                next_q_target_all = next_q_target_all.masked_fill(next_legal_masks == 0, -1e9)
                next_q_target = next_q_target_all.max(dim=1, keepdim=True).values

            target = rewards + (1.0 - dones) * cfg.gamma * next_q_target

        loss = nn.SmoothL1Loss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), cfg.grad_clip_norm)
        self.optimizer.step()

        if self.total_steps % cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    def train(self, num_episodes: int = 1000, verbose_every: int = 20):
        episode_returns = []
        episode_lengths = []
        losses = []
        success_flags = []
        rescued_counts = []
        found_counts = []

        best_avg_rescued = -1.0
        best_avg_success = -1.0

        with self.log_path.open("w", encoding="utf-8") as f:
            f.write("=== DQN Training Log ===\n")

        for ep in range(1, num_episodes + 1):
            state = self.env.reset()
            ep_return = 0.0
            ep_loss_values = []
            episode_steps = []

            episode_position_counter = Counter()
            episode_position_counter[state.agent_pos] += 1
            self.position_counter[state.agent_pos] += 1

            for t in range(self.config.max_steps_per_episode):
                self.total_steps += 1

                action = self.select_action(state, greedy=False)
                next_state, reward, done = self.env.step(action)

                train_reward = self.shape_reward(state, action, reward, next_state, done)
                episode_steps.append({
                    "state": state,
                    "action": action,
                    "reward": train_reward,
                    "next_state": next_state,
                    "done": done,
                })

                if self.total_steps % self.config.train_freq == 0:
                    loss = self.train_step()
                    if loss is not None:
                        ep_loss_values.append(loss)

                ep_return += reward
                state = next_state

                episode_position_counter[state.agent_pos] += 1
                self.position_counter[state.agent_pos] += 1

                if done:
                    break

            self.last_episode_position_counter = episode_position_counter

            ep_success = state.success(self.env.total_victims)
            ep_rescued = len(state.victims_rescued)
            ep_found = len(state.victims_found)

            episode_score = self._episode_quality_score(state)
            episode_len = len(episode_steps)

            for idx, step in enumerate(episode_steps):
                prio = self.compute_transition_priority(
                    state=step["state"],
                    action=step["action"],
                    next_state=step["next_state"],
                    done=step["done"],
                    episode_score=episode_score,
                    step_idx=idx,
                    episode_len=episode_len,
                )
                tr = self.build_transition(
                    step["state"], step["action"], step["reward"], step["next_state"], step["done"]
                )

                # always into main buffer
                self.buffer.push(*tr)

                # optional tier buffer based on transition priority
                tier = self._priority_to_tier(prio)
                if tier == 1:
                    self.tier1_buffer.push(*tr)
                elif tier == 2:
                    self.tier2_buffer.push(*tr)
                elif tier == 3:
                    self.tier3_buffer.push(*tr)

            episode_returns.append(ep_return)
            episode_lengths.append(t + 1)
            success_flags.append(1 if ep_success else 0)
            rescued_counts.append(ep_rescued)
            found_counts.append(ep_found)

            if ep_loss_values:
                losses.append(sum(ep_loss_values) / len(ep_loss_values))

            if ep % verbose_every == 0:
                avg_ret = np.mean(episode_returns[-verbose_every:])
                avg_len = np.mean(episode_lengths[-verbose_every:])
                avg_loss = np.mean(losses[-verbose_every:]) if losses else float("nan")
                avg_success = np.mean(success_flags[-verbose_every:])
                avg_rescued = np.mean(rescued_counts[-verbose_every:])
                avg_found = np.mean(found_counts[-verbose_every:])

                found_positions = sorted(state.victims_found)
                rescued_positions = sorted(state.victims_rescued)

                self._log(
                    f"Episode {ep:4d} | "
                    f"avg_return={avg_ret:8.2f} | "
                    f"avg_len={avg_len:6.1f} | "
                    f"epsilon={self.epsilon():.3f} | "
                    f"avg_loss={avg_loss:.4f} | "
                    f"avg_found={avg_found:.2f} | "
                    f"avg_rescued={avg_rescued:.2f} | "
                    f"avg_success={avg_success:.2%} | "
                    f"main_buf={len(self.buffer)} | "
                    f"tier1_buf={len(self.tier1_buffer)} | "
                    f"tier2_buf={len(self.tier2_buffer)} | "
                    f"tier3_buf={len(self.tier3_buffer)} | "
                    f"last_o2={state.oxygen} | "
                    f"found_pos={found_positions} | "
                    f"rescued_pos={rescued_positions}"
                )

                improved = False
                if avg_rescued > best_avg_rescued:
                    improved = True
                elif avg_rescued == best_avg_rescued and avg_success > best_avg_success:
                    improved = True

                if improved:
                    best_avg_rescued = avg_rescued
                    best_avg_success = avg_success
                    self.save("dqn_sar_best.pt")
                    self._log(
                        f"  -> Saved new best checkpoint: "
                        f"avg_rescued={avg_rescued:.2f}, avg_success={avg_success:.2%}"
                    )

        self.print_final_visit_map()
        return episode_returns, episode_lengths, losses

    def save(self, path: str = "dqn_sar.pt") -> None:
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "feature_dim": self.encoder.feature_dim,
            "num_actions": self.num_actions,
            "total_steps": self.total_steps,
        }, path)

    def load(self, path: str = "dqn_sar.pt") -> None:
        import os
        try:
            ckpt = torch.load(path, map_location=self.device)
            self.online_net.load_state_dict(ckpt["online_net"])
            self.target_net.load_state_dict(ckpt["target_net"])
            self.total_steps = ckpt.get("total_steps", 0)
            print(f"Loaded checkpoint from {path}")
        except Exception as e:
            print(f"Failed to load checkpoint from {path}: {e}")
            print(f"Creating a new checkpoint at {path}")

            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            torch.save(
                {
                    "online_net": self.online_net.state_dict(),
                    "target_net": self.target_net.state_dict(),
                    "feature_dim": self.encoder.feature_dim,
                    "num_actions": self.num_actions,
                    "total_steps": 0,
                },
                path,
            )
            self.total_steps = 0

    @torch.no_grad()
    def evaluate(
        self,
        episodes: int = 5,
        render: bool = False,
        env: SearchRescueEnv | None = None,
        checkpoint_path: str = "dqn_sar_best.pt",
    ):
        if env is not None:
            self.env = env
            self.encoder = SARStateEncoder(self.env)

        if checkpoint_path is not None:
            self.load(checkpoint_path)

        def print_row(label, success, steps, reward, rescued, total_victims):
            row = (
                f"{label:<12} "
                f"{'YES' if success else 'NO':<9} "
                f"{steps:<7} "
                f"{reward:<9.1f} "
                f"{rescued}/{total_victims:<8}"
            )
            print(row)

        print("\n=== DQN Greedy Evaluation ===\n")
        header = f"{'Episode':<12} {'Success':<9} {'Steps':<7} {'Reward':<9} {'Rescued':<8}"
        print(header)
        print("-" * len(header))

        results = []

        for ep in range(episodes):
            state = self.env.reset()
            total_reward = 0.0
            actions_taken = []

            if render:
                print(f"\n=== Evaluation Episode {ep + 1} ===")
                print(self.env.render(state))

            for step in range(self.config.max_steps_per_episode):
                action = self.select_action(state, greedy=True)
                next_state, reward, done = self.env.step(action)

                total_reward += reward
                actions_taken.append(action)

                if render:
                    print(f"\nstep={step+1} action={Action(action).name} reward={reward:.1f}")
                    print(self.env.render(next_state))

                state = next_state
                if done:
                    break

            success = state.success(self.env.total_victims)
            rescued = len(state.victims_rescued)

            print_row(
                label=f"Episode {ep + 1}",
                success=success,
                steps=len(actions_taken),
                reward=total_reward,
                rescued=rescued,
                total_victims=self.env.total_victims,
            )

            results.append({
                "episode": ep + 1,
                "success": success,
                "steps": len(actions_taken),
                "total_reward": total_reward,
                "rescued": rescued,
                "total_victims": self.env.total_victims,
                "final_state": state,
                "actions": actions_taken,
            })

        avg_reward = np.mean([x["total_reward"] for x in results]) if results else 0.0
        success_rate = np.mean([1.0 if x["success"] else 0.0 for x in results]) if results else 0.0
        avg_rescued = np.mean([x["rescued"] for x in results]) if results else 0.0
        avg_steps = np.mean([x["steps"] for x in results]) if results else 0.0

        print("\n=== Evaluation Summary ===")
        print(f"avg_reward   : {avg_reward:.2f}")
        print(f"success_rate : {success_rate:.2%}")
        print(f"avg_rescued  : {avg_rescued:.2f}")
        print(f"avg_steps    : {avg_steps:.2f}")
        print(f"map_size     : {self.env.rows}x{self.env.cols}")

        return results


# ============================================================
# Main
# ============================================================

def main():
    set_seed(42)

    env = build_default_env(seed=42, rows=30, cols=40)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Map size: {env.rows}x{env.cols}")
    print(f"Victims: {env.total_victims}")


    config = DQNConfig(epsilon_decay_steps = 70_000, max_steps_per_episode = env.state.oxygen, shape_no_progress_penalty = -0.7, 
    main_fraction = 0.40,
    tier1_fraction = 0.10,
    tier2_fraction = 0.15,
    tier3_fraction = 0.35,
    tier3_buffer_capacity = 200_00)
    agent = DQNAgent(env=env, device=device, config=config)

    path = "dqn_sar_best30_40.pt"
    agent.load(path)
    agent.prefill_with_landmark(num_episodes=13, max_nodes=300_000)
    agent.train(num_episodes=400, verbose_every=5)
    agent.save(path)
    agent.evaluate(episodes=3, render=False, env=env, checkpoint_path=path)


if __name__ == "__main__":
    main()