from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from env import Action, SearchRescueEnv, State
from qlearning import LinearQFunction


# ---------------------------------------------------------------------------
# Search node
# ---------------------------------------------------------------------------

@dataclass(order=True)
class Node:
    f: float
    g: float = field(compare=False)
    state: State = field(compare=False)
    parent: Optional["Node"] = field(compare=False, default=None)
    action: Optional[Action] = field(compare=False, default=None)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PlanResult:
    success: bool
    actions: List[Action]
    final_state: Optional[State]
    nodes_expanded: int
    nodes_generated: int
    elapsed_sec: float
    total_reward: float

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "FAILURE"
        return (
            f"[{status}] steps={len(self.actions)}  "
            f"expanded={self.nodes_expanded}  "
            f"generated={self.nodes_generated}  "
            f"reward={self.total_reward:.1f}  "
            f"time={self.elapsed_sec:.3f}s"
        )


# ---------------------------------------------------------------------------
# Classical heuristic
# ---------------------------------------------------------------------------

def _manhattan(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def heuristic_landmark(state: State, env: SearchRescueEnv) -> float:
    """
    TSP-style landmark heuristic (admissible).

    Computes a greedy lower-bound tour cost:
      current pos -> nearest unvisited victim -> ... -> exit
    using nearest-neighbour insertion on Manhattan distances.
    """
    pos = state.agent_pos
    unvisited = [v for v in env.victim_positions if v not in state.victims_found]
    exits = env.exit_positions

    if not exits:
        return 0.0
    if not unvisited:
        unrescued = state.victims_found - state.victims_rescued
        if not unrescued:
            return 0.0
        return float(min(_manhattan(pos, e) for e in exits))

    # Nearest-neighbour greedy tour
    remaining = list(unvisited)
    current   = pos
    total_dist = 0.0

    while remaining:
        nearest = min(remaining, key=lambda v: _manhattan(current, v))
        total_dist += _manhattan(current, nearest)
        current = nearest
        remaining.remove(nearest)

    # From last victim to nearest exit
    total_dist += min(_manhattan(current, e) for e in exits)
    return total_dist


# ---------------------------------------------------------------------------
# Q-learning feature extractor
# ---------------------------------------------------------------------------

class RescueFeatureExtractor:
    def __init__(self, env: SearchRescueEnv):
        self.env = env

    def num_features(self) -> int:
        return 11


    def extract(self, state: State, action: Action):
        next_state, reward, done = self.env.transition(state, action)

        total_victims = max(1, self.env.total_victims)

        found_before = len(state.victims_found)
        found_after = len(next_state.victims_found)

        rescued_before = len(state.victims_rescued)
        rescued_after = len(next_state.victims_rescued)

        features = [
            1.0,  # bias
            reward / 100.0,  # reward progress
            # oxygen progress
            state.oxygen / self.env.max_oxygen,
            next_state.oxygen / self.env.max_oxygen,

            # local event features
            1.0 if found_after > found_before else 0.0,
            1.0 if rescued_after > rescued_before else 0.0,

            # global progress features
            found_after / total_victims,
            rescued_after / total_victims,
            (total_victims - rescued_after) / total_victims,

            # terminal/success
            1.0 if done else 0.0,
            1.0 if done and next_state.success(self.env.total_victims) else 0.0,
        ]
        return features

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state_key(state: State):
    """
    Duplicate-detection key excluding oxygen, matching the earlier planner style.
    """
    return (
        state.agent_pos,
        state.inventory,
        state.doors_open,
        state.debris_cleared,
        state.victims_found,
        state.victims_rescued,
        state.medkits_taken,
    )


def q_value_to_cost_heuristic(state: State, env: SearchRescueEnv, qfunc: LinearQFunction) -> float:
    """
    Convert learned Q-values to a cost-style heuristic term.

    Since larger Q means more promising future reward, we flip the sign:
        h_q(s) = - max_a Q(s, a)

    We clamp at 0 to avoid strongly negative heuristics destabilizing search.
    """
    if state.success(env.total_victims):
        return 0.0

    legal_actions = env.actions(state)
    if not legal_actions:
        return 0.0 if state.is_terminal(env.total_victims) else float("inf")

    best_q = max(qfunc.get_q_value(state, action) for action in legal_actions)
    return max(0.0, -best_q)


def hybrid_heuristic(
    state: State,
    env: SearchRescueEnv,
    qfunc: LinearQFunction,
    q_weight: float = 0.5,
) -> float:
    """
    h(s) = h_manhattan(s) + q_weight * (-max_a Q(s,a))
    """
    h_m = heuristic_landmark(state, env)
    h_q = q_value_to_cost_heuristic(state, env, qfunc)
    return h_m * (1 - q_weight) + q_weight * h_q


# ---------------------------------------------------------------------------
# A* planner using hybrid heuristic
# ---------------------------------------------------------------------------

class QHeuristicAStarPlanner:
    def __init__(
        self,
        env: SearchRescueEnv,
        qfunc: LinearQFunction,
        max_nodes: int = 500_000,
        q_weight: float = 0.5,
    ):
        self.env = env
        self.qfunc = qfunc
        self.max_nodes = max_nodes
        self.q_weight = q_weight

    def heuristic(self, state: State) -> float:
        return hybrid_heuristic(
            state=state,
            env=self.env,
            qfunc=self.qfunc,
            q_weight=self.q_weight,
        )

    def plan(self, initial_state: Optional[State] = None) -> PlanResult:
        t0 = time.perf_counter()
        env = self.env
        start = initial_state if initial_state is not None else env.get_initial_state()

        h0 = self.heuristic(start)
        open_heap: List[Node] = []
        heapq.heappush(open_heap, Node(f=h0, g=0.0, state=start))

        best_g: Dict[tuple, float] = {_state_key(start): 0.0}
        nodes_expanded = 0
        nodes_generated = 1

        while open_heap:
            node = heapq.heappop(open_heap)

            # stale entry check
            if node.g > best_g.get(_state_key(node.state), math.inf):
                continue

            nodes_expanded += 1

            # success
            if node.state.success(env.total_victims):
                elapsed = time.perf_counter() - t0
                actions = self._extract_actions(node)
                reward = self._replay_reward(start, actions)
                return PlanResult(
                    success=True,
                    actions=actions,
                    final_state=node.state,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    elapsed_sec=elapsed,
                    total_reward=reward,
                )

            # failed terminal
            if node.state.is_terminal(env.total_victims):
                continue

            if nodes_expanded >= self.max_nodes:
                break

            for action in env.actions(node.state):
                next_state, reward, done = env.transition(node.state, action)

                # keep A* in cost space
                step_cost = max(0.0, -reward)
                new_g = node.g + step_cost

                key = _state_key(next_state)
                if new_g < best_g.get(key, math.inf):
                    best_g[key] = new_g
                    h = self.heuristic(next_state)

                    child = Node(
                        f=new_g + h,
                        g=new_g,
                        state=next_state,
                        parent=node,
                        action=action,
                    )
                    heapq.heappush(open_heap, child)
                    nodes_generated += 1

        elapsed = time.perf_counter() - t0
        return PlanResult(
            success=False,
            actions=[],
            final_state=None,
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            elapsed_sec=elapsed,
            total_reward=0.0,
        )

    @staticmethod
    def _extract_actions(node: Node) -> List[Action]:
        actions = []
        cur = node
        while cur.action is not None:
            actions.append(cur.action)
            cur = cur.parent
        actions.reverse()
        return actions

    def _replay_reward(self, start: State, actions: List[Action]) -> float:
        state = start
        total = 0.0
        for a in actions:
            state, r, done = self.env.transition(state, a)
            total += r
            if done:
                break
        return total