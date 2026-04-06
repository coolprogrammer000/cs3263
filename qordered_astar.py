from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from env import Action, SearchRescueEnv, State, Tile, DIRECTIONS
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


def heuristic_manhattan(state: State, env: SearchRescueEnv) -> float:
    """
    Lower-bound cost:
      - if there are unvisited victims:
            min distance(current -> victim -> exit)
      - if all victims found but not all rescued:
            distance(current -> nearest exit)
      - if all rescued:
            0
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

    best = math.inf
    for v in unvisited:
        d = _manhattan(pos, v) + min(_manhattan(v, e) for e in exits)
        if d < best:
            best = d
    return float(best)


# ---------------------------------------------------------------------------
# Feature extractor for linear Q-learning
# ---------------------------------------------------------------------------

class RescueFeatureExtractor:
    def __init__(self, env: SearchRescueEnv):
        self.env = env

    def num_features(self) -> int:
        return 10


    def extract(self, state: State, action: Action):
        next_state, reward, done = self.env.transition(state, action)

        total_victims = max(1, self.env.total_victims)

        found_before = len(state.victims_found)
        found_after = len(next_state.victims_found)

        rescued_before = len(state.victims_rescued)
        rescued_after = len(next_state.victims_rescued)

        features = [
            1.0,  # bias

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
    Duplicate-detection key excluding oxygen, same logic as before.
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


def compute_step_cost(
    env: SearchRescueEnv,
    state: State,
    action: Action,
    next_state: State,
) -> float:
    """
    Classical planner cost, independent of RL reward.

    Suggested design:
      - every action costs 1
      - moving onto smoke costs extra
      - failed terminal states are extremely bad and can be pruned naturally
    """
    cost = 1.0

    if action in DIRECTIONS:
        r, c = next_state.agent_pos
        tile = env._effective_tile(r, c, next_state)
        if tile == Tile.SMOKE:
            cost += 2.0

    return cost


def order_actions_by_q(
    env: SearchRescueEnv,
    qfunc: LinearQFunction,
    state: State,
    legal_actions: List[Action],
) -> List[Action]:
    """
    Sort actions by descending Q-value.
    Higher Q means more promising, so expand those children first.
    """
    return sorted(
        legal_actions,
        key=lambda a: qfunc.get_q_value(state, a),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# A* planner with Q-guided expansion order
# ---------------------------------------------------------------------------

class QOrderedAStarPlanner:
    """
    Standard A* with a classical heuristic, but legal actions are ordered
    by descending Q-value before successors are generated.

    So:
      - g stays planner cost
      - h stays classical heuristic
      - Q only influences expansion order
    """

    def __init__(
        self,
        env: SearchRescueEnv,
        qfunc: LinearQFunction,
        max_nodes: int = 500_000,
    ):
        self.env = env
        self.qfunc = qfunc
        self.max_nodes = max_nodes

    def heuristic(self, state: State) -> float:
        return heuristic_manhattan(state, self.env)

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

            # stale entry
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

            legal_actions = env.actions(node.state)
            ordered_actions = order_actions_by_q(
                env=env,
                qfunc=self.qfunc,
                state=node.state,
                legal_actions=legal_actions,
            )

            for action in ordered_actions:
                next_state, reward, done = env.transition(node.state, action)

                step_cost = compute_step_cost(env, node.state, action, next_state)
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