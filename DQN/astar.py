"""
A* Planner with pluggable heuristics for the Search-and-Rescue MDP.

The planner treats the MDP state as a search node and uses f(n) = g(n) + h(n)
where:
  g(n) = accumulated cost (oxygen consumed + step penalties)
  h(n) = heuristic estimate of remaining cost to goal

Pluggable heuristics (all admissible unless noted):
  1. zero            – Dijkstra baseline (h=0)
  2. manhattan       – min Manhattan distance to nearest unvisited victim + exit
  3. resource_aware  – Manhattan + oxygen-budget penalty term (slightly inadmissible,
                       useful for pruning oxygen-infeasible branches early)
  4. landmark        – TSP lower-bound over victim waypoints (admissible)

Usage
-----
    from env import build_default_env
    from astar import AStarPlanner, heuristic_manhattan

    env = build_default_env()
    planner = AStarPlanner(env, heuristic=heuristic_manhattan)
    result = planner.plan()
    print(result)
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from env import Action, SearchRescueEnv, State, Tile, DIRECTIONS


# ---------------------------------------------------------------------------
# Search node
# ---------------------------------------------------------------------------

@dataclass(order=True)
class Node:
    f:      float          # priority = g + h
    g:      float          = field(compare=False)
    state:  State          = field(compare=False)
    parent: Optional[Node] = field(compare=False, default=None)
    action: Optional[Action] = field(compare=False, default=None)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PlanResult:
    success:        bool
    actions:        List[Action]
    final_state:    Optional[State]
    nodes_expanded: int
    nodes_generated: int
    elapsed_sec:    float
    total_reward:   float

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
# Heuristic functions
# ---------------------------------------------------------------------------

def _manhattan(pos1: Tuple[int,int], pos2: Tuple[int,int]) -> int:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def heuristic_zero(state: State, env: SearchRescueEnv) -> float:
    """Trivial h=0 – degrades A* to Dijkstra's algorithm."""
    return 0.0


def heuristic_manhattan(state: State, env: SearchRescueEnv) -> float:
    """
    Admissible Manhattan heuristic.

    Lower-bound cost = distance to nearest unvisited victim
                     + distance from that victim to exit.
    When all victims found: distance to nearest exit.
    """
    pos = state.agent_pos
    unvisited = [v for v in env.victim_positions if v not in state.victims_found]
    exits = env.exit_positions

    if not exits:
        return 0.0

    if not unvisited:
        # All found – must reach exit to rescue
        unrescued = state.victims_found - state.victims_rescued
        if not unrescued:
            return 0.0
        return float(min(_manhattan(pos, e) for e in exits))

    # Min over victims: dist(pos -> victim) + dist(victim -> nearest exit)
    best = math.inf
    for v in unvisited:
        d = _manhattan(pos, v) + min(_manhattan(v, e) for e in exits)
        if d < best:
            best = d
    return float(best)


def heuristic_resource_aware(state: State, env: SearchRescueEnv) -> float:
    """
    Manhattan heuristic augmented with an oxygen-feasibility penalty.

    If the Manhattan lower-bound already exceeds remaining oxygen, add a
    large penalty to push these branches to the back of the queue.

    Note: slightly inadmissible in practice due to the penalty term, but
    produces faster search by pruning clearly infeasible paths.
    """
    h_base = heuristic_manhattan(state, env)
    o2 = state.oxygen
    if o2 <= 0:
        return float("inf")
    # Penalise when oxygen is tight relative to estimated cost
    if h_base > o2:
        return h_base + (h_base - o2) * 2.0
    return h_base


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


# Registry for easy lookup by name
HEURISTICS: Dict[str, Callable] = {
    "zero":           heuristic_zero,
    "manhattan":      heuristic_manhattan,
    "resource_aware": heuristic_resource_aware,
    "landmark":       heuristic_landmark,
}


# ---------------------------------------------------------------------------
# A* Planner
# ---------------------------------------------------------------------------

def _state_key(state: State):
    """
    Canonical key for A* duplicate-detection that excludes oxygen.

    Rationale: two paths reaching the same (pos, inventory, doors, debris,
    victims, medkits) differ only in cost (g).  The cheaper path always
    dominates — it has equal or better oxygen AND lower g.  Including
    oxygen in the key would create O(max_oxygen) copies of every logical
    state, exploding the search space.
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


class AStarPlanner:
    """
    A* planner that operates directly on MDP states.

    Parameters
    ----------
    env         : SearchRescueEnv
    heuristic   : callable(state, env) -> float
    max_nodes   : hard cap on nodes expanded (prevents runaway search)
    """

    def __init__(
        self,
        env: SearchRescueEnv,
        heuristic: Callable = heuristic_manhattan,
        max_nodes: int = 500_000,
    ):
        self.env       = env
        self.heuristic = heuristic
        self.max_nodes = max_nodes

    def plan(
        self,
        initial_state: Optional[State] = None,
    ) -> PlanResult:
        """
        Run A* from *initial_state* (defaults to env.get_initial_state()).

        Returns a PlanResult with the action sequence, stats, and outcome.
        """
        t0 = time.perf_counter()
        env = self.env

        start = initial_state if initial_state is not None else env.get_initial_state()

        h0 = self.heuristic(start, env)
        open_heap: List[Node] = []
        heapq.heappush(open_heap, Node(f=h0, g=0.0, state=start))

        # best g-value seen for each *logical* state (oxygen excluded from key)
        best_g: Dict[tuple, float] = {_state_key(start): 0.0}

        nodes_expanded  = 0
        nodes_generated = 1

        while open_heap:
            node = heapq.heappop(open_heap)

            # Skip stale entries
            if node.g > best_g.get(_state_key(node.state), math.inf):
                continue

            nodes_expanded += 1

            # Goal check
            if node.state.success(env.total_victims):
                elapsed = time.perf_counter() - t0
                actions = self._extract_actions(node)
                reward  = self._replay_reward(start, actions)
                return PlanResult(
                    success=True,
                    actions=actions,
                    final_state=node.state,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    elapsed_sec=elapsed,
                    total_reward=reward,
                )

            # Terminal-but-failed check
            if node.state.is_terminal(env.total_victims):
                continue

            # Budget cap
            if nodes_expanded >= self.max_nodes:
                break

            # Expand
            for action in env.actions(node.state):
                next_state, reward, done = env.transition(node.state, action)

                # A* cost = -reward (we minimise cost = maximise reward)
                step_cost = max(0.0, -reward)
                new_g = node.g + step_cost

                key = _state_key(next_state)
                if new_g < best_g.get(key, math.inf):
                    best_g[key] = new_g
                    h = self.heuristic(next_state, env)
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
