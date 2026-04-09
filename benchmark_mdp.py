"""
benchmark_mdp.py
Unified benchmark: Value Iteration, Policy Iteration, classical A*,
and Q-learning A* on the Search-and-Rescue MDP.

Usage
-----
    python benchmark_mdp.py                   # seed=42, 9-room A*, 4-room VI/PI
    python benchmark_mdp.py --seed 7          # different map
    python benchmark_mdp.py --vi-rooms 4      # override VI/PI map size
    python benchmark_mdp.py --no-vi --no-pi   # A*/Q only
    python benchmark_mdp.py --no-astar --no-q # model-based only
    python benchmark_mdp.py --all-rooms 4     # same map for everything
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import build_default_env
from mdp_solvers import SARMDPWrapper, ValueIterationSolver, PolicyIterationSolver

_Q_WEIGHT_CANDIDATES = [
    "q-learning-weights.json",
    "qlearning_astar/q-learning-weights.json",
    "q_learning_weights.json",
]
_Q_ORDER_CANDIDATES = [
    "q-order-weights.json",
    "qlearning_astar/q-order-weights.json",
    "q_order_weights.json",
]

def _find_weights(candidates: list, override: Optional[str]) -> Optional[str]:
    if override is not None:
        return override
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None



try:
    from astar import AStarPlanner, HEURISTICS
    HAS_ASTAR = True
except ImportError:
    HAS_ASTAR = False

try:
    from qlearning import LinearQFunction
    from qlearning_astar import RescueFeatureExtractor, QHeuristicAStarPlanner
    HAS_Q_ASTAR = True
except ImportError:
    HAS_Q_ASTAR = False

try:
    from qlearning_astar import RescueFeatureExtractor as RFE2
    from qordered_astar import QOrderedAStarPlanner
    HAS_Q_ORDERED = True
except ImportError:
    HAS_Q_ORDERED = False



COLS = (
    ("Solver",    30),
    ("Success",    9),
    ("Steps",      7),
    ("Expanded",  12),
    ("Reachable", 12),
    ("Reward",    10),
    ("Time(s)",    9),
)

def _header() -> str:
    h = "".join(f"{name:<{w}}" for name, w in COLS)
    return h + "\n" + "-" * len(h)

def _row(label: str, result) -> str:
    return (
        f"{label:<30}"
        f"{'YES' if result.success else 'NO':<9}"
        f"{len(result.actions):<7}"
        f"{result.nodes_expanded:<12}"
        f"{result.nodes_generated:<12}"
        f"{result.total_reward:<10.1f}"
        f"{result.elapsed_sec:<9.3f}"
    )

def _skip_row(label: str, reason: str) -> str:
    return f"{label:<30}{'--':<9}{'--':<7}{'--':<12}{'--':<12}{'--':<10}{reason}"


# ── VI / PI map-size probe ────────────────────────────────────────────────────

# Maximum states we'll enumerate for VI/PI (~20 s build on a laptop)
_VI_MAX_STATES = 400_000

def _state_count(seed: int, rooms: int):
    """Return (nS, truncated) for the given map, capped at _VI_MAX_STATES."""
    env = build_default_env(seed=seed, num_rooms=rooms)
    old_max = SARMDPWrapper.MAX_STATES
    SARMDPWrapper.MAX_STATES = _VI_MAX_STATES
    wrapper = SARMDPWrapper(env)
    SARMDPWrapper.MAX_STATES = old_max
    return wrapper.nS, wrapper.truncated

def _find_vi_rooms(seed: int, requested_rooms: int) -> int:
    """Largest rooms ≤ min(requested, 4) whose state space fits."""
    for rooms in range(min(requested_rooms, 4), 0, -1):
        n, trunc = _state_count(seed, rooms)
        if not trunc:
            return rooms
    return 0



def _run_vi(seed: int, rooms: int, gamma: float, theta: float):
    env    = build_default_env(seed=seed, num_rooms=rooms)
    solver = ValueIterationSolver(env, gamma=gamma, theta=theta)
    return f"Value Iteration ({rooms}-room)", solver.plan()

def _run_pi(seed: int, rooms: int, gamma: float, theta: float):
    env    = build_default_env(seed=seed, num_rooms=rooms)
    solver = PolicyIterationSolver(env, gamma=gamma, vi_theta=theta, eval_theta=theta)
    return f"Policy Iteration ({rooms}-room)", solver.plan()



def main() -> None:
    parser = argparse.ArgumentParser(description="SAR MDP benchmark")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--num-rooms",  type=int,   default=9,
                        help="Rooms for A*/Q planners (default 9)")
    parser.add_argument("--vi-rooms",   type=int,   default=None,
                        help="Rooms for VI/PI (auto-probed if omitted)")
    parser.add_argument("--all-rooms",  type=int,   default=None,
                        help="Override room count for ALL solvers")
    parser.add_argument("--gamma",      type=float, default=0.99)
    parser.add_argument("--theta",      type=float, default=0.1,
                        help="Convergence threshold for VI and PI (default 0.1)")
    parser.add_argument("--no-vi",      action="store_true")
    parser.add_argument("--no-pi",      action="store_true")
    parser.add_argument("--no-astar",   action="store_true")
    parser.add_argument("--no-q",       action="store_true")
    parser.add_argument("--q-weights",         type=str, default=None)
    parser.add_argument("--q-ordered-weights", type=str, default=None)
    args = parser.parse_args()

    if args.all_rooms is not None:
        args.num_rooms = args.all_rooms
        args.vi_rooms  = args.all_rooms

    astar_rooms = args.num_rooms

    print(f"\n=== Search-and-Rescue MDP Benchmark  (seed={args.seed}) ===\n")
    print(f"  A*/Q planners : {astar_rooms} rooms")

    # Probe VI/PI map size
    vi_rooms = args.vi_rooms
    need_probe = vi_rooms is None and (not args.no_vi or not args.no_pi)
    if need_probe:
        print(f"  Probing VI/PI map size (cap {_VI_MAX_STATES:,} states)…",
              end=" ", flush=True)
        vi_rooms = _find_vi_rooms(args.seed, astar_rooms)
        print(f"using {vi_rooms} rooms" if vi_rooms else "too large — VI/PI skipped")
    elif vi_rooms is not None:
        print(f"  VI/PI planners: {vi_rooms} rooms (specified)")

    print()
    print(_header())

    # ── Value Iteration ───────────────────────────────────────────────
    if not args.no_vi:
        if vi_rooms:
            label, result = _run_vi(args.seed, vi_rooms, args.gamma, args.theta)
            print(_row(label, result))
        else:
            print(_skip_row("Value Iteration", "state space too large"))

    # ── Policy Iteration ──────────────────────────────────────────────
    if not args.no_pi:
        if vi_rooms:
            label, result = _run_pi(args.seed, vi_rooms, args.gamma, args.theta)
            print(_row(label, result))
        else:
            print(_skip_row("Policy Iteration", "state space too large"))

    # ── Classical A* ──────────────────────────────────────────────────
    if not args.no_astar:
        if HAS_ASTAR:
            for name in ("manhattan", "resource_aware", "landmark"):
                if name not in HEURISTICS:
                    continue
                env = build_default_env(seed=args.seed, num_rooms=astar_rooms)
                planner = AStarPlanner(env, heuristic=HEURISTICS[name], max_nodes=200_000)
                print(_row(f"A* {name}", planner.plan()))
        else:
            print(_skip_row("A* (all)", "astar.py not found"))

    # ── Q-heuristic A* ────────────────────────────────────────────────
    if not args.no_q and HAS_Q_ASTAR:
        env   = build_default_env(seed=args.seed, num_rooms=astar_rooms)
        fe    = RescueFeatureExtractor(env)
        qfunc = LinearQFunction(feature_extractor=fe, alpha=0.05)
        path  = _find_weights(_Q_WEIGHT_CANDIDATES, args.q_weights)
        if path is None:
            print(_skip_row("Q-Heuristic A*", "weights not found (--q-weights PATH)"))
        else:
            try:
                qfunc.load_weights(path)
                result = QHeuristicAStarPlanner(env=env, qfunc=qfunc,
                                                max_nodes=200_000).plan()
                print(_row("Q-Heuristic A*", result))
            except (FileNotFoundError, ValueError) as e:
                print(_skip_row("Q-Heuristic A*", str(e)))

    if not args.no_q and HAS_Q_ORDERED:
        env    = build_default_env(seed=args.seed, num_rooms=astar_rooms)
        fe2    = RFE2(env)
        qfunc2 = LinearQFunction(feature_extractor=fe2, alpha=0.05)
        path2  = _find_weights(_Q_ORDER_CANDIDATES, args.q_ordered_weights)
        if path2 is None:
            print(_skip_row("Q-Ordered A*", "weights not found (--q-ordered-weights PATH)"))
        else:
            try:
                qfunc2.load_weights(path2)
                result = QOrderedAStarPlanner(env=env, qfunc=qfunc2,
                                              max_nodes=200_000).plan()
                print(_row("Q-Ordered A*", result))
            except (FileNotFoundError, ValueError) as e:
                print(_skip_row("Q-Ordered A*", str(e)))

    print()
    _print_notes(astar_rooms, vi_rooms)


def _print_notes(astar_rooms: int, vi_rooms) -> None:
    print("Notes:")
    if vi_rooms and vi_rooms < astar_rooms:
        print(f"  * VI/PI ran on a {vi_rooms}-room map: the {astar_rooms}-room structural")
        print(f"    state space exceeds {_VI_MAX_STATES:,} states.  VI/PI are exact but")
        print(f"    do not scale to large MDPs — motivating A* and RL-hybrid planners.")
    print(f"  * 'Expanded'  = Bellman backups (VI/PI) or nodes popped (A*).")
    print(f"  * 'Reachable' = structural states enumerated (VI/PI) or generated (A*).")
    print()


if __name__ == "__main__":
    main()