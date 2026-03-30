"""
Main entry point: runs A* with each heuristic on the default environment
and prints a comparison table.

Usage
-----
    python main.py
    python main.py --heuristic landmark
    python main.py --render
    python main.py --seed 7
"""

import argparse
import sys

from env import build_default_env
from astar import AStarPlanner, HEURISTICS


def run_comparison(seed: int, render: bool) -> None:
    print(f"\n=== Search-and-Rescue A* Benchmark  (seed={seed}) ===\n")

    env = build_default_env(seed=seed)
    print(env.render(env.get_initial_state()))
    print(f"\nVictims: {env.total_victims}  |  "
          f"Max O2: {env.max_oxygen}  |  "
          f"Grid: {env.rows}x{env.cols}\n")

    header = f"{'Heuristic':<18} {'Success':<9} {'Steps':<7} {'Expanded':<10} "
    header += f"{'Generated':<11} {'Reward':<9} {'Time(s)':<8}"
    print(header)
    print("-" * len(header))

    for name, h_fn in HEURISTICS.items():
        env2 = build_default_env(seed=seed)   # fresh env per run
        planner = AStarPlanner(env2, heuristic=h_fn, max_nodes=200_000)
        result  = planner.plan()

        row = (
            f"{name:<18} "
            f"{'YES' if result.success else 'NO':<9} "
            f"{len(result.actions):<7} "
            f"{result.nodes_expanded:<10} "
            f"{result.nodes_generated:<11} "
            f"{result.total_reward:<9.1f} "
            f"{result.elapsed_sec:<8.3f}"
        )
        print(row)

        if render and result.success:
            print(f"\n  --- Replay for '{name}' ---")
            state = env2.get_initial_state()
            for i, action in enumerate(result.actions):
                state, reward, done = env2.transition(state, action)
                if i % 10 == 0 or done:
                    print(f"\n  Step {i+1}  action={action.name}  r={reward:.1f}")
                    print(env2.render(state))
                if done:
                    break
            print()


def run_single(heuristic_name: str, seed: int, render: bool) -> None:
    if heuristic_name not in HEURISTICS:
        print(f"Unknown heuristic '{heuristic_name}'. "
              f"Choices: {list(HEURISTICS.keys())}")
        sys.exit(1)

    env = build_default_env(seed=seed)
    h_fn = HEURISTICS[heuristic_name]
    planner = AStarPlanner(env, heuristic=h_fn, max_nodes=200_000)

    print(f"\n=== A* with '{heuristic_name}' heuristic  (seed={seed}) ===")
    print(env.render(env.get_initial_state()))
    result = planner.plan()
    print(f"\n{result}")

    if render and result.success:
        state = env.get_initial_state()
        for i, action in enumerate(result.actions):
            state, reward, done = env.transition(state, action)
            print(f"Step {i+1:3d}  {action.name:<14} r={reward:+.1f}  O2={state.oxygen}")
            if i % 5 == 4:
                print(env.render(state))
            if done:
                break
        print("\nFinal state:")
        print(env.render(state))


def main() -> None:
    parser = argparse.ArgumentParser(description="Search-and-Rescue A* planner")
    parser.add_argument("--heuristic", default=None,
                        help="Run a single heuristic (zero/manhattan/resource_aware/landmark). "
                             "Omit to run all and compare.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for environment generation")
    parser.add_argument("--render", action="store_true",
                        help="Print grid at each step of the solution")
    args = parser.parse_args()

    if args.heuristic:
        run_single(args.heuristic, args.seed, args.render)
    else:
        run_comparison(args.seed, args.render)


if __name__ == "__main__":
    main()
