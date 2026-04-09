import argparse
import sys

from env import build_default_env
from astar import AStarPlanner, HEURISTICS
from qlearning import LinearQFunction
from qlearning_astar import RescueFeatureExtractor, QHeuristicAStarPlanner


def run_normal_astar(name: str, seed: int):
    env = build_default_env(seed=seed)
    planner = AStarPlanner(env, heuristic=HEURISTICS[name], max_nodes=200_000)
    result = planner.plan()
    return result


def run_qlearning_astar(seed: int, weights_path: str):
    env = build_default_env(seed=seed)

    feature_extractor = RescueFeatureExtractor(env)
    qfunc = LinearQFunction(feature_extractor=feature_extractor, alpha=0.05)
    qfunc.load_weights("qlearning_astar/q-learning-weights.json")

    planner = QHeuristicAStarPlanner(env=env, qfunc=qfunc, max_nodes=200_000)
    result = planner.plan()
    return result


def print_row(label, result):
    row = (
        f"{label:<20} "
        f"{'YES' if result.success else 'NO':<9} "
        f"{len(result.actions):<7} "
        f"{result.nodes_expanded:<10} "
        f"{result.nodes_generated:<11} "
        f"{result.total_reward:<9.1f} "
        f"{result.elapsed_sec:<8.3f}"
    )
    print(row)


def main():
    parser = argparse.ArgumentParser(description="Compare normal A* with Q-guided A*")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", type=str, default="linear_q_weights.json")
    args = parser.parse_args()

    print(f"\n=== Search-and-Rescue Planner Comparison (seed={args.seed}) ===\n")

    header = f"{'Planner':<20} {'Success':<9} {'Steps':<7} {'Expanded':<10} "
    header += f"{'Generated':<11} {'Reward':<9} {'Time(s)':<8}"
    print(header)
    print("-" * len(header))

    # normal A*
    manhattan_result = run_normal_astar("manhattan", args.seed)
    print_row("A* Manhattan", manhattan_result)

    landmark_result = run_normal_astar("landmark", args.seed)
    print_row("A* Landmark", landmark_result)

    # q-guided A*
    qlearning_result = run_qlearning_astar(args.seed, args.weights)
    print_row("Q-Learning A*", qlearning_result)


if __name__ == "__main__":
    main()