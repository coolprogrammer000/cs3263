from __future__ import annotations

import random
from statistics import mean

from env import build_default_env
from qlearning import LinearQFunction
from qlearning_astar import (
    RescueFeatureExtractor,
    QHeuristicAStarPlanner,
)


def epsilon_greedy_action(env, qfunc, state, epsilon: float):
    """
    Choose an action using epsilon-greedy over legal actions only.
    """
    legal_actions = env.actions(state)
    if not legal_actions:
        return None

    if random.random() < epsilon:
        return random.choice(legal_actions)

    best_action = None
    best_q = float("-inf")
    for action in legal_actions:
        q = qfunc.get_q_value(state, action)
        if q > best_q:
            best_q = q
            best_action = action
    return best_action


def train_q_learning(
    env,
    qfunc,
    episodes: int = 500,
    max_steps_per_episode: int = 300,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    seed: int = 42,
):
    """
    Train the linear Q-function on the environment using Q-learning.
    """
    random.seed(seed)

    epsilon = epsilon_start
    episode_returns = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0

        for _ in range(max_steps_per_episode):
            legal_actions = env.actions(state)
            if not legal_actions:
                break

            action = epsilon_greedy_action(env, qfunc, state, epsilon)
            next_state, reward, done = env.step(action)
            total_reward += reward

            current_q = qfunc.get_q_value(state, action)

            if done:
                target = reward
            else:
                next_legal_actions = env.actions(next_state)
                if next_legal_actions:
                    max_next_q = max(
                        qfunc.get_q_value(next_state, next_action)
                        for next_action in next_legal_actions
                    )
                else:
                    max_next_q = 0.0
                target = reward + gamma * max_next_q

            delta = target - current_q
            qfunc.update(state, action, delta)

            state = next_state
            if done:
                break

        episode_returns.append(total_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if episode % 25 == 0:
            avg_return = mean(episode_returns[-25:])
            print(
                f"Episode {episode:4d} | "
                f"avg_return(last 25) = {avg_return:8.2f} | "
                f"epsilon = {epsilon:.3f}"
            )

    return episode_returns


def evaluate_greedy_policy(env, qfunc, episodes: int = 5, max_steps_per_episode: int = 300):
    """
    Evaluate the learned Q-function as a greedy policy (no exploration).
    """
    print("\n=== Greedy policy evaluation ===")
    returns = []

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0

        print(f"\nEpisode {ep}")
        print(env.render(state))

        for step in range(max_steps_per_episode):
            legal_actions = env.actions(state)
            if not legal_actions:
                print("No legal actions available.")
                break

            # greedy action
            action = max(legal_actions, key=lambda a: qfunc.get_q_value(state, a))
            next_state, reward, done = env.step(action)
            total_reward += reward

            print(
                f"Step {step+1:3d} | action={action.name:12s} | "
                f"reward={reward:6.1f} | total={total_reward:7.1f}"
            )
            print(env.render(next_state))

            state = next_state
            if done:
                print("Episode finished.")
                break

        returns.append(total_reward)

    avg_return = mean(returns) if returns else 0.0
    print(f"\nAverage greedy-policy return over {episodes} episodes: {avg_return:.2f}")
    return returns


def run_q_astar(env, qfunc, use_hybrid: bool = True, q_weight: float = 0.5, classical_weight: float = 1.0):
    """
    Run the Q-guided A* planner and print the result.
    """
    print("\n=== Q-guided A* planning ===")
    planner = QHeuristicAStarPlanner(
        env=env,
        qfunc=qfunc,
        q_weight=q_weight,
    )

    result = planner.plan()
    print(result)

    if result.success:
        print("Planned actions:")
        print([a.name for a in result.actions])

        state = env.get_initial_state()
        print("\nReplay planned trajectory:")
        print(env.render(state))

        for i, action in enumerate(result.actions, start=1):
            state, reward, done = env.transition(state, action)
            print(
                f"Plan step {i:3d} | action={action.name:12s} | "
                f"reward={reward:6.1f}"
            )
            print(env.render(state))
            if done:
                break
    else:
        print("Planner did not find a solution.")


def main():
    # Build environment
    env = build_default_env()

    # Create feature extractor and linear Q-function
    feature_extractor = RescueFeatureExtractor(env)
    qfunc = LinearQFunction(
        feature_extractor=feature_extractor,
        alpha=0.05,
        default_q_value=0.0,
    )

    try:
        qfunc.load_weights("qlearning_astar/q-learning-weights.json")
    except (FileNotFoundError, KeyError, ValueError):
        pass


    # Train
    returns = train_q_learning(
        env=env,
        qfunc=qfunc,
        episodes=400,
        max_steps_per_episode=300,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.99,
        seed=242,
    )

    print("\nTraining finished.")
    print(f"Final average return over last 20 episodes: {mean(returns[-20:]):.2f}")

    # Test learned greedy policy
    evaluate_greedy_policy(
        env=env,
        qfunc=qfunc,
        episodes=2,
        max_steps_per_episode=200,
    )

    # Test Q-guided A*
    run_q_astar(
        env=env,
        qfunc=qfunc,
        use_hybrid=True,
        q_weight=0.5,
        classical_weight=1.0,
    )

    qfunc.save_weights("qlearning_astar/q-learning-weights.json")


if __name__ == "__main__":
    main()