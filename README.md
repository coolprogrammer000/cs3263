# CS3263 – Search-and-Rescue With Tools

Multi-room search-and-rescue project comparing classical A* planning against a hybrid A* + Reinforcement Learning planner.

## Project overview

A grid-based MDP environment (MiniGrid-style) where an agent must locate 3 victims, navigate locked doors, clear debris with tools, avoid oxygen-depleting smoke, and evacuate everyone to the exit within an oxygen budget.

### Approach

| Component | Status |
|---|---|
| Grid environment + MDP | Done |
| A* planner + classical heuristics | Done |
| Improved classical heuristics (landmark, resource-aware) | Done (baseline) |
| RL value function + hybrid heuristic | Pending |
| Integration & experiments | Pending |

---

## Setup

```bash
conda env create -f environment.yml
conda activate cs3263-sar
python main.py
```

## Usage

```bash
# Run all heuristics and print comparison table
python main.py

# Run a single heuristic with step-by-step render
python main.py --heuristic landmark --render

# Change the random seed
python main.py --seed 7
```

Available heuristics: `zero`, `manhattan`, `resource_aware`, `landmark`

---

## File structure

```
env.py      – Grid environment and MDP (State, actions, transition, reward)
astar.py    – A* planner with pluggable heuristics
main.py     – Benchmark runner and CLI entry point
```

### MDP formulation (env.py)

| Element | Definition |
|---|---|
| **State** | `(agent_pos, oxygen, inventory, doors_open, debris_cleared, victims_found, victims_rescued, medkits_taken)` |
| **Actions** | `MOVE_*` (4 dirs), `PICKUP`, `OPEN_DOOR`, `CLEAR_DEBRIS`, `RESCUE` |
| **Transition** | Deterministic — P(s'|s,a) = 1 |
| **Reward** | +10 find victim, +20 rescue victim, +5 medkit, −1/step, −2/smoke step, −50 oxygen out, +100 all rescued |
| **Terminal** | All victims rescued **or** oxygen <= 0 |

### Map legend

```
@  agent          #  wall          .  empty floor
D  door (open)    L  locked door   X  debris (needs crowbar)
~  smoke          V  victim        +  medkit
k  key            T  crowbar       E  exit
```

---

## Current results (seed=42, 15x20 map, 3 victims, O2=300)

| Heuristic | Success | Steps | Nodes expanded | Reward | Time |
|---|---|---|---|---|---|
| zero (Dijkstra) | No | — | 200,000 | — | ~4.5 s |
| manhattan | No | — | 200,000 | — | ~5.3 s |
| resource_aware | No | — | 200,000 | — | ~5.2 s |
| **landmark** | **Yes** | **62** | **28,236** | **128.0** | **0.6 s** |

`zero` and `manhattan` fail within the node budget because they cannot guide the agent through multi-subgoal dependencies (find key -> unlock door -> clear debris -> reach victim). This motivates the RL hybrid.
