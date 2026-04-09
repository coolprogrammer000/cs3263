"""
mdp_solvers.py
Model-based MDP solvers for the Search-and-Rescue environment.


    get_action_value(s, a, V, gamma, env_transition)
    get_max_action_value(s, env_nA, env_transition, V, gamma)
    get_policy(env_nS, env_nA, env_transition, gamma, V)
    policy_evaluation(env_nS, env_transition, V, gamma, theta, policy)
    policy_improvement(env_nS, env_nA, env_transition, policy, V, gamma)
    value_iteration(gamma, theta, env_nS, env_nA, env_transition)
    policy_iteration(gamma, theta, env_nS, env_nA, env_transition)


"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from env import Action, SearchRescueEnv, State

def _struct_key(state: State) -> tuple:
    return (
        state.agent_pos,
        state.inventory,
        state.doors_open,
        state.debris_cleared,
        state.victims_found,
        state.victims_rescued,
        state.medkits_taken,
    )



@dataclass
class PlanResult:
    success:         bool
    actions:         List[Action]
    nodes_expanded:  int    # Bellman backups
    nodes_generated: int    # reachable states = nS
    elapsed_sec:     float
    total_reward:    float

    def __str__(self) -> str:
        s = "SUCCESS" if self.success else "FAILURE"
        return (
            f"[{s}] steps={len(self.actions)}  "
            f"backups={self.nodes_expanded}  "
            f"states={self.nodes_generated}  "
            f"reward={self.total_reward:.1f}  "
            f"time={self.elapsed_sec:.3f}s"
        )



class SARMDPWrapper:
    """
    Exposes SearchRescueEnv as an integer-indexed tabular MDP.

    Attributes
    nS, nA          : state and action counts
    initial_sid      : integer index of the start state
    truncated        : True if MAX_STATES was hit
    transition(s, a) : [(prob=1.0, next_s, reward, done)]  — deterministic
    _np              : NumpyMDP  (numpy arrays for vectorized VI/PI)
    """

    MAX_STATES = 500_000

    def __init__(self, env: SearchRescueEnv) -> None:
        self.env  = env
        self.nA   = len(Action)

        self._sid_to_sk: List[tuple]      = []
        self._sk_to_sid: Dict[tuple, int] = {}
        # _raw[s][a] = (next_s, reward, done) | None
        self._raw: List[List[Optional[Tuple[int, float, bool]]]] = []

        self.initial_sid = -1
        self.truncated   = False
        self.nS          = 0

        self._bfs_build()
        self._np = _NumpyMDP(self)   # pre-built numpy arrays

    def _bfs_build(self) -> None:
        env   = self.env
        start = env.get_initial_state()
        sk0   = _struct_key(start)

        self._sk_to_sid[sk0] = 0
        self._sid_to_sk.append(sk0)
        self.initial_sid = 0

        _sk_state: Dict[tuple, State] = {sk0: start}
        queue: deque[int] = deque([0])

        while queue:
            if len(self._sid_to_sk) >= self.MAX_STATES:
                self.truncated = True
                break
            sid   = queue.popleft()
            sk    = self._sid_to_sk[sid]
            state = _sk_state[sk]
            row   = [None] * self.nA

            if not state.is_terminal(env.total_victims):
                legal = set(env.actions(state))
                for action in Action:
                    if action not in legal:
                        continue
                    ns, reward, done = env.transition(state, action)
                    nsk = _struct_key(ns)
                    if nsk not in self._sk_to_sid:
                        new_sid = len(self._sid_to_sk)
                        self._sk_to_sid[nsk] = new_sid
                        self._sid_to_sk.append(nsk)
                        _sk_state[nsk] = ns
                        queue.append(new_sid)
                    row[int(action)] = (self._sk_to_sid[nsk], reward, done)

            self._raw.append(row)

        # Fill rows for states queued but not yet expanded
        while len(self._raw) < len(self._sid_to_sk):
            self._raw.append([None] * self.nA)

        self.nS = len(self._sid_to_sk)

    def transition(self, s: int, a: int) -> List[Tuple[float, int, float, bool]]:
        """
        env_transition(s, a) → [(prob, next_state, reward, done)]

        Deterministic MDP: always one tuple with prob = 1.0.
        Illegal actions return a self-loop with reward = -1.
        """
        entry = self._raw[s][a]
        if entry is None:
            return [(1.0, s, -1.0, False)]
        ns, reward, done = entry
        return [(1.0, ns, reward, done)]

    def extract_plan(
        self, policy: np.ndarray, max_steps: int = 2000
    ) -> Tuple[List[Action], float, bool]:
        """
        Follow policy (int array indexed by state id) from initial_sid.
        Returns (action_list, total_reward, success).
        """
        sid          = self.initial_sid
        seen         = set()
        action_list  = []
        total_reward = 0.0

        for _ in range(max_steps):
            if all(e is None for e in self._raw[sid]):
                break
            if sid in seen:
                break
            seen.add(sid)
            a = int(policy[sid])
            entry = self._raw[sid][a]
            if entry is None:
                break
            ns, reward, done = entry
            action_list.append(Action(a))
            total_reward += reward
            sid = ns
            if done:
                break

        # Success: check victims_rescued count in final structural key
        final_sk = self._sid_to_sk[sid]
        success  = len(final_sk[5]) >= self.env.total_victims
        return action_list, total_reward, success



class _NumpyMDP:
    """
    Pre-computes three arrays of shape (nS, nA):

        next_s   int32    successor state index (self-loop if illegal)
        reward   float32  immediate reward
        done     bool     episode-terminal flag

    """

    def __init__(self, wrapper: SARMDPWrapper) -> None:
        nS, nA = wrapper.nS, wrapper.nA
        self.next_s  = np.arange(nS, dtype=np.int32)[:, None].repeat(nA, axis=1)
        self.reward  = np.full((nS, nA), -1.0, dtype=np.float32)
        self.done    = np.zeros((nS, nA), dtype=bool)

        for s, row in enumerate(wrapper._raw):
            for a, entry in enumerate(row):
                if entry is not None:
                    ns, r, d = entry
                    self.next_s[s, a] = ns
                    self.reward[s, a] = r
                    self.done[s, a]   = d



class MDPSolver:

    def __init__(self, verbose: bool = False) -> None:
        self.iteration = 0
        if verbose:
            print("MDPSolver initialized!")

    def get_action_value(
        self,
        s: int,
        a: int,
        V: np.ndarray,
        gamma: float,
        env_transition: Callable,
    ) -> float:
        """
        Q(s, a) = Σ_{s'} p(s'|s,a) · [r + γ · V(s')]
        """
        value = 0.0
        for prob, next_state, reward, done in env_transition(s, a):
            future = 0.0 if done else gamma * V[next_state]
            value += prob * (reward + future)
        return value

    def get_max_action_value(
        self,
        s: int,
        env_nA: int,
        env_transition: Callable,
        V: np.ndarray,
        gamma: float,
    ) -> Tuple[float, int]:
        """
        a* = argmax_a Q(s, a).  Returns (max_value, best_action).
        """
        max_value  = -np.inf
        max_action = 0
        for a in range(env_nA):
            q = self.get_action_value(s, a, V, gamma, env_transition)
            if q > max_value:
                max_value  = q
                max_action = a
        return max_value, max_action

    def get_policy(
        self,
        env_nS: int,
        env_nA: int,
        env_transition: Callable,
        gamma: float,
        V: np.ndarray,
    ) -> np.ndarray:
        """
        π(s) = argmax_a Q(s, a)  for all s.
        """
        policy = np.zeros(env_nS, dtype=int)
        for s in range(env_nS):
            _, best = self.get_max_action_value(s, env_nA, env_transition, V, gamma)
            policy[s] = best
        return policy

    def policy_evaluation(
        self,
        env_nS: int,
        env_transition: Callable,
        V: np.ndarray,
        gamma: float,
        theta: float,
        policy: np.ndarray,
    ) -> np.ndarray:
        """
        Iterative policy evaluation (Gauss-Seidel, in-place).
        Runs until max_s |ΔV(s)| < theta.
        """
        while True:
            delta = 0.0
            for s in range(env_nS):
                v_old = V[s]
                V[s]  = self.get_action_value(s, int(policy[s]), V, gamma, env_transition)
                delta = max(delta, abs(v_old - V[s]))
            if delta < theta:
                break
        return V

    def policy_improvement(
        self,
        env_nS: int,
        env_nA: int,
        env_transition: Callable,
        policy: np.ndarray,
        V: np.ndarray,
        gamma: float,
    ) -> Tuple[bool, np.ndarray]:
        """
        One pass of greedy policy improvement.
        Returns (policy_stable, updated_policy).
        """
        policy_stable = True
        for s in range(env_nS):
            old  = int(policy[s])
            _, best = self.get_max_action_value(s, env_nA, env_transition, V, gamma)
            policy[s] = best
            if old != best:
                policy_stable = False
        return policy_stable, policy


    def value_iteration(
        self,
        gamma: float,
        theta: float,
        env_nS: int,
        env_nA: int,
        env_transition: Callable,
        _np: Optional[_NumpyMDP] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Value Iteration:

            V(s) ← max_a Σ_{s'} p(s'|s,a)·[r + γ·V(s')]

        """
        V = np.zeros(env_nS, dtype=np.float64)

        if _np is not None:
            next_s  = _np.next_s          # (nS, nA) int32
            reward  = _np.reward          # (nS, nA) float32
            done    = _np.done            # (nS, nA) bool

            while True:
                future = np.where(done, 0.0, gamma * V[next_s])
                Q      = reward + future       # (nS, nA)
                V_new  = Q.max(axis=1)
                delta  = float(np.abs(V_new - V).max())
                V      = V_new
                if delta < theta:
                    break

            policy = Q.argmax(axis=1).astype(int)

        else:
            while True:
                delta = 0.0
                for s in range(env_nS):
                    v_old     = V[s]
                    V[s], _   = self.get_max_action_value(
                        s, env_nA, env_transition, V, gamma
                    )
                    delta = max(delta, abs(v_old - V[s]))
                if delta < theta:
                    break

            policy = self.get_policy(env_nS, env_nA, env_transition, gamma, V)

        return policy, V

    def policy_iteration(
        self,
        gamma: float,
        theta: float,
        env_nS: int,
        env_nA: int,
        env_transition: Callable,
        _np: Optional[_NumpyMDP] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Policy Iteration 
        """
        # Cold start 
        policy = np.zeros(env_nS, dtype=int)
        V      = np.zeros(env_nS, dtype=np.float64)

        if _np is not None:
            # Vectorised path
            next_s = _np.next_s   # (nS, nA) int32
            reward = _np.reward   # (nS, nA) float32
            done   = _np.done    
            idx    = np.arange(env_nS)

            while True:
                # policy iteration
                # Iterate V^pi(s) = r(s,pi(s)) + gamma*V^pi(s') until converged
                while True:
                    r_pi  = reward[idx, policy]           # r(s, pi(s))
                    ns_pi = next_s[idx, policy]           # next state s'
                    d_pi  = done[idx, policy]             # terminal flag
                    V_new = r_pi + np.where(d_pi, 0.0, gamma * V[ns_pi])
                    delta = float(np.abs(V_new - V).max())
                    V = V_new
                    if delta < theta:
                        break
                future     = np.where(done, 0.0, gamma * V[next_s])
                Q          = reward + future              # (nS, nA)
                new_policy = Q.argmax(axis=1).astype(int)

                #Stability check
                if np.array_equal(new_policy, policy):
                    policy = new_policy
                    break
                policy = new_policy

        else:
            while True:
                # policy Evaluation
                V = self.policy_evaluation(
                    env_nS, env_transition, V, gamma, theta, policy
                )
                stable, policy = self.policy_improvement(
                    env_nS, env_nA, env_transition, policy, V, gamma
                )
                if stable:
                    break

        return policy, V


def _run_solver(env, method, gamma=0.99, theta=0.1):
    t0      = time.perf_counter()
    wrapper = SARMDPWrapper(env)

    if wrapper.truncated:
        return PlanResult(False, [], 0, wrapper.nS,
                          time.perf_counter() - t0, 0.0)

    solver = MDPSolver()
    np_mdp = wrapper._np      # pre-built numpy arrays

    if method == "vi":
        policy, V = solver.value_iteration(
            gamma=gamma, theta=theta,
            env_nS=wrapper.nS, env_nA=wrapper.nA,
            env_transition=wrapper.transition,
            _np=np_mdp,
        )
    else:
        policy, V = solver.policy_iteration(
            gamma=gamma, theta=theta,
            env_nS=wrapper.nS, env_nA=wrapper.nA,
            env_transition=wrapper.transition,
            _np=np_mdp,
        )

    actions, total_reward, success = wrapper.extract_plan(policy)

    return PlanResult(
        success=success,
        actions=actions,
        nodes_expanded=wrapper.nS, 
        nodes_generated=wrapper.nS,
        elapsed_sec=time.perf_counter() - t0,
        total_reward=total_reward,
    )


class ValueIterationSolver:
    def __init__(self, env, gamma=0.99, theta=0.1, **_):
        self.env = env; self.gamma = gamma; self.theta = theta

    def plan(self, initial_state=None):
        return _run_solver(self.env, "vi", self.gamma, self.theta)


class PolicyIterationSolver:
    def __init__(self, env, gamma=0.99, vi_theta=0.1, eval_theta=0.1, **_):
        self.env = env; self.gamma = gamma
        self.theta = min(vi_theta, eval_theta)

    def plan(self, initial_state=None):
        return _run_solver(self.env, "pi", self.gamma, self.theta)