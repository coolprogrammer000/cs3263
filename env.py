"""
Search-and-Rescue Grid Environment with MDP formulation.

Tile legend (grid values):
  0  = empty floor
  1  = wall
  2  = door (unlocked)
  3  = door (locked) – color stored separately in door_colors
  4  = debris (requires crowbar in inventory to clear)
  5  = smoke (traversable but costs extra oxygen)
  6  = victim
  7  = medkit
  8  = key    – color stored in item_colors
  9  = crowbar (tool)
  10 = exit zone

MDP formulation
---------------
State  : (agent_pos, inventory, doors_open, victims_rescued,
          victims_found, oxygen, debris_cleared)
Action : {MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT,
          PICKUP, OPEN_DOOR, CLEAR_DEBRIS, RESCUE_VICTIM}
Transition : deterministic (P(s'|s,a) = 1)
Reward :
  +20  per victim rescued to exit
  +5   per medkit picked up
  +10  per victim found (first visit)
  -2   per smoke tile step
  -1   per normal step (time pressure)
  -50  oxygen exhausted (terminal penalty)
  +100 all victims rescued (terminal bonus)
Terminal : all victims rescued  OR  oxygen <= 0
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, FrozenSet, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Tile and action constants
# ---------------------------------------------------------------------------

class Tile(IntEnum):
    EMPTY   = 0
    WALL    = 1
    DOOR    = 2   # unlocked
    DOOR_L  = 3   # locked
    DEBRIS  = 4
    SMOKE   = 5
    VICTIM  = 6
    MEDKIT  = 7
    KEY     = 8
    CROWBAR = 9
    EXIT    = 10


class Action(IntEnum):
    MOVE_UP      = 0
    MOVE_DOWN    = 1
    MOVE_LEFT    = 2
    MOVE_RIGHT   = 3
    PICKUP       = 4   # pick up item on current tile
    OPEN_DOOR    = 5   # open adjacent unlocked door (agent must be next to it)
    CLEAR_DEBRIS = 6   # clear debris on adjacent tile (needs crowbar)
    RESCUE       = 7   # evacuate carried victim if on exit tile


DIRECTIONS: Dict[Action, Tuple[int, int]] = {
    Action.MOVE_UP:    (-1,  0),
    Action.MOVE_DOWN:  ( 1,  0),
    Action.MOVE_LEFT:  ( 0, -1),
    Action.MOVE_RIGHT: ( 0,  1),
}

SMOKE_OXYGEN_COST = 3   # extra oxygen consumed per smoke step
STEP_OXYGEN_COST  = 1   # base oxygen per step


# ---------------------------------------------------------------------------
# MDP State
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class State:
    """Immutable MDP state – hashable for use in A* open/closed sets."""
    agent_pos:       Tuple[int, int]
    oxygen:          int
    # inventory: frozenset of item strings, e.g. {"key_red", "crowbar", "medkit"}
    inventory:       FrozenSet[str]              = field(default_factory=frozenset)
    # which (row, col) door positions have been opened
    doors_open:      FrozenSet[Tuple[int,int]]   = field(default_factory=frozenset)
    # which debris positions have been cleared
    debris_cleared:  FrozenSet[Tuple[int,int]]   = field(default_factory=frozenset)
    # victim positions the agent has visited (found)
    victims_found:   FrozenSet[Tuple[int,int]]   = field(default_factory=frozenset)
    # victim positions already evacuated to exit
    victims_rescued: FrozenSet[Tuple[int,int]]   = field(default_factory=frozenset)
    # medkit positions already collected
    medkits_taken:   FrozenSet[Tuple[int,int]]   = field(default_factory=frozenset)

    def is_terminal(self, total_victims: int) -> bool:
        return self.oxygen <= 0 or len(self.victims_rescued) >= total_victims

    def success(self, total_victims: int) -> bool:
        return len(self.victims_rescued) >= total_victims


# ---------------------------------------------------------------------------
# Environment / MDP
# ---------------------------------------------------------------------------

class SearchRescueEnv:
    """
    Grid-based MDP environment for the Search-and-Rescue task.

    Parameters
    ----------
    grid        : 2-D list of Tile values
    door_colors : mapping from (row, col) -> color string for DOOR_L tiles
    item_colors : mapping from (row, col) -> color string for KEY tiles
    start_pos   : agent starting (row, col)
    max_oxygen  : initial oxygen budget
    """

    def __init__(
        self,
        grid: List[List[int]],
        door_colors: Dict[Tuple[int,int], str],
        item_colors: Dict[Tuple[int,int], str],
        start_pos: Tuple[int, int],
        max_oxygen: int = 200,
    ):
        self.grid        = grid
        self.rows        = len(grid)
        self.cols        = len(grid[0])
        self.door_colors = door_colors   # (r,c) -> color for locked doors
        self.item_colors = item_colors   # (r,c) -> color for keys
        self.start_pos   = start_pos
        self.max_oxygen  = max_oxygen

        # Locate static objects once
        self.victim_positions: List[Tuple[int,int]] = []
        self.exit_positions:   List[Tuple[int,int]] = []
        for r in range(self.rows):
            for c in range(self.cols):
                t = grid[r][c]
                if t == Tile.VICTIM:
                    self.victim_positions.append((r, c))
                elif t == Tile.EXIT:
                    self.exit_positions.append((r, c))

        self.total_victims = len(self.victim_positions)

        # Current episode state (mutable; reset each episode)
        self.state: State = self._make_initial_state()

    # ------------------------------------------------------------------
    # MDP interface
    # ------------------------------------------------------------------

    def _make_initial_state(self) -> State:
        return State(
            agent_pos=self.start_pos,
            oxygen=self.max_oxygen,
        )

    def reset(self) -> State:
        """Reset environment to initial state and return it."""
        self.state = self._make_initial_state()
        return self.state

    def get_initial_state(self) -> State:
        return self._make_initial_state()

    def actions(self, state: State) -> List[Action]:
        """Return list of actions legal from *state*."""
        legal = []
        r, c = state.agent_pos

        # Movement
        for act, (dr, dc) in DIRECTIONS.items():
            nr, nc = r + dr, c + dc
            if self._passable(nr, nc, state):
                legal.append(act)

        # Pickup: item on current tile
        tile = self._effective_tile(r, c, state)
        if tile in (Tile.KEY, Tile.CROWBAR, Tile.MEDKIT):
            legal.append(Action.PICKUP)

        # Open door: any adjacent unlocked door
        for dr, dc in DIRECTIONS.values():
            nr, nc = r + dr, c + dc
            if self._in_bounds(nr, nc):
                t = self._effective_tile(nr, nc, state)
                if t == Tile.DOOR:
                    legal.append(Action.OPEN_DOOR)
                    break

        # Clear debris: adjacent debris + crowbar in inventory
        if "crowbar" in state.inventory:
            for dr, dc in DIRECTIONS.values():
                nr, nc = r + dr, c + dc
                if self._in_bounds(nr, nc):
                    t = self._effective_tile(nr, nc, state)
                    if t == Tile.DEBRIS:
                        legal.append(Action.CLEAR_DEBRIS)
                        break

        # Rescue: on exit with at least one found-but-not-yet-rescued victim
        if (r, c) in self.exit_positions:
            unrescued = state.victims_found - state.victims_rescued
            if unrescued:
                legal.append(Action.RESCUE)

        return legal

    def transition(self, state: State, action: Action) -> Tuple[State, float, bool]:
        """
        MDP transition function.

        Returns
        -------
        next_state : State
        reward     : float
        done       : bool
        """
        r, c = state.agent_pos
        reward = 0.0

        # ---- Movement ----
        if action in DIRECTIONS:
            dr, dc = DIRECTIONS[action]
            nr, nc = r + dr, c + dc

            if not self._passable(nr, nc, state):
                # Illegal move: no state change, small penalty
                next_state = state
                return next_state, -1.0, state.is_terminal(self.total_victims)

            # Oxygen cost
            tile = self._effective_tile(nr, nc, state)
            o2_cost = STEP_OXYGEN_COST
            if tile == Tile.SMOKE:
                o2_cost += SMOKE_OXYGEN_COST
                reward -= 2.0

            new_oxygen = state.oxygen - o2_cost
            reward -= 1.0  # step cost

            new_victims_found = set(state.victims_found)
            if tile == Tile.VICTIM and (nr, nc) not in state.victims_found:
                new_victims_found.add((nr, nc))
                reward += 10.0

            next_state = State(
                agent_pos=(nr, nc),
                oxygen=max(0, new_oxygen),
                inventory=state.inventory,
                doors_open=state.doors_open,
                debris_cleared=state.debris_cleared,
                victims_found=frozenset(new_victims_found),
                victims_rescued=state.victims_rescued,
                medkits_taken=state.medkits_taken,
            )

            if next_state.oxygen <= 0:
                reward -= 50.0
                return next_state, reward, True

            done = next_state.success(self.total_victims)
            if done:
                reward += 100.0
            return next_state, reward, done

        # ---- Pickup ----
        if action == Action.PICKUP:
            tile = self._effective_tile(r, c, state)
            new_inv = set(state.inventory)
            new_medkits = state.medkits_taken

            if tile == Tile.KEY:
                color = self.item_colors.get((r, c), "unknown")
                new_inv.add(f"key_{color}")
            elif tile == Tile.CROWBAR:
                new_inv.add("crowbar")
            elif tile == Tile.MEDKIT:
                if (r, c) not in state.medkits_taken:
                    new_inv.add("medkit")
                    new_medkits = state.medkits_taken | {(r, c)}
                    reward += 5.0

            next_state = State(
                agent_pos=state.agent_pos,
                oxygen=max(0, state.oxygen - STEP_OXYGEN_COST),
                inventory=frozenset(new_inv),
                doors_open=state.doors_open,
                debris_cleared=state.debris_cleared,
                victims_found=state.victims_found,
                victims_rescued=state.victims_rescued,
                medkits_taken=new_medkits,
            )
            return next_state, reward - 1.0, next_state.is_terminal(self.total_victims)

        # ---- Open door ----
        if action == Action.OPEN_DOOR:
            new_doors = set(state.doors_open)
            for dr, dc in DIRECTIONS.values():
                nr, nc = r + dr, c + dc
                if self._in_bounds(nr, nc):
                    t = self._effective_tile(nr, nc, state)
                    if t == Tile.DOOR:
                        new_doors.add((nr, nc))
                        break

            next_state = State(
                agent_pos=state.agent_pos,
                oxygen=max(0, state.oxygen - STEP_OXYGEN_COST),
                inventory=state.inventory,
                doors_open=frozenset(new_doors),
                debris_cleared=state.debris_cleared,
                victims_found=state.victims_found,
                victims_rescued=state.victims_rescued,
                medkits_taken=state.medkits_taken,
            )
            return next_state, reward - 1.0, next_state.is_terminal(self.total_victims)

        # ---- Clear debris ----
        if action == Action.CLEAR_DEBRIS:
            new_debris = set(state.debris_cleared)
            for dr, dc in DIRECTIONS.values():
                nr, nc = r + dr, c + dc
                if self._in_bounds(nr, nc):
                    t = self._effective_tile(nr, nc, state)
                    if t == Tile.DEBRIS:
                        new_debris.add((nr, nc))
                        break

            next_state = State(
                agent_pos=state.agent_pos,
                oxygen=max(0, state.oxygen - STEP_OXYGEN_COST),
                inventory=state.inventory,
                doors_open=state.doors_open,
                debris_cleared=frozenset(new_debris),
                victims_found=state.victims_found,
                victims_rescued=state.victims_rescued,
                medkits_taken=state.medkits_taken,
            )
            return next_state, reward - 1.0, next_state.is_terminal(self.total_victims)

        # ---- Rescue ----
        if action == Action.RESCUE:
            unrescued = state.victims_found - state.victims_rescued
            if unrescued and (r, c) in self.exit_positions:
                new_rescued = state.victims_rescued | unrescued
                reward += 20.0 * len(unrescued)
                next_state = State(
                    agent_pos=state.agent_pos,
                    oxygen=max(0, state.oxygen - STEP_OXYGEN_COST),
                    inventory=state.inventory,
                    doors_open=state.doors_open,
                    debris_cleared=state.debris_cleared,
                    victims_found=state.victims_found,
                    victims_rescued=new_rescued,
                    medkits_taken=state.medkits_taken,
                )
                done = next_state.success(self.total_victims)
                if done:
                    reward += 100.0
                return next_state, reward - 1.0, done

        return state, -1.0, state.is_terminal(self.total_victims)

    def step(self, action: Action) -> Tuple[State, float, bool]:
        """Apply action to current episode state in-place."""
        next_state, reward, done = self.transition(self.state, action)
        self.state = next_state
        return next_state, reward, done

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _effective_tile(self, r: int, c: int, state: State) -> int:
        """Return the tile value accounting for dynamic state changes."""
        base = self.grid[r][c]
        if base == Tile.DOOR_L:
            color = self.door_colors.get((r, c), "")
            if (r, c) in state.doors_open or f"key_{color}" in state.inventory:
                return Tile.DOOR
            return Tile.DOOR_L
        if base == Tile.DEBRIS and (r, c) in state.debris_cleared:
            return Tile.EMPTY
        if base == Tile.DOOR and (r, c) in state.doors_open:
            return Tile.EMPTY
        return base

    def _passable(self, r: int, c: int, state: State) -> bool:
        if not self._in_bounds(r, c):
            return False
        t = self._effective_tile(r, c, state)
        return t not in (Tile.WALL, Tile.DOOR_L, Tile.DEBRIS)

    def render(self, state: Optional[State] = None) -> str:
        """Return ASCII string of the grid for the given state."""
        if state is None:
            state = self.state
        symbols = {
            Tile.EMPTY:   ".",
            Tile.WALL:    "#",
            Tile.DOOR:    "D",
            Tile.DOOR_L:  "L",
            Tile.DEBRIS:  "X",
            Tile.SMOKE:   "~",
            Tile.VICTIM:  "V",
            Tile.MEDKIT:  "+",
            Tile.KEY:     "k",
            Tile.CROWBAR: "T",
            Tile.EXIT:    "E",
        }
        rows = []
        for r in range(self.rows):
            row_str = ""
            for c in range(self.cols):
                if (r, c) == state.agent_pos:
                    row_str += "@"
                else:
                    t = self._effective_tile(r, c, state)
                    row_str += symbols.get(t, "?")
            rows.append(row_str)
        info = (f"O2={state.oxygen}  inv={set(state.inventory)}  "
                f"found={len(state.victims_found)}  "
                f"rescued={len(state.victims_rescued)}")
        return "\n".join(rows) + "\n" + info


# ---------------------------------------------------------------------------
# Default map builder
# ---------------------------------------------------------------------------

def build_default_env(seed: int = 42) -> SearchRescueEnv:
    """
    Build a hand-crafted 15x20 map with 8 rooms for reproducible testing.

    Room layout (approximate):
      [Room A] [Room B] [Room C] [Room D]
      [Room E] [Room F] [Room G] [Room H]
    Connected by doors; some locked, some blocked by debris.
    """
    rng = random.Random(seed)

    W  = Tile.WALL
    E  = Tile.EMPTY
    D  = Tile.DOOR
    DL = Tile.DOOR_L
    DB = Tile.DEBRIS
    SM = Tile.SMOKE
    V  = Tile.VICTIM
    MK = Tile.MEDKIT
    K  = Tile.KEY
    CR = Tile.CROWBAR
    EX = Tile.EXIT

    # 15 rows x 20 cols
    grid = [
        #0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19
        [W,  W,  W,  W,  W,  W,  W,  W,  W,  W,  W,  W,  W,  W,  W,  W,  W,  W,  W,  W ],  # 0
        [W,  E,  E,  E,  E,  W,  E,  E,  E,  E,  W,  E,  E,  E,  E,  W,  E,  E,  EX, W ],  # 1
        [W,  E,  E,  E,  E,  W,  E,  SM, E,  E,  W,  E,  E,  E,  E,  D,  E,  E,  E,  W ],  # 2
        [W,  E,  K,  E,  E,  DL, E,  E,  E,  V,  W,  E,  E,  MK, E,  W,  E,  E,  E,  W ],  # 3
        [W,  E,  E,  E,  E,  W,  E,  SM, E,  E,  D,  E,  E,  E,  E,  W,  E,  E,  E,  W ],  # 4
        [W,  W,  D,  W,  W,  W,  W,  W,  W,  W,  W,  W,  D,  W,  W,  W,  W,  W,  W,  W ],  # 5
        [W,  E,  E,  E,  E,  W,  E,  E,  E,  E,  W,  E,  E,  E,  E,  W,  E,  E,  E,  W ],  # 6
        [W,  E,  E,  E,  E,  D,  E,  E,  CR, E,  W,  E,  V,  E,  E,  DL, E,  E,  E,  W ],  # 7
        [W,  E,  E,  E,  E,  W,  E,  E,  E,  E,  D,  E,  E,  E,  E,  W,  E,  E,  E,  W ],  # 8
        [W,  W,  W,  DB, W,  W,  W,  W,  W,  W,  W,  W,  W,  W,  DB, W,  W,  W,  W,  W ],  # 9
        [W,  E,  E,  E,  E,  W,  E,  E,  E,  E,  W,  E,  E,  E,  E,  W,  E,  E,  E,  W ],  # 10
        [W,  E,  MK, E,  E,  D,  E,  SM, SM, E,  W,  E,  E,  E,  E,  D,  E,  E,  E,  W ],  # 11
        [W,  E,  E,  E,  E,  W,  E,  E,  E,  E,  DL, E,  E,  E,  E,  W,  E,  V,  E,  W ],  # 12
        [W,  E,  E,  E,  E,  W,  E,  E,  E,  E,  W,  E,  E,  E,  E,  W,  E,  E,  E,  W ],  # 13
        [W,  W,  W,  W,  W,  W,  W,  W,  W,  W,  W,  W,  W,  W,  W,  W,  W,  W,  W,  W ],  # 14
    ]

    door_colors = {
        (3,  5):  "red",    # locked door between room A and B
        (7,  15): "blue",   # locked door between room C and D
        (12, 10): "green",  # locked door between lower rooms
    }

    item_colors = {
        (3, 2): "red",   # red key in room A
    }

    # Blue key hidden in room B (row 6-8, col 6-9)
    grid[6][8] = K
    item_colors[(6, 8)] = "blue"

    # Green key in room H (row 10-13, col 16-18)
    grid[10][17] = K
    item_colors[(10, 17)] = "green"

    return SearchRescueEnv(
        grid=grid,
        door_colors=door_colors,
        item_colors=item_colors,
        start_pos=(1, 1),
        max_oxygen=300,
    )
