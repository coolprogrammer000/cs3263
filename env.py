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
import math
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
# Procedural map builder
# ---------------------------------------------------------------------------

def build_default_env(seed: int = 42, num_rooms: int = 9) -> SearchRescueEnv:
    """
    Procedurally generate a multi-room Search-and-Rescue environment.

    Parameters
    ----------
    seed      : RNG seed for full reproducibility
    num_rooms : number of rooms (4-16 recommended; default 9).
                Different seeds produce different room sizes, connections,
                and item placements within the same room count.

    Layout
    ------
    Rooms are arranged in a roughly square grid of varying sizes.
    Adjacent rooms are connected through full carved corridors (1 cell wide)
    that hold an unlocked door, a locked door, or debris.

    Key dependencies form chains naturally:
        room_0 holds key_A  ->  door_A unlocks room_1
        room_1 holds key_B  ->  door_B unlocks room_2   (key in locked room)
        ...

    Solvability guarantee
    ---------------------
    - All rooms are connected via a random spanning tree.
    - Keys for locked spanning-tree doors are placed in the BFS ancestor room,
      so no key is ever locked behind the door it opens.
    - Debris is placed only on non-spanning-tree (optional) edges, so every
      room is reachable without a crowbar; the crowbar unlocks shortcuts.
    - Oxygen budget scales generously with grid area.
    """
    rng = random.Random(seed)
    num_rooms = max(4, min(num_rooms, 20))

    # ------------------------------------------------------------------
    # 1. Room grid dimensions
    # ------------------------------------------------------------------
    n_cols_r = max(2, round(math.sqrt(num_rooms)))
    n_rows_r = math.ceil(num_rooms / n_cols_r)
    total_slots = n_rows_r * n_cols_r

    # Independent random height/width per room slot
    room_h = [rng.randint(4, 7) for _ in range(total_slots)]
    room_w = [rng.randint(4, 8) for _ in range(total_slots)]

    # Each strip shares its boundary at the widest/tallest room in that strip
    row_max_h = [
        max(room_h[ri * n_cols_r + ci] for ci in range(n_cols_r))
        for ri in range(n_rows_r)
    ]
    col_max_w = [
        max(room_w[ri * n_cols_r + ci] for ri in range(n_rows_r))
        for ci in range(n_cols_r)
    ]

    # Top-left interior corner of each room strip
    row_starts = [1 + sum(row_max_h[i] + 1 for i in range(ri)) for ri in range(n_rows_r)]
    col_starts = [1 + sum(col_max_w[i] + 1 for i in range(ci)) for ci in range(n_cols_r)]

    total_rows = 1 + sum(h + 1 for h in row_max_h)
    total_cols = 1 + sum(w + 1 for w in col_max_w)

    # ------------------------------------------------------------------
    # 2. Carve room interiors
    # ------------------------------------------------------------------
    grid: List[List[int]] = [[Tile.WALL] * total_cols for _ in range(total_rows)]
    room_rect: Dict[int, Tuple[int, int, int, int]] = {}  # (top, left, bot, right)

    for ri in range(n_rows_r):
        for ci in range(n_cols_r):
            idx = ri * n_cols_r + ci
            if idx >= num_rooms:
                continue
            t = row_starts[ri]
            l = col_starts[ci]
            h = room_h[idx]
            w = room_w[idx]
            for r in range(t, t + h):
                for c in range(l, l + w):
                    grid[r][c] = Tile.EMPTY
            room_rect[idx] = (t, l, t + h - 1, l + w - 1)

    # ------------------------------------------------------------------
    # 3. Build corridors between adjacent rooms
    # ------------------------------------------------------------------
    # Each edge: (room_a, room_b, door_pos, extra_empty_cells)
    # door_pos       — the single tile that becomes DOOR / DOOR_L / DEBRIS
    # extra_empty_cells — remaining corridor tiles, always carved as EMPTY
    # The door is placed immediately adjacent to room_a so the agent can
    # stand inside room_a and operate it.
    #
    # Why full corridors?
    #   Rooms can be narrower than their strip's maximum width, leaving
    #   multiple wall tiles between them.  A single door tile would be
    #   unreachable from the far room without carving the whole gap.

    def _build_corridor(
        idx_a: int, idx_b: int, horizontal: bool
    ) -> Tuple[Optional[Tuple[int, int]], List[Tuple[int, int]]]:
        ta, la, ba, ra = room_rect[idx_a]
        tb, lb, bb, rb = room_rect[idx_b]
        if horizontal:                      # idx_a is left room, idx_b is right
            ov_top = max(ta, tb)
            ov_bot = min(ba, bb)
            if ov_top > ov_bot:
                return None, []
            mid = (ov_top + ov_bot) // 2
            door_pos = (mid, ra + 1)        # immediately right of room_a
            extra    = [(mid, c) for c in range(ra + 2, lb)]
        else:                               # idx_a is top room, idx_b is bottom
            ov_left  = max(la, lb)
            ov_right = min(ra, rb)
            if ov_left > ov_right:
                return None, []
            mid = (ov_left + ov_right) // 2
            door_pos = (ba + 1, mid)        # immediately below room_a
            extra    = [(r, mid) for r in range(ba + 2, tb)]
        return door_pos, extra

    all_edges: List[Tuple[int, int, Tuple[int, int], List[Tuple[int, int]]]] = []
    for ri in range(n_rows_r):
        for ci in range(n_cols_r):
            idx = ri * n_cols_r + ci
            if idx >= num_rooms:
                continue
            if ci + 1 < n_cols_r:
                idx_r = ri * n_cols_r + (ci + 1)
                if idx_r < num_rooms:
                    dp, ex = _build_corridor(idx, idx_r, horizontal=True)
                    if dp:
                        all_edges.append((idx, idx_r, dp, ex))
            if ri + 1 < n_rows_r:
                idx_d = (ri + 1) * n_cols_r + ci
                if idx_d < num_rooms:
                    dp, ex = _build_corridor(idx, idx_d, horizontal=False)
                    if dp:
                        all_edges.append((idx, idx_d, dp, ex))

    # ------------------------------------------------------------------
    # 4. Randomised Kruskal spanning tree
    # ------------------------------------------------------------------
    rng.shuffle(all_edges)
    uf_parent = list(range(num_rooms))

    def _find(x: int) -> int:
        while uf_parent[x] != x:
            uf_parent[x] = uf_parent[uf_parent[x]]
            x = uf_parent[x]
        return x

    def _union(a: int, b: int) -> bool:
        pa, pb = _find(a), _find(b)
        if pa == pb:
            return False
        uf_parent[pa] = pb
        return True

    spanning: List[Tuple] = []
    extras:   List[Tuple] = []
    for e in all_edges:
        (spanning if _union(e[0], e[1]) else extras).append(e)

    # Extra edges (cycles) add complexity without breaking connectivity
    n_extra = min(len(extras), max(1, num_rooms // 3))
    connections = spanning + rng.sample(extras, n_extra)

    # Set of door positions that belong to the spanning tree
    spanning_door_set = {e[2] for e in spanning}

    # ------------------------------------------------------------------
    # 5. BFS order from room 0 on the spanning tree
    # ------------------------------------------------------------------
    tree_adj: Dict[int, List[int]] = {i: [] for i in range(num_rooms)}
    for a, b, _, __ in spanning:
        tree_adj[a].append(b)
        tree_adj[b].append(a)

    bfs_order: List[int] = []
    visited_bfs: set = {0}
    queue = [0]
    while queue:
        cur = queue.pop(0)
        bfs_order.append(cur)
        for nb in tree_adj[cur]:
            if nb not in visited_bfs:
                visited_bfs.add(nb)
                queue.append(nb)

    bfs_rank = {r: i for i, r in enumerate(bfs_order)}

    # ------------------------------------------------------------------
    # 6. Carve all corridors; assign door types
    # ------------------------------------------------------------------
    door_colors: Dict[Tuple[int, int], str] = {}
    item_colors: Dict[Tuple[int, int], str] = {}

    color_pool = ["red", "blue", "green", "yellow", "purple",
                  "orange", "cyan", "magenta"]
    rng.shuffle(color_pool)
    color_iter = iter(color_pool)

    key_placements: Dict[str, int] = {}   # color -> room_idx where key goes
    needs_crowbar = False

    # Scale locked-door probability down for larger maps to keep state space
    # tractable: each locked door exponentially multiplies inventory states.
    locked_prob = max(0.15, 0.28 - 0.01 * max(0, num_rooms - 9))

    for a, b, door_pos, extra_cells in connections:
        # Carve the full corridor gap as EMPTY first
        for cell in extra_cells:
            grid[cell[0]][cell[1]] = Tile.EMPTY

        # Orient so `a` is the shallower (BFS-closer-to-start) room
        if bfs_rank.get(a, 0) > bfs_rank.get(b, 0):
            a, b = b, a

        roll = rng.random()
        if roll < locked_prob:
            # Locked door — key placed in room a (always accessible before b)
            try:
                color = next(color_iter)
            except StopIteration:
                grid[door_pos[0]][door_pos[1]] = Tile.DOOR
                continue
            door_colors[door_pos] = color
            grid[door_pos[0]][door_pos[1]] = Tile.DOOR_L
            key_placements[color] = a
        elif roll < 0.40 and door_pos not in spanning_door_set:
            # Debris only on non-spanning (optional) edges: every room stays
            # reachable via spanning-tree doors; debris just blocks shortcuts
            grid[door_pos[0]][door_pos[1]] = Tile.DEBRIS
            needs_crowbar = True
        else:
            grid[door_pos[0]][door_pos[1]] = Tile.DOOR

    # ------------------------------------------------------------------
    # 7. Place items
    # ------------------------------------------------------------------
    occupied: set = set()

    def _room_empty_cells(room_idx: int) -> List[Tuple[int, int]]:
        t, l, b, r = room_rect[room_idx]
        return [
            (rr, cc)
            for rr in range(t, b + 1)
            for cc in range(l, r + 1)
            if grid[rr][cc] == Tile.EMPTY and (rr, cc) not in occupied
        ]

    def _place(room_idx: int, tile_type: int) -> Optional[Tuple[int, int]]:
        candidates = _room_empty_cells(room_idx)
        if not candidates:
            return None
        p = rng.choice(candidates)
        grid[p[0]][p[1]] = tile_type
        occupied.add(p)
        return p

    # Keys — placed in ancestor rooms, naturally creating key-chain dependencies.
    # If a room is full (no EMPTY cells), downgrade the locked door to an
    # unlocked door rather than leaving an unacquirable key on the map.
    for color, room_idx in key_placements.items():
        p = _place(room_idx, Tile.KEY)
        if p:
            item_colors[p] = color
        else:
            # Fallback: unlock the door so the map stays solvable
            for door_pos, c in list(door_colors.items()):
                if c == color:
                    grid[door_pos[0]][door_pos[1]] = Tile.DOOR
                    del door_colors[door_pos]
                    break

    # Crowbar in room 0 (always immediately accessible)
    if needs_crowbar:
        _place(0, Tile.CROWBAR)

    # Victims — concentrated in later BFS rooms for challenge.
    # Cap at 6 to keep state space tractable (each victim adds a dimension).
    n_victims = max(3, min(num_rooms // 2, 6))
    later_half = bfs_order[max(1, len(bfs_order) // 2):]
    victim_rooms = (later_half * math.ceil(n_victims / max(1, len(later_half))))[:n_victims]
    rng.shuffle(victim_rooms)
    for room_idx in victim_rooms:
        _place(room_idx, Tile.VICTIM)

    # Medkits — scattered across all rooms
    for _ in range(max(2, num_rooms // 3)):
        _place(rng.choice(bfs_order), Tile.MEDKIT)

    # Smoke patches — traversable, not added to `occupied`
    for _ in range(max(2, num_rooms // 3)):
        room_idx = rng.choice(bfs_order)
        for _ in range(rng.randint(1, 3)):
            candidates = _room_empty_cells(room_idx)
            if candidates:
                p = rng.choice(candidates)
                grid[p[0]][p[1]] = Tile.SMOKE

    # Exit — in the deepest BFS room
    for room_idx in reversed(bfs_order):
        if _place(room_idx, Tile.EXIT):
            break

    # ------------------------------------------------------------------
    # 8. Start position in room 0
    # ------------------------------------------------------------------
    candidates = _room_empty_cells(0)
    start_pos = rng.choice(candidates) if candidates else room_rect[0][:2]

    # ------------------------------------------------------------------
    # 9. Oxygen budget — generous scaling with map size and victim count
    # ------------------------------------------------------------------
    max_oxygen = max(300, (total_rows + total_cols) * 3 * n_victims + 200)

    return SearchRescueEnv(
        grid=grid,
        door_colors=door_colors,
        item_colors=item_colors,
        start_pos=start_pos,
        max_oxygen=max_oxygen,
    )
