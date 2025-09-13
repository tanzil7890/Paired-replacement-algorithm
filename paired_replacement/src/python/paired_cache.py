import numpy as np
from typing import Optional, Tuple


class PairedCache:
    """
    Numpy reference implementation of the Paired Replacement algorithm.

    Pools:
      - up_pool: shape [N, up_cols]
      - down_pool_T: shape [N, hidden_dim]  (transposed once for row-major access)
      - gate_pool: optional, shape [N, gate_cols]

    Active buffers:
      - active_up: shape [m, up_cols]
      - active_down_T: shape [m, hidden_dim]
      - active_gate: optional, shape [m, gate_cols]

    Exposes:
      - get_concat_weight(): [m, up_cols + (gate_cols if present)]
      - get_active_down_weight(): [hidden_dim, m]
    """

    def __init__(
        self,
        up_pool: np.ndarray,
        down_pool: np.ndarray,
        gate_pool: Optional[np.ndarray] = None,
        init_mask: Optional[np.ndarray] = None,
    ) -> None:
        assert up_pool.ndim == 2
        assert down_pool.ndim == 2
        if gate_pool is not None:
            assert gate_pool.ndim == 2

        self.N = up_pool.shape[0]
        self.up_cols = up_pool.shape[1]
        self.hidden_dim = down_pool.shape[0]
        assert down_pool.shape[1] == self.N, "down_pool must be [hidden_dim, N]"

        # Store pools (transpose down to row-major per-row access)
        self.up_pool = up_pool
        self.down_pool_T = down_pool.T.copy()  # [N, hidden_dim]
        self.gate_pool = gate_pool
        self.gate_cols = gate_pool.shape[1] if gate_pool is not None else 0

        # Active state
        self.active_indices: list[int] = []
        self.index_to_pos: dict[int, int] = {}

        # Active buffers (start empty, grow as needed)
        self.active_up = np.empty((0, self.up_cols), dtype=up_pool.dtype)
        self.active_down_T = np.empty((0, self.hidden_dim), dtype=down_pool.dtype)
        self.active_gate = (
            np.empty((0, self.gate_cols), dtype=gate_pool.dtype) if gate_pool is not None else None
        )

        self.current_mask = np.zeros((self.N,), dtype=bool)

        if init_mask is not None:
            self.update(init_mask)

    def _ensure_capacity(self, new_m: int) -> None:
        # Grow buffers if needed
        cur = self.active_up.shape[0]
        if new_m <= cur:
            return
        grow_to = new_m
        self.active_up = self._grow(self.active_up, (grow_to, self.up_cols))
        self.active_down_T = self._grow(self.active_down_T, (grow_to, self.hidden_dim))
        if self.active_gate is not None:
            self.active_gate = self._grow(self.active_gate, (grow_to, self.gate_cols))

    @staticmethod
    def _grow(arr: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
        new_arr = np.empty(new_shape, dtype=arr.dtype)
        if arr.size:
            new_arr[: arr.shape[0], : arr.shape[1]] = arr
        return new_arr

    def update(self, new_mask: np.ndarray) -> None:
        new_mask = np.asarray(new_mask, dtype=bool).reshape(-1)
        assert new_mask.shape[0] == self.N

        old_mask = self.current_mask
        added_mask = new_mask & (~old_mask)
        removed_mask = old_mask & (~new_mask)

        added = np.flatnonzero(added_mask)
        removed = np.flatnonzero(removed_mask)

        num_add = added.shape[0]
        num_rem = removed.shape[0]
        pairs = min(num_add, num_rem)

        # Ensure capacity for potential growth
        m = len(self.active_indices)
        self._ensure_capacity(m + max(0, num_add - num_rem))

        # Phase 1: Pair removals with additions
        for i in range(pairs):
            r_idx = int(removed[i])
            a_idx = int(added[i])
            pos = self.index_to_pos.get(r_idx, None)
            if pos is None:
                continue
            # Overwrite at same position
            self.active_up[pos, :] = self.up_pool[a_idx, :]
            self.active_down_T[pos, :] = self.down_pool_T[a_idx, :]
            if self.active_gate is not None:
                self.active_gate[pos, :] = self.gate_pool[a_idx, :]
            # Update maps
            del self.index_to_pos[r_idx]
            self.active_indices[pos] = a_idx
            self.index_to_pos[a_idx] = pos

        # Phase 2: Remaining additions (append)
        for i in range(pairs, num_add):
            a_idx = int(added[i])
            pos = len(self.active_indices)
            self.active_up[pos, :] = self.up_pool[a_idx, :]
            self.active_down_T[pos, :] = self.down_pool_T[a_idx, :]
            if self.active_gate is not None:
                self.active_gate[pos, :] = self.gate_pool[a_idx, :]
            self.active_indices.append(a_idx)
            self.index_to_pos[a_idx] = pos

        # Phase 3: Remaining removals (swap-with-last)
        for i in range(pairs, num_rem):
            r_idx = int(removed[i])
            pos = self.index_to_pos.get(r_idx, None)
            if pos is None:
                continue
            last_pos = len(self.active_indices) - 1
            if pos != last_pos:
                last_idx = self.active_indices[last_pos]
                # Move last row into pos
                self.active_up[pos, :] = self.active_up[last_pos, :]
                self.active_down_T[pos, :] = self.active_down_T[last_pos, :]
                if self.active_gate is not None:
                    self.active_gate[pos, :] = self.active_gate[last_pos, :]
                self.active_indices[pos] = last_idx
                self.index_to_pos[last_idx] = pos
            # Pop last
            self.active_indices.pop()
            del self.index_to_pos[r_idx]

        self.current_mask = new_mask.copy()

    def get_concat_weight(self) -> np.ndarray:
        m = len(self.active_indices)
        if m == 0:
            if self.active_gate is not None:
                return np.empty((0, self.gate_cols + self.up_cols), dtype=self.up_pool.dtype)
            return np.empty((0, self.up_cols), dtype=self.up_pool.dtype)
        if self.active_gate is not None:
            return np.concatenate(
                [self.active_gate[:m, :], self.active_up[:m, :]], axis=1
            )
        return self.active_up[:m, :]

    def get_active_down_weight(self) -> np.ndarray:
        m = len(self.active_indices)
        if m == 0:
            return np.empty((self.hidden_dim, 0), dtype=self.down_pool_T.dtype)
        return self.active_down_T[:m, :].T  # [hidden_dim, m]


def baseline_full_rebuild(
    up_pool: np.ndarray, down_pool: np.ndarray, gate_pool: Optional[np.ndarray], mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Rebuild active tensors from scratch via indexing.

    Returns (concat_weight, down_weight)
    """
    idx = np.flatnonzero(mask)
    up_act = up_pool[idx, :]
    down_T_act = down_pool.T[idx, :]  # [m, hidden]
    gate_act = gate_pool[idx, :] if gate_pool is not None else None
    if gate_act is not None:
        concat = np.concatenate([gate_act, up_act], axis=1)
    else:
        concat = up_act
    down = down_T_act.T  # [hidden, m]
    return concat, down

