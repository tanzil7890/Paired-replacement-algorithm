# Paired Replacement Algorithm — Pseudocode and Invariants

I keep this file as a compact reference for how I implement the algorithm and the invariants I rely on during updates.

## Data Structures and Layout

- Memory pools (row-major unless specified):
  - `gate_pool[N, gate_cols]` (optional)
  - `up_pool[N, up_cols]`
  - `down_pool_T[N, hidden_dim]` (transpose of `down[hidden_dim, N]` for row access)
- Active buffers (always tightly packed):
  - `active_gate[m, gate_cols]` (optional)
  - `active_up[m, up_cols]`
  - `active_down_T[m, hidden_dim]`
- Indexing metadata:
  - `active_indices[m]`: row ids currently active
  - `index_to_position`: map row id → position in `active_*`
  - `current_mask[N]`: boolean mask of the active set

Invariants:
- Shapes consistent with initialization; dtype float32.
- `active_*` are contiguous views; when device is CPU, tensor views are zero-copy via `from_blob`.
- `active_indices` and `index_to_position` are consistent (bijective over the active set).

## Phase 0 — Initialization
```
// inputs: init_mask[N], hidden_dim, gate/up/down weights
// allocate pools (64B-aligned), transpose down → down_pool_T
// pre-allocate active buffers sized for up to N rows
current_mask ← zeros(N)
update_active_weights(init_mask)
```

## Phase 1 — Compute Mask Diff (added, removed)
```
function compute_mask_diff(old_mask, new_mask):
    added_mask   ← new_mask & ~old_mask
    removed_mask ← old_mask & ~new_mask
    added   ← nonzero(added_mask)
    removed ← nonzero(removed_mask)
    return (added, removed)
```

## Phase 2 — Paired Replacement (overwrites)
```
function paired_replace(added, removed):
    P ← min(|added|, |removed|)
    // collect valid (pos, new_id) pairs to avoid map races
    pairs ← []
    for i in [0..P):
        r_id ← removed[i]
        if r_id ∈ index_to_position:
            pos ← index_to_position[r_id]
            pairs.append((pos, added[i], r_id))

    // parallel memcpy over pairs
    parallel_for (pos, a_id, r_id) in pairs:
        if has_gate: memcpy(active_gate[pos], gate_pool[a_id])
        memcpy(active_up[pos],   up_pool[a_id])
        memcpy(active_down_T[pos], down_pool_T[a_id])

    // sequential metadata updates
    for (pos, a_id, r_id) in pairs:
        erase(index_to_position[r_id])
        active_indices[pos] ← a_id
        index_to_position[a_id] ← pos
```

## Phase 3 — Residual Additions (append)
```
function append_remaining(added, start=P):
    to_append ← clamp(|added|-P, 0, capacity-left)
    // parallel memcpy
    parallel_for t in [0..to_append):
        pos ← size(active_indices) + t
        a_id ← added[P + t]
        if has_gate: memcpy(active_gate[pos], gate_pool[a_id])
        memcpy(active_up[pos],   up_pool[a_id])
        memcpy(active_down_T[pos], down_pool_T[a_id])
    // sequential metadata
    resize active_indices by +to_append
    for t in [0..to_append):
        pos ← old_size + t
        a_id ← added[P + t]
        active_indices[pos] ← a_id
        index_to_position[a_id] ← pos
```

## Phase 4 — Residual Removals (swap-with-last)
```
function remove_remaining(removed, start=P):
    for i in [P..|removed|):
        r_id ← removed[i]
        if r_id ∈ index_to_position:
            pos ← index_to_position[r_id]
            last ← size(active_indices)-1
            if pos ≠ last:
                l_id ← active_indices[last]
                // move last row to pos
                if has_gate: memcpy(active_gate[pos], active_gate[last])
                memcpy(active_up[pos], active_up[last])
                memcpy(active_down_T[pos], active_down_T[last])
                active_indices[pos] ← l_id
                index_to_position[l_id] ← pos
            pop_back(active_indices)
            erase(index_to_position[r_id])
```

## Phase 5 — Rebuild Tensor Views (zero-copy on CPU)
```
up_tensor    ← from_blob(active_up)       // [m, up_cols]
down_packed  ← from_blob(active_down_T)  // [m, hidden]
down_tensor  ← transpose(down_packed)    // [hidden, m]
if has_gate:
    gate_tensor ← from_blob(active_gate)  // [m, gate_cols]
    concat ← cat([gate_tensor, up_tensor], dim=1)
else:
    concat ← up_tensor
if device==CPU: cache ← views
else:          cache ← to(device)
```

## Complexity and Optimality
- Time: O(k) data movement and metadata updates, where k = |added|+|removed|.
- Bytes moved: exactly `max(|added|,|removed|)` rows (minimal by lower bound under contiguity).
- Cache lines: with B-aligned rows and sizes multiple of B, writes/reads are `max(A,R)·S/B`.
- Parallelism: `at::parallel_for` on pair/appends; environment knobs:
  - `WEIGHT_CACHE_GRAIN` (default 64)
  - `WEIGHT_CACHE_PAR_THRESHOLD` (default 64)

## Hybrid Baseline (for comparison)
- Heuristic: if delta `k > τ·m`, rebuild via index_select; else use paired update.
- Control via `WEIGHT_CACHE_HYBRID_TAU_FRAC` or `--hybrid_tau` in microbench scripts.

## Notes
- Device: current implementation is CPU-first; on non-CPU devices, tensors are materialized via `.to(device)`.
- Types: currently float32; templating/specialization planned.
