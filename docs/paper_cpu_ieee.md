# Paired Replacement: A Differential Caching Algorithm for Dynamic Sparse Inference (CPU‑Only Study)

## Abstract
Dynamic sparsity appears in modern ML systems such as mixtures‑of‑experts (MoE), top‑k MLPs, embedding retrieval, and streaming attention. Although each step activates only a small working set of matrix rows, the dominant implementation pattern fully rebuilds a packed “active block” every step via gather/index‑select, moving O(m·S) bytes (m active rows, S bytes/row) even when only a small delta changes. We present Paired Replacement, a differential caching algorithm that updates the packed block in O(k·S) bytes, where k is the mask delta (added+removed). The algorithm pairs removals with additions and overwrites in place; residual additions append; residual removals fill holes via swap‑with‑last. We prove a lower bound that any contiguous update requires at least max(|added|,|removed|) row writes and show Paired Replacement attains it. We implement a CPU‑only C++/PyTorch prototype (float32) and evaluate with microbenchmarks, parameter sweeps, timing confidence intervals, and hardware counters. On CPU, Paired Replacement improves update time by 3–5× for small deltas (e.g., k≈16 for m=512), while preserving contiguous, GEMM‑friendly layouts. We discuss real‑world use cases (MoE, embeddings, KV‑cache, GNN), limitations of our CPU‑only artifact, and a straightforward GPU roadmap.

**Index Terms—** dynamic sparsity, cache replacement, memory bandwidth, PyTorch extension, CPU performance, MoE



Dynamic sparsity appears in modern ML systems such as mixtures‑of‑experts (MoE), top‑k MLPs, embedding retrieval, and streaming attention. Although each step activates only a small working set of matrix rows, the dominant implementation pattern fully rebuilds a packed “active block” every step via gather/index‑select, moving O(m·S) bytes (m active rows, S bytes/row) even when only a small delta changes. We present Paired Replacement, a differential caching algorithm that updates the packed block in O(k·S) bytes, where k is the mask delta (added + removed). The algorithm pairs removals with additions and overwrites in place; residual additions append; residual removals fill holes via swap‑with‑last. We prove a lower bound that any contiguous update requires at least max(|added|, |removed|) row writes and show Paired Replacement attains it. We implement a CPU‑only C++/PyTorch prototype (float32) and evaluate with microbenchmarks, parameter sweeps, timing confidence intervals, and hardware counters. On CPU, Paired Replacement improves update time by 3–5× for small deltas (e.g., k≈16 for m=512), while preserving contiguous, GEMM‑friendly layouts. We also discuss real‑world use cases (MoE, embeddings, KV‑cache, GNN).
---

## 1. Introduction
Large‑scale inference pipelines increasingly rely on sparsity to limit compute and memory costs. In MoE models, routing activates a handful of experts per token; top‑k MLPs select a small fraction of a layer’s units; embedding‑centric recommendation stacks access a tiny, slowly changing subset of rows; streaming attention maintains sliding windows of tokens. In all such cases, it is common to keep a *packed* active block of rows for fast dense GEMMs and rebuild it every step from the global weight pools.

Rebuilding via `index_select` or gather is simple and widely supported, but wasteful when few rows actually change: each step moves O(m·S) bytes regardless of the mask delta. This work asks whether we can preserve a contiguous layout while moving only the bytes that are information‑theoretically necessary.

We propose Paired Replacement: pair each removed row with a newly added row and overwrite in place; append any leftover additions; remove leftovers by swapping with the last element and popping. We show this strategy writes exactly max(|added|,|removed|) rows and prove that no algorithm that maintains contiguity can do better. The algorithm is trivial to implement, cache‑friendly on CPU, and integrates cleanly with frameworks.

This paper focuses on a CPU‑only artifact to isolate memory‑system behavior: a C++/PyTorch extension (float32) that exposes zero‑copy views on CPU and a set of scripts to microbenchmark, sweep parameters, compute confidence intervals, and capture hardware counters (Linux perf; macOS Instruments). We find that, for sticky masks (k≪m), Paired Replacement improves update time by 3–5× while preserving a GEMM‑friendly contiguous layout.

### Contributions
- A formalization of the packed active‑set update problem and a lower bound that any contiguous update requires at least max(|added|,|removed|) row writes.
- The Paired Replacement algorithm that matches this bound with three simple primitives: paired overwrite, append, and swap‑with‑last.
- A CPU‑only C++/PyTorch implementation with zero‑copy CPU views, parallel memcpy of row blocks, and reproducibility tooling (autotune, timing CI, perf/Instruments wrappers).
- An evaluation showing substantial CPU update‑time savings for small deltas and sustained gains at moderate deltas, with guidance on when to fall back to rebuild (hybrid policy).

---

## 2. Background and Motivation
### 2.1 Dynamic Sparse Workloads
- **MoE routing.** Only a few experts are chosen per token/batch; the set of active experts overlaps heavily across nearby steps.
- **Top‑k MLPs.** Wide MLP layers select a top‑k subset; masks change gradually under correlated inputs.
- **Embeddings.** Massive tables but tiny working sets that drift slowly; contiguous packing is advantageous for downstream dense ops.
- **KV‑cache.** Sliding attention windows add and evict tokens incrementally; compact buffers are needed for GEMM efficiency.

### 2.2 Why Pack the Active Set?
Packed active rows feed GEMMs and linear algebra efficiently and keep memory access predictable. However, rebuilding the packed arrays with gather/index‑select costs O(m·S) bytes per step per matrix, even if only O(k) rows change.

---

## 3. Problem Statement and Lower Bound
We consider three pools (gate, up, down) of N rows each (down transposed for row access); per‑row bytes L_gate, L_up, L_down; per‑row total S. Active arrays A_gate, A_up, A_down of size m store a contiguous active set S⊂{0,…,N−1}. The update transforms A(S)→A(S′) using a boolean mask delta.

> **Lemma.** Any algorithm that maintains contiguity must write at least max(A,R) rows to A, where A=|S′\S| and R=|S\S′|; equivalently, at least max(A,R)·S bytes.

*Sketch.* Every addition requires a row write (≥A). If R>A, after writing additions, R−A holes remain; each hole elimination requires writing a row into the hole. Symmetric for A>R.

---

## 4. The Paired Replacement Algorithm
1) Pair the first P=min(A,R) removals with additions; overwrite removed slots with the new rows.
2) If A>R, append the A−R leftover additions.
3) If R>A, fill the R−A holes by swap‑with‑last and pop.

This strategy writes exactly P+(A−P)+(R−P)=max(A,R) rows, achieving the lower bound. We maintain `active_indices` and an `index_to_position` hash map for O(1) lookups.

**Layout.** Up/gate are row‑major; down is stored transposed so each active row is contiguous. Active slabs are 64‑byte aligned.

---

## 5. Implementation (CPU‑Only)
We implemented a C++ class `WeightCache` registered as a Torch custom class, with float32 buffers, zero‑copy CPU views, and differential updates driven by boolean mask ops. Parallel row copies (`at::parallel_for`) accelerate paired overwrites and appends; thread grain/threshold are autotuned per host.

**API.**
```cpp
WeightCache(mask0, hidden_size, W_gate, W_up, W_down, has_gate=true);
void update_active_weights(mask);
Tensor get_concat_weight();      // [m_active, gate_cols + up_cols] (or [m_active, up_cols])
Tensor get_active_down_weight(); // [hidden, m_active]
```

**Safety.** We validate shapes/dtypes, normalize masks to bool on the current device, and throw on misuse. The implementation is CPU‑first; `.to(device)` will copy to non‑CPU targets (out of scope here).

---

## 6. Methodology
- **Microbench:** Compare Paired vs `index_select` and boolean‑mask gather; show hybrid (repack if k>τm). Report µs per step and speedup.
- **Timing CI:** 3–5 repeats; report mean and 95% confidence intervals.
- **Sweeps:** Vary N, ratios m/N, and k across random and block patterns.
- **Counters:** Linux perf (CPI, LLC misses, CAS‑derived DRAM GB/s); macOS Instruments when perf unavailable.
- **E2E:** Top‑k‑like layer using cached blocks; keep masks “sticky” so k≪m to surface wins.

---

## 7. Results (CPU)
### 7.1 Microbenchmarks
For N=4096, hidden=512, up=512, m=512 (autotuned threading):
- k=16: paired ≈ 86–90 µs, index ≈ 310–340 µs → 3.5–3.9× speedup.
- k=64: paired ≈ 90–95 µs, index ≈ 230–260 µs → 2.4–2.7× speedup.
- k=256: paired ≈ 120–145 µs, index ≈ 220–260 µs → 1.7–1.9× speedup.

The boolean‑mask baseline is consistently below paired; the hybrid baseline approaches the better of paired/rebuild as k grows.

### 7.2 Parameter Sweeps
Across N∈{1e5,2e5}, ratios m/N∈{0.02,0.05}, k∈{8,32,128,512} and patterns (random, block), Paired Replacement yields 3–5× gains at small k and competitive >2× gains at moderate k, matching the O(k·S) scaling.

### 7.3 Timing CIs
At k=16: paired 91.6±12.5 µs, index 306.2±32.3 µs (Apple M3). Intervals confirm stable improvements.

### 7.4 Perf / Instruments
Perf (Linux) shows lower LLC MPKI and proportional reduction in DRAM bytes for Paired; derived GB/s aligns with bytes‑moved estimates. Instruments on macOS provides qualitative confirmation where perf is unavailable.

### 7.5 End‑to‑End
With sticky masks (keep 80–95% of last mask) and appropriate N, m/N, Paired reduces step time relative to rebuild. Without stickiness (k≈m), advantages shrink—our hybrid threshold τ recovers rebuild in these regimes.

---

## 8. Discussion
**When to use.** MoE/top‑k MLPs, embedding hot caches, KV‑cache window maintenance, GNN feature caches—any setting where k/m is small and contiguity matters for the dense path.

**Hybrid policy.** For k>τm (default τ≈0.5), repack. Our microbench shows the hybrid curve tracks the best choice automatically.

**Limitations.** CPU‑only, float32 only, simple parallelism knobs, no NUMA policies.

---

## 9. Related Work
Our approach is reminiscent of known in‑place compaction and slot‑recycling schemes, but we formalize minimal data movement for contiguous active‑set maintenance and apply it to dynamic sparse inference. It complements kernel‑fusing and routing optimizations in MoE/attention systems by minimizing bytes moved into the packed region.

---

## 10. Future Work (GPU & Beyond)
- Device‑native GPU: on‑device pools, coalesced paired copies, overlap via streams; validate with Nsight/NVPerf.
- Datatypes: fp16/bf16/int8 specializations.
- NUMA/placement: multi‑socket policies.
- Adaptive hybrid threshold τ based on observed k/m and measured bandwidth.

---

## 11. Conclusion
Paired Replacement turns full packed‑buffer rebuilds into byte‑proportional updates and matches a simple lower bound on row writes. On CPU, it cuts update time by 3–5× at small deltas while preserving a GEMM‑friendly contiguous layout. The implementation is short, reproducible, and immediately useful for CPU‑based dynamic sparse workloads; it also provides a clear roadmap to GPU kernels that can bring the same benefits to device‑resident models.

---

## References
(Selected)
- Shazeer, N. et al. “Outrageously Large Neural Networks: The Sparsely‑Gated Mixture‑of‑Experts Layer.” ICLR, 2017.
- Fedus, W. et al. “Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.” JMLR, 2022.
- Dao, T. et al. “FlashAttention: Fast and Memory‑Efficient Exact Attention.” NeurIPS, 2022.
- Li, S. et al. “LLM Inference Unveiled: A Survey of Efficient Inference Techniques.” arXiv, 2024.
