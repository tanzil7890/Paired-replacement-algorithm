Below is a concrete, step‑by‑step plan to turn docs/paper_cpu_ieee.md into a complete, CPU‑only, submission‑ready paper. I include exact commands to regenerate results, what figures/tables to include, ready‑to‑paste Markdown blocks to extend the paper, and final IEEE formatting/export steps.

High‑level checklist

Regenerate results (microbench, CI, sweeps, optional perf).
Insert figures and tables with captions and hardware specs.
Expand paper sections (Background, Proof, Algorithm, Implementation, Methodology, Results, Discussion, Threats, Artifact & Reproducibility).
Final polish (abstract length, related work, references).
IEEE formatting (LaTeX/IEEEtran or Pandoc export).
Regenerate the CPU artifact (from repo root)
Install and build:
python install_dev.py
Autotune per host (loads automatically later):
python3 paired_replacement/benchmarks/scripts/run_autotune.py --N 8192 --hidden_dim 1024 --up_cols 1024 --m 1024 --steps 50 --outfile paired_replacement/benchmarks/configs/autotune_result.json
Microbench + overlays:
python3 paired_replacement/benchmarks/scripts/run_microbench_torch.py --N 8192 --hidden_dim 1024 --up_cols 1024 --m 1024 --ks 16,64,256,1024 --steps_per_k 50 --device cpu --outfile paired_replacement/benchmarks/data/results_microbench_torch.csv
python3 paired_replacement/benchmarks/scripts/plot_microbench_overlays.py --torch_csv paired_replacement/benchmarks/data/results_microbench_torch.csv --out paired_replacement/benchmarks/plots/microbench_speedup_overlays.png
Timing CI:
python3 paired_replacement/benchmarks/scripts/run_timing_ci.py --N 8192 --hidden_dim 1024 --up_cols 1024 --m 1024 --ks 16,64,256,1024 --steps_per_k 50 --repeats 5 --device cpu --outfile paired_replacement/benchmarks/data/results_timing_ci.csv
Parameter sweeps + plots:
python3 paired_replacement/benchmarks/scripts/run_sweep.py --Ns 100000,200000 --ratios 0.02,0.05 --ks 8,32,128,512 --patterns random,block --repeats 3 --up_cols 256 --hidden_dim 256 --outfile paired_replacement/benchmarks/data/results_sweep.csv
python3 paired_replacement/benchmarks/scripts/plot_sweep.py --csv paired_replacement/benchmarks/data/results_sweep.csv --out paired_replacement/benchmarks/plots/sweep_plots.png
Optional (Linux perf + plots):
python3 paired_replacement/benchmarks/scripts/run_perf_torch.py --N 8192 --hidden_dim 1024 --up_cols 1024 --m 1024 --ks 16,64,256,1024 --steps_per_k 50 --device cpu --bw_preset intel --bytes_per_cas 64 --outfile paired_replacement/benchmarks/data/results_perf_torch.csv
python3 paired_replacement/benchmarks/scripts/plot_perf.py --perf_csv paired_replacement/benchmarks/data/results_perf_torch.csv --out paired_replacement/benchmarks/plots/perf_plots.png
Optional end‑to‑end (CPU):
python3 paired_replacement/benchmarks/scripts/run_e2e_sweep.py --N 8192 --hidden_dim 1024 --up_cols 1024 --m_list 512,1024,2048 --batch_list 8,16,32 --steps 50 --device cpu --outfile paired_replacement/benchmarks/data/results_e2e_torch.csv
python3 paired_replacement/benchmarks/scripts/plot_e2e.py --csv paired_replacement/benchmarks/data/results_e2e_torch.csv --out paired_replacement/benchmarks/plots/e2e_speedup.png
Insert figures and captions (CPU only)
Add these figures into docs/paper_cpu_ieee.md with short, precise captions including hardware, N, m/N, k, dims, and whether autotune was used:
paired_replacement/benchmarks/plots/microbench_speedup_overlays.png
paired_replacement/benchmarks/plots/sweep_plots.png
(Linux only) paired_replacement/benchmarks/plots/perf_plots.png
paired_replacement/benchmarks/plots/e2e_speedup.png
Example figure block to paste:
Figure 1: Microbench speedups (Paired vs index_select/boolean‑mask; Hybrid vs index_select). CPU: [Apple M3 or x86 spec], N=8192, hidden=1024, up=1024, m=1024, autotuned grain/threshold (see CSV banner).
Figure 2: Parameter sweeps (speedup vs k) across m/N and random vs block patterns. Same hardware; repeated 3×; error bars omitted for clarity (CIs reported in Table 1).
Figure 3 (Linux): Perf counters — LLC miss rate, CPI, derived DRAM GB/s vs k; normalized per step; paired vs rebuild.
Add a CI table (ready-to-paste template)
Compute means ±95% CIs from paired_replacement/benchmarks/data/results_timing_ci.csv.
Paste a Markdown table like:
k	Paired (µs)	Index (µs)	Speedup (×)
16	91.6 ± 12.5	306.2 ± 32.3	3.34
64	94.9 ± 8.8	252.7 ± 26.7	2.66
256	145.5 ± 22.7	258.6 ± 31.6	1.78
Expand sections by adding these content blocks
Paste the blocks into docs/paper_cpu_ieee.md in the indicated sections. You can adapt numbers to your latest CSVs.
Section 2 (Background and Motivation; ~800–1200 words)

Expand with short surveys on MoE routing, activation sparsity, embedding caches, and their packed‑buffer rebuild patterns. Contrast gather/index_select vs differential update. Cite 3–5 key papers.
Section 3 (Proof of Lower Bound; ~500–800 words)

Paste a detailed proof sketch:
Define S, S′, A=|S′\S|, R=|S\S′|; show any contiguous update that transforms A(S)→A(S′) must write ≥max(A,R) rows. Provide case analysis and argue no metadata trick can avoid writing into holes while preserving contiguity.
Section 4 (Algorithm; ~500–800 words)

Include pseudocode (you can adapt from docs/algorithm_pseudocode.md) and a paragraph on why pairing minimizes random access (contiguous memcpy) and why swap‑with‑last is cache‑friendly.
Section 5 (Implementation; ~700–1000 words)

Describe memory layout (row‑major up/gate; down transposed), 64‑B alignment, zero‑copy on CPU, parallel_for usage, grain/threshold autotune, safety checks, dtype/shape validation, and hybrid baseline support.
Section 6 (Methodology; ~700–1000 words)

State hardware, OS, toolchains; describe scripts used and their parameters; justify metrics (µs per step, speedup, CPI/LLC/BW). Note that perf is only for Linux; Instruments on macOS collected qualitatively.
Section 7 (Results; ~1200–1800 words)

Microbench findings: present 3–4 key points (small k → 3–5×; moderate k → ~2×). Link to overlays figure.
Sweeps: discuss trends, crossovers, and random vs block patterns. Link to sweeps figure.
CIs: present the table; emphasize stable improvements.
Perf/Instruments: describe counter trends; show DRAM BW reduction proportional to k where available. Link to perf figure.
E2E CPU: explain conditions for “sticky masks” and give at least one tuned configuration with wins.
Section 8 (Discussion; ~500–800 words)

Where to deploy (MoE/top‑k, embeddings, KV‑cache, GNN); when to flip to hybrid; how to expand beyond CPU.
Section 9 (Related Work; ~600–900 words)

Summarize array compaction and slot recycling; compare to sparse gather/scatter libraries; mention MoE and attention optimization work that’s orthogonal.
Section 10 (Future Work; ~400–600 words)

GPU design plan: on‑device pools; coalesced, vectorized row copies; stream overlap; Nsight/NVPerf evaluation; datatype specializations.
Section 12–16 (Appendices, Artifact & Reproducibility, Threats)

Add step‑by‑step reproducibility (below), threats to validity, and multi‑CPU merge instructions.
Append step‑by‑step reproducibility (paste after References)
Below is a block you can copy‑paste under a new section “12. Artifact & Reproducibility (CPU‑Only)”. It references your existing scripts and paths.
— begin paste —

12. Artifact & Reproducibility (CPU‑Only)
Environment

Python ≥3.10; PyTorch ≥2.1 (CPU). Optional: matplotlib, pandas; perf on Linux.
Install

python install_dev.py
pip install -e .[benchmarks] (optional plots/tools)
Build

Torch scripts JIT‑build on first run: python3 paired_replacement/benchmarks/scripts/run_microbench_torch.py --help
Microbench + overlays

Autotune: see paired_replacement/benchmarks/configs/autotune_result.json
Run microbench, plot overlays (commands in Section 1)
Timing CI, sweeps, and perf

Run the commands in Section 1 to generate results_timing_ci.csv, results_sweep.csv, and perf plots (Linux).
End‑to‑end (CPU)

python3 paired_replacement/benchmarks/scripts/run_e2e_torch.py --N 65536 --hidden_dim 1024 --up_cols 1024 --m 2048 --batch 32 --steps 50 --device cpu
Figure & table inventory

overlays → benchmarks/plots/microbench_speedup_overlays.png
sweeps → benchmarks/plots/sweep_plots.png
perf → benchmarks/plots/perf_plots.png
CI table → benchmarks/data/results_timing_ci.csv
13. Threats to Validity (CPU‑Only)
Platform variance (cache sizes, memory freq, prefetchers) can shift crossovers.
Synthetic deltas approximate but do not fully capture production routing; O(k) trend remains robust.
Float32 only; half/bfloat/int paths may alter bandwidth/latency.
No GPU kernels; device‑native effects (SM occupancy, kernel overhead) not captured here.
14. Submission Checklist (Step‑by‑Step)
Regenerate figures/CSVs with commands in Section 1.
Ensure captions list hardware and parameters (N, m/N, k, dims, autotune).
Include a CI table with means ±95% at k=16/64/256 (and one larger N).
If Linux: include a perf figure; otherwise describe Instruments counters qualitatively.
Review: reproduce microbench+CI on a second CPU where possible.
Package: include a one‑command README for figure regeneration.
Submit to CPU‑focused venue (MLSys artifact, PPoPP experience, IPDPS/SC workshop, ML systems workshop).
15. Appendix A: Experimental Settings
OS/toolchain: macOS 15.6.1 (Apple M3), Python 3.12, PyTorch 2.8.0 (CPU). Linux servers for perf.
Threads: autotuned grain/threshold; recorded in CSV banners.
Memory layout: up/gate row‑major; down transposed; 64‑B aligned.
16. Appendix B: Script Index
run_microbench_torch.py (paired vs baselines), run_timing_ci.py (repeats, CI)
run_sweep.py/plot_sweep.py (parameter sweeps)
run_perf_torch.py/plot_perf.py (Linux perf)
run_e2e_torch.py (E2E top‑k layer)
run_multi_cpu_suite.py/merge_multi_cpu.py (cross‑machine suite)
— end paste —
IEEE formatting and export
Option A — Overleaf (IEEEtran):
Create an IEEEtran LaTeX project on Overleaf.
Split your Markdown into sections and paste into LaTeX (or use Pandoc to generate a .tex draft; see Option B).
Include figures (PNG/PDF) and tables; add \caption with hardware and parameters; use \bibliography with BibTeX (extract citations from your References and add more as needed).
Option B — Pandoc to IEEEtran:

Install pandoc and texlive-full (or a minimal LaTeX with IEEEtran).
Convert Markdown → IEEEtran .tex (then compile):
pandoc docs/paper_cpu_ieee.md -s -o paper.tex --from gfm --to latex --template=ieee -V documentclass=IEEEtran
pdflatex paper.tex (repeat as needed)
You may need to hand‑fix figure placements and references for camera‑ready quality.
Final polish
Abstract ~200–250 words with clear CPU‑only scope.
Background + Related Work: expand to 1.5–2.5 pages with citations.
Add error bars to at least one figure; include a CI table.
Check that every figure/table is referenced in text and has a self‑contained caption.
Want me to auto‑append the Artifact & Reproducibility/Threats/Checklist appendices into docs/paper_cpu_ieee.md for you? I can patch them directly and add figure references using your current file paths.
