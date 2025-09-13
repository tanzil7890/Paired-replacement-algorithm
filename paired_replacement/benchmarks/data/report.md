# Paired Replacement Benchmark Report

## Microbench Overlays
- Data points: 2
- Min-k speedup: k=16, idx=43.27x, mask=18.03x
- Max speedup vs index_select: 43.27x at k=16

![Speedup overlays](benchmarks/microbench_speedup_overlays.png)

## Timing (mean and 95% CI)
- k=16: paired=54.2±6.2us, index=179.0±112.4us, mask=153.8±23.3us, speedup idx=3.30x, mask=2.84x
- k=64: paired=63.7±7.4us, index=92.5±12.5us, mask=122.8±13.8us, speedup idx=1.45x, mask=1.93x

