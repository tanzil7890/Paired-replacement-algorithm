# Cross-Machine Comparison

## Hosts

- Mohammads-MacBook-Air-2112.local_20250912_212451Z: Apple M3 | cores=8 | mem=24.0 GB

## k=16

host | cpu | speedup_idx_mean | speedup_mask_mean | paired_us_mean | index_us_mean | mask_us_mean
--- | --- | ---:| ---:| ---:| ---:| ---:
Mohammads-MacBook-Air-2112.local_20250912_212451Z | Apple M3 | 2.79 | 2.33 | 54.5 | 152.3 | 127.3

- Average speedup vs index_select across hosts: 2.79x (sd 0.00)


## k=64

host | cpu | speedup_idx_mean | speedup_mask_mean | paired_us_mean | index_us_mean | mask_us_mean
--- | --- | ---:| ---:| ---:| ---:| ---:
Mohammads-MacBook-Air-2112.local_20250912_212451Z | Apple M3 | 1.33 | 1.89 | 75.6 | 100.8 | 143.2

- Average speedup vs index_select across hosts: 1.33x (sd 0.00)

