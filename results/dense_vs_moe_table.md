| | dense (control arm) | upcycled MoE, bias balancing (as planned) | upcycled MoE, bias + aux loss 0.01 (fallback) |
|---|---:|---:|---:|
| total parameters | 17,308,032 | 50,387,328 (2.91x) | 50,387,328 (2.91x) |
| active parameters per token | 17,308,032 | 17,357,184 (1.003x) | 17,357,184 (1.003x) |
| FFN width per layer, total / active | 1024 / 1024 | 4608 / 1024 | 4608 / 1024 |
| training tokens/s, median over stage 2 (A10G) | 175,987 | 86,296 (0.49x) | 88,960 (0.51x) |
| peak GPU memory allocated, GiB | 8.01 | 10.93 (1.36x) | 10.95 (1.37x) |
