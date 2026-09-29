# Hardware utilisation and real-world operation

GPU and CPU utilisation, GPU memory, repeated timing and batch-size-1 latency of 2D early exit (**2D EE**), layer-only early exit (**1D EE**) and the same model without early exit (**full**).

## Setup

| | |
|---|---|
| GPU | 1× NVIDIA A100-SXM4-40GB (40 GB), driver 575.57.08 – exclusive, verified idle before each run |
| CPU | AMD EPYC 7742 64-Core Processor (128 logical cores) |
| Model | Qwen2.5-3B-Instruct, fp16, 36 layers |
| Software | PyTorch 2.6.0+cu124 (CUDA 12.4), FlashInfer 0.2.5, Transformers 4.51.1, Python 3.12.3 |
| Data | MMS, SCOTUS, arXiv (test split, random sample) |
| Batched operation | batch 32, 128 documents per dataset |
| Batch size 1 | 64 documents per dataset, one request at a time |
| Timing | 1 warm-up batch, then 3 timed passes over the same documents |
| Monitoring | NVML every 100 ms (GPU utilisation, memory), psutil (process CPU, RSS), PyTorch peak memory |
| Date | 2026-09-29 |

## Measurements

### Batched operation

| dataset | mode | time / doc (ms), mean ± std | GPU util (%) | CPU (% of 1 core), mean / max | GPU memory peak (GiB) | host RSS (GiB) |
|---|---|--:|--:|--:|--:|--:|
| MMS | full | 23.6 ± 0.1 | 96 | 101 / 119 | 8.5 | 1.3 |
| MMS | 1D EE | 16.9 ± 0.1 | 96 | 100 / 118 | 8.4 | 1.3 |
| MMS | 2D EE | 18.4 ± 0.3 | 90 | 101 / 109 | 8.0 | 1.3 |
| SCOTUS | full | 697.2 ± 0.6 | 98 | 102 / 129 | 26.7 | 1.4 |
| SCOTUS | 1D EE | 321.3 ± 0.1 | 98 | 101 / 119 | 23.1 | 1.4 |
| SCOTUS | 2D EE | 143.1 ± 1.3 | 95 | 101 / 118 | 17.8 | 1.4 |
| arXiv | full | 912.4 ± 8.4 | 98 | 102 / 119 | 30.4 | 1.4 |
| arXiv | 1D EE | 416.4 ± 1.3 | 98 | 102 / 119 | 24.0 | 1.5 |
| arXiv | 2D EE | 137.8 ± 0.4 | 97 | 101 / 119 | 21.7 | 1.5 |

### Batch size 1 (single request)

| dataset | mode | latency p50 (ms) | p90 (ms) | max (ms) | GPU util (%) | CPU (% of 1 core) | GPU memory peak (GiB) |
|---|---|--:|--:|--:|--:|--:|--:|
| MMS | full | 56 | 78 | 97 | 54 | 100 | 6.0 |
| MMS | 1D EE | 41 | 52 | 74 | 55 | 100 | 6.0 |
| MMS | 2D EE | 49 | 83 | 139 | 45 | 101 | 6.0 |
| SCOTUS | full | 624 | 1549 | 2219 | 97 | 101 | 10.3 |
| SCOTUS | 1D EE | 290 | 703 | 995 | 98 | 100 | 9.9 |
| SCOTUS | 2D EE | 175 | 370 | 567 | 80 | 100 | 8.2 |
| arXiv | full | 750 | 2004 | 2228 | 97 | 101 | 10.2 |
| arXiv | 1D EE | 333 | 906 | 1008 | 97 | 101 | 9.8 |
| arXiv | 2D EE | 78 | 321 | 1401 | 82 | 100 | 9.6 |

Latency percentiles over 64 documents × 3 passes.

### Repeated timing

Time per document (ms) in each timed pass.

| phase | dataset | mode | pass 1 | pass 2 | pass 3 | std / mean |
|---|---|---|--:|--:|--:|--:|
| batched | MMS | full | 23.5 | 23.7 | 23.6 | 0.49 % |
| batched | MMS | 1D EE | 16.9 | 16.7 | 16.9 | 0.83 % |
| batched | MMS | 2D EE | 18.2 | 18.2 | 18.7 | 1.47 % |
| batched | SCOTUS | full | 696.9 | 696.8 | 697.9 | 0.09 % |
| batched | SCOTUS | 1D EE | 321.4 | 321.2 | 321.3 | 0.02 % |
| batched | SCOTUS | 2D EE | 144.6 | 142.2 | 142.5 | 0.92 % |
| batched | arXiv | full | 922.0 | 906.7 | 908.5 | 0.92 % |
| batched | arXiv | 1D EE | 417.7 | 416.4 | 415.1 | 0.31 % |
| batched | arXiv | 2D EE | 137.5 | 138.2 | 137.7 | 0.30 % |
| batch 1 | MMS | full | 57.0 | 65.9 | 60.7 | 7.29 % |
| batch 1 | MMS | 1D EE | 39.2 | 39.2 | 41.6 | 3.43 % |
| batch 1 | MMS | 2D EE | 47.4 | 49.5 | 52.9 | 5.53 % |
| batch 1 | SCOTUS | full | 703.1 | 700.3 | 704.8 | 0.33 % |
| batch 1 | SCOTUS | 1D EE | 329.1 | 326.8 | 326.4 | 0.44 % |
| batch 1 | SCOTUS | 2D EE | 196.3 | 195.5 | 196.8 | 0.32 % |
| batch 1 | arXiv | full | 905.6 | 915.0 | 906.8 | 0.56 % |
| batch 1 | arXiv | 1D EE | 416.6 | 415.2 | 413.5 | 0.37 % |
| batch 1 | arXiv | 2D EE | 168.7 | 165.7 | 165.5 | 1.07 % |

## Figures

### CPU and GPU utilisation

![CPU and GPU utilisation](figures/hw_utilization.png)

*Mean GPU utilisation (NVML) and mean CPU usage of the process during the timed passes.*

### GPU memory

![GPU memory](figures/gpu_memory.png)

*Peak GPU memory reserved by PyTorch; dashed line = model weights.*

### Time per document

![Time per document](figures/latency_ms_per_doc.png)

*Time per document, mean ± std over the timed passes.*

### Batch size 1 latency

![Batch size 1 latency](figures/batch1_latency_distribution.png)

*Batch size 1: per-request latency p50, p90 and max (log scale).*

## Metrics

- **GPU util** – NVML utilisation: share of time at least one kernel is running on the GPU, sampled every 100 ms.
- **CPU** – CPU usage of the inference process (psutil); 100 % = one core.
- **GPU memory peak** – `torch.cuda.max_memory_reserved` (weights + KV cache + activations).
- **host RSS** – peak resident memory of the process.
- **Batch size 1 latency** – wall-clock time of one request, synchronised per request.
