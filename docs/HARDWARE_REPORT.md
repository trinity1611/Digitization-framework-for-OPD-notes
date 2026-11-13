# Hardware report

Generated 2026-11-13T11:23:28+00:00 by `python -m rxbench hardware-report` on host `dev-laptop`. Every value below was read from this machine at that time; nothing is assumed from a product specification.

> Development laptop on which this project was built and tested. It is NOT the target machine: it is neither an NVIDIA GB10 system nor a Slurm cluster node. The target's report is produced on the target by jobs/00_hardware_discovery.sbatch (cluster) or scripts/setup_gb10.sh (GB10).

## Summary

| item | detected |
| --- | --- |
| GPU(s) | NVIDIA GeForce RTX 4060 Laptop GPU |
| NVIDIA GB10 | no (not detected) |
| Unified memory | no |
| CPU architecture | AMD64 (x86_64) |
| CPU | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel |
| CPU cores | 16 physical / 22 logical |
| System RAM | 15.4 GB total, 3.9 GB available |
| Operating system | Windows-11-10.0.26300-SP0 |
| Kernel | 11 |
| Python | 3.13.13 |
| PyTorch | 2.6.0+cu124 |
| CUDA (PyTorch build) | 12.4 |
| CUDA usable from PyTorch | True |
| NVIDIA driver | 610.91 |
| CUDA (driver) | 13.3 |

## GB10 identification

- GB10 detected: **no** (confidence: not detected).
- Unified memory: **no**. Discrete GPU memory: nvidia-smi and torch.cuda.mem_get_info describe the GPU; system RAM is separate.
- Evidence: No GB10 evidence found (GPU name, DMI product, device tree, /etc/dgx-release).

## GPU memory

| index | name | total (nvidia-smi) | used | free | compute capability |
| --- | --- | --- | --- | --- | --- |
| 0 | NVIDIA GeForce RTX 4060 Laptop GPU | 8.0 GB | 0.0 GB | 7.8 GB | 8.9 |

| index | name (PyTorch) | total_memory | mem_get_info free / total |
| --- | --- | --- | --- |
| 0 | NVIDIA GeForce RTX 4060 Laptop GPU | 8.0 GB | 6.9 GB / 8.0 GB |

`nvidia-smi` is not used alone: on unified memory it may print `[N/A]` for memory, and the usable budget is the system's available memory shown below.

## System memory

- Total 15.4 GB, available 3.9 GB, swap 20.0 GB.

## Storage

| location | path | total | free |
| --- | --- | --- | --- |
| project | C:\path\to\Baseline-OCR\prescription_vlm_baseline | 952.9 GB | 381.8 GB |
| experiments | C:\path\to\Baseline-OCR\prescription_vlm_baseline\experiments | 952.9 GB | 381.8 GB |
| model_cache | C:\Users\<you>\.cache\huggingface\hub | 952.9 GB | 381.8 GB |

## CPU

- Intel64 Family 6 Model 170 Stepping 4, GenuineIntel (AMD64); `lscpu` not available on this system.

## NVIDIA software

- docker: Docker version 29.5.3, build d1c06ef

## Scheduler

- Not inside a Slurm job.

## Python packages

| package | version |
| --- | --- |
| torch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| transformers | 5.12.1 |
| accelerate | 1.14.0 |
| bitsandbytes | 0.49.2 |
| huggingface_hub | 1.20.1 |
| hf_xet | 1.5.1 |
| safetensors | 0.8.0 |
| tokenizers | 0.22.2 |
| pillow | 12.2.0 |
| pypdfium2 | 5.14.0 |
| numpy | 2.4.6 |
| psutil | 7.2.2 |
| PyYAML | 6.0.3 |
| streamlit | not installed |
| pandas | 3.0.3 |
| rapidfuzz | 3.14.5 |
| pytest | 9.1.1 |
| vllm | not installed |
| sglang | not installed |
| llama_cpp_python | not installed |

## Compatibility assessment

- **ok**: CPU architecture AMD64 (x86_64) is supported by PyTorch, Transformers and the pure-Python parts of this application.
- **ok**: PyTorch 2.6.0+cu124 sees CUDA (build 12.4).

## Raw nvidia-smi output

```
Fri Nov 13 16:53:25 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 610.91                 KMD Version: 610.91        CUDA UMD Version: 13.3     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4060 ...  WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   55C    P3             13W /   50W |       0MiB /   8188MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

## uname -a

```
MINGW64_NT-10.0-26300 dev-laptop 3.6.7-fb42d713.x86_64 2026-03-29 11:44 UTC x86_64 Msys
```
