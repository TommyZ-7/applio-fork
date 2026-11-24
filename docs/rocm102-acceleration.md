# ROCm 10.2 acceleration on gfx1100 (RX 7900 XTX)

Measured on 2026-11-24. CachyOS, kernel 7.3.5, Ryzen (16 threads), 46 GB RAM,
Radeon RX 7900 XTX 24 GB (`gfx1100`), Python 3.12.15.

## Result summary

| Workload | Baseline | This branch | Speedup |
|---|---|---|---|
| Inference, 30 s clip | 1.03 s | 0.55 s | **1.89x** |
| Inference, 10 s clip | 0.38 s | 0.18 s | **2.08x** |
| Inference, 3 s clip | 0.15 s | 0.08 s | **1.92x** |
| Training step, batch 8, BF16 | 633 ms (ROCm 10.1) | 265 ms (ROCm 10.2) | **2.39x** vs 10.1 / 2.19x vs FP32 on 10.2 |
| Training peak VRAM, batch 8 | 4.92 GB | 3.43 GB | -30% |
| Realtime block, 250 ms, FP16 | 52.7 ms | 26.0 ms | **2.03x** |
| Realtime block, 250 ms, FP16 + graphs | 52.7 ms | 19.2 ms | **2.74x** |

Inference quality against the FP32 baseline: **SI-SDR 51.2 dB / 51.5 dB**
(10 s / 30 s), correlation 0.999996. That is inaudible.

Realtime CUDA graphs are bit-identical to eager (max abs difference 0.0 with
model noise disabled, FP32 and FP16). With noise enabled both paths differ from
each other only because a captured graph freezes the RNG stream; repeating one
block drifts in eager mode too, so that is SOLA feedback, not the graph.

## What actually pays

1. **FP16 decoder autocast** (biggest win, ~1.75x of the total).
   `Synthesizer.infer` already had a `decoder_fp16` switch, but nothing on the
   offline path passed it, so inference always ran FP32. Autocast must wrap
   **only the vocoder**: wrapping the whole `infer()` call also puts the text
   encoder and flow in FP16, which is slower and cancels the gain
   (measured 0.99 s vs 0.59 s for 30 s audio).

2. **FP16 HuBERT autocast** (10-25% on the embedder stage). The features are
   cast back to FP32 before retrieval and interpolation.

3. **ROCm 10.2 over 10.1 for training** (1.6-1.8x, batch-size dependent).
   Kernel microbenchmarks are nearly identical between the two stacks; the win
   is concentrated in the generator/discriminator backward pass on BF16 conv1d
   (1.5 ms -> 0.9 ms per op). Inference is a wash between the two versions.

4. **BF16 for training** (2.19x over FP32, -30% VRAM). No loss scaling needed.

5. **Realtime: FP16 decoder + CUDA graphs** (2.74x together).
   The precision setting was not wired into realtime either, and
   `configuration.py` / `client.py` actively forced `decoder_fp16 = False`
   whenever the browser sent a config update, because the UI never sends that
   key. Both now leave the precision-derived value alone unless the client
   sends an explicit choice.

   CUDA graphs were gated to NVIDIA. They work on ROCm 10.2/gfx1100: 52.7 ->
   47.8 ms eager, and combined with FP16, 19.2 ms. The gate now admits ROCm,
   still opt-in via `APPLIO_ENABLE_CUDA_GRAPHS=1`. Because the client does an
   `os._exit(86)` to recover from a bad capture and nothing flips the env
   var on restart, a failure would retry forever; captures now write
   `~/.cache/applio/realtime-cuda-graphs-disabled` and the gate honours it.
   Delete that file to retry.

## What does not pay

- **`torch.compile`**: 1.01-1.06x on the decoder, +6% on top of BF16 training,
  and it needs 25-77 s of warmup. Not worth the complexity by default.
  Numerically it matches the eager path (SI-SDR 79.5 dB with noise disabled);
  the apparent difference with noise enabled is just Inductor's separate RNG
  stream for `randn_like`.
- **SDPA / Flash Attention / AOTriton**: 0-1% end to end. RVC's vocoder is
  convolution-bound, not attention-bound.
- **compile on RMVPE**: 0%. The salience net is conv/MLP-bound and small.
- **BF16 autocast for F0 (RMVPE)**: 1.1-1.3x on the pitch stage, but it changes
  the decoded contour enough (9 of 1001 coarse bins, max 1 bin) that the
  synthesised audio lands at -2.3 dB SI-SDR. Rejected: F0 is the model's
  conditioning signal, not a place to trade quality for 7% of total time.
  In realtime the F0 window is under 500 ms and takes 12.8 ms; FP16/BF16 there
  measured 13.1 / 12.0 ms, i.e. no win, so the stage is left as is.
- **HuBERT FP16 autocast in realtime**: slower. 250 ms blocks go from 52.7 ms
  to 53.9 ms because the win does not cover the per-call autocast overhead.
  Enabled for the offline path only (where it is worth 10-25% of the embedder
  stage); realtime keeps FP32 features.
- **Dilated-conv patch from upstream PR #1291**: not needed on ROCm 10.2.
  Dilated conv1d runs at the same speed as undilated here (0.92 ms vs 1.30 ms),
  so the startup probe correctly keeps the native MIOpen kernel. On ROCm 10.1
  the same probe also keeps native, and the dilated kernel is *not* 30-60x
  slower on either stack (that was measured on ROCm 7.2.4).

## Bug found: `precision: fp16` did not converge in training

The default in `assets/config_template.json` is `fp16`. With a fixed seed and
synthetic batches the generator loss stalled near 560 for 10 steps before the
GradScaler recovered, while BF16 and FP32 fell monotonically:

| step | fp32 | bf16 | fp16 |
|---|---|---|---|
| 1 | 552.1 | 546.6 | 552.2 |
| 4 | 287.8 | 286.9 | 551.0 |
| 8 | 201.9 | 201.0 | 558.7 |
| 12 | 136.3 | 134.5 | 326.5 |

`rvc/train/train.py` now maps `fp16` to BF16 on ROCm and prints why. CUDA keeps
its previous FP16 path, which was not measured here.

## Reproduce

```bash
uv venv .venv --python 3.12
uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
  "torch[device-gfx1100]==2.15.0+rocm10.2.0" "torchaudio==2.11.1+rocm10.2.0"
# torch/torchaudio are filtered out of requirements-engine.txt; see run-install.sh
```

`run-install.sh` now detects an AMD GPU via `lspci`/`amdgpu` and uses AMD's wheel
index instead of the CUDA one. Before this change the Linux path unconditionally
passed `--extra-index-url .../cu128`, so an AMD host silently installed a CUDA
wheel and failed at runtime.

ROCm 10.2 wheels ship PyTorch **2.15** only; the repo pin is 2.11.0, which has
no ROCm 10.2 build (2.13+ does). The installer overrides the pin on AMD hosts.

## Environment notes

- The system ROCm is not used: the `rocm` pip metapackage pulls the runtime,
  so CachyOS's packaged ROCm 10.1 does not conflict.
- Approximate download size: torch 218 MB + Triton 445 MB + ROCm SDK ~960 MB
  + device kernels ~340 MB.
- AMD's ROCm 10.2 hipBLASLt has no gfx1100 tuning table, so GEMM-heavy
  workloads may fall back. RVC did not regress in these measurements, but it is
  worth re-testing if a future workload becomes GEMM-bound.
