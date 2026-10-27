# FA2 GQA decode query addressing: reproducibility notes

The measured increment is query input packing removal **on top of** Daniel Kristjansson's output fix (#207). Do not compare it to unpatched main and attribute both improvements to this increment.

- Baseline: `78335fb69d81fc96b5b83a615b6e177109155b99`.
- Candidate public source: `376061daa147a6ba692364d12785124190dc1b15`; identical source tree to the locally tested `b2f1a6c59f37fb9c5a13cf68a222c6d9a6803626` (`117fdb8c67cb5c5b0f083d71eb1863d23544efbc`). Public commit metadata differs because publication used the GitHub API.
- vLLM runtime: `ae3bc684e021f4b8c17c86a42e149b4482bc4ebe`, actually built native extensions, MRV2.
- CUTLASS: `62750a2b75c802660e4894434dc55e839f322277`.
- Hardware: two RTX 4090 24GB, power limits 450W; each server uses one card/TP1.
- Software: Python 3.12, PyTorch 2.14.0+cu126, Triton 3.8.0, CUDA compiler 12.9, driver 550.127.05, GCC 13.4.

## Build and select each arm

Use a separate development environment and source worktree for each arm. vLLM's CMake integration accepts `VLLM_FLASH_ATTN_SRC_DIR` for the selected source checkout. Build against the same vLLM/Torch/toolchain in both arms. The measured full native libraries retained the same CUDA object files and rebuilt only host `flash_api.cpp`; all other vLLM native libraries were identical. `FULL_EXTENSION_aligned_v2_COMPLETE.json` records the exact native-library and C++ SHA256 values. Serving process mappings were checked to confirm those libraries were loaded; experimental host-wrapper injection was not used.

The raw microbenchmark `source_commit` field identifies the shared Python checkout, **not** which C++ arm was loaded. Use the build record to identify the arm; raw JSON fields have not been rewritten.

## Public API tests and microbenchmark

In the source checkout with its matching FA2 native library selected:

```bash
python -m pytest -q -x tests/test_varlen_decode_query_strides.py
python benchmark_query_strides.py --repeats 1 --output micro.json
```

The first command passes 18 tests on the candidate. The baseline is expected to fail the new no-query-clone assertion; this is not a numerical baseline failure. Independent full-native comparisons cover 216 cases per arm and are bit-exact for output and LSE.

Run the benchmark in three independent processes per arm, reversing arm order on pass 2. It uses the public API, BF16, 32 Q/8 KV heads, head dimension 128, page size 16, heterogeneous KV lengths, preallocated output, and independent layer banks. CUDA Graph replay includes attention, split-KV combine, and any remaining packing operations. It is **not** an isolated attention-kernel timer. Batch one is a negative control: the original layout normally avoids a query materialization, so its timing variation must not be attributed to eliminating a copy.

## Unprofiled serving

Use the exact Qwen3-4B-Instruct-2507 snapshot at `cdbee75f17c01a7cc42f958dc650907174af0554`. With the selected native library:

```bash
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_NO_USAGE_STATS=1
export VLLM_USE_RUST_FRONTEND=0
export VLLM_USE_RUST_BENCH=0
VLLM_USE_V2_MODEL_RUNNER=1 VLLM_GPU_SYNC_CHECK=error vllm serve "$MODEL_SNAPSHOT" \
  --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
  --host 127.0.0.1 --port 18520 --tensor-parallel-size 1 --dtype bfloat16 \
  --max-model-len 10240 --max-num-seqs 32 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 --kv-cache-memory-bytes 6442450944 \
  --seed 42 --no-enable-prefix-caching --enable-chunked-prefill \
  --generation-config vllm --attention-backend FLASH_ATTN \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,24,32]}'

vllm bench serve --backend vllm --base-url http://127.0.0.1:18520 \
  --endpoint /v1/completions --model Qwen/Qwen3-4B-Instruct-2507 \
  --tokenizer "$MODEL_SNAPSHOT" --dataset-name random \
  --random-input-len 8192 --random-output-len 1024 --random-range-ratio 0 \
  --ignore-eos --temperature 0 --request-rate inf --seed 42 \
  --num-prompts 32 --num-warmups 0 --max-concurrency 4 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
  --save-result --save-detailed --result-dir results --result-filename B_1.json \
  --disable-tqdm
```

The measured servers are pinned to disjoint CPU sets (0-15 / 16-31), use separate compilation cache directories per arm, and select their physical GPU with CUDA_VISIBLE_DEVICES. The server also has an inactive PyTorch profiler configuration for later separate trace capture. Formal timing never starts the profiler. Complete literal argv/env/logs are retained with the raw experiment.

A/B/C/D are respectively 8192/1024/C1, 8192/1024/C4, 8192/128/C1, 8192/128/C4. Each C1 run has 16 formal requests; each C4 run has 32. Three repeats per card/arm, card roles exchanged. Initial thermal warmup is 8 long-output C1 requests; before each workload, warm up with 2*concurrency requests at that shape. Warmups and CLI sanity requests are excluded from formal counts. Exact input/output lengths and zero failures are checked for every run.

Within-card matching is used because the two cards have different performance/power characteristics. Trials are not randomized. Quantile estimates, particularly p95 with 16 requests, and small end-to-end changes should be interpreted with the observed repeat ranges.

The original round-1 NVML series was accidentally overwritten when resuming round 2. Its timing JSONs are retained as additional evidence, but **rounds 2 and 3 were selected for the primary complete timing/energy comparison before recovery results were available**. No raw timing files were replaced. This is recorded in `energy_recovery_protocol.json`.

Board energy uses 2-second NVML samples and approximate boundaries reconstructed from the CLI's rounded UTC save timestamp and duration. It excludes host energy and is not a calibrated power-meter result. Do not claim statistically resolved sub-percent energy improvements from this estimate.

## Trace evidence

The two public C4 `.trace.json.gz` files are original, unchanged PyTorch traces from a separate fixed-source round-1 execution. Decompress them to open in Perfetto. They each contain 8 decode iterations x 36 layers: baseline has 288 BF16 query-copy calls; candidate has zero; both retain 288 split-KV attention calls. The separate C32 profiles retain the FP32 LSE layout copies in both arms. `trace_copy_evidence.json` includes original file SHA256 values and distinguishes these operations. Profiler timing is not the end-to-end qualification.

## Model generation comparison

Four request lengths (17/129/513/1025), greedy 32 output tokens, are checked with the existing vLLM `check_logprobs_close` defaults. Default asynchronous serving is not guaranteed to reproduce exact text: native-vs-native controls reproduced the same punctuation near-tie in 6/10 paired calls. GPU output/LSE bitwise comparisons and their tolerances were not weakened. The default helper checks mutual top-5 membership at the first divergent token and then stops; it does not bound numeric logprob error or certify the divergent suffix. This check is not an accuracy benchmark.

AI assistance: Codex assisted with implementation, experiments and these notes. Human review remains required before marking the Draft ready.

## Final validation and artifact provenance

`alignment_controls.json` records 10 fresh-process/fresh-cache trials per build on a representative unaligned-head-stride reproducer. The guarded measured candidate passes all 10 exactly; the earlier unguarded candidate and its neutral-comment rebuild each reproduce 10 expected errors. These are candidate-regression controls, not baseline #207 failures. `baseline_cuda_code.json` proves the baseline and candidate embed identical CUDA code.

The first post-benchmark binary-inspection helper used objcopy without an explicit output ELF, which rewrote its input ELF container after all serving runs had stopped. `artifact_inspection_provenance.json` records the unchanged host .text/CUDA-code hashes, deterministic restoration of the exact original full-file SHA256 containers, and preserved historical evidence. The final 30 controls were rerun against the original restored binaries using read-only inspection with input-hash assertions. No raw benchmark JSON or original build record was edited.

Primary four-concurrency throughput changes are +0.366% (8K/1K) and +0.198% (8K/128). Single-concurrency controls are approximately unchanged. These are modest improvements. D TPOT p50 has a +1.442% point estimate and a wide paired range; neither uniform latency nor energy improvement is claimed. The additional round-1/round-2 timing comparison is separately labelled in `secondary_timing_pairs.json` and does not substitute for the primary complete timing/energy comparison.
