"""Portable FA2 varlen decode benchmark; run unchanged on each source revision."""
import argparse
import json
from pathlib import Path
import subprocess

import torch
import triton.testing

from vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batches', nargs='+', type=int, default=[1, 4, 16])
    parser.add_argument('--contexts', nargs='+', type=int, default=[128, 2048, 8192])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except subprocess.CalledProcessError:
        commit = None
    rows = []
    for batch in args.batches:
        for context in args.contexts:
            pages = (context + 15) // 16
            banks = []
            for _ in range(36 if batch == 1 else 12):
                query = torch.randn(batch, 32, 128, device='cuda', dtype=torch.bfloat16)
                key = torch.randn(batch * pages, 16, 8, 128, device='cuda', dtype=query.dtype)
                value = torch.randn_like(key)
                destination = torch.empty_like(query)
                cu_query = torch.arange(batch + 1, device='cuda', dtype=torch.int32)
                lengths = torch.tensor([max(1, context-i*7) for i in range(batch)],
                                       device='cuda', dtype=torch.int32)
                table = torch.arange(batch*pages, device='cuda', dtype=torch.int32).view(batch, pages)
                banks.append((query, key, value, destination, cu_query, lengths, table))

            def run():
                for query, key, value, destination, cu_query, lengths, table in banks:
                    flash_attn_varlen_func(
                        query, key, value, max_seqlen_q=1, cu_seqlens_q=cu_query,
                        max_seqlen_k=context, seqused_k=lengths, block_table=table,
                        causal=True, out=destination, fa_version=2,
                    )

            for repeat in range(1, args.repeats+1):
                latency = triton.testing.do_bench_cudagraph(run, rep=100) * 1000 / len(banks)
                row = dict(batch=batch, context=context, repeat=repeat, mean_us=latency,
                           layer_banks=len(banks))
                rows.append(row)
                print(json.dumps(row), flush=True)
            del banks
            torch.cuda.empty_cache()
    args.output.write_text(json.dumps(dict(
        torch=torch.__version__, gpu=torch.cuda.get_device_name(), source_commit=commit,
        shape='32 Q /8 KV heads, head_dim128, BF16, page16, heterogeneous lengths',
        timing='CUDA Graph replay; preallocated output; independent KV banks',
        rows=rows,
    ), indent=2)+'\n')


if __name__ == '__main__':
    main()
