"""
Disaggregated Inference Example for vLLM-Spyre

This example demonstrates how to use KV connectors with vLLM-Spyre for
disaggregated serving (prefill-decode separation) or KV cache reuse.

Three connector types are available:
  1. ExampleConnector - Disk-based debug connector (saves/loads to files)
  2. NixlConnector   - Network-based P/D disaggregation (via NIXL)
  3. OffloadingConnector - CPU memory offloading for KV reuse

Usage:
    # Example: Run with ExampleConnector (disk-based)
    python disaggregated_inference.py \
        --model ibm-granite/granite-3.3-8b-instruct \
        --connector ExampleConnector \
        --storage-path /tmp/kv_cache

    # Example: Run with OffloadingConnector
    python disaggregated_inference.py \
        --model ibm-granite/granite-3.3-8b-instruct \
        --connector OffloadingConnector

    # Example: Run as prefill instance with NIXL
    python disaggregated_inference.py \
        --model ibm-granite/granite-3.3-8b-instruct \
        --connector NixlConnector \
        --role kv_producer \
        --kv-rank 0 \
        --kv-parallel-size 2

Environment Variables:
    VLLM_SPYRE_DYNAMO_BACKEND: Set to "eager" for CPU testing, "sendnn" for AIU
    VLLM_SPYRE_USE_CB: Set to "1" to enable continuous batching (required)
"""

import argparse
import json

from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(
        description="Disaggregated inference with vLLM-Spyre KV connectors"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ibm-granite/granite-3.3-8b-instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--connector",
        type=str,
        default="ExampleConnector",
        choices=[
            "ExampleConnector",
            "NixlConnector",
            "OffloadingConnector",
        ],
        help="KV connector type",
    )
    parser.add_argument(
        "--role",
        type=str,
        default="kv_both",
        choices=["kv_producer", "kv_consumer", "kv_both"],
        help="Role for disaggregated serving (producer=prefill, consumer=decode)",
    )
    parser.add_argument(
        "--kv-rank",
        type=int,
        default=0,
        help="Rank within KV transfer group",
    )
    parser.add_argument(
        "--kv-parallel-size",
        type=int,
        default=1,
        help="Total number of instances in disaggregated setup",
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        default="/tmp/spyre_kv_cache",
        help="Storage path for ExampleConnector",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Maximum model context length",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=4,
        help="Maximum number of sequences",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate",
    )
    return parser.parse_args()


def build_kv_transfer_config(args) -> dict:
    """Build the kv_transfer_config dict from CLI arguments."""
    connector_module_map = {
        "ExampleConnector": "vllm_spyre.v1.kv_connector.example_connector",
        "NixlConnector": "vllm_spyre.v1.kv_connector.nixl_connector",
        "OffloadingConnector": "vllm_spyre.v1.kv_connector.offloading_connector",
    }

    config = {
        "kv_connector": args.connector,
        "kv_connector_module_path": connector_module_map[args.connector],
        "kv_role": args.role,
        "kv_rank": args.kv_rank,
        "kv_parallel_size": args.kv_parallel_size,
    }

    # Add connector-specific extra config
    if args.connector == "ExampleConnector":
        config["kv_connector_extra_config"] = json.dumps(
            {"shared_storage_path": args.storage_path}
        )
    elif args.connector == "OffloadingConnector":
        config["kv_connector_extra_config"] = json.dumps(
            {"max_cpu_cache_entries": 100}
        )

    return config


def main():
    args = parse_args()

    kv_config = build_kv_transfer_config(args)

    print(f"Initializing vLLM with {args.connector}...")
    print(f"  Model: {args.model}")
    print(f"  Role: {args.role}")
    print(f"  KV transfer config: {json.dumps(kv_config, indent=2)}")

    # Initialize the LLM with KV transfer config
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        kv_transfer_config=kv_config,
    )

    # Sample prompts
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about programming.",
    ]

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=args.max_tokens,
    )

    # First pass: generates and (for ExampleConnector/OffloadingConnector) caches KV
    print("\n--- First Pass (KV cache will be saved) ---")
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated = output.outputs[0].text
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated[:200]}...")

    # Second pass with same prompts: should hit KV cache
    print("\n--- Second Pass (should use cached KV) ---")
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated = output.outputs[0].text
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated[:200]}...")

    print("\nDone!")


if __name__ == "__main__":
    main()
