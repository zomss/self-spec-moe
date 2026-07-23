# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import json
import logging
import os
import sys
import tempfile
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    VLLM_HOST_IP: str = ""
    VLLM_PORT: int | None = None
    VLLM_RPC_BASE_PATH: str = tempfile.gettempdir()
    VLLM_USE_MODELSCOPE: bool = False
    VLLM_USE_FASTOKENS: bool = False
    VLLM_RINGBUFFER_WARNING_INTERVAL: int = 60
    VLLM_NCCL_SO_PATH: str | None = None
    LD_LIBRARY_PATH: str | None = None
    VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE: int = 256
    LOCAL_RANK: int = 0
    CUDA_VISIBLE_DEVICES: str | None = None
    VLLM_ENGINE_ITERATION_TIMEOUT_S: int = 60
    VLLM_ENGINE_READY_TIMEOUT_S: int = 600
    VLLM_API_KEY: str | None = None
    VLLM_DEBUG_LOG_API_SERVER_RESPONSE: bool = False
    S3_ACCESS_KEY_ID: str | None = None
    S3_SECRET_ACCESS_KEY: str | None = None
    S3_ENDPOINT_URL: str | None = None
    VLLM_MODEL_REDIRECT_PATH: str | None = None
    VLLM_CACHE_ROOT: str = os.path.expanduser("~/.cache/vllm")
    VLLM_CONFIG_ROOT: str = os.path.expanduser("~/.config/vllm")
    VLLM_USAGE_STATS_SERVER: str = "https://stats.vllm.ai"
    VLLM_NO_USAGE_STATS: bool = False
    VLLM_DO_NOT_TRACK: bool = False
    VLLM_USAGE_SOURCE: str = "production"
    VLLM_CONFIGURE_LOGGING: bool = True
    VLLM_LOGGING_LEVEL: str = "INFO"
    VLLM_LOGGING_PREFIX: str = ""
    VLLM_LOGGING_STREAM: str = "ext://sys.stdout"
    VLLM_LOGGING_CONFIG_PATH: str | None = None
    VLLM_LOGGING_COLOR: str = "auto"
    NO_COLOR: bool = False
    VLLM_LOG_STATS_INTERVAL: float = 10.0
    VLLM_TRACE_FUNCTION: int = 0
    VLLM_USE_FLASHINFER_SAMPLER: bool = True
    VLLM_PP_LAYER_PARTITION: str | None = None
    VLLM_CPU_KVCACHE_SPACE: int | None = 0
    VLLM_CPU_OMP_THREADS_BIND: str = "auto"
    VLLM_CPU_NUM_OF_RESERVED_CPU: int | None = None
    VLLM_CPU_SGL_KERNEL: bool = False
    VLLM_CPU_ATTN_SPLIT_KV: bool = True
    VLLM_ZENTORCH_WEIGHT_PREPACK: bool = True
    VLLM_CPU_INT4_W4A8: bool = True
    VLLM_XLA_CACHE_PATH: str = os.path.join(VLLM_CACHE_ROOT, "xla_cache")
    VLLM_XLA_CHECK_RECOMPILATION: bool = False
    VLLM_SPARSE_INDEXER_MAX_LOGITS_MB: int = 512
    VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE: Literal["auto", "nccl", "shm"] = "auto"
    VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM: bool = False
    VLLM_USE_RAY_WRAPPED_PP_COMM: bool = True
    VLLM_USE_RAY_V2_EXECUTOR_BACKEND: bool = False
    VLLM_DISTRIBUTED_USE_SPLIT_GROUP: bool = False
    VLLM_XLA_USE_SPMD: bool = False
    VLLM_WORKER_MULTIPROC_METHOD: Literal["fork", "spawn"] = "fork"
    VLLM_ASSETS_CACHE: str = os.path.join(VLLM_CACHE_ROOT, "assets")
    VLLM_ASSETS_CACHE_MODEL_CLEAN: bool = False
    VLLM_IMAGE_FETCH_TIMEOUT: int = 5
    VLLM_VIDEO_FETCH_TIMEOUT: int = 30
    VLLM_AUDIO_FETCH_TIMEOUT: int = 10
    VLLM_MEDIA_CACHE: str = ""
    VLLM_MEDIA_CACHE_MAX_SIZE_MB: int = 5120
    VLLM_MEDIA_CACHE_TTL_HOURS: float = 24
    VLLM_MEDIA_FETCH_MAX_RETRIES: int = 3
    VLLM_MEDIA_URL_ALLOW_REDIRECTS: bool = True
    VLLM_MEDIA_LOADING_THREAD_COUNT: int = 8
    VLLM_MAX_AUDIO_CLIP_FILESIZE_MB: int = 25
    VLLM_MAX_AUDIO_DECODE_DURATION_S: int = 600
    VLLM_MAX_AUDIO_PREPROCESS_WORKERS: int = max(1, min(os.cpu_count() or 1, 2))
    VLLM_VIDEO_LOADER_BACKEND: str = "opencv"
    VLLM_MEDIA_CONNECTOR: str = "http"
    VLLM_MM_HASHER_ALGORITHM: str = "blake3"
    VLLM_TARGET_DEVICE: str = "cuda"
    VLLM_MAIN_CUDA_VERSION: str = "13.0"
    VLLM_FLOAT32_MATMUL_PRECISION: Literal["highest", "high", "medium"] = "highest"
    VLLM_BATCH_INVARIANT: bool = False
    VLLM_TRITON_ATTN_USE_TD: bool | None = None
    MAX_JOBS: str | None = None
    NVCC_THREADS: str | None = None
    VLLM_USE_PRECOMPILED: bool = False
    VLLM_USE_PRECOMPILED_RUST: bool = False
    VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX: bool = False
    VLLM_DOCKER_BUILD_CONTEXT: bool = False
    VLLM_KEEP_ALIVE_ON_ENGINE_DEATH: bool = False
    CMAKE_BUILD_TYPE: Literal["Debug", "Release", "RelWithDebInfo"] | None = None
    VERBOSE: bool = False
    VLLM_ALLOW_LONG_MAX_MODEL_LEN: bool = False
    VLLM_HTTP_TIMEOUT_KEEP_ALIVE: int = 5  # seconds
    VLLM_MAX_N_SEQUENCES: int = 16384
    VLLM_PLUGINS: list[str] | None = None
    VLLM_LORA_RESOLVER_CACHE_DIR: str | None = None
    VLLM_LORA_RESOLVER_HF_REPO_LIST: str | None = None
    VLLM_USE_AOT_COMPILE: bool = False
    VLLM_USE_BYTECODE_HOOK: bool = True
    VLLM_FORCE_AOT_LOAD: bool = False
    VLLM_USE_MEGA_AOT_ARTIFACT: bool = False
    VLLM_USE_TRITON_AWQ: bool = False
    VLLM_FASTSAFETENSORS_QUEUE_SIZE: int = 0
    VLLM_TRITON_FORCE_FIRST_CONFIG: bool = False
    VLLM_ALLOW_RUNTIME_LORA_UPDATING: bool = False
    VLLM_SKIP_P2P_CHECK: bool = False
    VLLM_DISABLED_KERNELS: list[str] = []
    VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE: bool = True
    VLLM_DISABLE_PYNCCL: bool = False
    VLLM_USE_OINK_OPS: bool = False
    VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD: bool = True
    VLLM_ROCM_USE_AITER: bool = False
    VLLM_ROCM_USE_AITER_PAGED_ATTN: bool = False
    VLLM_ROCM_USE_AITER_LINEAR: bool = True
    VLLM_ROCM_USE_AITER_LINEAR_HIPBMM: bool = False
    VLLM_ROCM_USE_AITER_MOE: bool = True
    VLLM_ROCM_AITER_MOE_DISPATCH_POLICY: int = 0
    VLLM_ROCM_USE_AITER_RMSNORM: bool = True
    VLLM_ROCM_USE_AITER_MLA: bool = True
    VLLM_ROCM_USE_AITER_MHA: bool = True
    VLLM_ROCM_USE_AITER_FP4_ASM_GEMM: bool = False
    VLLM_ROCM_USE_AITER_TRITON_ROPE: bool = False
    VLLM_ROCM_USE_AITER_FP8BMM: bool = True
    VLLM_ROCM_USE_AITER_FP4BMM: bool = True
    VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION: bool = False
    VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS: bool = False
    VLLM_ROCM_USE_AITER_TRITON_GEMM: bool = True
    VLLM_ROCM_USE_SKINNY_GEMM: bool = True
    VLLM_ROCM_FP8_PADDING: bool = True
    VLLM_ROCM_MOE_PADDING: bool = True
    VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT: bool = False
    VLLM_ENABLE_V1_MULTIPROCESSING: bool = True
    VLLM_LOG_BATCHSIZE_INTERVAL: float = -1
    VLLM_DISABLE_COMPILE_CACHE: bool = False
    VLLM_USE_LAYERNAME: bool = True
    Q_SCALE_CONSTANT: int = 200
    K_SCALE_CONSTANT: int = 200
    V_SCALE_CONSTANT: int = 100
    VLLM_USE_RUST_FRONTEND: bool = False
    VLLM_RUST_FRONTEND_PATH: str | None = "auto"
    VLLM_SERVER_DEV_MODE: bool = False
    VLLM_V1_OUTPUT_PROC_CHUNK_SIZE: int = 128
    VLLM_MLA_DISABLE: bool = False
    VLLM_RAY_PER_WORKER_GPUS: float = 1.0
    VLLM_RAY_BUNDLE_INDICES: str = ""
    VLLM_CUDART_SO_PATH: str | None = None
    VLLM_DP_RANK: int = 0
    VLLM_DP_RANK_LOCAL: int = -1
    VLLM_DP_SIZE: int = 1
    VLLM_USE_STANDALONE_COMPILE: bool = True
    VLLM_ENABLE_PREGRAD_PASSES: bool = True
    VLLM_USE_BREAKABLE_CUDAGRAPH: bool = False
    VLLM_DP_MASTER_IP: str = ""
    VLLM_DP_MASTER_PORT: int = 0
    VLLM_RANDOMIZE_DP_DUMMY_INPUTS: bool = False
    VLLM_RAY_DP_PACK_STRATEGY: Literal["strict", "fill", "span"] = "strict"
    VLLM_RAY_DP_PLACEMENT_NODE_IPS: str = ""
    VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY: str = ""
    VLLM_RAY_EXTRA_ENV_VARS_TO_COPY: str = ""
    VLLM_MARLIN_USE_ATOMIC_ADD: bool = False
    VLLM_MARLIN_INPUT_DTYPE: Literal["int8", "fp8"] | None = None
    VLLM_HUMMING_ONLINE_QUANT_CONFIG: dict[str, Any] | None = None
    VLLM_HUMMING_INPUT_QUANT_CONFIG: dict[str, Any] | None = None
    VLLM_HUMMING_USE_F16_ACCUM: bool = False
    VLLM_HUMMING_MOE_GEMM_TYPE: Literal["indexed", "grouped", "auto"] | None = None
    VLLM_DEEPEPLL_NVFP4_DISPATCH: bool = False
    VLLM_V1_USE_OUTLINES_CACHE: bool = False
    VLLM_TPU_BUCKET_PADDING_GAP: int = 0
    VLLM_TPU_MOST_MODEL_LEN: int | None = None
    VLLM_TPU_USING_PATHWAYS: bool = False
    VLLM_USE_DEEP_GEMM: bool = True
    VLLM_MOE_USE_DEEP_GEMM: bool = True
    VLLM_USE_DEEP_GEMM_E8M0: bool = True
    VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES: bool = True
    VLLM_DEEP_GEMM_WARMUP: Literal[
        "skip",
        "full",
        "relax",
    ] = "relax"
    VLLM_USE_FUSED_MOE_GROUPED_TOPK: bool = True
    VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER: bool = True
    VLLM_USE_FLASHINFER_MOE_INT4: bool = False
    VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR: str | None = None
    VLLM_FLASHINFER_ALLREDUCE_BACKEND: Literal["auto", "trtllm", "mnnvl"] = "auto"
    VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE: int = 394 * 1024 * 1024
    VLLM_XGRAMMAR_CACHE_MB: int = 0
    VLLM_REGEX_COMPILATION_TIMEOUT_S: int = 5
    VLLM_MSGPACK_ZERO_COPY_THRESHOLD: int = 256
    VLLM_ALLOW_INSECURE_SERIALIZATION: bool = False
    VLLM_DISABLE_REQUEST_ID_RANDOMIZATION: bool = False
    VLLM_NIXL_SIDE_CHANNEL_HOST: str = "localhost"
    VLLM_NIXL_SIDE_CHANNEL_PORT: int = 5600
    VLLM_MOONCAKE_BOOTSTRAP_PORT: int = 8998
    VLLM_MOONCAKE_STORE_TIER_LOG: bool = False
    VLLM_MOONCAKE_DISK_STAGING_USABLE_RATIO: float = 0.9
    MOONCAKE_PREFERRED_SEGMENT: str | None = None
    MOONCAKE_REQUESTER_LOCAL_HOSTNAME: str | None = None
    VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE: int = 163840
    VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS: int = 1
    VLLM_ENFORCE_STRICT_TOOL_CALLING: bool = True
    VLLM_MQ_MAX_CHUNK_BYTES_MB: int = 16
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS: int = 300
    VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS: int = 5
    VLLM_KV_CACHE_LAYOUT: Literal["NHD", "HND"] | None = None
    VLLM_USE_PACKED_HMA_KV_CACHE: bool = False
    VLLM_SSM_CONV_STATE_LAYOUT: Literal["SD", "DS"] | None = None
    VLLM_COMPUTE_NANS_IN_LOGITS: bool = False
    VLLM_ROCM_QUICK_REDUCE_QUANTIZATION: Literal[
        "FP", "INT8", "INT6", "INT4", "NONE"
    ] = "NONE"
    VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16: bool = True
    VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB: int | None = None
    VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB: int | None = None
    VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB: int | None = None
    VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT: int = 480
    VLLM_ENABLE_CUDAGRAPH_GC: bool = False
    VLLM_LOOPBACK_IP: str = ""
    VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE: bool = True
    VLLM_ENABLE_RESPONSES_API_STORE: bool = False
    VLLM_HAS_FLASHINFER_CUBIN: bool = False
    VLLM_ROCM_FP8_MFMA_PAGE_ATTN: bool = False
    VLLM_ALLREDUCE_USE_SYMM_MEM: bool = True
    VLLM_ALLREDUCE_USE_FLASHINFER: bool = False
    VLLM_TUNED_CONFIG_FOLDER: str | None = None
    VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS: set[str] = set()
    VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT: bool = False
    VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS: bool = False
    VLLM_SYSTEM_START_DATE: str | None = None
    VLLM_TOOL_JSON_ERROR_AUTOMATIC_RETRY: bool = False
    VLLM_CUSTOM_SCOPES_FOR_PROFILING: bool = False
    VLLM_NVTX_SCOPES_FOR_PROFILING: bool = False
    VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES: bool = True
    VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME: str = "VLLM_OBJECT_STORAGE_SHM_BUFFER"
    VLLM_DEEPEP_BUFFER_SIZE_MB: int = 1024
    VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE: bool = False
    VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL: bool = False
    VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE: bool = True
    VLLM_DEEPEP_V2_PREFER_OVERLAP: bool = False
    VLLM_DEEPEP_V2_ALLOW_MULTIPLE_REDUCTION: bool = False
    VLLM_SELF_SPEC_EMULATE_A2A_DELAY_MS: float = 0.0
    VLLM_SELF_SPEC_EMULATE_A2A_DELAY_US: float = 0.0
    VLLM_SELF_SPEC_LOG_A2A_COUNTS: bool = False
    VLLM_SELF_SPEC_A2A_COUNT_ACTIVE_FILE: str = ""
    VLLM_SELF_SPEC_SKIP_A2A: bool = False
    VLLM_SELF_SPEC_LOCAL_ROUTE: bool = False
    VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE: bool = False
    VLLM_SELF_SPEC_DRAFT_RESIDENT_SETS: str = ""
    VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS: str = ""
    VLLM_SELF_SPEC_BANDIT: bool = False
    VLLM_SELF_SPEC_DRAFT_PARTIAL_REPLICA: str = ""
    VLLM_SELF_SPEC_DRAFT_TOPC: int = 0
    VLLM_SELF_SPEC_DRAFT_KV_WINDOW: int = 0
    VLLM_SELF_SPEC_DRAFT_KV_SINKS: int = 16
    VLLM_SELF_SPEC_ACCEPT_OFF_THRESHOLD: float = 0.0
    VLLM_SELF_SPEC_ACCEPT_PROBE_INTERVAL: int = 64
    VLLM_SELF_SPEC_ACCEPT_PROBE_BURST: int = 2
    VLLM_SELF_SPEC_SHORTCTX_OFF: str = ""
    VLLM_SELF_SPEC_ACCEPT_ON_THRESHOLD: float = 0.0
    VLLM_SELF_SPEC_GATE_DEBUG: bool = False
    VLLM_SELF_SPEC_ACCEPT_GATE_MIN_BATCH: int = 1
    VLLM_SELF_SPEC_ACCEPT_THRESH_LONG: str = ""
    VLLM_SELF_SPEC_POLICY_FILE: str = ""
    VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT: bool = False
    VLLM_SELF_SPEC_DRAFT_WHOLECHAIN: bool = False
    VLLM_SELF_SPEC_SHARED_KV: bool = False
    VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE: bool = False
    VLLM_SELF_SPEC_DRAFT_FULL_REPLICA: bool = False
    VLLM_SELF_SPEC_DRAFT_FULL_CG: bool = False
    VLLM_SELF_SPEC_DRAFT_FULLCG: bool = False
    VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE: bool = False
    VLLM_SELF_SPEC_DRAFT_EAGER: bool = False
    VLLM_SELF_SPEC_COMPILE_CONSISTENT: bool = False
    VLLM_SELF_SPEC_CPU_ORCH: bool = False
    VLLM_SELF_SPEC_SHADOW_CHAIN: int = 0
    VLLM_SELF_SPEC_DRAFT_GRAPH_POOL: bool = False
    VLLM_SELF_SPEC_DRAFT_WORKSPACE: bool = False
    VLLM_SELF_SPEC_AHEAD_CHAIN: bool = False
    VLLM_SELF_SPEC_CONSUME_AHEAD: bool = False
    VLLM_SELF_SPEC_DRAFT_SKIP_DP_COORD: bool = False
    VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU: bool = False
    VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD: bool = False
    VLLM_SELF_SPEC_DRAFT_NODE_LOCAL: bool = False
    VLLM_SELF_SPEC_DRAFT_AMORTIZE_DP_COORD: bool = False
    VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG: bool = False
    VLLM_SELF_SPEC_NODE_LOCAL: bool = False
    VLLM_SELF_SPEC_FAST_PARSE: bool = False
    VLLM_SELF_SPEC_PROFILE: bool = False
    VLLM_SELF_SPEC_PROFILE_FINE: bool = False
    VLLM_SELF_SPEC_MOE_NUM_DUMP: str = ""
    VLLM_SELF_SPEC_MOE_DUMP_LAYER: int = 0
    VLLM_SELF_SPEC_MOE_FP32_ACCUM: bool = False
    VLLM_DBO_COMM_SMS: int = 20
    VLLM_PATTERN_MATCH_DEBUG: str | None = None
    VLLM_DEBUG_DUMP_PATH: str | None = None
    VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE: bool = True
    VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING: bool = True
    VLLM_USE_NCCL_SYMM_MEM: bool = False
    VLLM_NCCL_INCLUDE_PATH: str | None = None
    VLLM_GC_DEBUG: str = ""
    VLLM_DEBUG_WORKSPACE: bool = False
    VLLM_DISABLE_SHARED_EXPERTS_STREAM: bool = False
    VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD: int = 256
    VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD: int = 1024
    VLLM_COMPILE_CACHE_SAVE_FORMAT: Literal["binary", "unpacked"] = "binary"
    VLLM_USE_V2_MODEL_RUNNER: bool | None = None
    VLLM_LOG_MODEL_INSPECTION: bool = False
    VLLM_DEBUG_MFU_METRICS: bool = False
    VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY: bool = False
    VLLM_WEIGHT_OFFLOADING_DISABLE_UVA: bool = False
    VLLM_WSL2_ENABLE_PIN_MEMORY: bool = False
    VLLM_DISABLE_LOG_LOGO: bool = False
    VLLM_LORA_DISABLE_PDL: bool = False
    VLLM_ENABLE_CUDA_COMPATIBILITY: bool = False
    VLLM_CUDA_COMPATIBILITY_PATH: str | None = None
    VLLM_SKIP_MODEL_NAME_VALIDATION: bool = False
    """If set, vLLM will skip model name validation in API requests.
    This allows any model name to be accepted in the 'model' field of requests,
    making the server model-name agnostic. Useful for proxy/gateway scenarios."""
    VLLM_ELASTIC_EP_SCALE_UP_LAUNCH: bool = False
    VLLM_ELASTIC_EP_DRAIN_REQUESTS: bool = False
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: bool = True
    VLLM_NIXL_EP_MAX_NUM_RANKS: int = 32
    VLLM_XPU_ENABLE_XPU_GRAPH: bool = False
    VLLM_XPU_USE_SAMPLER_KERNEL: bool = True
    VLLM_LORA_ENABLE_DUAL_STREAM: bool = False
    VLLM_GPU_NIC_PCIE_MAPPING: str = ""
    VLLM_NIC_SELECTION_VARS: str = ""
    VLLM_PREFIX_CACHE_RETENTION_INTERVAL: int | None = None


def get_default_cache_root():
    return os.getenv(
        "XDG_CACHE_HOME",
        os.path.join(os.path.expanduser("~"), ".cache"),
    )


def get_default_config_root():
    return os.getenv(
        "XDG_CONFIG_HOME",
        os.path.join(os.path.expanduser("~"), ".config"),
    )


def maybe_convert_int(value: str | None) -> int | None:
    if value is None:
        return None
    return int(value)


def maybe_convert_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return bool(int(value))


def maybe_convert_json_str_or_file(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if os.path.exists(value):
        with open(value) as f:
            return json.load(f)
    return json.loads(value)


def disable_compile_cache() -> bool:
    return bool(int(os.getenv("VLLM_DISABLE_COMPILE_CACHE", "0")))


def use_aot_compile() -> bool:
    from vllm.utils.torch_utils import is_torch_equal_or_newer

    default_value = (
        "1"
        if is_torch_equal_or_newer("2.10.0") and not disable_compile_cache()
        else "0"
    )

    return os.environ.get("VLLM_USE_AOT_COMPILE", default_value) == "1"


def use_mega_aot_artifact():
    from vllm.utils.torch_utils import is_torch_equal_or_newer

    default_value = (
        "1" if is_torch_equal_or_newer("2.12.0.dev") and use_aot_compile() else "0"
    )

    return os.environ.get("VLLM_USE_MEGA_AOT_ARTIFACT", default_value) == "1"


def env_with_choices(
    env_name: str,
    default: str | None,
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], str | None]:
    """
    Create a lambda that validates environment variable against allowed choices

    Args:
        env_name: Name of the environment variable
        default: Default value if not set (can be None)
        choices: List of valid string options or callable that returns list
        case_sensitive: Whether validation should be case sensitive

    Returns:
        Lambda function for environment_variables dict
    """

    def _get_validated_env() -> str | None:
        value = os.getenv(env_name)
        if value is None:
            return default

        # Resolve choices if it's a callable (for lazy loading)
        actual_choices = choices() if callable(choices) else choices

        if not case_sensitive:
            check_value = value.lower()
            check_choices = [choice.lower() for choice in actual_choices]
        else:
            check_value = value
            check_choices = actual_choices

        if check_value not in check_choices:
            raise ValueError(
                f"Invalid value '{value}' for {env_name}. "
                f"Valid options: {actual_choices}."
            )

        return value

    return _get_validated_env


def env_list_with_choices(
    env_name: str,
    default: list[str],
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], list[str]]:
    """
    Create a lambda that validates environment variable
    containing comma-separated values against allowed choices

    Args:
        env_name: Name of the environment variable
        default: Default list of values if not set
        choices: List of valid string options or callable that returns list
        case_sensitive: Whether validation should be case sensitive

    Returns:
        Lambda function for environment_variables
        dict that returns list of strings
    """

    def _get_validated_env_list() -> list[str]:
        value = os.getenv(env_name)
        if value is None:
            return default

        # Split comma-separated values and strip whitespace
        values = [v.strip() for v in value.split(",") if v.strip()]

        if not values:
            return default

        # Resolve choices if it's a callable (for lazy loading)
        actual_choices = choices() if callable(choices) else choices

        # Validate each value
        for val in values:
            if not case_sensitive:
                check_value = val.lower()
                check_choices = [choice.lower() for choice in actual_choices]
            else:
                check_value = val
                check_choices = actual_choices

            if check_value not in check_choices:
                raise ValueError(
                    f"Invalid value '{val}' in {env_name}. "
                    f"Valid options: {actual_choices}."
                )

        return values

    return _get_validated_env_list


def env_set_with_choices(
    env_name: str,
    default: list[str],
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], set[str]]:
    """
    Creates a lambda which that validates environment variable
    containing comma-separated values against allowed choices which
    returns choices as a set.
    """

    def _get_validated_env_set() -> set[str]:
        return set(env_list_with_choices(env_name, default, choices, case_sensitive)())

    return _get_validated_env_set


def get_vllm_port() -> int | None:
    """Get the port from VLLM_PORT environment variable.

    Returns:
        The port number as an integer if VLLM_PORT is set, None otherwise.

    Raises:
        ValueError: If VLLM_PORT is a URI, suggest k8s service discovery issue.
    """
    if "VLLM_PORT" not in os.environ:
        return None

    port = os.getenv("VLLM_PORT", "0")

    try:
        return int(port)
    except ValueError as err:
        from urllib3.util import parse_url

        parsed = parse_url(port)
        if parsed.scheme:
            raise ValueError(
                f"VLLM_PORT '{port}' appears to be a URI. "
                "This may be caused by a Kubernetes service discovery issue,"
                "check the warning in: https://docs.vllm.ai/en/latest/configuration/env_vars.html"
            ) from None
        raise ValueError(f"VLLM_PORT '{port}' must be a valid integer") from err


def get_env_or_set_default(
    env_name: str,
    default_factory: Callable[[], str],
) -> Callable[[], str]:
    """
    Create a lambda that returns an environment variable value if set,
    or generates and sets a default value using the provided factory function.
    """

    def _get_or_set_default() -> str:
        value = os.getenv(env_name)
        if value is not None:
            return value

        default_value = default_factory()
        os.environ[env_name] = default_value
        return default_value

    return _get_or_set_default


# The start-* and end* here are used by the documentation generator
# to extract the used env vars.

# --8<-- [start:env-vars-definition]

logger = logging.getLogger(__name__)


def _resolve_rust_frontend_path() -> str | None:
    """Resolve the Rust frontend binary path.

    Returns None if VLLM_USE_RUST_FRONTEND is not enabled.
    When enabled, resolves VLLM_RUST_FRONTEND_PATH ("auto" by default)
    to the actual binary path.
    """
    use_rust = bool(int(os.environ.get("VLLM_USE_RUST_FRONTEND", "0")))
    raw = os.environ.get("VLLM_RUST_FRONTEND_PATH", "auto")

    if not use_rust:
        if os.environ.get("VLLM_RUST_FRONTEND_PATH") is not None:
            logger.warning(
                "VLLM_RUST_FRONTEND_PATH is set but VLLM_USE_RUST_FRONTEND "
                "is not enabled. The Rust frontend will not be used. "
                "Set VLLM_USE_RUST_FRONTEND=1 to enable it."
            )
        return None

    if raw.lower() in ("auto", "1", "true"):
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(pkg_dir, "vllm-rs")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

        raise FileNotFoundError(
            "VLLM_RUST_FRONTEND_PATH=auto but the vllm-rs binary was "
            f"not found at {candidate}. "
            "Build with setuptools-rust or set the path explicitly."
        )
    return raw


environment_variables: dict[str, Callable[[], Any]] = {
    # ================== Installation Time Env Vars ==================
    # Target device of vLLM, supporting [cuda (by default),
    # rocm, cpu]
    "VLLM_TARGET_DEVICE": lambda: os.getenv("VLLM_TARGET_DEVICE", "cuda").lower(),
    # Main CUDA version of vLLM. This follows PyTorch but can be overridden.
    "VLLM_MAIN_CUDA_VERSION": lambda: (
        os.getenv("VLLM_MAIN_CUDA_VERSION", "").lower() or "13.0"
    ),
    # Controls PyTorch float32 matmul precision mode within vLLM workers.
    # Valid options mirror torch.set_float32_matmul_precision
    "VLLM_FLOAT32_MATMUL_PRECISION": env_with_choices(
        "VLLM_FLOAT32_MATMUL_PRECISION",
        "highest",
        ["highest", "high", "medium"],
        case_sensitive=False,
    ),
    # Enable batch-invariant mode: deterministic results regardless of
    # batch composition. Requires NVIDIA GPU with compute capability >= 9.0.
    "VLLM_BATCH_INVARIANT": lambda: bool(int(os.getenv("VLLM_BATCH_INVARIANT", "0"))),
    # Use tensor descriptors for Q/K/V loads and output stores in the
    # Triton unified-attention kernel.  Enables HW 2D block reads on
    # Intel Xe2/Xe3; the non-TD branch is dead-code-eliminated at Triton
    # compile time so other platforms see no overhead.  Tri-state override:
    # unset (default) lets the `triton_attn` backend auto-select per
    # platform (currently auto-enabled on XPU only); ``1`` forces TD on;
    # ``0`` forces TD off.  Useful for A/B benchmarking the TD path.
    "VLLM_TRITON_ATTN_USE_TD": lambda: {"1": True, "0": False}.get(
        os.getenv("VLLM_TRITON_ATTN_USE_TD", "").strip()
    ),
    # Maximum number of compilation jobs to run in parallel.
    # By default this is the number of CPUs
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    # Number of threads to use for nvcc
    # By default this is 1.
    # If set, `MAX_JOBS` will be reduced to avoid oversubscribing the CPU.
    "NVCC_THREADS": lambda: os.getenv("NVCC_THREADS", None),
    # If set, vllm will use precompiled native binaries (*.so and vllm-rs).
    "VLLM_USE_PRECOMPILED": lambda: (
        os.environ.get("VLLM_USE_PRECOMPILED", "").strip().lower() in ("1", "true")
        or bool(os.environ.get("VLLM_PRECOMPILED_WHEEL_LOCATION"))
    ),
    # If set, vllm will use the precompiled Rust frontend binary (vllm-rs).
    "VLLM_USE_PRECOMPILED_RUST": lambda: (
        os.environ.get("VLLM_USE_PRECOMPILED_RUST", "").strip().lower() in ("1", "true")
    ),
    # If set, skip adding +precompiled suffix to version string
    "VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX": lambda: bool(
        int(os.environ.get("VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX", "0"))
    ),
    # Used to mark that setup.py is running in a Docker build context,
    # in order to force the use of precompiled binaries.
    "VLLM_DOCKER_BUILD_CONTEXT": lambda: (
        os.environ.get("VLLM_DOCKER_BUILD_CONTEXT", "").strip().lower() in ("1", "true")
    ),
    # CMake build type
    # If not set, defaults to "Debug" or "RelWithDebInfo"
    # Available options: "Debug", "Release", "RelWithDebInfo"
    "CMAKE_BUILD_TYPE": env_with_choices(
        "CMAKE_BUILD_TYPE", None, ["Debug", "Release", "RelWithDebInfo"]
    ),
    # If set, vllm will print verbose logs during installation
    "VERBOSE": lambda: bool(int(os.getenv("VERBOSE", "0"))),
    # Root directory for vLLM configuration files
    # Defaults to `~/.config/vllm` unless `XDG_CONFIG_HOME` is set
    # Note that this not only affects how vllm finds its configuration files
    # during runtime, but also affects how vllm installs its configuration
    # files during **installation**.
    "VLLM_CONFIG_ROOT": lambda: os.path.expanduser(
        os.getenv(
            "VLLM_CONFIG_ROOT",
            os.path.join(get_default_config_root(), "vllm"),
        )
    ),
    # ================== Runtime Env Vars ==================
    # Root directory for vLLM cache files
    # Defaults to `~/.cache/vllm` unless `XDG_CACHE_HOME` is set
    "VLLM_CACHE_ROOT": lambda: os.path.expanduser(
        os.getenv(
            "VLLM_CACHE_ROOT",
            os.path.join(get_default_cache_root(), "vllm"),
        )
    ),
    # used in distributed environment to determine the ip address
    # of the current node, when the node has multiple network interfaces.
    # If you are using multi-node inference, you should set this differently
    # on each node.
    "VLLM_HOST_IP": lambda: os.getenv("VLLM_HOST_IP", ""),
    # used in distributed environment to manually set the communication port
    # Note: if VLLM_PORT is set, and some code asks for multiple ports, the
    # VLLM_PORT will be used as the first port, and the rest will be generated
    # by incrementing the VLLM_PORT value.
    "VLLM_PORT": get_vllm_port,
    # path used for ipc when the frontend api server is running in
    # multi-processing mode to communicate with the backend engine process.
    "VLLM_RPC_BASE_PATH": lambda: os.getenv(
        "VLLM_RPC_BASE_PATH", tempfile.gettempdir()
    ),
    # If true, will load models from ModelScope instead of Hugging Face Hub.
    # note that the value is true or false, not numbers
    "VLLM_USE_MODELSCOPE": lambda: (
        os.environ.get("VLLM_USE_MODELSCOPE", "False").lower() == "true"
    ),
    # If true, replace the Rust BPE backend that powers HF fast tokenizers
    # with the `fastokens` (https://github.com/crusoecloud/fastokens) shim.
    # Applies to any tokenizer mode that loads an HF fast tokenizer
    # (`hf`, `deepseek_v32`, `deepseek_v4`, …). The `fastokens`
    # Python package must be installed.
    "VLLM_USE_FASTOKENS": lambda: bool(int(os.getenv("VLLM_USE_FASTOKENS", "0"))),
    # Interval in seconds to log a warning message when the ring buffer is full
    "VLLM_RINGBUFFER_WARNING_INTERVAL": lambda: int(
        os.environ.get("VLLM_RINGBUFFER_WARNING_INTERVAL", "60")
    ),
    # path to cudatoolkit home directory, under which should be bin, include,
    # and lib directories.
    "CUDA_HOME": lambda: os.environ.get("CUDA_HOME", None),
    # Path to the NCCL library file. It is needed because nccl>=2.19 brought
    # by PyTorch contains a bug: https://github.com/NVIDIA/nccl/issues/1234
    "VLLM_NCCL_SO_PATH": lambda: os.environ.get("VLLM_NCCL_SO_PATH", None),
    # when `VLLM_NCCL_SO_PATH` is not set, vllm will try to find the nccl
    # library file in the locations specified by `LD_LIBRARY_PATH`
    "LD_LIBRARY_PATH": lambda: os.environ.get("LD_LIBRARY_PATH", None),
    # flag to control the chunk size (in MB) for sleeping memory allocations under ROCm
    "VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE": lambda: int(
        os.environ.get("VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE", "256")
    ),
    # Feature flag to enable/disable Inductor standalone compile.
    # In torch <= 2.7 we ignore this flag; in torch >= 2.9 this is
    # enabled by default.
    "VLLM_USE_STANDALONE_COMPILE": lambda: (
        os.environ.get("VLLM_USE_STANDALONE_COMPILE", "1") == "1"
    ),
    # Inductor's pre-grad passes don't do anything for vLLM.
    # The pre-grad passes get run even on cache-hit and negatively impact
    # vllm cold compile times by O(1s)
    # Can remove this after the following issue gets fixed
    # TODO(luka): maybe_inplace requires this
    # https://github.com/pytorch/pytorch/issues/174502
    "VLLM_ENABLE_PREGRAD_PASSES": lambda: (
        os.environ.get("VLLM_ENABLE_PREGRAD_PASSES", "1") == "1"
    ),
    # Experimental: breakable cudagraph does not rely on torch.compile
    "VLLM_USE_BREAKABLE_CUDAGRAPH": lambda: (
        os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH", "0") == "1"
    ),
    # Debug pattern matching inside custom passes.
    # Should be set to the fx.Node name (e.g. 'getitem_34' or 'scaled_mm_3').
    "VLLM_PATTERN_MATCH_DEBUG": lambda: os.environ.get(
        "VLLM_PATTERN_MATCH_DEBUG", None
    ),
    # Dump fx graphs to the given directory.
    # It will override CompilationConfig.debug_dump_path if set.
    "VLLM_DEBUG_DUMP_PATH": lambda: os.environ.get("VLLM_DEBUG_DUMP_PATH", None),
    # Feature flag to enable/disable AOT compilation. This will ensure
    # compilation is done in warmup phase and the compilation will be
    # reused in subsequent calls.
    "VLLM_USE_AOT_COMPILE": use_aot_compile,
    # Feature flag to enable/disable bytecode in
    # TorchCompileWithNoGuardsWrapper.
    "VLLM_USE_BYTECODE_HOOK": lambda: bool(
        int(os.environ.get("VLLM_USE_BYTECODE_HOOK", "1"))
    ),
    # Force vllm to always load AOT compiled models from disk. Failure
    # to load will result in a hard error when this is enabled.
    # Will be ignored when VLLM_USE_AOT_COMPILE is disabled.
    "VLLM_FORCE_AOT_LOAD": lambda: os.environ.get("VLLM_FORCE_AOT_LOAD", "0") == "1",
    # Enable loading compiled models directly from cached standalone compile artifacts
    # without re-splitting graph modules. This reduces overhead during model
    # loading by using reconstruct_serializable_fn_from_mega_artifact.
    "VLLM_USE_MEGA_AOT_ARTIFACT": use_mega_aot_artifact,
    # local rank of the process in the distributed setting, used to determine
    # the GPU device id
    "LOCAL_RANK": lambda: int(os.environ.get("LOCAL_RANK", "0")),
    # used to control the visible devices in the distributed setting
    "CUDA_VISIBLE_DEVICES": lambda: os.environ.get("CUDA_VISIBLE_DEVICES", None),
    # timeout for each iteration in the engine
    "VLLM_ENGINE_ITERATION_TIMEOUT_S": lambda: int(
        os.environ.get("VLLM_ENGINE_ITERATION_TIMEOUT_S", "60")
    ),
    # Timeout in seconds for waiting for engine cores to become ready
    # during startup. Default is 600 seconds (10 minutes).
    "VLLM_ENGINE_READY_TIMEOUT_S": lambda: int(
        os.environ.get("VLLM_ENGINE_READY_TIMEOUT_S", "600")
    ),
    # API key for vLLM API server
    "VLLM_API_KEY": lambda: os.environ.get("VLLM_API_KEY", None),
    # Whether to log responses from API Server for debugging
    "VLLM_DEBUG_LOG_API_SERVER_RESPONSE": lambda: (
        os.environ.get("VLLM_DEBUG_LOG_API_SERVER_RESPONSE", "False").lower() == "true"
    ),
    # S3 access information, used for tensorizer to load model from S3
    "S3_ACCESS_KEY_ID": lambda: os.environ.get("S3_ACCESS_KEY_ID", None),
    "S3_SECRET_ACCESS_KEY": lambda: os.environ.get("S3_SECRET_ACCESS_KEY", None),
    "S3_ENDPOINT_URL": lambda: os.environ.get("S3_ENDPOINT_URL", None),
    # Usage stats collection
    "VLLM_USAGE_STATS_SERVER": lambda: os.environ.get(
        "VLLM_USAGE_STATS_SERVER", "https://stats.vllm.ai"
    ),
    "VLLM_NO_USAGE_STATS": lambda: os.environ.get("VLLM_NO_USAGE_STATS", "0") == "1",
    "VLLM_DO_NOT_TRACK": lambda: (
        (
            os.environ.get("VLLM_DO_NOT_TRACK", None)
            or os.environ.get("DO_NOT_TRACK", None)
            or "0"
        )
        == "1"
    ),
    "VLLM_USAGE_SOURCE": lambda: os.environ.get("VLLM_USAGE_SOURCE", "production"),
    # Logging configuration
    # If set to 0, vllm will not configure logging
    # If set to 1, vllm will configure logging using the default configuration
    #    or the configuration file specified by VLLM_LOGGING_CONFIG_PATH
    "VLLM_CONFIGURE_LOGGING": lambda: bool(
        int(os.getenv("VLLM_CONFIGURE_LOGGING", "1"))
    ),
    "VLLM_LOGGING_CONFIG_PATH": lambda: os.getenv("VLLM_LOGGING_CONFIG_PATH"),
    # this is used for configuring the default logging level
    "VLLM_LOGGING_LEVEL": lambda: os.getenv("VLLM_LOGGING_LEVEL", "INFO").upper(),
    # this is used for configuring the default logging stream
    "VLLM_LOGGING_STREAM": lambda: os.getenv("VLLM_LOGGING_STREAM", "ext://sys.stdout"),
    # if set, VLLM_LOGGING_PREFIX will be prepended to all log messages
    "VLLM_LOGGING_PREFIX": lambda: os.getenv("VLLM_LOGGING_PREFIX", ""),
    # Controls colored logging output. Options: "auto" (default, colors when terminal),
    # "1" (always use colors), "0" (never use colors)
    "VLLM_LOGGING_COLOR": lambda: os.getenv("VLLM_LOGGING_COLOR", "auto"),
    # Standard unix flag for disabling ANSI color codes
    "NO_COLOR": lambda: os.getenv("NO_COLOR", "0") != "0",
    # If set, vllm will log stats at this interval in seconds
    # If not set, vllm will log stats every 10 seconds.
    "VLLM_LOG_STATS_INTERVAL": lambda: (
        val
        if (val := float(os.getenv("VLLM_LOG_STATS_INTERVAL", "10."))) > 0.0
        else 10.0
    ),
    # Trace function calls
    # If set to 1, vllm will trace function calls
    # Useful for debugging
    "VLLM_TRACE_FUNCTION": lambda: int(os.getenv("VLLM_TRACE_FUNCTION", "0")),
    # Whether to use the FlashInfer top-k / top-p sampler on CUDA. Enabled
    # by default when the hardware supports it — set to 0 to opt out
    # explicitly, which forces the PyTorch-native (Triton for bs>=8) path.
    "VLLM_USE_FLASHINFER_SAMPLER": lambda: (
        bool(int(os.environ["VLLM_USE_FLASHINFER_SAMPLER"]))
        if "VLLM_USE_FLASHINFER_SAMPLER" in os.environ
        else True
    ),
    # Pipeline stage partition strategy
    "VLLM_PP_LAYER_PARTITION": lambda: os.getenv("VLLM_PP_LAYER_PARTITION", None),
    # (CPU backend only) CPU key-value cache space.
    # default is None and will be set as 4 GB
    "VLLM_CPU_KVCACHE_SPACE": lambda: (
        int(os.getenv("VLLM_CPU_KVCACHE_SPACE", "0"))
        if "VLLM_CPU_KVCACHE_SPACE" in os.environ
        else None
    ),
    # (CPU backend only) CPU core ids bound by OpenMP threads, e.g., "0-31",
    # "0,1,2", "0-31,33". CPU cores of different ranks are separated by '|'.
    "VLLM_CPU_OMP_THREADS_BIND": lambda: os.getenv("VLLM_CPU_OMP_THREADS_BIND", "auto"),
    # (CPU backend only) CPU cores not used by OMP threads .
    # Those CPU cores will not be used by OMP threads of a rank.
    "VLLM_CPU_NUM_OF_RESERVED_CPU": lambda: (
        int(os.getenv("VLLM_CPU_NUM_OF_RESERVED_CPU", "0"))
        if "VLLM_CPU_NUM_OF_RESERVED_CPU" in os.environ
        else None
    ),
    # (CPU backend only) whether to use SGL kernels, optimized for small batch.
    "VLLM_CPU_SGL_KERNEL": lambda: bool(int(os.getenv("VLLM_CPU_SGL_KERNEL", "0"))),
    # (CPU backend only) whether to enable attention spilt KV.
    "VLLM_CPU_ATTN_SPLIT_KV": lambda: bool(
        int(os.getenv("VLLM_CPU_ATTN_SPLIT_KV", "1"))
    ),
    # (Zen CPU backend) eagerly prepack weights into ZenDNN blocked layout
    # at model load time. Eliminates per-inference layout conversion overhead.
    "VLLM_ZENTORCH_WEIGHT_PREPACK": lambda: bool(
        int(os.getenv("VLLM_ZENTORCH_WEIGHT_PREPACK", "1"))
    ),
    # (CPU backend only) whether to use SGLang INT4 W4A8 kernels for AWQ.
    "VLLM_CPU_INT4_W4A8": lambda: bool(int(os.getenv("VLLM_CPU_INT4_W4A8", "1"))),
    # If the env var is set, Ray Compiled Graph uses the specified
    # channel type to communicate between workers belonging to
    # different pipeline-parallel stages.
    # Available options:
    # - "auto": use the default channel type
    # - "nccl": use NCCL for communication
    # - "shm": use shared memory and gRPC for communication
    "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE": env_with_choices(
        "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE", "auto", ["auto", "nccl", "shm"]
    ),
    # If the env var is set, it enables GPU communication overlap
    # (experimental feature) in Ray's Compiled Graph.
    "VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM": lambda: bool(
        int(os.getenv("VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM", "0"))
    ),
    # If the env var is set, it uses a Ray Communicator wrapping
    # vLLM's pipeline parallelism communicator to interact with Ray's
    # Compiled Graph. Otherwise, it uses Ray's NCCL communicator.
    "VLLM_USE_RAY_WRAPPED_PP_COMM": lambda: bool(
        int(os.getenv("VLLM_USE_RAY_WRAPPED_PP_COMM", "1"))
    ),
    # When True and distributed_executor_backend="ray", use RayExecutorV2
    # (MQ-based) instead of RayDistributedExecutor (compiled-graph backend).
    "VLLM_USE_RAY_V2_EXECUTOR_BACKEND": lambda: bool(
        int(os.getenv("VLLM_USE_RAY_V2_EXECUTOR_BACKEND", "1"))
    ),
    # When True, GroupCoordinator constructs its CPU/device subgroups via
    # ``torch.distributed.split_group(backend=...)``
    # and ``init_distributed_environment`` initializes the default PG with
    # mixed ``cpu:gloo,cuda:nccl`` backend + eager ``device_id`` binding.
    "VLLM_DISTRIBUTED_USE_SPLIT_GROUP": lambda: bool(
        int(os.getenv("VLLM_DISTRIBUTED_USE_SPLIT_GROUP", "0"))
    ),
    # Use dedicated multiprocess context for workers.
    # Both spawn and fork work
    "VLLM_WORKER_MULTIPROC_METHOD": env_with_choices(
        "VLLM_WORKER_MULTIPROC_METHOD", "fork", ["spawn", "fork"]
    ),
    # Path to the cache for storing downloaded assets
    "VLLM_ASSETS_CACHE": lambda: os.path.expanduser(
        os.getenv(
            "VLLM_ASSETS_CACHE",
            os.path.join(get_default_cache_root(), "vllm", "assets"),
        )
    ),
    # If the env var is set, we will clean model file in
    # this path $VLLM_ASSETS_CACHE/model_streamer/$model_name
    "VLLM_ASSETS_CACHE_MODEL_CLEAN": lambda: bool(
        int(os.getenv("VLLM_ASSETS_CACHE_MODEL_CLEAN", "0"))
    ),
    # Timeout for fetching images when serving multimodal models
    # Default is 5 seconds
    "VLLM_IMAGE_FETCH_TIMEOUT": lambda: int(os.getenv("VLLM_IMAGE_FETCH_TIMEOUT", "5")),
    # Timeout for fetching videos when serving multimodal models
    # Default is 30 seconds
    "VLLM_VIDEO_FETCH_TIMEOUT": lambda: int(
        os.getenv("VLLM_VIDEO_FETCH_TIMEOUT", "30")
    ),
    # Timeout for fetching audio when serving multimodal models
    # Default is 10 seconds
    "VLLM_AUDIO_FETCH_TIMEOUT": lambda: int(
        os.getenv("VLLM_AUDIO_FETCH_TIMEOUT", "10")
    ),
    # Directory for caching media downloads (images, video, audio fetched
    # from URLs during inference). Empty string disables caching.
    "VLLM_MEDIA_CACHE": lambda: os.getenv("VLLM_MEDIA_CACHE", ""),
    # Maximum cache size in MB. When exceeded, least-recently-used entries
    # are evicted. Default is 5120 (5 GB).
    "VLLM_MEDIA_CACHE_MAX_SIZE_MB": lambda: int(
        os.getenv("VLLM_MEDIA_CACHE_MAX_SIZE_MB", "5120")
    ),
    # Time-to-live in hours for cached media files. Entries older than this
    # are evicted regardless of cache size. Default is 24 hours.
    "VLLM_MEDIA_CACHE_TTL_HOURS": lambda: float(
        os.getenv("VLLM_MEDIA_CACHE_TTL_HOURS", "24")
    ),
    # Maximum number of retries for fetching media (images, audio, video)
    # from URLs. Each retry quadruples the timeout. Default is 3.
    "VLLM_MEDIA_FETCH_MAX_RETRIES": lambda: int(
        os.getenv("VLLM_MEDIA_FETCH_MAX_RETRIES", "3")
    ),
    # Whether to allow HTTP redirects when fetching from media URLs.
    # Default to True
    "VLLM_MEDIA_URL_ALLOW_REDIRECTS": lambda: bool(
        int(os.getenv("VLLM_MEDIA_URL_ALLOW_REDIRECTS", "1"))
    ),
    # Max number of workers for the thread pool handling
    # media bytes loading. Set to 1 to disable parallel processing.
    # Default is 8
    "VLLM_MEDIA_LOADING_THREAD_COUNT": lambda: int(
        os.getenv("VLLM_MEDIA_LOADING_THREAD_COUNT", "8")
    ),
    # Maximum filesize in MB for a single audio file when processing
    # speech-to-text requests. Files larger than this will be rejected.
    # Default is 25 MB
    "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB": lambda: int(
        os.getenv("VLLM_MAX_AUDIO_CLIP_FILESIZE_MB", "25")
    ),
    # Maximum decoded audio duration in seconds.  Compressed audio files
    # (e.g. OPUS at very low bitrate) can expand into gigabytes of float32
    # PCM.  This limit is enforced *during* decoding so the memory is never
    # allocated.  Default is 600s (10 minutes).
    "VLLM_MAX_AUDIO_DECODE_DURATION_S": lambda: int(
        os.getenv("VLLM_MAX_AUDIO_DECODE_DURATION_S", "600")
    ),
    # Maximum number of worker threads used for STT preprocessing. The default
    # intentionally caps at 2 because that performed best in profiling.
    # https://github.com/vllm-project/vllm/pull/44612#issuecomment-4662757781
    "VLLM_MAX_AUDIO_PREPROCESS_WORKERS": lambda: int(
        os.getenv(
            "VLLM_MAX_AUDIO_PREPROCESS_WORKERS",
            str(max(1, min(os.cpu_count() or 1, 2))),
        )
    ),
    # Backend for Video IO — selects the frame-sampling algorithm.
    # - "opencv": uniform sampling.
    # - "opencv_dynamic": duration-aware dynamic sampling.
    # - "identity": returns raw video bytes for model processor to handle.
    #
    # Custom backend implementations can be registered
    # via `@VIDEO_LOADER_REGISTRY.register("my_custom_video_loader")` and
    # imported at runtime.
    # If a non-existing backend is used, an AssertionError will be thrown.
    "VLLM_VIDEO_LOADER_BACKEND": lambda: os.getenv(
        "VLLM_VIDEO_LOADER_BACKEND", "opencv"
    ),
    # Media connector implementation.
    # - "http": Default connector that supports fetching media via HTTP.
    #
    # Custom implementations can be registered
    # via `@MEDIA_CONNECTOR_REGISTRY.register("my_custom_media_connector")` and
    # imported at runtime.
    # If a non-existing backend is used, an AssertionError will be thrown.
    "VLLM_MEDIA_CONNECTOR": lambda: os.getenv("VLLM_MEDIA_CONNECTOR", "http"),
    # Hash algorithm for multimodal content hashing.
    # - "blake3": Default, fast cryptographic hash (not FIPS 140-3 compliant)
    # - "sha256": FIPS 140-3 compliant, widely supported
    # - "sha512": FIPS 140-3 compliant, faster on 64-bit systems
    # Use sha256 or sha512 for FIPS compliance in government/enterprise deployments
    "VLLM_MM_HASHER_ALGORITHM": env_with_choices(
        "VLLM_MM_HASHER_ALGORITHM",
        "blake3",
        ["blake3", "sha256", "sha512"],
        case_sensitive=False,
    ),
    # Path to the XLA persistent cache directory.
    # Only used for XLA devices such as TPUs.
    "VLLM_XLA_CACHE_PATH": lambda: os.path.expanduser(
        os.getenv(
            "VLLM_XLA_CACHE_PATH",
            os.path.join(get_default_cache_root(), "vllm", "xla_cache"),
        )
    ),
    # If set, assert on XLA recompilation after each execution step.
    "VLLM_XLA_CHECK_RECOMPILATION": lambda: bool(
        int(os.getenv("VLLM_XLA_CHECK_RECOMPILATION", "0"))
    ),
    # Enable SPMD mode for TPU backend.
    "VLLM_XLA_USE_SPMD": lambda: bool(int(os.getenv("VLLM_XLA_USE_SPMD", "0"))),
    # Maximum size (in MB) for logits tensor in sparse MLA indexer prefill chunks.
    # Bounds the [M, N] float32 logits tensor to prevent CUDA OOM.
    # Default: 512 MB
    "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB": lambda: int(
        os.getenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "512")
    ),
    # If set, the OpenAI API server will stay alive even after the underlying
    # AsyncLLMEngine errors and stops serving requests
    "VLLM_KEEP_ALIVE_ON_ENGINE_DEATH": lambda: bool(
        int(os.getenv("VLLM_KEEP_ALIVE_ON_ENGINE_DEATH", "0"))
    ),
    # If the env var VLLM_ALLOW_LONG_MAX_MODEL_LEN is set, it allows
    # the user to specify a max sequence length greater than
    # the max length derived from the model's config.json.
    # To enable this, set VLLM_ALLOW_LONG_MAX_MODEL_LEN=1.
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": lambda: (
        os.environ.get("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "0").strip().lower()
        in ("1", "true")
    ),
    # If set, forces FP8 Marlin to be used for FP8 quantization regardless
    # of the hardware support for FP8 compute.
    "VLLM_TEST_FORCE_FP8_MARLIN": lambda: (
        os.environ.get("VLLM_TEST_FORCE_FP8_MARLIN", "0").strip().lower()
        in ("1", "true")
    ),
    "VLLM_TEST_FORCE_LOAD_FORMAT": lambda: os.getenv(
        "VLLM_TEST_FORCE_LOAD_FORMAT", "dummy"
    ),
    # Queue size for fastsafetensors ParallelLoader pipelined weight
    # loading. Peak load-time VRAM is roughly
    # model_weights + (1 + queue_size) * shard_size.
    # Default 0 preserves the non-pipelined memory footprint so this
    # change does not shrink the loadable-model envelope. Set to 1
    # (or higher) to overlap producing the next shard's device buffer
    # with the consumer copying the current shard into model params,
    # at the cost of `queue_size` extra shard-sized buffers resident
    # at peak during loading.
    "VLLM_FASTSAFETENSORS_QUEUE_SIZE": lambda: int(
        os.getenv("VLLM_FASTSAFETENSORS_QUEUE_SIZE", "0")
    ),
    # Timeout in seconds for keeping HTTP connections alive in API server
    "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": lambda: int(
        os.environ.get("VLLM_HTTP_TIMEOUT_KEEP_ALIVE", "5")
    ),
    # Maximum allowed value for the `n` sampling parameter (number of output
    # sequences per request). Limits resource consumption to prevent
    # denial-of-service via excessively large fan-out. Default: 16384.
    "VLLM_MAX_N_SEQUENCES": lambda: int(
        os.environ.get("VLLM_MAX_N_SEQUENCES", "16384")
    ),
    # a list of plugin names to load, separated by commas.
    # if this is not set, it means all plugins will be loaded
    # if this is set to an empty string, no plugins will be loaded
    "VLLM_PLUGINS": lambda: (
        None
        if "VLLM_PLUGINS" not in os.environ
        else os.environ["VLLM_PLUGINS"].split(",")
    ),
    # Retain local sliding-window KV checkpoints for prefix caching.
    # Unset (default) preserves the dense local checkpointing behavior. `0`
    # retains only the latest completed prompt boundary. Positive values retain
    # checkpoints at the specified interval boundaries (rounded up to the
    # prefix-cache alignment).
    # Applies to sliding-window attention for now but not yet Mamba/linear attention.
    "VLLM_PREFIX_CACHE_RETENTION_INTERVAL": lambda: (
        int(os.environ["VLLM_PREFIX_CACHE_RETENTION_INTERVAL"])
        if "VLLM_PREFIX_CACHE_RETENTION_INTERVAL" in os.environ
        else None
    ),
    # a local directory to look in for unrecognized LoRA adapters.
    # only works if plugins are enabled and
    # VLLM_ALLOW_RUNTIME_LORA_UPDATING is enabled.
    "VLLM_LORA_RESOLVER_CACHE_DIR": lambda: os.getenv(
        "VLLM_LORA_RESOLVER_CACHE_DIR", None
    ),
    # A remote HF repo(s) containing one or more LoRA adapters, which
    # may be downloaded and leveraged as needed. Only works if plugins
    # are enabled and VLLM_ALLOW_RUNTIME_LORA_UPDATING is enabled.
    # Values should be comma separated.
    "VLLM_LORA_RESOLVER_HF_REPO_LIST": lambda: os.getenv(
        "VLLM_LORA_RESOLVER_HF_REPO_LIST", None
    ),
    # If set, vLLM will use Triton implementations of AWQ.
    "VLLM_USE_TRITON_AWQ": lambda: bool(int(os.getenv("VLLM_USE_TRITON_AWQ", "0"))),
    # If set, monkey-patch triton.runtime.autotuner.Autotuner.run to skip
    # benchmarking and select the first valid config (walking past invalid
    # ones). Used to eliminate autotuning variability when measuring kernel
    # performance and applied before running any kernel.
    "VLLM_TRITON_FORCE_FIRST_CONFIG": lambda: (
        os.environ.get("VLLM_TRITON_FORCE_FIRST_CONFIG", "0").strip().lower()
        in ("1", "true")
    ),
    # If set, allow loading or unloading lora adapters in runtime,
    "VLLM_ALLOW_RUNTIME_LORA_UPDATING": lambda: (
        os.environ.get("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "0").strip().lower()
        in ("1", "true")
    ),
    # We assume drivers can report p2p status correctly.
    # If the program hangs when using custom allreduce,
    # potantially caused by a bug in the driver (535 series),
    # if might be helpful to set VLLM_SKIP_P2P_CHECK=0
    # so that vLLM can verify if p2p is actually working.
    # See https://github.com/vllm-project/vllm/blob/a9b15c606fea67a072416ea0ea115261a2756058/vllm/distributed/device_communicators/custom_all_reduce_utils.py#L101-L108 for details. # noqa
    "VLLM_SKIP_P2P_CHECK": lambda: os.getenv("VLLM_SKIP_P2P_CHECK", "1") == "1",
    # List of quantization kernels that should be disabled, used for testing
    # and performance comparisons. Currently only affects MPLinearKernel
    # selection
    # (kernels: MacheteLinearKernel, MarlinLinearKernel, ExllamaLinearKernel)
    "VLLM_DISABLED_KERNELS": lambda: (
        []
        if "VLLM_DISABLED_KERNELS" not in os.environ
        else os.environ["VLLM_DISABLED_KERNELS"].split(",")
    ),
    "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE": lambda: bool(
        int(os.getenv("VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE", "1"))
    ),
    # Disable pynccl (using torch.distributed instead)
    "VLLM_DISABLE_PYNCCL": lambda: (
        os.getenv("VLLM_DISABLE_PYNCCL", "False").lower() in ("true", "1")
    ),
    # Optional: enable external Oink custom ops (e.g., Blackwell RMSNorm).
    # Disabled by default.
    "VLLM_USE_OINK_OPS": lambda: (
        os.getenv("VLLM_USE_OINK_OPS", "False").lower() in ("true", "1")
    ),
    # Disable aiter ops unless specifically enabled.
    # Acts as a parent switch to enable the rest of the other operations.
    # On hardware without a native MXFP8 kernel (e.g. ROCm gfx942 / MI300), the
    # MXFP8 emulation path dequantizes weights MXFP8->BF16 once at load time and
    # runs as a BF16 checkpoint (no per-step dequant). Set to 0 to fall back to
    # per-step dequant: keeps the 1-byte MXFP8 weights (~half the weight memory)
    # at the cost of dequantizing every forward step (much slower). Default on.
    "VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD": lambda: (
        os.getenv("VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD", "True").lower()
        in ("true", "1")
    ),
    "VLLM_ROCM_USE_AITER": lambda: (
        os.getenv("VLLM_ROCM_USE_AITER", "False").lower() in ("true", "1")
    ),
    # Whether to use aiter paged attention.
    # By default is disabled.
    "VLLM_ROCM_USE_AITER_PAGED_ATTN": lambda: (
        os.getenv("VLLM_ROCM_USE_AITER_PAGED_ATTN", "False").lower() in ("true", "1")
    ),
    # use aiter linear op if aiter ops are enabled
    # The following list of related ops
    # - scaled_mm (per-tensor / rowwise)
    # - use aiter tuned gemms for unquantized gemms
    "VLLM_ROCM_USE_AITER_LINEAR": lambda: (
        os.getenv("VLLM_ROCM_USE_AITER_LINEAR", "True").lower() in ("true", "1")
    ),
    "VLLM_ROCM_USE_AITER_LINEAR_HIPBMM": lambda: (
        os.getenv("VLLM_ROCM_USE_AITER_LINEAR_HIPBMM", "False").lower() in ("true", "1")
    ),
    # Whether to use aiter moe ops.
    # By default is enabled.
    "VLLM_ROCM_USE_AITER_MOE": lambda: (
        os.getenv("VLLM_ROCM_USE_AITER_MOE", "True").lower() in ("true", "1")
    ),
    # MoE sorting dispatch policy for AITER fused MoE kernels.
    #   0 = auto (default): single-pass for small batches, multi-pass
    #       for large batches
    #   1 = always single-pass: one kernel launch, no workspace,
    #       may be preferred for low-concurrency decode workloads
    #   2 = always multi-pass: can be faster for MoE-heavy models
    #       (e.g., +2-5% on Qwen3-Next, +1.5% on DeepSeek-V3 at TP4,
    #       see PR #39177 for benchmarks)
    "VLLM_ROCM_AITER_MOE_DISPATCH_POLICY": lambda: int(
        os.getenv("VLLM_ROCM_AITER_MOE_DISPATCH_POLICY", "0")
    ),
    # use aiter rms norm op if aiter ops are enabled.
    "VLLM_ROCM_USE_AITER_RMSNORM": lambda: (
        os.getenv("VLLM_ROCM_USE_AITER_RMSNORM", "True").lower() in ("true", "1")
    ),
    # Whether to use aiter mla ops.
    # By default is enabled.
    "VLLM_ROCM_USE_AITER_MLA": lambda: (
        os.getenv("VLLM_ROCM_USE_AITER_MLA", "True").lower() in ("true", "1")
    ),
    # Whether to use aiter mha ops.
    # By default is enabled.
    "VLLM_ROCM_USE_AITER_MHA": lambda: (
        os.getenv("VLLM_ROCM_USE_AITER_MHA", "True").lower() in ("true", "1")
    ),
    # Whether to use aiter fp4 gemm asm.
    # By default is disabled.
    "VLLM_ROCM_USE_AITER_FP4_ASM_GEMM": lambda: (
        os.getenv("VLLM_ROCM_USE_AITER_FP4_ASM_GEMM", "False").lower() in ("true", "1")
    ),
    # Whether to use aiter rope.
    # By default is disabled.
    "VLLM_ROCM_USE_AITER_TRITON_ROPE": lambda: (
        os.getenv("VLLM_ROCM_USE_AITER_TRITON_ROPE", "False").lower() in ("true", "1")
    ),
    # Whether to use aiter triton fp8 bmm kernel
    # By default is enabled.
    "VLLM_ROCM_USE_AITER_FP8BMM": lambda: (
        os.getenv("VLLM_ROCM_USE_AITER_FP8BMM", "True").lower() in ("true", "1")
    ),
    # Whether to use aiter triton fp4 bmm kernel
    # By default is enabled.
    "VLLM_ROCM_USE_AITER_FP4BMM": lambda: (
        os.getenv("VLLM_ROCM_USE_AITER_FP4BMM", "True").lower() in ("true", "1")
    ),
    # Use AITER triton unified attention for V1 attention
    "VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION": lambda: (
        os.getenv("VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION", "False").lower()
        in ("true", "1")
    ),
    # Whether to use aiter fusion shared experts ops.
    # By default is disabled.
    "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": lambda: (
        os.getenv("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS", "False").lower()
        in ("true", "1")
    ),
    # Whether to use aiter triton kernels for gemm ops.
    # By default is enabled.
    "VLLM_ROCM_USE_AITER_TRITON_GEMM": lambda: (
        os.getenv("VLLM_ROCM_USE_AITER_TRITON_GEMM", "True").lower() in ("true", "1")
    ),
    # use rocm skinny gemms
    "VLLM_ROCM_USE_SKINNY_GEMM": lambda: (
        os.getenv("VLLM_ROCM_USE_SKINNY_GEMM", "True").lower() in ("true", "1")
    ),
    # Pad the fp8 weights to 256 bytes for ROCm
    "VLLM_ROCM_FP8_PADDING": lambda: bool(int(os.getenv("VLLM_ROCM_FP8_PADDING", "1"))),
    # Pad the weights for the moe kernel
    "VLLM_ROCM_MOE_PADDING": lambda: bool(int(os.getenv("VLLM_ROCM_MOE_PADDING", "1"))),
    # Whether to use the shuffled kv cache layout
    "VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT": lambda: (
        os.getenv("VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT", "False").lower() in ("true", "1")
    ),
    # Custom quick allreduce kernel for MI3* cards
    # Choice of quantization level: FP, INT8, INT6, INT4 or NONE
    # Recommended for large models to get allreduce
    "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION": env_with_choices(
        "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION",
        "NONE",
        ["FP", "INT8", "INT6", "INT4", "NONE"],
    ),
    # Custom quick allreduce kernel for MI3* cards
    # Due to the lack of the bfloat16 asm instruction, bfloat16
    # kernels are slower than fp16,
    # If environment variable is set to 1, the input is converted to fp16
    "VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16": lambda: (
        os.getenv("VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16", "True").lower()
        in ("true", "1")
    ),
    # Custom quick allreduce kernel for MI3* cards.
    # Controls the maximum allowed number of data bytes(MB) for custom quick
    # allreduce communication.
    # Default: 2048 MB.
    # Data exceeding this size will use either custom allreduce or RCCL
    # communication.
    "VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB": lambda: maybe_convert_int(
        os.environ.get("VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB", None)
    ),
    # Custom quick allreduce kernel for MI3* cards.
    # Controls the minimum allowed number of data bytes(MB) required to use
    # custom quick allreduce communication.
    # If unset, use the built-in threshold table.
    "VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB": lambda: maybe_convert_int(
        os.environ.get("VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB", None)
    ),
    # Controls the minimum tensor size (KB, where 1 KB = 1024 bytes) required
    # to use the configured QuickReduce codec. Smaller tensors use FP
    # QuickReduce. This does not affect QuickReduce eligibility.
    "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB": lambda: maybe_convert_int(
        os.environ.get("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB", None)
    ),
    # Divisor for dynamic query scale factor calculation for FP8 KV Cache
    "Q_SCALE_CONSTANT": lambda: int(os.getenv("Q_SCALE_CONSTANT", "200")),
    # Divisor for dynamic key scale factor calculation for FP8 KV Cache
    "K_SCALE_CONSTANT": lambda: int(os.getenv("K_SCALE_CONSTANT", "200")),
    # Divisor for dynamic value scale factor calculation for FP8 KV Cache
    "V_SCALE_CONSTANT": lambda: int(os.getenv("V_SCALE_CONSTANT", "100")),
    # If set, enable multiprocessing in LLM for the V1 code path.
    "VLLM_ENABLE_V1_MULTIPROCESSING": lambda: bool(
        int(os.getenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1"))
    ),
    "VLLM_LOG_BATCHSIZE_INTERVAL": lambda: float(
        os.getenv("VLLM_LOG_BATCHSIZE_INTERVAL", "-1")
    ),
    "VLLM_DISABLE_COMPILE_CACHE": disable_compile_cache,
    # If set to "0", disable LayerName opaque type for layer_name
    # parameters in custom ops.  Defaults to enabled on torch >= 2.11.
    "VLLM_USE_LAYERNAME": lambda: bool(int(os.getenv("VLLM_USE_LAYERNAME", "1"))),
    # If set, use the Rust frontend binary instead of the Python API server
    # process(es).
    "VLLM_USE_RUST_FRONTEND": lambda: bool(
        int(os.getenv("VLLM_USE_RUST_FRONTEND", "0"))
    ),
    # Path to the Rust frontend binary. Defaults to "auto" which discovers
    # the binary installed with the vllm package. Only used when
    # VLLM_USE_RUST_FRONTEND=1.
    "VLLM_RUST_FRONTEND_PATH": lambda: _resolve_rust_frontend_path(),
    # If set, vllm will run in development mode, which will enable
    # some additional endpoints for developing and debugging,
    # e.g. `/reset_prefix_cache`
    "VLLM_SERVER_DEV_MODE": lambda: bool(int(os.getenv("VLLM_SERVER_DEV_MODE", "0"))),
    # Controls the maximum number of requests to handle in a
    # single asyncio task when processing per-token outputs in the
    # V1 AsyncLLM interface. It is applicable when handling a high
    # concurrency of streaming requests.
    # Setting this too high can result in a higher variance of
    # inter-message latencies. Setting it too low can negatively impact
    # TTFT and overall throughput.
    "VLLM_V1_OUTPUT_PROC_CHUNK_SIZE": lambda: int(
        os.getenv("VLLM_V1_OUTPUT_PROC_CHUNK_SIZE", "128")
    ),
    # If set, vLLM will disable the MLA attention optimizations.
    "VLLM_MLA_DISABLE": lambda: bool(int(os.getenv("VLLM_MLA_DISABLE", "0"))),
    # If set, vLLM will pick up the provided Flash Attention MLA
    # Number of GPUs per worker in Ray, if it is set to be a fraction,
    # it allows ray to schedule multiple actors on a single GPU,
    # so that users can colocate other actors on the same GPUs as vLLM.
    "VLLM_RAY_PER_WORKER_GPUS": lambda: float(
        os.getenv("VLLM_RAY_PER_WORKER_GPUS", "1.0")
    ),
    # Bundle indices for Ray, if it is set, it can control precisely
    # which indices are used for the Ray bundle, for every worker.
    # Format: comma-separated list of integers, e.g. "0,1,2,3"
    "VLLM_RAY_BUNDLE_INDICES": lambda: os.getenv("VLLM_RAY_BUNDLE_INDICES", ""),
    # In some system, find_loaded_library() may not work. So we allow users to
    # specify the path through environment variable VLLM_CUDART_SO_PATH.
    "VLLM_CUDART_SO_PATH": lambda: os.getenv("VLLM_CUDART_SO_PATH", None),
    # Rank of the process in the data parallel setting
    "VLLM_DP_RANK": lambda: int(os.getenv("VLLM_DP_RANK", "0")),
    # Rank of the process in the data parallel setting.
    # Defaults to VLLM_DP_RANK when not set.
    "VLLM_DP_RANK_LOCAL": lambda: int(
        os.getenv("VLLM_DP_RANK_LOCAL", sys.modules[__name__].VLLM_DP_RANK)
    ),
    # World size of the data parallel setting
    "VLLM_DP_SIZE": lambda: int(os.getenv("VLLM_DP_SIZE", "1")),
    # IP address of the master node in the data parallel setting
    "VLLM_DP_MASTER_IP": lambda: os.getenv("VLLM_DP_MASTER_IP", "127.0.0.1"),
    # Port of the master node in the data parallel setting
    "VLLM_DP_MASTER_PORT": lambda: int(os.getenv("VLLM_DP_MASTER_PORT", "0")),
    # Randomize inputs during dummy runs when using Data Parallel
    "VLLM_RANDOMIZE_DP_DUMMY_INPUTS": lambda: (
        os.environ.get("VLLM_RANDOMIZE_DP_DUMMY_INPUTS", "0") == "1"
    ),
    # Strategy to pack the data parallel ranks for Ray.
    # Available options:
    # - "fill":
    #   for DP master node, allocate exactly data-parallel-size-local DP ranks,
    #   for non-master nodes, allocate as many DP ranks as can fit;
    # - "strict":
    #   allocate exactly data-parallel-size-local DP ranks to each picked node;
    # - "span":
    #   Should be used only when a single DP rank requires multiple nodes.
    #   allocate one DP rank over as many nodes as required for set world_size;
    # This environment variable is ignored if data-parallel-backend is not Ray.
    "VLLM_RAY_DP_PACK_STRATEGY": lambda: os.getenv(
        "VLLM_RAY_DP_PACK_STRATEGY", "strict"
    ),
    # Optional comma-separated list of node IPs that Ray data-parallel
    # placement groups may use. When set, create_dp_placement_groups only
    # considers these nodes (the DP master node is always included).
    # This environment variable is ignored if data-parallel-backend is not Ray.
    "VLLM_RAY_DP_PLACEMENT_NODE_IPS": lambda: os.getenv(
        "VLLM_RAY_DP_PLACEMENT_NODE_IPS", ""
    ),
    # Comma-separated *additional* prefixes of env vars to copy from the
    # driver to Ray workers.  These are merged with the built-in defaults
    # defined in ``vllm.ray.ray_env`` (VLLM_, etc.).  Example: "MYLIB_,OTHER_"
    "VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY": lambda: os.getenv(
        "VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY", ""
    ),
    # Comma-separated *additional* individual env var names to copy from
    # the driver to Ray workers.  Merged with the built-in defaults
    # defined in ``vllm.ray.ray_env`` (PYTHONHASHSEED).
    # Example: "MY_SECRET,MY_FLAG"
    "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY": lambda: os.getenv(
        "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", ""
    ),
    # Whether to use S3 path for model loading in CI via RunAI Streamer
    "VLLM_CI_USE_S3": lambda: os.environ.get("VLLM_CI_USE_S3", "0") == "1",
    # Use model_redirect to redirect the model name to a local folder.
    # `model_redirect` can be a json file mapping the model between
    # repo_id and local folder:
    # {"meta-llama/Llama-3.2-1B": "/tmp/Llama-3.2-1B"}
    # or a space separated values table file:
    # meta-llama/Llama-3.2-1B   /tmp/Llama-3.2-1B
    "VLLM_MODEL_REDIRECT_PATH": lambda: os.environ.get(
        "VLLM_MODEL_REDIRECT_PATH", None
    ),
    # Whether to use atomicAdd reduce in gptq/awq marlin kernel.
    "VLLM_MARLIN_USE_ATOMIC_ADD": lambda: (
        os.environ.get("VLLM_MARLIN_USE_ATOMIC_ADD", "0") == "1"
    ),
    # The activation dtype for marlin kernel
    "VLLM_MARLIN_INPUT_DTYPE": env_with_choices(
        "VLLM_MARLIN_INPUT_DTYPE", None, ["int8", "fp8"]
    ),
    # The online quantization dtype for humming kernel
    "VLLM_HUMMING_ONLINE_QUANT_CONFIG": lambda: maybe_convert_json_str_or_file(
        os.environ.get("VLLM_HUMMING_ONLINE_QUANT_CONFIG", None)
    ),
    # The activation dtype config for humming kernel
    "VLLM_HUMMING_INPUT_QUANT_CONFIG": lambda: maybe_convert_json_str_or_file(
        os.environ.get("VLLM_HUMMING_INPUT_QUANT_CONFIG", None)
    ),
    # Whether to use fp16 accumulator mma
    "VLLM_HUMMING_USE_F16_ACCUM": lambda: maybe_convert_bool(
        os.environ.get("VLLM_HUMMING_USE_F16_ACCUM", "0")
    ),
    # Whether to use indexed gemm for humming moe
    # if 1, force use indexed gemm
    # if 0, force use grouped gemm
    # if None, choose better gemm type automatically
    "VLLM_HUMMING_MOE_GEMM_TYPE": lambda: os.environ.get(
        "VLLM_HUMMING_MOE_GEMM_TYPE", None
    ),
    # Whether to use DeepEPLL kernels for NVFP4 quantization and dispatch method
    # only supported on Blackwell GPUs and with
    # https://github.com/deepseek-ai/DeepEP/pull/341
    "VLLM_DEEPEPLL_NVFP4_DISPATCH": lambda: bool(
        int(os.getenv("VLLM_DEEPEPLL_NVFP4_DISPATCH", "0"))
    ),
    # Whether to turn on the outlines cache for V1
    # This cache is unbounded and on disk, so it's not safe to use in
    # an environment with potentially malicious users.
    "VLLM_V1_USE_OUTLINES_CACHE": lambda: (
        os.environ.get("VLLM_V1_USE_OUTLINES_CACHE", "0") == "1"
    ),
    # Gap between padding buckets for the forward pass. So we have
    # 8, we will run forward pass with [16, 24, 32, ...].
    "VLLM_TPU_BUCKET_PADDING_GAP": lambda: (
        int(os.environ["VLLM_TPU_BUCKET_PADDING_GAP"])
        if "VLLM_TPU_BUCKET_PADDING_GAP" in os.environ
        else 0
    ),
    "VLLM_TPU_MOST_MODEL_LEN": lambda: maybe_convert_int(
        os.environ.get("VLLM_TPU_MOST_MODEL_LEN", None)
    ),
    # Whether using Pathways
    "VLLM_TPU_USING_PATHWAYS": lambda: bool(
        "proxy" in os.getenv("JAX_PLATFORMS", "").lower()
    ),
    # Allow use of DeepGemm kernels for fused moe ops.
    "VLLM_USE_DEEP_GEMM": lambda: bool(int(os.getenv("VLLM_USE_DEEP_GEMM", "1"))),
    # Allow use of DeepGemm specifically for MoE fused ops (overrides only MoE).
    "VLLM_MOE_USE_DEEP_GEMM": lambda: bool(
        int(os.getenv("VLLM_MOE_USE_DEEP_GEMM", "1"))
    ),
    # Whether to use E8M0 scaling when DeepGEMM is used on Blackwell GPUs.
    "VLLM_USE_DEEP_GEMM_E8M0": lambda: bool(
        int(os.getenv("VLLM_USE_DEEP_GEMM_E8M0", "1"))
    ),
    # Whether to create TMA-aligned scale tensor when DeepGEMM is used.
    "VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES": lambda: bool(
        int(os.getenv("VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES", "1"))
    ),
    # DeepGemm JITs the kernels on-demand. The warmup attempts to make DeepGemm
    # JIT all the required kernels before model execution so there is no
    # JIT'ing in the hot-path. However, this warmup increases the engine
    # startup time by a couple of minutes.
    # Available options:
    #  - "skip"  : Skip warmup.
    #  - "full"  : Warmup deepgemm by running all possible gemm shapes the
    #   engine could encounter.
    #  - "relax" : Select gemm shapes to run based on some heuristics. The
    #   heuristic aims to have the same effect as running all possible gemm
    #   shapes, but provides no guarantees.
    "VLLM_DEEP_GEMM_WARMUP": env_with_choices(
        "VLLM_DEEP_GEMM_WARMUP",
        "relax",
        [
            "skip",
            "full",
            "relax",
        ],
    ),
    # Whether to use fused grouped_topk used for MoE expert selection.
    "VLLM_USE_FUSED_MOE_GROUPED_TOPK": lambda: bool(
        int(os.getenv("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "1"))
    ),
    # Allow use of FlashInfer FP8 block-scale GEMM for linear layers.
    # This uses TensorRT-LLM kernels and requires SM90+ (Hopper).
    "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER": lambda: bool(
        int(os.getenv("VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER", "1"))
    ),
    # Allow use of FlashInfer MxInt4 MoE kernels for fused moe ops.
    "VLLM_USE_FLASHINFER_MOE_INT4": lambda: bool(
        int(os.getenv("VLLM_USE_FLASHINFER_MOE_INT4", "0"))
    ),
    # Control the cache sized used by the xgrammar compiler. The default
    # of 512 MB should be enough for roughly 1000 JSON schemas.
    # It can be changed with this variable if needed for some reason.
    "VLLM_XGRAMMAR_CACHE_MB": lambda: int(os.getenv("VLLM_XGRAMMAR_CACHE_MB", "512")),
    # Maximum time in seconds allowed for regex compilation in structured
    # output backends (xgrammar, outlines). Prevents ReDoS attacks where
    # adversarial patterns cause exponential DFA state-space explosion.
    # Set to 0 to disable the timeout (not recommended in production).
    "VLLM_REGEX_COMPILATION_TIMEOUT_S": lambda: int(
        os.getenv("VLLM_REGEX_COMPILATION_TIMEOUT_S", "5")
    ),
    # Control the threshold for msgspec to use 'zero copy' for
    # serialization/deserialization of tensors. Tensors below
    # this limit will be encoded into the msgpack buffer, and
    # tensors above will instead be sent via a separate message.
    # While the sending side still actually copies the tensor
    # in all cases, on the receiving side, tensors above this
    # limit will actually be zero-copy decoded.
    "VLLM_MSGPACK_ZERO_COPY_THRESHOLD": lambda: int(
        os.getenv("VLLM_MSGPACK_ZERO_COPY_THRESHOLD", "256")
    ),
    # If set, allow insecure serialization using pickle.
    # This is useful for environments where it is deemed safe to use the
    # insecure method and it is needed for some reason.
    "VLLM_ALLOW_INSECURE_SERIALIZATION": lambda: bool(
        int(os.getenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "0"))
    ),
    # Temporary: skip adding random suffix to internal request IDs. May be
    # needed for KV connectors that match request IDs across instances.
    "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": lambda: bool(
        int(os.getenv("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "0"))
    ),
    # IP address used for NIXL handshake between remote agents.
    "VLLM_NIXL_SIDE_CHANNEL_HOST": lambda: os.getenv(
        "VLLM_NIXL_SIDE_CHANNEL_HOST", "localhost"
    ),
    # Port used for NIXL handshake between remote agents.
    "VLLM_NIXL_SIDE_CHANNEL_PORT": lambda: int(
        os.getenv("VLLM_NIXL_SIDE_CHANNEL_PORT", "5600")
    ),
    # Port used for Mooncake handshake between remote agents.
    "VLLM_MOONCAKE_BOOTSTRAP_PORT": lambda: int(
        os.getenv("VLLM_MOONCAKE_BOOTSTRAP_PORT", "8998")
    ),
    # Log per-batch memory/disk tier breakdown on external GETs.
    "VLLM_MOONCAKE_STORE_TIER_LOG": lambda: (
        os.getenv("VLLM_MOONCAKE_STORE_TIER_LOG", "False").lower() in ("true", "1")
    ),
    # Fraction of the owner's DirectIO staging buffer to fill per GET batch.
    "VLLM_MOONCAKE_DISK_STAGING_USABLE_RATIO": lambda: float(
        os.getenv("VLLM_MOONCAKE_DISK_STAGING_USABLE_RATIO", "0.9")
    ),
    # Pin this rank to a specific owner segment ("host:port").
    "MOONCAKE_PREFERRED_SEGMENT": lambda: os.getenv("MOONCAKE_PREFERRED_SEGMENT"),
    # Override the hostname the rank registers as a Mooncake requester.
    "MOONCAKE_REQUESTER_LOCAL_HOSTNAME": lambda: os.getenv(
        "MOONCAKE_REQUESTER_LOCAL_HOSTNAME"
    ),
    # Override the directory for the FlashInfer autotune config cache.
    "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": lambda: os.getenv(
        "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR", None
    ),
    # Flashinfer fused allreduce backend.
    "VLLM_FLASHINFER_ALLREDUCE_BACKEND": env_with_choices(
        "VLLM_FLASHINFER_ALLREDUCE_BACKEND",
        "auto",
        ["auto", "trtllm", "mnnvl"],
    ),
    # Control the workspace buffer size for the FlashInfer backend.
    "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE": lambda: int(
        os.getenv("VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE", str(394 * 1024 * 1024))
    ),
    # Control the maximum number of tokens per expert supported by the
    # NVFP4 MoE CUTLASS Kernel. This value is used to create a buffer for
    # the blockscale tensor of activations NVFP4 Quantization.
    # This is used to prevent the kernel from running out of memory.
    "VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE": lambda: int(
        os.getenv("VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE", "163840")
    ),
    # Specifies the thresholds of the communicated tensor sizes under which
    # vllm should use flashinfer fused allreduce. The variable should be a
    # JSON with the following format:
    #     { <world size>: <max size in mb> }
    # Unspecified world sizes will fall back to
    #     { 2: 64, 4: 1, <everything else>: 0.5 }
    "VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB": lambda: json.loads(
        os.getenv("VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB", "{}")
    ),
    # MoE routing strategy selector.
    # See `RoutingSimulator.get_available_strategies()` # for available
    # strategies.
    # Custom routing strategies can be registered by
    # RoutingSimulator.register_strategy()
    # Note: custom strategies may not produce correct model outputs
    "VLLM_MOE_ROUTING_SIMULATION_STRATEGY": lambda: os.environ.get(
        "VLLM_MOE_ROUTING_SIMULATION_STRATEGY", ""
    ).lower(),
    # Regex timeout for use by the vLLM tool parsing plugins.
    "VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS": lambda: int(
        os.getenv("VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS", "1")
    ),
    # Enforce function parameter schemas in structural-tag based tool calling.
    "VLLM_ENFORCE_STRICT_TOOL_CALLING": lambda: os.getenv(
        "VLLM_ENFORCE_STRICT_TOOL_CALLING", "True"
    ).lower()
    in ("true", "1"),
    # Control the max chunk bytes (in MB) for the rpc message queue.
    # Object larger than this threshold will be broadcast to worker
    # processes via zmq.
    "VLLM_MQ_MAX_CHUNK_BYTES_MB": lambda: int(
        os.getenv("VLLM_MQ_MAX_CHUNK_BYTES_MB", "16")
    ),
    # Timeout in seconds for execute_model RPC calls in multiprocessing
    # executor (only applies when TP > 1).
    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": lambda: int(
        os.getenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "300")
    ),
    # Timeout in seconds for engine and worker process shutdown
    "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS": lambda: int(
        os.getenv("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", "5")
    ),
    # KV Cache layout used throughout vllm.
    # Some common values are:
    # - NHD
    # - HND
    # Where N=num_blocks, H=num_heads and D=head_size. The default value will
    # leave the layout choice to the backend. Mind that backends may only
    # implement and support a subset of all possible layouts.
    "VLLM_KV_CACHE_LAYOUT": env_with_choices(
        "VLLM_KV_CACHE_LAYOUT", None, ["NHD", "HND"]
    ),
    # Opt into packed per-block KV cache allocation for multi-group
    # attention-only HMA models (e.g. gpt-oss, Gemma 3/4).
    "VLLM_USE_PACKED_HMA_KV_CACHE": lambda: bool(
        int(os.getenv("VLLM_USE_PACKED_HMA_KV_CACHE", "0"))
    ),
    # SSM conv state layout used for Mamba models.
    # - SD: (state_len, dim) — dim contiguous (default)
    # - DS: (dim, state_len) — TP-sharded dim on dim1,
    #   consistent with SSM temporal state and HND KV cache layout.
    "VLLM_SSM_CONV_STATE_LAYOUT": env_with_choices(
        "VLLM_SSM_CONV_STATE_LAYOUT", None, ["SD", "DS"]
    ),
    # Enable checking whether the generated logits contain NaNs,
    # indicating corrupted output. Useful for debugging low level bugs
    # or bad hardware but it may add compute overhead.
    "VLLM_COMPUTE_NANS_IN_LOGITS": lambda: bool(
        int(os.getenv("VLLM_COMPUTE_NANS_IN_LOGITS", "0"))
    ),
    # Timeout (in seconds) for MooncakeConnector in PD disaggregated setup.
    "VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT": lambda: int(
        os.getenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "480")
    ),
    # If set, it means we pre-downloaded cubin files and flashinfer will
    # read the cubin files directly.
    "VLLM_HAS_FLASHINFER_CUBIN": lambda: bool(
        int(os.getenv("VLLM_HAS_FLASHINFER_CUBIN", "0"))
    ),
    # Controls garbage collection during CUDA graph capture.
    # If set to 0 (default), enables GC freezing to speed up capture time.
    # If set to 1, allows GC to run during capture.
    "VLLM_ENABLE_CUDAGRAPH_GC": lambda: bool(
        int(os.getenv("VLLM_ENABLE_CUDAGRAPH_GC", "0"))
    ),
    # Used to force set up loopback IP
    "VLLM_LOOPBACK_IP": lambda: os.getenv("VLLM_LOOPBACK_IP", ""),
    # Used to set the process name prefix for vLLM processes.
    # This is useful for debugging and monitoring purposes.
    # The default value is "VLLM".
    "VLLM_PROCESS_NAME_PREFIX": lambda: os.getenv("VLLM_PROCESS_NAME_PREFIX", "VLLM"),
    # Allow chunked local attention with hybrid kv cache manager.
    # Currently using the Hybrid KV cache manager with chunked local attention
    # in the Llama4 models (the only models currently using chunked local attn)
    # causes a latency regression. For this reason, we disable it by default.
    # This flag is used to allow users to enable it if they want to (to save on
    # kv-cache memory usage and enable longer contexts)
    # TODO(lucas): Remove this flag once latency regression is resolved.
    "VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE": lambda: bool(
        int(os.getenv("VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE", "1"))
    ),
    # Enables support for the "store" option in the OpenAI Responses API.
    # When set to 1, vLLM's OpenAI server will retain the input and output
    # messages for those requests in memory. By default, this is disabled (0),
    # and the "store" option is ignored.
    # NOTE/WARNING:
    # 1. Messages are kept in memory only (not persisted to disk) and will be
    #    lost when the vLLM server shuts down.
    # 2. Enabling this option will cause a memory leak, as stored messages are
    #    never removed from memory until the server terminates.
    "VLLM_ENABLE_RESPONSES_API_STORE": lambda: bool(
        int(os.getenv("VLLM_ENABLE_RESPONSES_API_STORE", "0"))
    ),
    # If set, use the fp8 mfma in rocm paged attention.
    "VLLM_ROCM_FP8_MFMA_PAGE_ATTN": lambda: bool(
        int(os.getenv("VLLM_ROCM_FP8_MFMA_PAGE_ATTN", "0"))
    ),
    # Whether to use pytorch symmetric memory for allreduce
    "VLLM_ALLREDUCE_USE_SYMM_MEM": lambda: bool(
        int(os.getenv("VLLM_ALLREDUCE_USE_SYMM_MEM", "1"))
    ),
    # Whether to use FlashInfer allreduce
    "VLLM_ALLREDUCE_USE_FLASHINFER": lambda: bool(
        int(os.getenv("VLLM_ALLREDUCE_USE_FLASHINFER", "0"))
    ),
    # Experimental: use this to enable MCP tool calling for non harmony models
    "VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT": lambda: bool(
        int(os.getenv("VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT", "0"))
    ),
    # User override folder for tuned Triton-kernel configs. Shared by MoE,
    # Mamba SSU, and LoRA. Filenames are distinct so one folder can hold all.
    # Each component first checks this folder, then the configs shipped with
    # vLLM (if any). If no JSON matches, it uses a hard-coded heuristic.
    "VLLM_TUNED_CONFIG_FOLDER": lambda: os.getenv("VLLM_TUNED_CONFIG_FOLDER", None),
    # Valid values are container,code_interpreter,web_search_preview
    # ex VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS=container,code_interpreter
    # If the server_label of your mcp tool is not in this list it will
    # be completely ignored.
    "VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS": env_set_with_choices(
        "VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS",
        default=[],
        choices=["container", "code_interpreter", "web_search_preview"],
    ),
    # Allows harmony instructions to be injected on system messages
    "VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS": lambda: bool(
        int(os.getenv("VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS", "0"))
    ),
    # Pin the conversation start date injected into the Harmony system
    # message. When unset the current date is used, which introduces
    # non-determinism (different tokens -> different model behaviour at
    # temperature=0). Set to an ISO date string, e.g. "2023-09-12",
    # for reproducible inference or testing.
    "VLLM_SYSTEM_START_DATE": lambda: os.getenv("VLLM_SYSTEM_START_DATE", None),
    # Enable automatic retry when tool call JSON parsing fails
    # If enabled, returns an error message to the model to retry
    # If disabled (default), raises an exception and fails the request
    "VLLM_TOOL_JSON_ERROR_AUTOMATIC_RETRY": lambda: bool(
        int(os.getenv("VLLM_TOOL_JSON_ERROR_AUTOMATIC_RETRY", "0"))
    ),
    # Add optional custom scopes for profiling, disable to avoid overheads
    "VLLM_CUSTOM_SCOPES_FOR_PROFILING": lambda: bool(
        int(os.getenv("VLLM_CUSTOM_SCOPES_FOR_PROFILING", "0"))
    ),
    # Add optional nvtx scopes for profiling, disable to avoid overheads
    "VLLM_NVTX_SCOPES_FOR_PROFILING": lambda: bool(
        int(os.getenv("VLLM_NVTX_SCOPES_FOR_PROFILING", "0"))
    ),
    # Represent block hashes in KV cache events as 64-bit integers instead of
    # raw bytes. Defaults to True for backward compatibility.
    "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES": lambda: bool(
        int(os.getenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", "1"))
    ),
    # Name of the shared memory buffer used for object storage.
    # Only effective when mm_config.mm_processor_cache_type == "shm".
    # Automatically generates a unique UUID-based name per process tree
    # if not explicitly set.
    "VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME": get_env_or_set_default(
        "VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME",
        lambda: f"VLLM_OBJECT_STORAGE_SHM_BUFFER_{uuid.uuid4().hex}",
    ),
    # The size in MB of the buffers (NVL and RDMA) used by DeepEP
    "VLLM_DEEPEP_BUFFER_SIZE_MB": lambda: int(
        os.getenv("VLLM_DEEPEP_BUFFER_SIZE_MB", "1024")
    ),
    # Force DeepEP to use intranode kernel for inter-node communication in
    # high throughput mode. This is useful archive higher prefill throughput
    # on system supports multi-node nvlink (e.g GB200).
    "VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE": lambda: bool(
        int(os.getenv("VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE", "0"))
    ),
    # Allow DeepEP to use MNNVL (multi-node nvlink) for internode_ll kernel,
    # turn this for better latency on GB200 like system
    "VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL": lambda: bool(
        int(os.getenv("VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL", "0"))
    ),
    # DeepEP v2: enable two-tier NVLink+RDMA hybrid mode
    "VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE": lambda: bool(
        int(os.getenv("VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE", "0"))
    ),
    # DeepEP v2: use fewer SMs at slight throughput cost
    "VLLM_DEEPEP_V2_PREFER_OVERLAP": lambda: bool(
        int(os.getenv("VLLM_DEEPEP_V2_PREFER_OVERLAP", "0"))
    ),
    # DeepEP v2: trade precision for transfer size in combine
    "VLLM_DEEPEP_V2_ALLOW_MULTIPLE_REDUCTION": lambda: bool(
        int(os.getenv("VLLM_DEEPEP_V2_ALLOW_MULTIPLE_REDUCTION", "0"))
    ),
    # Research-only: inject host-side delay after MoE EP allgather/reducescatter
    # to emulate exposed multi-node communication.
    "VLLM_SELF_SPEC_EMULATE_A2A_DELAY_MS": lambda: float(
        os.getenv("VLLM_SELF_SPEC_EMULATE_A2A_DELAY_MS", "0")
    ),
    # Research-only: per-collective exposed-A2A delay in microseconds, injected
    # on the GPU stream (torch.cuda._sleep) so it is captured into CUDA graphs
    # and replays every decode step. Use this (not the MS host-sleep) to emulate
    # inter-node all-to-all latency under graph mode. MS and US are additive.
    "VLLM_SELF_SPEC_EMULATE_A2A_DELAY_US": lambda: float(
        os.getenv("VLLM_SELF_SPEC_EMULATE_A2A_DELAY_US", "0")
    ),
    # Research-only: log AgRs MoE EP allgather/reducescatter call counts.
    "VLLM_SELF_SPEC_LOG_A2A_COUNTS": lambda: bool(
        int(os.getenv("VLLM_SELF_SPEC_LOG_A2A_COUNTS", "0"))
    ),
    "VLLM_SELF_SPEC_A2A_COUNT_ACTIVE_FILE": lambda: os.getenv(
        "VLLM_SELF_SPEC_A2A_COUNT_ACTIVE_FILE", ""
    ),
    # Research-only: skip the AgRs MoE EP allgather/reducescatter collective and
    # replace it with a shape-preserving local op (no cross-rank comm). For the
    # comm-free local-routing DRAFT step -- TIMING ONLY (dummy weights); produces
    # incorrect values, so do not use for correctness. Default off.
    "VLLM_SELF_SPEC_SKIP_A2A": lambda: bool(
        int(os.getenv("VLLM_SELF_SPEC_SKIP_A2A", "0"))
    ),
    # Research-only: CORRECT comm-free local-routing MoE forward. When set, each
    # rank routes its LOCAL tokens to only its RESIDENT experts (the EP shard,
    # via expert_map) and skips BOTH the AgRs dispatch (allgather) and combine
    # (reducescatter). Unlike VLLM_SELF_SPEC_SKIP_A2A (timing-only, tiled
    # garbage), this masks the router to the resident experts (skip-cold) so the
    # output is the correct local-routing result. At resident=all-experts this
    # is identical to normal full routing. The flag can also be driven per
    # forward via ForwardContext.additional_kwargs["self_spec_local_route"]
    # (this env is the default when that key is absent). Default off.
    "VLLM_SELF_SPEC_LOCAL_ROUTE": lambda: bool(
        int(os.getenv("VLLM_SELF_SPEC_LOCAL_ROUTE", "0"))
    ),
    # Self-spec W0 PRODUCER opt-in: when set, the V1 draft_model proposer drives
    # the comm-free local-routing flag per DRAFT forward (via
    # ForwardContext.additional_kwargs["self_spec_local_route"]=True), so the
    # draft step routes to resident experts and skips the AgRs collectives while
    # the VERIFY forward (a separate forward context, no key) stays full-EP. This
    # is intentionally distinct from VLLM_SELF_SPEC_LOCAL_ROUTE (the reader's
    # env fallback): keep that one at 0 so the verify does NOT inherit local
    # routing. Default off -> no change to default draft_model behavior.
    "VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE": lambda: bool(
        int(os.getenv("VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE", "0"))
    ),
    # Self-spec W7 top-C draft prune (W2b probe): when > 0 AND the comm-free
    # draft flag is active (the self_spec_local_route path), the DRAFT MoE keeps
    # only the C highest-gate-weight experts per token (C < top_k), renormalizes
    # those C to sum to 1, and zeros the rest -> only C experts are computed per
    # token in the draft. The VERIFY forward (no draft flag) keeps full top_k, so
    # the draft stays lossless regardless of C (verify corrects). Isolates the
    # accept/compute tradeoff of a "smart" prune WITHOUT building the bounded
    # expert cache (experts stay resident). 0 -> off (full top_k). Only affects
    # the modular-kernel draft path (select_experts). Default off.
    # Phase 83: path to a torch.load-able {layer_idx: LongTensor expert ids}
    # (or list) of frequency-profiled RESIDENT expert sets for the self-spec
    # draft's local routing. Requires the draft full replica (expert_map None);
    # masks draft routing to the per-layer set instead of the EP shard.
    "VLLM_SELF_SPEC_DRAFT_RESIDENT_SETS": lambda: os.getenv(
        "VLLM_SELF_SPEC_DRAFT_RESIDENT_SETS", ""
    ),
    # Phase 89: skip-set draft (KnapSpec-style layer skip in the self-spec
    # framework). Comma list of ORIGINAL decoder-layer indices to skip in
    # the draft (e.g. "2,4,7,16"); indices are preserved so shared-KV
    # name-binding maps remaining layers to their target twins.
    "VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS": lambda: os.getenv(
        "VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS", ""
    ),
    # Phase 91: Thompson-sampling scheduler policy (stage-3 bandit).
    "VLLM_SELF_SPEC_BANDIT": lambda: os.getenv(
        "VLLM_SELF_SPEC_BANDIT", "0") == "1",
    # Phase 83: draft PARTIAL replica -- load ONLY the per-layer resident
    # expert sets ({layer_idx: LongTensor} file) into the draft's replica
    # (requires VLLM_SELF_SPEC_DRAFT_FULL_REPLICA=1). bf16, comm-free,
    # ~frac of expert bytes; routing masks via the layer's own expert_map.
    "VLLM_SELF_SPEC_DRAFT_PARTIAL_REPLICA": lambda: os.getenv(
        "VLLM_SELF_SPEC_DRAFT_PARTIAL_REPLICA", ""
    ),
    "VLLM_SELF_SPEC_DRAFT_TOPC": lambda: int(
        os.getenv("VLLM_SELF_SPEC_DRAFT_TOPC", "0")
    ),
    # Self-spec W7 window-KV draft (Phase 62, StreamingLLM-style): when > 0,
    # DRAFT forward steps attend only over the attention-sink pages (the first
    # ceil(KV_SINKS/block_size) blocks) plus the trailing pages covering ~this
    # many tokens. Per draft step, each request's block-table row is compacted
    # into a proposer-owned buffer and the draft-side kv seq_len shrunk to the
    # token count actually present in the kept pages. Paged KV entries carry
    # their true RoPE'd positions and FA aligns the causal mask at the END of
    # the provided KV sequence, so this is exactly sinks+window attention (no
    # position surgery). The verify pass metadata is untouched (full-KV,
    # lossless). Requires the eager/PIECEWISE draft chain (per-step metadata
    # rebuild); under the FULL-CG skip-rebuild chain it is ignored with a
    # warning. 0 -> off (default). See research/62_window_kv_draft.
    "VLLM_SELF_SPEC_DRAFT_KV_WINDOW": lambda: int(
        os.getenv("VLLM_SELF_SPEC_DRAFT_KV_WINDOW", "0")
    ),
    # Phase 82 E1: accept-feedback OFF gate (content axis of the regime
    # detector). When the EMA of the per-step draft acceptance fraction
    # (accepted/drafted) drops below this threshold, the scheduler stops
    # scheduling spec tokens; periodic probe bursts re-measure so the gate
    # can recover (hysteresis). 0.0 -> disabled.
    "VLLM_SELF_SPEC_ACCEPT_OFF_THRESHOLD": lambda: float(
        os.getenv("VLLM_SELF_SPEC_ACCEPT_OFF_THRESHOLD", "0.0")
    ),
    # Probe cadence for the accept gate: every INTERVAL scheduler steps,
    # BURST consecutive steps run speculation regardless of the gate.
    "VLLM_SELF_SPEC_ACCEPT_PROBE_INTERVAL": lambda: int(
        os.getenv("VLLM_SELF_SPEC_ACCEPT_PROBE_INTERVAL", "64")
    ),
    "VLLM_SELF_SPEC_ACCEPT_PROBE_BURST": lambda: int(
        os.getenv("VLLM_SELF_SPEC_ACCEPT_PROBE_BURST", "2")
    ),
    "VLLM_SELF_SPEC_GATE_DEBUG": lambda: os.getenv(
        "VLLM_SELF_SPEC_GATE_DEBUG", "0") == "1",
    # Phase 82 E1: "ctx:thresh" -- above this mean context, the accept
    # gate's OFF threshold switches to the given (lower) value. The
    # break-even acceptance is a per-(batch, ctx) cell quantity: the
    # window draft's R shrinks with context, so long-ctx cells win at
    # much lower acceptance. Empty -> single threshold.
    "VLLM_SELF_SPEC_ACCEPT_THRESH_LONG": lambda: os.getenv(
        "VLLM_SELF_SPEC_ACCEPT_THRESH_LONG", ""
    ),
    # Phase 82 E1: the accept gate engages only at running batch >= this.
    # Below it (e.g. b1) the per-step accept signal lacks the volume to
    # separate content regimes from noise -- follow the map prior (ON).
    "VLLM_SELF_SPEC_ACCEPT_GATE_MIN_BATCH": lambda: int(
        os.getenv("VLLM_SELF_SPEC_ACCEPT_GATE_MIN_BATCH", "1")
    ),
    # Phase 82 (gap G-A): skip the draft's propose on prefill chunks.
    # The draft otherwise forwards EVERY prefill token (measured 2x
    # prefill wall time); with shared-KV + window scratchpad the chain
    # reads target KV, so that forward is pure waste (validated: requests
    # whose prompts were never draft-prefilled draft correctly). Costs
    # one AR step after each prefill (drafting resumes next step).
    "VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT": lambda: os.getenv(
        "VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT", "0") == "1",
    # Phase 82: capture the K-1 draft chain steps as ONE CUDA graph
    # (single launch per spec cycle -- removes per-step CPU dispatch from
    # the chain's critical path). Requires the scratchpad chain
    # (VLLM_SELF_SPEC_DRAFT_FULLCG) and greedy sampling.
    "VLLM_SELF_SPEC_DRAFT_WHOLECHAIN": lambda: os.getenv(
        "VLLM_SELF_SPEC_DRAFT_WHOLECHAIN", "0") == "1",
    # Phase 82 F4: path to a COMPILED policy table (compile_policy.py
    # --solve): per-(batch, ctx) cells of {K, accept_off_thresh} measured
    # on the deployment path. Replaces the hand-rule envs (schedule /
    # SHORTCTX_OFF / THRESH_LONG); the accept gate's thresholds and the
    # per-step K both come from the table.
    "VLLM_SELF_SPEC_POLICY_FILE": lambda: os.getenv(
        "VLLM_SELF_SPEC_POLICY_FILE", ""
    ),
    # Phase 82 E1: hysteresis upper bound -- once gated OFF, speculation
    # re-enables only when the probe EMA rises above this (defaults to the
    # OFF threshold when 0, i.e. no hysteresis band).
    "VLLM_SELF_SPEC_ACCEPT_ON_THRESHOLD": lambda: float(
        os.getenv("VLLM_SELF_SPEC_ACCEPT_ON_THRESHOLD", "0.0")
    ),
    # Phase 82 E1: "ctx:batch" -- speculation OFF when the mean effective
    # context of running requests is below ctx AND the running batch is
    # >= batch (the map's verify-width OFF region at short context).
    # Empty -> disabled.
    "VLLM_SELF_SPEC_SHORTCTX_OFF": lambda: os.getenv(
        "VLLM_SELF_SPEC_SHORTCTX_OFF", ""
    ),
    # Number of attention-sink tokens the window-KV draft keeps at the start
    # of the KV sequence (rounded up to whole blocks). Only meaningful when
    # VLLM_SELF_SPEC_DRAFT_KV_WINDOW > 0.
    "VLLM_SELF_SPEC_DRAFT_KV_SINKS": lambda: int(
        os.getenv("VLLM_SELF_SPEC_DRAFT_KV_SINKS", "16")
    ),
    # Phase 66 shared-KV self-draft: the draft_model proposer's attention
    # layers BIND to the target layers' KV cache tensors instead of
    # registering their own (draft KV rode the same block tables already;
    # this removes the duplicate 48-layer allocation, halving the per-token
    # KV cost back to target-only). Draft KV writes are restricted to slots
    # the verify pass has not yet written (the appended sampled-token slot +
    # chain-drafted slots); those slots are overwritten with target-exact KV
    # by the next verify forward. draft_model (self-spec) method only.
    # Default off. See research/66_shared_kv.
    "VLLM_SELF_SPEC_SHARED_KV": lambda: bool(
        int(os.getenv("VLLM_SELF_SPEC_SHARED_KV", "0"))
    ),
    # Self-spec W7 (phase 67): with SHARED_KV, compact the step-0 draft forward
    # from a q=(K+2) query to a q=1 decode of only the appended sampled token
    # per request. Under shared KV the K+1 re-ingested verify tokens have their
    # draft KV writes PAD-masked and their hidden states discarded, so their
    # forward is wasted FLOPs -- only the appended token's hidden state is
    # consumed (to sample draft-1). The compacted decode reuses the SAME
    # windowed seq_lens/block_table so the appended token attends to an
    # identical key set (the same bf16 cached KV). CAVEAT: this switches
    # draft-1's FA3 kernel from the varlen (q=K+2) path to the decode (q=1)
    # path; bit-exact only at window=0 (canary W0 K2 stays 3.000), but in the
    # windowed regime the kernel-path change perturbs accept (measured W64 bf16
    # -0.014, W512 fp8 -0.040 at ~93% draft-1 acceptance). The exposed cycle
    # saving is small (~2.4 ms single-node) and the accept drop nearly cancels
    # it. Requires SHARED_KV + draft_model + K>1 + a decode-shaped propose.
    # Default off. See research/67_propose_fixed_opt.
    "VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE": lambda: bool(
        int(os.getenv("VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE", "0"))
    ),
    # Self-spec W2a FULL REPLICA opt-in: when set, the draft_model parallel
    # config is built with enable_expert_parallel=False (overriding the W0 EP
    # propagation from the target). The draft's FusedMoE then builds use_ep=False
    # -> expert_map=None -> EVERY rank holds ALL experts (a full bf16 replica)
    # and routes over all of them. This is comm-free by replication (non-EP MoE
    # has no all-to-all) and the W1 router mask becomes a no-op (expert_map=None),
    # so the draft gets full expert coverage -> acceptance approaches 1.0 (vs the
    # low coverage of an EP shard draft). Cost: full E experts/rank in bf16. The
    # draft stays data-parallel (DP) when the target is DP. Default off -> W0
    # (EP shard) draft behavior unchanged.
    "VLLM_SELF_SPEC_DRAFT_FULL_REPLICA": lambda: bool(
        int(os.getenv("VLLM_SELF_SPEC_DRAFT_FULL_REPLICA", "0"))
    ),
    # Self-spec W7: when set, the draft_model proposer uses FULL cudagraphs for
    # its decode-step forwards (mirroring the verify model's FULL decode graph).
    # By default the draft_model proposer's cudagraph dispatcher keys are never
    # initialized, so the draft runs fully eager; this flag both initializes the
    # draft's keys and FULL-wraps the draft model, capturing the whole draft
    # decode forward as one graph and removing the per-forward kernel-launch
    # overhead. Only takes effect when the engine's decode cudagraph mode is
    # FULL. Default off (draft eager, unchanged). See research/34_worldA_system.
    "VLLM_SELF_SPEC_DRAFT_FULL_CG": lambda: bool(
        int(os.getenv("VLLM_SELF_SPEC_DRAFT_FULL_CG", "0"))
    ),
    # Self-spec Phase 69: window-scratchpad FULL cudagraph for the draft chain.
    # Composes with VLLM_SELF_SPEC_DRAFT_KV_WINDOW (+ SHARED_KV): each chain-step
    # attention gathers the sinks+window KV from the (shared) paged cache into a
    # FIXED-shape dense scratchpad and runs a masked SDPA-style attention instead
    # of the paged FA3 decode kernel. The paged FA3 decode kernel freezes its
    # host-side work distribution at capture and is not replay-safe for the
    # draft's growing sequence, forcing eager attention (_draft_chain_force_eager
    # _attn) and PIECEWISE dispatch (~98 graph-piece launches/step). The dense
    # scratchpad attention is pure shape-driven tensor ops (gather + matmul +
    # softmax + matmul), fully CUDA-graph-capturable, so the whole per-step draft
    # forward replays as ONE FULL graph. Bit-exact greedy (same window key set,
    # same causal-at-end mask). Default off. See research/69_draft_fullcg.
    "VLLM_SELF_SPEC_DRAFT_FULLCG": lambda: bool(
        int(os.getenv("VLLM_SELF_SPEC_DRAFT_FULLCG", "0"))
    ),
    # Self-spec W7-piecewise: when the FA3 (GQA / non-MLA) draft chain runs its
    # attention eagerly (VLLM_SELF_SPEC_DRAFT_FULL_CG=1 -> _draft_chain_force_
    # eager_attn), by default the WHOLE chain forward runs eager (CUDAGraphMode.
    # NONE) because the FULL decode graph is not replay-safe for the draft's
    # growing sequence. That leaves the model BODY (GEMM/MoE) running op-by-op
    # eagerly (~67 ms/step on Qwen3-30B) instead of on a captured graph (~18 ms
    # for the identical-shape step-0 PIECEWISE forward). This flag switches the
    # chain to PIECEWISE cudagraphs: the body runs on captured graph pieces while
    # attention stays a splitting op (eager, live seq_lens -> byte-identical
    # numerics / accept to the NONE chain). Only takes effect when the chain is
    # already forcing eager attention (i.e. with DRAFT_FULL_CG on a non-MLA
    # backend). Default off -> unchanged (NONE chain). See research W7-draft-graph.
    "VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE": lambda: bool(
        int(os.getenv("VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE", "0"))
    ),
    # Self-spec W7-dp: when set, the draft_model is loaded UNCOMPILED (eager,
    # CompilationMode.NONE, no cudagraphs) while the target/verify model keeps
    # its normal torch.compile config. Partial mitigation for a torch.compile
    # numerical inconsistency in the self-spec MoE forward for non-MLA MoE
    # models (e.g. Qwen2-MoE / Qwen3-MoE under FLASH_ATTN): with compile on, the
    # separately-compiled draft and verify disagree per position so acceptance
    # collapses (Qwen1.5-MoE K=4: per-token 0.94 eager vs 0.14 compiled). Dense
    # and MLA-MoE drafts are unaffected. Eager draft alone only partly recovers
    # (~1.6->1.9) since the verify is still compiled; full recovery (->4.8)
    # needs enforce_eager. Default off. See research/36_dp_accept.
    "VLLM_SELF_SPEC_DRAFT_EAGER": lambda: bool(
        int(os.getenv("VLLM_SELF_SPEC_DRAFT_EAGER", "0"))
    ),
    # Self-spec W7 compile-consistency: when set, enables batch-invariant
    # numerics (the same machinery as VLLM_BATCH_INVARIANT) so the COMPILED
    # draft and the COMPILED verify produce per-position-identical greedy
    # tokens. Root cause it fixes: the draft runs its MoE/GEMM/attention at the
    # decode batch shape (~1 token/seq) while the verify runs at the larger
    # verify shape; the default kernels (cuBLAS split-k, shape-tiled Triton MoE)
    # are batch-variant, so the draft's tokens drift from the compiled verify's
    # and acceptance collapses (Qwen1.5-MoE K=4: 1.57 compiled vs 4.78 eager).
    # Disabling batch variance makes draft==verify with BOTH compiled, recovering
    # acceptance to ~5.0 while keeping the draft compiled/cudagraphed (fast).
    # Default off -> unchanged. See research/37_compile_consistency.
    "VLLM_SELF_SPEC_COMPILE_CONSISTENT": lambda: bool(
        int(os.getenv("VLLM_SELF_SPEC_COMPILE_CONSISTENT", "0"))
    ),
    # Self-spec W7 CPU-orchestration reductions (phase 45): master opt-in that
    # enables the lossless per-cycle CPU-overhead cuts on the comm-free
    # self-spec (draft_model) decode path. Individual cuts can also be toggled
    # by their own flags below; this one turns the whole set on. Default off ->
    # the outer/bookkeeping path is byte-identical to before. Only affects the
    # draft_model + sync-scheduling verify path.
    "VLLM_SELF_SPEC_CPU_ORCH": lambda: bool(
        int(os.getenv("VLLM_SELF_SPEC_CPU_ORCH", "0"))
    ),
    # Self-spec OV0b (phase 49): shadow-chain overlap probe. When >0, after the
    # verify forward is ENQUEUED the proposer replays this many draft-chain
    # decode-step forwards on a side CUDA stream, using the previous cycle's
    # cached dispatch state, with the slot-mapping buffer filled with
    # PADDING_SLOT_ID so every shadow KV write is discarded (timing-only,
    # output-neutral). The main stream waits on the shadow-done event before the
    # real propose touches shared buffers, so the step time measures
    # max(verify, shadow) + rest. Probes (a) whether draft compute can hide in
    # the verify's comm window and (b) cudagraph memory-pool aliasing between
    # concurrently replaying draft/verify graphs (accept_len collapse = alias).
    # Default 0 = off, byte-identical.
    "VLLM_SELF_SPEC_SHADOW_CHAIN": lambda: int(
        os.getenv("VLLM_SELF_SPEC_SHADOW_CHAIN", "0")
    ),
    # Self-spec OV0b/OV1 (phase 49): capture draft-tagged graphs (compilation
    # prefix "draft_model") into a DEDICATED cudagraph memory pool instead of
    # the shared global pool. Required for any schedule that replays draft
    # graphs concurrently with verify graphs (shadow chain / overlap): with a
    # shared pool their workspaces alias and concurrent replay races (sticky
    # CUDA illegal-instruction). Costs the draft's activation workspace as
    # extra memory. Default off = shared pool, unchanged behavior.
    "VLLM_SELF_SPEC_DRAFT_GRAPH_POOL": lambda: bool(
        int(os.getenv("VLLM_SELF_SPEC_DRAFT_GRAPH_POOL", "0"))
    ),
    # Self-spec OV1 (phase 49): dedicated WorkspaceManager for the comm-free
    # draft forward. The fused-MoE modular kernel takes its gemm/activation
    # scratch from the global workspace; draft and verify graphs bake pointers
    # into the SAME buffer, so concurrent replay corrupts the verify's MoE
    # output (OV0b root cause). Routes by the draft forward-context flag.
    # Costs one extra workspace buffer. Default off = shared, unchanged.
    "VLLM_SELF_SPEC_DRAFT_WORKSPACE": lambda: bool(
        int(os.getenv("VLLM_SELF_SPEC_DRAFT_WORKSPACE", "0"))
    ),
    # Self-spec OV1(a) (phase 49): free-running ahead-chain, VALIDATION mode.
    # After each propose, the draft chain CONTINUES for K+1 steps on the side
    # stream during the verify (assuming full acceptance; the first ahead token
    # is the draft's bonus guess), with real KV writes covered by the extended
    # scheduler lookahead. Outputs are DISCARDED (the normal propose still
    # runs; served output byte-identical); per-request hit rate
    # P(all-K accepted AND bonus == guess) is measured and logged. Requires the
    # phase-49 isolation stack (DRAFT_GRAPH_POOL + DRAFT_WORKSPACE). Greedy
    # only. Default off.
    "VLLM_SELF_SPEC_AHEAD_CHAIN": lambda: bool(
        int(os.getenv("VLLM_SELF_SPEC_AHEAD_CHAIN", "0"))
    ),
    # Self-spec OV1(b) (phase 49): CONSUME the ahead chain's drafts -- the
    # full free-running mode. The lockstep propose is skipped in steady state;
    # each verify's drafts come from the previous cycle's ahead chain, and the
    # ahead chain RE-ANCHORS every cycle on the actual committed token
    # (identical to continuation on hits; automatic correction on misses,
    # which cost one low-accept cycle and are losslessly rejected by the
    # verify). Requires VLLM_SELF_SPEC_AHEAD_CHAIN=1 and the phase-49
    # isolation stack. Greedy only. Default off.
    "VLLM_SELF_SPEC_CONSUME_AHEAD": lambda: bool(
        int(os.getenv("VLLM_SELF_SPEC_CONSUME_AHEAD", "0"))
    ),
    # Self-spec Phase 55: memoize coordinate_batch_across_dp within a spec
    # cycle -- ONE DP rendezvous per distinct draft shape per propose() instead
    # of one per chain forward (the traced dominant wait, Phases 52/54). Safe
    # for lockstep uniform batches; dummy runs and capture always coordinate.
    # Default off.
    "VLLM_SELF_SPEC_DRAFT_AMORTIZE_DP_COORD": lambda: bool(
        int(os.getenv("VLLM_SELF_SPEC_DRAFT_AMORTIZE_DP_COORD", "0"))
    ),
    # Self-spec Phase 55: capture the draft's STEP-0 forward as a FULL
    # cudagraph (K=1 only; requires VLLM_SELF_SPEC_DRAFT_FULL_CG). At K=1 the
    # step-0 forward (q=K+1 padded-uniform) is the ONLY draft forward, yet it
    # dispatched PIECEWISE by design -- running per-layer collectives/glue in
    # Python (~1 ms x layers; the measured 60 ms/cycle at 236B for dispatching
    # drafts, and harmless only for comm-free ones). Repoints the draft
    # dispatcher's uniform_decode_query_len to K+1 and captures step0-shaped
    # FULL graphs. Default off.
    "VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG": lambda: bool(
        int(os.getenv("VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG", "0"))
    ),
    # Self-spec Phase 54: NODE-LOCAL draft routing (the Phase 11/19
    # hierarchical draft). The draft routes to any expert resident on its NODE
    # and dispatches via AllGather+ReduceScatter over an INTRA-NODE subgroup
    # (NVLink only), skipping the inter-node hop; the verify stays full-EP.
    # Producer flag, draft_model method only; mutually exclusive with
    # VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE (which wins if both are set). Note
    # VLLM_SELF_SPEC_DRAFT_SKIP_DP_COORD is only safe with this when per-rank
    # batches are uniform (the node collective needs real per-rank sizes).
    # Default off.
    "VLLM_SELF_SPEC_DRAFT_NODE_LOCAL": lambda: bool(
        int(os.getenv("VLLM_SELF_SPEC_DRAFT_NODE_LOCAL", "0"))
    ),
    # Reader-side fallback for the node-local flag (mirrors
    # VLLM_SELF_SPEC_LOCAL_ROUTE): applies node-local routing to EVERY forward
    # when set. Research/debug only; the spec draft path uses the
    # forward-context key set by the producer flag above. Default off.
    "VLLM_SELF_SPEC_NODE_LOCAL": lambda: bool(
        int(os.getenv("VLLM_SELF_SPEC_NODE_LOCAL", "0"))
    ),
    # Self-spec W7 (phase 52): skip coordinate_batch_across_dp for the LOCKSTEP
    # comm-free draft chain (requires VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE). The
    # coordination all_reduce + .item() readback is a DP-wide rendezvous per
    # chain forward -- traced at 66% of the 2-node DP16 spec cycle -- and the
    # comm-free draft has no collectives that need DP-consistent dispatch.
    # Per-rank dispatch divergence is safe (same argument as CONSUME_AHEAD's
    # skip); the verify's coordination is untouched. Default off.
    "VLLM_SELF_SPEC_DRAFT_SKIP_DP_COORD": lambda: bool(
        int(os.getenv("VLLM_SELF_SPEC_DRAFT_SKIP_DP_COORD", "0"))
    ),
    # Self-spec W7 (phase 65): run the DRAFT path's coordinate_batch_across_dp
    # all_reduce on the DP CPU (gloo) group instead of NCCL. The NCCL
    # coordination's .item() readbacks are cudaStreamSynchronize points that
    # drain all previously enqueued GPU work (the chain coordination drains
    # the whole step-0 draft forward) -- traced at ~32 ms/step at 16k DP16
    # (Phase 64). On the CPU group the readbacks are free and the rendezvous
    # blocks only on rank skew, never on the GPU stream; the coordination
    # RESULT is identical. Applies to every draft-proposer coordination
    # (propose, dummy runs, capture) so busy/idle ranks stay collective-
    # symmetric. The verify's coordination is untouched. Default off.
    "VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU": lambda: bool(
        int(os.getenv("VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU", "0"))
    ),
    # Self-spec W7 (phase 65): lightweight per-step attention metadata for the
    # PIECEWISE/eager draft chain (FlashAttention backends only). The chain
    # rebuilds the per-layer attn metadata every step, but every tensor field
    # of the built FlashAttentionMetadata (seq_lens / block_table /
    # slot_mapping / query_start_loc) already lives in proposer-owned
    # persistent buffers that the per-step updates rewrite IN PLACE; only the
    # max_seq_len scalar actually changes value. Build once on the first
    # chain step, then per step run only the data updates (window compaction
    # writes + max_seq_len scalar sync) and reuse the metadata objects.
    # Non-FA backends and the FULL-CG chain fall back unchanged. Default off.
    "VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD": lambda: bool(
        int(os.getenv("VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD", "0"))
    ),
    # Self-spec W7 (phase 45): vectorized, non-blocking rejection-output parse.
    # Replaces the per-request Python list-comprehension + blocking .cpu() D2H
    # in RejectionSampler.parse_output with a single pinned-buffer async copy
    # (event-synced, does not stall other CUDA streams) + one flat .tolist()
    # split by per-row valid counts. Byte-identical output. Implied by
    # VLLM_SELF_SPEC_CPU_ORCH. Default off.
    "VLLM_SELF_SPEC_FAST_PARSE": lambda: bool(
        int(os.getenv("VLLM_SELF_SPEC_FAST_PARSE", "0"))
    ),
    # Self-spec W7 micro-benchmark: when set, the proposer and target-verify
    # forwards record additive, CUDA-synchronized wall-clock timings (whole
    # draft chain, a single draft forward, and the verify forward) into a
    # process-local profiler (vllm.v1.spec_decode.self_spec_profiler). Timing
    # only -- no behavior change. Default off. See research/34_worldA_system.
    "VLLM_SELF_SPEC_PROFILE": lambda: bool(
        int(os.getenv("VLLM_SELF_SPEC_PROFILE", "0"))
    ),
    # Self-spec W7: record fine-grained per-step / per-cycle CPU sub-regions
    # (step_*, cpu_*) in the profiler. Extra timers -- default off so the clean
    # draft_chain/verify measurement is undisturbed. Read directly by the
    # profiler; listed here for discoverability.
    "VLLM_SELF_SPEC_PROFILE_FINE": lambda: bool(
        int(os.getenv("VLLM_SELF_SPEC_PROFILE_FINE", "0"))
    ),
    # Self-spec W7 numerical-divergence instrumentation (Step 1): when set to a
    # directory path, the MoE runner dumps, for the FIRST decode forward on DP
    # rank 0 at layer VLLM_SELF_SPEC_MOE_DUMP_LAYER, the per-token (a) router
    # logits, (b) selected expert ids + weights, and (c) post-combine MoE
    # output, to "{path}/moe_dump_L{layer}.pt". Lets configs A (comm-free
    # full-replica) and C (EP all-to-all) be diffed on the SAME prompt to prove
    # the comm-free-vs-EP bf16 reduce structure is the divergence cause. Timing-
    # neutral, default off (empty -> no dump). See research/40_num_divergence.
    "VLLM_SELF_SPEC_MOE_NUM_DUMP": lambda: os.getenv(
        "VLLM_SELF_SPEC_MOE_NUM_DUMP", ""
    ),
    "VLLM_SELF_SPEC_MOE_DUMP_LAYER": lambda: int(
        os.getenv("VLLM_SELF_SPEC_MOE_DUMP_LAYER", "0")
    ),
    # Self-spec W7 FP32-accumulation fix (Step 2): when set, force FP32
    # accumulation in BOTH MoE reduce paths so the bf16 summation associativity
    # stops mattering and the comm-free full-replica sum and the EP all-to-all
    # sum both converge to the true FP32 sum (-> match). Affects:
    #   * the LOCAL moe_sum (config A path): topk experts are summed in fp32 and
    #     the result cast back to the activation dtype;
    #   * the reduce_scatterv cross-rank combine (config C path): the cross-rank
    #     partials are accumulated in fp32. The wire payload stays bf16 only when
    #     fp32 transport is not needed; here we upcast the partials to fp32 for a
    #     faithful fp32 cross-rank sum (correctness probe; see results for the
    #     bandwidth note). Default off -> both paths unchanged (bf16-accum).
    "VLLM_SELF_SPEC_MOE_FP32_ACCUM": lambda: bool(
        int(os.getenv("VLLM_SELF_SPEC_MOE_FP32_ACCUM", "0"))
    ),
    # The number of SMs/CUs to allocate for communication kernels when
    # running DBO; the rest will be allocated to compute.
    # Default: 20 on CUDA (SMs), 64 on ROCm (CUs).
    "VLLM_DBO_COMM_SMS": lambda: int(
        os.getenv(
            "VLLM_DBO_COMM_SMS",
            "64"
            if hasattr(__import__("torch").version, "hip")
            and __import__("torch").version.hip is not None
            else "20",
        )
    ),
    # Enable max_autotune & coordinate_descent_tuning in inductor_config
    # to compile static shapes passed from compile_sizes in compilation_config
    # If set to 1, enable max_autotune; By default, this is enabled (1)
    "VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE": lambda: bool(
        int(os.getenv("VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE", "1"))
    ),
    # If set to 1, enable coordinate_descent_tuning;
    # By default, this is enabled (1)
    "VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING": lambda: bool(
        int(os.getenv("VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING", "1"))
    ),
    # Flag to enable NCCL symmetric memory allocation and registration
    "VLLM_USE_NCCL_SYMM_MEM": lambda: bool(
        int(os.getenv("VLLM_USE_NCCL_SYMM_MEM", "0"))
    ),
    # NCCL header path
    "VLLM_NCCL_INCLUDE_PATH": lambda: os.environ.get("VLLM_NCCL_INCLUDE_PATH", None),
    # GC debug config
    # - VLLM_GC_DEBUG=0: disable GC debugger
    # - VLLM_GC_DEBUG=1: enable GC debugger with gc.collect elpased times
    # - VLLM_GC_DEBUG='{"top_objects":5}': enable GC debugger with
    #                                      top 5 collected objects
    "VLLM_GC_DEBUG": lambda: os.getenv("VLLM_GC_DEBUG", ""),
    # Debug workspace allocations.
    # logging of workspace resize operations.
    "VLLM_DEBUG_WORKSPACE": lambda: bool(int(os.getenv("VLLM_DEBUG_WORKSPACE", "0"))),
    # Disables parallel execution of shared_experts via separate cuda stream
    "VLLM_DISABLE_SHARED_EXPERTS_STREAM": lambda: bool(
        int(os.getenv("VLLM_DISABLE_SHARED_EXPERTS_STREAM", "0"))
    ),
    # Limits when we run shared_experts in a separate stream.
    # We found out that for large batch sizes, the separate stream
    # execution is not beneficial (most likely because of the input clone)
    # TODO(alexm-redhat): Tune to be more dynamic based on GPU type
    "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD": lambda: int(
        int(os.getenv("VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD", 256))
    ),
    # Token-count cutoff for multi-stream overlap of the attention input
    # GEMM with auxiliary GEMMs (e.g. fused_wqa_wkv overlapped with indexer
    # weights / kv-score projections in DeepSeek-V4). At or below this many
    # tokens the FP8 main GEMM has idle SMs to share with the bf16 aux GEMMs
    # and overlap is a 5-45% win; above it the FP8 GEMM saturates the device
    # and the cross-stream sync becomes pure overhead. Set to 0 to disable
    # the multi-stream path entirely. See #PR 41526 for the empirical result
    # for the default value of 1024 tokens.
    "VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD": lambda: int(
        os.getenv("VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD", "1024")
    ),
    # Format for saving torch.compile cache artifacts
    # - "binary": saves as binary file
    #     Safe for multiple vllm serve processes accessing the same torch compile cache.
    # - "unpacked": saves as directory structure (for inspection/debugging)
    #     NOT multiprocess safe - race conditions may occur with multiple processes.
    #     Allows viewing and setting breakpoints in Inductor's code output files.
    "VLLM_COMPILE_CACHE_SAVE_FORMAT": env_with_choices(
        "VLLM_COMPILE_CACHE_SAVE_FORMAT", "binary", ["binary", "unpacked"]
    ),
    # Flag to control the v2 model runner. If unset, use config defaults.
    "VLLM_USE_V2_MODEL_RUNNER": lambda: maybe_convert_bool(
        os.getenv("VLLM_USE_V2_MODEL_RUNNER", None)
    ),
    # Log model inspection after loading.
    # If enabled, logs a transformers-style hierarchical view of the model
    # with quantization methods and attention backends.
    "VLLM_LOG_MODEL_INSPECTION": lambda: bool(
        int(os.getenv("VLLM_LOG_MODEL_INSPECTION", "0"))
    ),
    # Debug logging for --enable-mfu-metrics
    "VLLM_DEBUG_MFU_METRICS": lambda: bool(
        int(os.getenv("VLLM_DEBUG_MFU_METRICS", "0"))
    ),
    # Disable using pytorch's pin memory for CPU offloading.
    "VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY": lambda: bool(
        int(os.getenv("VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY", "0"))
    ),
    # Disable using UVA (Unified Virtual Addressing) for CPU offloading.
    "VLLM_WEIGHT_OFFLOADING_DISABLE_UVA": lambda: bool(
        int(os.getenv("VLLM_WEIGHT_OFFLOADING_DISABLE_UVA", "0"))
    ),
    # On WSL2 with a compatible kernel (>= 4.19.121), pinned memory is
    # supported but disabled by default due to a small performance regression.
    # Set to 1 when pinned memory or UVA is required (e.g. CPU offloading
    # or v2 model runner).
    "VLLM_WSL2_ENABLE_PIN_MEMORY": lambda: bool(
        int(os.getenv("VLLM_WSL2_ENABLE_PIN_MEMORY", "0"))
    ),
    # Disable logging of vLLM logo at server startup time.
    "VLLM_DISABLE_LOG_LOGO": lambda: bool(int(os.getenv("VLLM_DISABLE_LOG_LOGO", "0"))),
    # Disable PDL for LoRA, as enabling PDL with LoRA on SM100 causes
    # Triton compilation to fail.
    "VLLM_LORA_DISABLE_PDL": lambda: bool(int(os.getenv("VLLM_LORA_DISABLE_PDL", "0"))),
    # Enable CUDA compatibility mode for datacenter GPUs with older
    # driver versions than the CUDA toolkit major version of vLLM.
    "VLLM_ENABLE_CUDA_COMPATIBILITY": lambda: (
        os.environ.get("VLLM_ENABLE_CUDA_COMPATIBILITY", "0").strip().lower()
        in ("1", "true")
    ),
    # Path to the CUDA compatibility libraries when CUDA compatibility is enabled.
    "VLLM_CUDA_COMPATIBILITY_PATH": lambda: os.environ.get(
        "VLLM_CUDA_COMPATIBILITY_PATH", None
    ),
    # Skip model name validation in OpenAI API requests.
    # When set to 1, any model name will be accepted in the 'model' field
    # of API requests. This is useful for proxy/gateway scenarios where
    # the actual model is served but different names may be used in requests.
    "VLLM_SKIP_MODEL_NAME_VALIDATION": lambda: (
        os.getenv("VLLM_SKIP_MODEL_NAME_VALIDATION", "0").strip().lower()
        in ("1", "true")
    ),
    # Whether it is a scale up launch engine for elastic EP,
    # Should only be set by EngineCoreClient.
    "VLLM_ELASTIC_EP_SCALE_UP_LAUNCH": lambda: bool(
        int(os.getenv("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH", "0"))
    ),
    # Whether to wait for all requests to drain before sending the
    # scaling command in elastic EP.
    "VLLM_ELASTIC_EP_DRAIN_REQUESTS": lambda: bool(
        int(os.getenv("VLLM_ELASTIC_EP_DRAIN_REQUESTS", "0"))
    ),
    # If set to 1, enable CUDA graph memory estimation during memory profiling.
    # This profiles CUDA graph memory usage to provide more accurate KV cache
    # memory allocation. Enabled by default as of v0.21.0
    "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": lambda: bool(
        int(os.getenv("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "1"))
    ),
    # NIXL EP environment variables
    "VLLM_NIXL_EP_MAX_NUM_RANKS": lambda: int(
        os.getenv("VLLM_NIXL_EP_MAX_NUM_RANKS", "32")
    ),
    # Whether enable XPU graph on Intel GPU
    "VLLM_XPU_ENABLE_XPU_GRAPH": lambda: bool(
        int(os.getenv("VLLM_XPU_ENABLE_XPU_GRAPH", "0"))
    ),
    # whether use xpu specific sample kernel
    "VLLM_XPU_USE_SAMPLER_KERNEL": lambda: bool(
        int(os.getenv("VLLM_XPU_USE_SAMPLER_KERNEL", "1"))
    ),
    # Enable simple KV offload.
    "VLLM_USE_SIMPLE_KV_OFFLOAD": lambda: bool(
        int(os.getenv("VLLM_USE_SIMPLE_KV_OFFLOAD", "0"))
    ),
    # Whether to enable dual cuda streams for LoRA computation
    # (used by both BaseLinearLayerWithLoRA and FusedMoEWithLoRA to
    # overlap the base layer compute with the LoRA fast path).
    "VLLM_LORA_ENABLE_DUAL_STREAM": lambda: bool(
        int(os.getenv("VLLM_LORA_ENABLE_DUAL_STREAM", "0"))
    ),
    # If set to 1, use Python spinloop extension to poll in a more efficient
    # way when using the mp backend.
    "VLLM_USE_SPINLOOP_EXT": lambda: bool(int(os.getenv("VLLM_USE_SPINLOOP_EXT", "0"))),
    # Comma-separated GPU_BDF=NIC_BDF pairs for RDMA NIC selection.
    # Must be set together with VLLM_NIC_SELECTION_VARS.
    "VLLM_GPU_NIC_PCIE_MAPPING": lambda: os.getenv("VLLM_GPU_NIC_PCIE_MAPPING", ""),
    # Comma-separated list of env vars to set from the GPU-NIC mapping.
    # Each entry is VAR_NAME or VAR_NAME:<suffix> (suffix appended to
    # RDMA device name). Must be set together with VLLM_GPU_NIC_PCIE_MAPPING.
    "VLLM_NIC_SELECTION_VARS": lambda: os.getenv("VLLM_NIC_SELECTION_VARS", ""),
}


# --8<-- [end:env-vars-definition]


def __getattr__(name: str):
    """
    Gets environment variables lazily.

    NOTE: After enable_envs_cache() invocation (which triggered after service
    initialization), all environment variables will be cached.
    """
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _is_envs_cache_enabled() -> bool:
    """Checked if __getattr__ is wrapped with functools.cache"""
    global __getattr__
    return hasattr(__getattr__, "cache_clear")


def enable_envs_cache() -> None:
    """
    Enables caching of environment variables. This is useful for performance
    reasons, as it avoids the need to re-evaluate environment variables on
    every call.

    NOTE: Currently, it's invoked after service initialization to reduce
    runtime overhead. This also means that environment variables should NOT
    be updated after the service is initialized.
    """
    if _is_envs_cache_enabled():
        # Avoid wrapping functools.cache multiple times
        return
    # Tag __getattr__ with functools.cache
    global __getattr__
    __getattr__ = functools.cache(__getattr__)

    # Cache all environment variables
    for key in environment_variables:
        __getattr__(key)


def disable_envs_cache() -> None:
    """
    Resets the environment variables cache. It could be used to isolate environments
    between unit tests.
    """
    global __getattr__
    # If __getattr__ is wrapped by functions.cache, unwrap the caching layer.
    if _is_envs_cache_enabled():
        assert hasattr(__getattr__, "__wrapped__")
        __getattr__ = __getattr__.__wrapped__


def __dir__():
    return list(environment_variables.keys())


def is_set(name: str):
    """Check if an environment variable is explicitly set."""
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def validate_environ(hard_fail: bool) -> None:
    for env in os.environ:
        if env.startswith("VLLM_") and env not in environment_variables:
            if hard_fail:
                raise ValueError(f"Unknown vLLM environment variable detected: {env}")
            else:
                logger.warning("Unknown vLLM environment variable detected: %s", env)


def compile_factors() -> dict[str, object]:
    """Return env vars used for torch.compile cache keys.

    Start with every known vLLM env var; drop entries in `ignored_factors`;
    hash everything else. This keeps the cache key aligned across workers."""

    ignored_factors: set[str] = {
        "MAX_JOBS",
        "VLLM_RPC_BASE_PATH",
        "VLLM_USE_MODELSCOPE",
        "VLLM_RINGBUFFER_WARNING_INTERVAL",
        "VLLM_DEBUG_DUMP_PATH",
        "VLLM_PORT",
        "VLLM_CACHE_ROOT",
        "LD_LIBRARY_PATH",
        "VLLM_SERVER_DEV_MODE",
        "VLLM_DP_MASTER_IP",
        "VLLM_DP_MASTER_PORT",
        "VLLM_NIXL_SIDE_CHANNEL_HOST",
        "VLLM_RANDOMIZE_DP_DUMMY_INPUTS",
        "VLLM_CI_USE_S3",
        "VLLM_MODEL_REDIRECT_PATH",
        "VLLM_HOST_IP",
        "VLLM_FORCE_AOT_LOAD",
        "S3_ACCESS_KEY_ID",
        "S3_SECRET_ACCESS_KEY",
        "S3_ENDPOINT_URL",
        "VLLM_USAGE_STATS_SERVER",
        "VLLM_NO_USAGE_STATS",
        "VLLM_DO_NOT_TRACK",
        "VLLM_LOGGING_LEVEL",
        "VLLM_LOGGING_PREFIX",
        "VLLM_LOGGING_STREAM",
        "VLLM_LOGGING_CONFIG_PATH",
        "VLLM_LOGGING_COLOR",
        "VLLM_LOG_STATS_INTERVAL",
        "VLLM_DEBUG_LOG_API_SERVER_RESPONSE",
        "VLLM_TUNED_CONFIG_FOLDER",
        "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
        "VLLM_ENGINE_ITERATION_TIMEOUT_S",
        "VLLM_HTTP_TIMEOUT_KEEP_ALIVE",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS",
        "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS",
        "VLLM_KEEP_ALIVE_ON_ENGINE_DEATH",
        "VLLM_IMAGE_FETCH_TIMEOUT",
        "VLLM_VIDEO_FETCH_TIMEOUT",
        "VLLM_AUDIO_FETCH_TIMEOUT",
        "VLLM_MEDIA_CACHE",
        "VLLM_MEDIA_CACHE_MAX_SIZE_MB",
        "VLLM_MEDIA_CACHE_TTL_HOURS",
        "VLLM_MEDIA_FETCH_MAX_RETRIES",
        "VLLM_MEDIA_URL_ALLOW_REDIRECTS",
        "VLLM_MEDIA_LOADING_THREAD_COUNT",
        "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB",
        "VLLM_MAX_AUDIO_DECODE_DURATION_S",
        "VLLM_MAX_AUDIO_PREPROCESS_WORKERS",
        "VLLM_VIDEO_LOADER_BACKEND",
        "VLLM_MEDIA_CONNECTOR",
        "VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME",
        "VLLM_ASSETS_CACHE",
        "VLLM_ASSETS_CACHE_MODEL_CLEAN",
        "VLLM_WORKER_MULTIPROC_METHOD",
        "VLLM_ENABLE_V1_MULTIPROCESSING",
        "VLLM_V1_OUTPUT_PROC_CHUNK_SIZE",
        "VLLM_CPU_KVCACHE_SPACE",
        "VLLM_CPU_MOE_PREPACK",
        "VLLM_ZENTORCH_WEIGHT_PREPACK",
        "VLLM_TEST_FORCE_LOAD_FORMAT",
        "VLLM_ENABLE_CUDA_COMPATIBILITY",
        "VLLM_CUDA_COMPATIBILITY_PATH",
        "VLLM_SKIP_MODEL_NAME_VALIDATION",
        "LOCAL_RANK",
        "CUDA_VISIBLE_DEVICES",
        "NO_COLOR",
    }

    from vllm.config.utils import normalize_value

    factors: dict[str, object] = {}
    for factor, getter in environment_variables.items():
        if factor in ignored_factors:
            continue

        try:
            raw = getter()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Skipping environment variable %s while hashing compile factors: %s",
                factor,
                exc,
            )
            continue

        factors[factor] = normalize_value(raw)

    ray_noset_env_vars = [
        # Refer to
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/nvidia_gpu.py#L11
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/amd_gpu.py#L11
        # https://github.com/ray-project/ray/blob/b97d21dab233c2bd8ed7db749a82a1e594222b5c/python/ray/_private/accelerators/amd_gpu.py#L10
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/npu.py#L12
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/hpu.py#L12
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/neuron.py#L14
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/tpu.py#L38
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/intel_gpu.py#L10
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/rbln.py#L10
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
        "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
        "RAY_EXPERIMENTAL_NOSET_RBLN_RT_VISIBLE_DEVICES",
    ]

    for var in ray_noset_env_vars:
        factors[var] = normalize_value(os.getenv(var))

    return factors
