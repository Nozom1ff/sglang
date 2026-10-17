"""Eligibility for declaring PRE_REPLAY on PD-prefill EXTEND forwards.

A backend may declare ``SharedReadEnds.PRE_REPLAY`` for EXTEND only after its
``prepare_prefill_shared_read_snapshot`` hook has confirmed that the forward
reads no scheduler-shared buffer past the read-done event. These helpers hold
the conditions shared by every participating backend; each backend still owns
its own late-read snapshot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import SharedReadEnds
from sglang.srt.layers.attention.kv_shard_hooks import get_kv_shard_pool
from sglang.srt.layers.cp.utils import is_cp_active
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.runtime_context import get_disagg, get_memory, get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def prefill_war_read_done_server_eligible(token_to_kv_pool) -> bool:
    """Server-level gates, evaluated once at backend init.

    Only the PD-prefill overlap loop re-fences its next scheduling round behind
    the whole forward after result processing; the other loops that consume the
    read-done event do not, so the fine-grained barrier is limited to it. DCP,
    the unified memory pool and page-interleaved KV pools read shared state on
    paths that have not been audited for an early read-done event.
    """
    return (
        envs.SGLANG_ENABLE_PREFILL_WAR_READ_DONE.get()
        and get_disagg().disaggregation_mode == "prefill"
        and not get_parallel().dcp_enabled
        and not get_memory().enable_unified_memory
        and get_kv_shard_pool(token_to_kv_pool) is None
    )


def prefill_war_read_done_batch_eligible(forward_batch: ForwardBatch) -> bool:
    """Per-forward gates. Graph replay and CP read shared state on paths that
    have not been audited, so they keep the coarse whole-forward fence."""
    return (
        forward_batch.forward_mode == ForwardMode.EXTEND
        and not is_in_breakable_cuda_graph()
        and not is_in_tc_piecewise_cuda_graph()
        and not is_cp_active(forward_batch)
        and forward_batch.extend_prefix_lens_cpu is not None
    )


def prefill_snapshot_read_ends(
    fm: ForwardMode, metadata, fallback: SharedReadEnds
) -> SharedReadEnds:
    """PRE_REPLAY iff this forward's snapshot hook marked its metadata.

    The graph-context check guards against a metadata object that a graph
    replay path reuses without calling the hook again.
    """
    if (
        fm == ForwardMode.EXTEND
        and metadata is not None
        and getattr(metadata, "prefill_shared_reads_snapshotted", False)
        and not is_in_breakable_cuda_graph()
        and not is_in_tc_piecewise_cuda_graph()
    ):
        return SharedReadEnds.PRE_REPLAY
    return fallback
