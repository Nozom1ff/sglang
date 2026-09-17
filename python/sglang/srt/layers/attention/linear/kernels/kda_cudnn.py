# SPDX-License-Identifier: Apache-2.0
"""cuDNN KDA prefill backend (``cudnn.linear_attention.kimi_delta_attention``).

Calls the cuDNN public KDA op directly instead of ``cudnn.fla.accelerate_fla()``:
that helper rebinds ``fla.ops.kda.chunk_kda``, and SGLang's hot path imports its
own vendored ``sglang.kernels.ops.attention.fla.kda``, so the patch has no
target here.

Prefill only, safe (bounded) gate only. The op takes the raw pre-activation
gate plus ``a_log``/``dt_bias`` and applies
``lower_bound * sigmoid(exp(a_log) * (g + dt_bias))`` in-kernel, so the
unbounded ``-exp(a_log) * softplus`` variant would have to be activated
here — a second copy of gate semantics that already live in the vendored
Triton kernel. Those models fall back to Triton instead.

The op is THD ``[total_tokens, heads, dim]`` with ``[N+1]`` ``cu_seqlens``,
which is SGLang's packed extend layout minus the size-1 batch dim, and its
state is ``[N, HO, V, K]`` — the same inner two dims as the Mamba pool slot, so
no transpose. What the op does not take is the pool's slot indices, hence the
gather/scatter around the call.

For interior prefix-cache tracking, request checkpoints every 64 tokens. Each
sequence starts with its initial state, followed by the states entering the
remaining chunks, matching Triton's intermediate-state layout.

Targets the cudnn-frontend v1.28.0 release envelope, not the develop branch:
head dims exactly 128 and ``g`` fp32-only, where develop also serves 64 and
bf16/fp16 ``g``. ``initial_state``/``final_state`` are fp32 or bf16; the pool
state is passed through in its native dtype.
"""

import logging
from typing import Optional

import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)

logger = logging.getLogger(__name__)

# FROST KDA engine envelope, mirroring the v1.28.0 release gates:
# SM100-SM103 or SM107 and q/k/v fp16/bf16 (frost/engine.py frost_la_gate),
# head dims exactly 128 -- "the recurrent state is 128x128" -- and g fp32
# (same function). The engine and public op allow fp32/bf16 state with matching
# initial and final dtypes.
#
# The develop branch relaxes two of these (head dims 64 or 128, g also
# bf16/fp16). Release is the target because pinning a develop commit means
# tracking a branch that forked from v1.26.0 and carries 397 commits release
# does not. Keep these mirroring whichever version is installed: too loose and
# every call pays a decline-then-fall-back round trip, too tight and shapes the
# engine would serve go to Triton instead.
_FROST_HEAD_DIMS = (128,)
_FROST_STATE_DTYPES = (torch.float32, torch.bfloat16)
_FROST_IO_DTYPES = (torch.bfloat16, torch.float16)
# g is upcast to this before the call; the engine accepts nothing else.
_FROST_GATE_DTYPE = torch.float32

# One pinned plan for now: FROST is the engine whose gates are mirrored above.
# Leaving it unpinned lets the planner also offer kda_cutile, whose envelope and
# performance here are unmeasured.
_PLAN_NAME = "kda_frost"

# SGLang's KDA tracking indexes states at 64-token chunk boundaries.
_KDA_CHECKPOINT_INTERVAL = 64


def _kda_checkpoint_rows(
    cu_seqlens: torch.Tensor,
    num_tokens: int,
    n_seqs: int,
    extend_seq_lens_cpu,
) -> int:
    """Valid checkpoint rows, excluding cuDNN's allocation-capacity tail."""
    interval = _KDA_CHECKPOINT_INTERVAL
    if n_seqs == 1:
        return (num_tokens + interval - 1) // interval

    lengths = extend_seq_lens_cpu
    if torch.is_tensor(lengths):
        lengths = lengths.tolist()
    if lengths is not None:
        lengths = [int(n) for n in lengths[:n_seqs]]
    if (
        lengths is None
        or len(lengths) != n_seqs
        or any(n < 0 for n in lengths)
        or sum(lengths) != num_tokens
    ):
        # Serving supplies CPU lengths. Direct callers without this metadata
        # need the packed boundaries to distinguish valid rows from capacity.
        boundaries = cu_seqlens.detach().cpu().tolist()
        lengths = [end - start for start, end in zip(boundaries, boundaries[1:])]
    return sum((n + interval - 1) // interval for n in lengths)


def _frost_sm_supported() -> bool:
    """True iff this device is in the FROST KDA engine's SM allowlist."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    return 100 <= sm <= 103 or sm == 107


class CuDNNKDAKernel(LinearAttnKernelBase):
    """cuDNN ``kimi_delta_attention`` KDA prefill (SM100-SM103/SM107).

    Decode and target_verify stay on their existing kernels: the public op
    returns a per-sequence final state but takes no pool slot indices,
    speculative rollback ports, or K3 fused output norm, so those phases have
    nothing to gain and state semantics to lose.
    """

    supports_safe_gate: bool = True

    def __init__(self, triton_fallback: LinearAttnKernelBase):
        self.supports_prefill = _frost_sm_supported()
        self._triton = triton_fallback
        self._kda_fn: Optional[callable] = None
        # cuDNN's decline signals; everything else is a real failure and rises.
        self._decline: tuple = (NotImplementedError,)
        # Detached fp32 views of the frozen gate params, keyed per source
        # tensor: this kernel instance is shared across KDA layers and each
        # layer owns its A_log/dt_bias.
        self._param_view: dict = {}
        self._unsupported_logged: set = set()
        self._engaged_logged: bool = False

    def _ensure_loaded(self) -> None:
        if self._kda_fn is not None:
            return
        try:
            import cudnn
            from cudnn.linear_attention import kimi_delta_attention
        except ImportError as e:
            raise ImportError(
                "The 'cudnn' KDA prefill backend requires the cuDNN frontend "
                "with the cutedsl extra, which is not installed. Install it "
                "with:\n"
                "    pip install 'nvidia-cudnn-frontend[cutedsl]'"
            ) from e

        self._kda_fn = kimi_delta_attention
        graph_not_supported = getattr(cudnn, "cudnnGraphNotSupportedError", None)
        if graph_not_supported is not None:
            self._decline = (graph_not_supported, NotImplementedError)
        logger.info("Using cuDNN KDA prefill (FROST, plan=%s)", _PLAN_NAME)

    def decode(self, *args, **kwargs):
        raise NotImplementedError("CuDNNKDAKernel is prefill-only")

    def target_verify(self, *args, **kwargs):
        raise NotImplementedError("CuDNNKDAKernel does not support target_verify")

    def _param_2d(
        self, t: Optional[torch.Tensor], *, rows: int, cols: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """Cached fp32 ``[rows]`` / ``[rows, cols]`` view of a gate parameter.

        K3 stores A_log as ``[1, 1, H, 1]`` and dt_bias flat as ``[H * K]``;
        the op wants ``[HO]`` and ``[HO, K]``. Returns None when the numel does
        not match, which the caller treats as ineligible.
        """
        if t is None:
            return None
        want = rows if cols is None else rows * cols
        if t.numel() != want:
            return None
        key = (t.data_ptr(), t.dtype, tuple(t.shape), rows, cols)
        view = self._param_view.get(key)
        if view is None:
            flat = t.detach().float()
            view = (
                flat.reshape(rows) if cols is None else flat.reshape(rows, cols)
            ).contiguous()
            self._param_view[key] = view
        return view

    def _live_sequence_count(
        self, query_start_loc: torch.Tensor, num_tokens: int, extend_seq_lens_cpu
    ) -> int:
        """Number of leading real sequences in ``query_start_loc``.

        Prefill CUDA graph replay under the Full backend pads the request tail
        with zero-length sentinels whose ``extend_start_loc`` sits at the flat
        end of the real tokens. cuDNN's behaviour on zero-length sequences is
        unverified, so the sentinels are trimmed off rather than handed over.
        Trimming needs the per-request lengths from the CPU side; without them
        (or if they do not account for every token) the boundaries are passed
        through as given.
        """
        n_seqs = query_start_loc.shape[0] - 1
        if extend_seq_lens_cpu is None:
            return n_seqs
        if torch.is_tensor(extend_seq_lens_cpu):
            lengths = [int(x) for x in extend_seq_lens_cpu.tolist()]
        else:
            lengths = [int(x) for x in extend_seq_lens_cpu]
        n_live = len(lengths)
        while n_live > 0 and lengths[n_live - 1] == 0:
            n_live -= 1
        if n_live > n_seqs or sum(lengths[:n_live]) != num_tokens:
            return n_seqs
        return n_live

    def _triton_extend(
        self,
        q,
        k,
        v,
        g,
        beta,
        *,
        ssm_states,
        cache_indices,
        query_start_loc,
        A_log,
        dt_bias,
        lower_bound,
        return_intermediate_states,
        kwargs,
    ):
        return self._triton.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            return_intermediate_states=return_intermediate_states,
            **kwargs,
        )

    def _log_unsupported(self, key: tuple, message: str, *args) -> None:
        if key in self._unsupported_logged:
            return
        self._unsupported_logged.add(key)
        logger.warning(message, *args)

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        lower_bound: Optional[float] = None,
        return_intermediate_states: bool = False,
        is_spec_decode: bool = False,
        extend_seq_lens_cpu: Optional[list] = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        # forward_extend expects (output, h) when tracking is requested and a
        # bare output otherwise, matching the Triton backend's contract.
        fallback = dict(
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            return_intermediate_states=return_intermediate_states,
            kwargs=dict(
                kwargs,
                is_spec_decode=is_spec_decode,
                extend_seq_lens_cpu=extend_seq_lens_cpu,
            ),
        )

        # Explicitly empty indices mean aligned tracking: only the committed
        # final state is needed. Without tracking metadata, a caller requesting
        # intermediate states receives the complete per-chunk series.
        track_h_src = kwargs.get("track_ssm_h_src")
        needs_interior_snapshot = return_intermediate_states and (
            track_h_src is None or track_h_src.numel() > 0
        )
        # draft_extend_v2 must stay rollback-able; this path commits the final
        # state into the pool.
        if is_spec_decode:
            return self._triton_extend(q, k, v, g, beta, **fallback)
        # Unbounded gate: see the module docstring.
        if lower_bound is None:
            return self._triton_extend(q, k, v, g, beta, **fallback)

        num_tokens = q.shape[1]
        H, K = q.shape[2], q.shape[3]
        HV, V = v.shape[2], v.shape[3]
        HO = max(H, HV)

        a_log_p = self._param_2d(A_log, rows=HO)
        dt_bias_p = self._param_2d(dt_bias, rows=HO, cols=K)
        eligible = (
            q.dtype in _FROST_IO_DTYPES
            and k.dtype == q.dtype
            and v.dtype == q.dtype
            and beta.dtype in (torch.float32, q.dtype)
            and K in _FROST_HEAD_DIMS
            and V in _FROST_HEAD_DIMS
            and ssm_states.dtype in _FROST_STATE_DTYPES
            and a_log_p is not None
            and dt_bias_p is not None
        )
        if not eligible:
            self._log_unsupported(
                (
                    q.dtype,
                    k.dtype,
                    v.dtype,
                    g.dtype,
                    beta.dtype,
                    K,
                    V,
                    ssm_states.dtype,
                    a_log_p is None,
                    dt_bias_p is None,
                ),
                "cuDNN KDA prefill supports fp16/bf16 q/k/v (matching), "
                "fp32-or-io beta, head dims in %s, fp32/bf16 pool state "
                "(passed in its native dtype), and "
                "A_log[%d]/dt_bias[%d,%d]; got q/k/v/beta=%s/%s/%s/%s, "
                "K/V=%d/%d, state=%s, A_log=%s, dt_bias=%s. "
                "Falling back to Triton.",
                _FROST_HEAD_DIMS,
                HO,
                HO,
                K,
                q.dtype,
                k.dtype,
                v.dtype,
                beta.dtype,
                K,
                V,
                ssm_states.dtype,
                tuple(A_log.shape) if A_log is not None else None,
                tuple(dt_bias.shape) if dt_bias is not None else None,
            )
            return self._triton_extend(q, k, v, g, beta, **fallback)

        n_seqs = self._live_sequence_count(
            query_start_loc, num_tokens, extend_seq_lens_cpu
        )
        # An all-sentinel batch (warmup / dummy run) would ask the op for zero
        # sequences with an empty state; Triton already absorbs it.
        if n_seqs == 0 or num_tokens == 0:
            return self._triton_extend(q, k, v, g, beta, **fallback)

        self._ensure_loaded()

        cu_seqlens = query_start_loc[: n_seqs + 1].to(torch.int32)

        # Padding requests (-1) land on slot 0, the pool's reserved dummy row
        # (memory_pool.py allocates size+1 rows and MambaSlotAllocator hands out
        # 1..size, so 0 is never owned by a request). NOT the trailing row: that
        # one IS allocatable, and scattering a padded final state there would
        # overwrite a live request's SSM state.
        slots = torch.where(
            cache_indices[:n_seqs] >= 0, cache_indices[:n_seqs], 0
        ).to(torch.int64)
        # The op takes a dense [N, HO, V, K] state, not pool slots. Gathering is
        # the boundary cost of that contract; the pool's own layout is a compile
        # contract for the kernels that read it in place and cannot be handed
        # over as a dense state.
        #
        # FROST accepts fp32 and bf16 state and returns final_state in the same
        # dtype. Preserve the pool dtype at the cuDNN boundary.
        initial_state = ssm_states.index_select(0, slots).contiguous()

        # [1, T, H, D] -> [T, H, D]. g/beta can carry padded rows past q's real
        # token count (the caller's [:real_num_tokens] slice narrows their batch
        # dim, not their tokens), so trim them.
        #
        # Keep the existing fp32 gate path used by this adapter.
        g_thd = g[0][:num_tokens].reshape(num_tokens, HO, K).to(_FROST_GATE_DTYPE)
        beta_thd = beta[0][:num_tokens].reshape(num_tokens, HO)

        checkpoint_kwargs = {}
        checkpoint_rows = 0
        if needs_interior_snapshot:
            checkpoint_kwargs["checkpoint_every_n_tokens"] = _KDA_CHECKPOINT_INTERVAL
            checkpoint_rows = _kda_checkpoint_rows(
                cu_seqlens, num_tokens, n_seqs, extend_seq_lens_cpu
            )

        h = None
        try:
            result = self._kda_fn(
                q[0],
                k[0],
                v[0],
                g_thd,
                beta_thd,
                cu_seqlens,
                scale=K**-0.5,
                initial_state=initial_state,
                output_final_state=True,
                # The KDA feature map, in-kernel; do not pre-norm q/k.
                use_qk_l2norm_in_kernel=True,
                # The extend path's beta is already post-sigmoid (see
                # KimiDeltaAttention.forward).
                use_beta_sigmoid_in_kernel=False,
                safe_gate=True,
                gate_lower_bound=lower_bound,
                a_log=a_log_p,
                dt_bias=dt_bias_p,
                plan_name=_PLAN_NAME,
                **checkpoint_kwargs,
            )
            if needs_interior_snapshot:
                o, final_state, checkpoints = result
                if (
                    checkpoints.ndim != 4
                    or tuple(checkpoints.shape[1:]) != (HO, V, K)
                    or checkpoints.shape[0] < checkpoint_rows
                ):
                    raise RuntimeError(
                        "Unexpected cuDNN KDA checkpoint layout: "
                        f"got {tuple(checkpoints.shape)}, expected at least "
                        f"{checkpoint_rows} rows of [{HO}, {V}, {K}]"
                    )
                # cuDNN packs sequences in cu_seqlens order. For length L,
                # ceil(L / 64) rows hold the states entering chunks 0, 1, ...;
                # row 0 is the initial state, not the state after 64 tokens.
                # This is exactly track_ssm_h_src's indexing convention.
                # Checkpoints already use the IO dtype and [HO, V, K] layout.
                h = checkpoints[:checkpoint_rows].unsqueeze(0)
            else:
                o, final_state = result
        except self._decline as e:
            # cuDNN declined this graph. Nothing has touched the pool yet, so
            # Triton can redo the batch from the committed state. A launch or
            # runtime error is a real failure and is left to propagate rather
            # than reported as a performance fallback.
            self._log_unsupported(
                ("declined", num_tokens, n_seqs, H, HV, K, V),
                "cuDNN declined the KDA prefill graph (T=%d sequences=%d "
                "H/HV=%d/%d K/V=%d/%d): %s. Falling back to Triton.",
                num_tokens,
                n_seqs,
                H,
                HV,
                K,
                V,
                e,
            )
            return self._triton_extend(q, k, v, g, beta, **fallback)

        if not self._engaged_logged:
            self._engaged_logged = True
            logger.info(
                "cuDNN KDA prefill engaged: T=%d sequences=%d H/HV=%d/%d "
                "K/V=%d/%d state=%s",
                num_tokens,
                n_seqs,
                H,
                HV,
                K,
                V,
                ssm_states.dtype,
            )

        ssm_states.index_copy_(0, slots, final_state.to(ssm_states.dtype))
        if return_intermediate_states:
            if h is None:
                # Aligned tracking reads the final state from the pool.
                h = q.new_empty((1, 0) + tuple(ssm_states.shape[1:]))
            return o.unsqueeze(0), h

        return o.unsqueeze(0)
