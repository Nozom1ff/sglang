# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton

from sglang.kernels.ops.attention.fla.utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def _chunk_indices_from_counts(num_chunks: list[int]) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in num_chunks])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1)


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    # The .tolist() reads cu_seqlens back to the host and syncs the stream.
    num_chunks = triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
    return _chunk_indices_from_counts(num_chunks).to(cu_seqlens)


def prepare_chunk_indices_from_lens(
    seq_lens: list[int], chunk_size: int
) -> torch.LongTensor:
    """Host-side ``prepare_chunk_indices`` from per-sequence lengths.

    Returns the same [num_chunks, 2] (sequence, chunk) table on the CPU, so a
    caller that already holds the lengths on the host can upload it without
    reading cu_seqlens back from the device.
    """
    return _chunk_indices_from_counts([triton.cdiv(n, chunk_size) for n in seq_lens])


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)
