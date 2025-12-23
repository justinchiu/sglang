import logging
from abc import ABC
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_dp_local_info,
    is_dp_attention_enabled,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.server_args import get_global_server_args

logger = logging.getLogger(__name__)

_GB = 1024 * 1024 * 1024
_MB = 1024 * 1024


def get_tensor_size_bytes(t: torch.Tensor):
    return np.prod(t.shape) * t.dtype.itemsize


class _RoutedExpertsDeviceCache:
    def __init__(
        self,
        max_running_requests: int,
        num_hidden_layers: int,
        num_experts_per_tok: int,
        num_fused_shared_experts: int,
        weights_dtype: torch.dtype,
        device: str,
    ) -> None:
        buffer_shape = (
            max(
                get_global_server_args().chunked_prefill_size
                * get_global_server_args().dp_size,
                max_running_requests,
            ),
            num_hidden_layers,
            num_experts_per_tok + num_fused_shared_experts,
        )
        self.buffer = torch.zeros(
            buffer_shape,
            dtype=torch.int32,
            device=device,
        )
        # Buffer for routing weights (topk_weights)
        self.weights_buffer = torch.zeros(
            buffer_shape,
            dtype=weights_dtype,
            device=device,
        )
        self._finalize_allocation_log()

    def get_buffer_size_bytes(self):
        assert hasattr(self, "buffer")
        return get_tensor_size_bytes(self.buffer) + get_tensor_size_bytes(self.weights_buffer)

    def capture_fwd_routed_experts(self, layer_id: int, topk_ids: torch.Tensor, topk_weights: torch.Tensor):
        assert layer_id is not None, "capturing routing experts but get layer_id None"
        batch, _ = topk_ids.shape
        self.buffer[:batch, layer_id, :] = topk_ids
        self.weights_buffer[:batch, layer_id, :] = topk_weights

    def _finalize_allocation_log(self):
        """Common logging and memory usage computation for captured experts buffers."""
        buffer_size_MB = self.get_buffer_size_bytes() / _MB
        logger.info(
            f"Routing experts device buffer allocated. #shape: {tuple(self.buffer.shape)}, size: {buffer_size_MB:.2f} MB"
        )


class _RoutedExpertsHostCache:
    def __init__(
        self,
        num_tokens: int,
        num_hidden_layers: int,
        num_experts_per_tok: int,
        weights_dtype: torch.dtype,
    ) -> None:
        self.num_tokens = num_tokens
        buffer_shape = (
            num_tokens,
            num_hidden_layers,
            num_experts_per_tok,
        )
        self.buffer = torch.zeros(
            buffer_shape,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        # Buffer for routing weights (topk_weights)
        self.weights_buffer = torch.zeros(
            buffer_shape,
            dtype=weights_dtype,
            device="cpu",
            pin_memory=True,
        )
        self._finalize_allocation_log()

    def get_buffer_size_bytes(self):
        assert hasattr(self, "buffer")
        return get_tensor_size_bytes(self.buffer) + get_tensor_size_bytes(self.weights_buffer)

    def _finalize_allocation_log(self):
        """Common logging and memory usage computation for captured experts buffers."""
        buffer_size_GB = self.get_buffer_size_bytes() / _GB
        logger.info(
            f"Routing experts host buffer allocated. #tokens: {self.num_tokens}, size: {buffer_size_GB:.2f} GB"
        )


class RoutedExpertsCapturer(ABC):
    @staticmethod
    def create(
        enable: bool,
        model_config: ModelConfig,
        num_fused_shared_experts: int,
        num_tokens: int,
        max_running_requests: int,
        device: str,
        weights_dtype: torch.dtype = torch.float16,
    ):
        if enable:
            return _RoutedExpertsCapturerReal(
                model_config,
                num_tokens=num_tokens,
                max_running_requests=max_running_requests,
                num_fused_shared_experts=num_fused_shared_experts,
                device=device,
                weights_dtype=weights_dtype,
            )
        else:
            return _RoutedExpertsCapturerNoop()

    def capture(self, layer_id: int, topk_ids: torch.Tensor, topk_weights: torch.Tensor):
        raise NotImplementedError

    def get_routed_experts(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        raise NotImplementedError

    def sync_fwd_experts_buffer_DtoH(
        self,
        device_loc: torch.Tensor,
        cpu_loc: torch.Tensor,
        can_run_graph: bool,
        cuda_graph_batch: int,
    ):
        raise NotImplementedError

    @contextmanager
    def with_forward(self, forward_batch):
        yield

    def get_host_cache(self):
        raise NotImplementedError

    def get_device_cache(self):
        raise NotImplementedError


class _RoutedExpertsCapturerReal(RoutedExpertsCapturer):
    """Capturer for routed experts with host buffer"""

    def __init__(
        self,
        model_config: ModelConfig,
        num_tokens: int,
        max_running_requests: int,
        num_fused_shared_experts: int,
        device: str,
        weights_dtype: torch.dtype,
    ):
        self.forward_batch = None
        self.num_fused_shared_experts = num_fused_shared_experts
        self.num_hidden_layers = model_config.hf_text_config.num_hidden_layers
        self.num_experts_per_tok = model_config.hf_text_config.num_experts_per_tok

        self.host_cache = _RoutedExpertsHostCache(
            num_tokens=num_tokens,
            num_hidden_layers=self.num_hidden_layers,
            num_experts_per_tok=self.num_experts_per_tok,
            weights_dtype=weights_dtype,
        )

        self.device_cache = _RoutedExpertsDeviceCache(
            max_running_requests=max_running_requests,
            num_hidden_layers=self.num_hidden_layers,
            num_experts_per_tok=self.num_experts_per_tok,
            num_fused_shared_experts=self.num_fused_shared_experts,
            weights_dtype=weights_dtype,
            device=device,
        )

    def capture(self, layer_id: int, topk_ids: torch.Tensor, topk_weights: torch.Tensor):
        self.device_cache.capture_fwd_routed_experts(layer_id, topk_ids, topk_weights)

    def sync_fwd_experts_buffer_DtoH(
        self,
        device_loc: torch.Tensor,
        cpu_loc: torch.Tensor,
        can_run_graph: bool,
        cuda_graph_batch: int,
    ):
        if is_dp_attention_enabled():
            local_start_pos, local_num_tokens = get_dp_local_info(self.forward_batch)
            # handle with cuda graph padding
            if can_run_graph:
                local_start_pos = get_attention_dp_rank() * cuda_graph_batch
                local_end_pos = local_start_pos + local_num_tokens
            else:
                local_end_pos = local_start_pos + local_num_tokens
        else:
            local_start_pos = 0
            local_end_pos = device_loc.shape[0]

        if self.forward_batch.num_token_non_padded is not None:
            assert local_end_pos - local_start_pos >= self.forward_batch.num_token_non_padded
            local_end_pos = local_start_pos + self.forward_batch.num_token_non_padded
            cpu_loc = cpu_loc[: self.forward_batch.num_token_non_padded]

        # Copy both ids and weights from device to host
        self.host_cache.buffer[cpu_loc] = self.device_cache.buffer[
            local_start_pos:local_end_pos, :, : self.num_experts_per_tok
        ].cpu()
        self.host_cache.weights_buffer[cpu_loc] = self.device_cache.weights_buffer[
            local_start_pos:local_end_pos, :, : self.num_experts_per_tok
        ].cpu()

    def get_routed_experts(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        cache_pool_idx = (
            req_to_token_pool.req_to_token[req_pool_idx][: seqlen - 1].cpu().clone()
        )
        # Return dict with both topk_ids and topk_weights
        return {
            "topk_ids": self.get_host_cache().buffer[cache_pool_idx],
            "topk_weights": self.get_host_cache().weights_buffer[cache_pool_idx],
        }

    @contextmanager
    def with_forward(self, forward_batch):
        self.forward_batch = forward_batch
        yield

    def get_host_cache(self):
        return self.host_cache

    def get_device_cache(self):
        return self.device_cache


class _RoutedExpertsCapturerNoop(RoutedExpertsCapturer):
    def __init__(self):
        pass

    def capture(self, layer_id: int, topk_ids: torch.Tensor, topk_weights: torch.Tensor):
        pass

    def get_routed_experts(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        return None

    def sync_fwd_experts_buffer_DtoH(
        self,
        device_loc: torch.Tensor,
        cpu_loc: torch.Tensor,
        can_run_graph: bool,
        cuda_graph_batch: int,
    ):
        pass

    @contextmanager
    def with_forward(self, forward_batch):
        yield

    def get_host_cache(self):
        pass

    def get_device_cache(self):
        pass


_global_expert_capturer: Optional[RoutedExpertsCapturer] = _RoutedExpertsCapturerNoop()


def get_global_experts_capturer():
    return _global_expert_capturer


def set_global_experts_capturer(capturer: RoutedExpertsCapturer):
    global _global_expert_capturer
    _global_expert_capturer = capturer