# Expert Routing Capture with TopK Weights

This document describes the expert routing capture feature added to SGLang for comparing FP8 sampler vs BF16 trainer routing decisions in MoE models.

## Branch

```
slime-dev-v0.5.6
```

Based on SGLang `v0.5.6` with SLIME patches applied.

## What Was Done

### 1. Applied SLIME sglang.patch (commit `d18fe2300`)

The main SLIME patch adds:
- `RoutedExpertsCapturer` class for capturing expert routing decisions
- RL on-policy training support
- Weight update mixins for online RL
- Various model compatibility fixes

### 2. Added TopK Weights Capture (commit `8bf77fed7`)

Extended the expert routing capture to include routing weights (not just indices):

#### Files Modified

| File | Changes |
|------|---------|
| `python/sglang/srt/layers/moe/routed_experts_capturer.py` | Added `weights_buffer` to device/host caches, `capture()` now accepts `topk_weights`, `get_routed_experts()` returns dict |
| `python/sglang/srt/layers/moe/topk.py` | Pass `topk_weights` to capturer |
| `python/sglang/srt/managers/io_struct.py` | Updated type annotations for dict format |
| `python/sglang/srt/managers/detokenizer_manager.py` | Serialize both ids and weights to lists |
| `python/sglang/srt/managers/schedule_batch.py` | Updated type annotation for `routed_experts` |

#### Data Format

Previously, `routed_experts` was a tensor of shape `(seqlen, num_layers, topk)` containing expert indices.

Now, `routed_experts` is a dict:
```python
{
    "topk_ids": tensor,      # shape (seqlen, num_layers, topk), dtype int32
    "topk_weights": tensor,  # shape (seqlen, num_layers, topk), dtype float16
}
```

When serialized (in HTTP responses), both are converted to nested lists.

## How to Use

### Option 1: Mount and Install Editable (Recommended)

```bash
# When running docker, mount the fork:
docker run ... -v /data/junxiong/slime/sglang:/data/junxiong/slime/sglang ...

# Inside container, install in editable mode:
pip install -e /data/junxiong/slime/sglang/python --no-deps
```

### Option 2: PYTHONPATH Override

```bash
export PYTHONPATH=/data/junxiong/slime/sglang/python:$PYTHONPATH
```

### Enable Expert Capture

In your SLIME config, set:
```yaml
capture_expert_routing: true
```

Or pass `--capture-expert-routing` to your training script.

This sets `enable_return_routed_experts=True` in SGLang ServerArgs.

## Testing

### 1. Basic Import Test

```bash
cd /data/junxiong/slime/sglang
python -c "
from sglang.srt.layers.moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    get_global_experts_capturer,
)
print('Import successful')

# Check that capture() accepts topk_weights
import inspect
sig = inspect.signature(RoutedExpertsCapturer.capture)
params = list(sig.parameters.keys())
assert 'topk_weights' in params, f'topk_weights not in {params}'
print('topk_weights parameter found in capture()')
"
```

### 2. Buffer Allocation Test

```bash
python -c "
import torch
from unittest.mock import MagicMock, patch

# Mock server args
mock_args = MagicMock()
mock_args.chunked_prefill_size = 512
mock_args.dp_size = 1

with patch('sglang.srt.layers.moe.routed_experts_capturer.get_global_server_args', return_value=mock_args):
    from sglang.srt.layers.moe.routed_experts_capturer import _RoutedExpertsDeviceCache

    cache = _RoutedExpertsDeviceCache(
        max_running_requests=256,
        num_hidden_layers=28,
        num_experts_per_tok=8,
        num_fused_shared_experts=0,
        weights_dtype=torch.float16,
        device='cpu',
    )

    print(f'buffer shape: {cache.buffer.shape}')
    print(f'weights_buffer shape: {cache.weights_buffer.shape}')
    print(f'buffer dtype: {cache.buffer.dtype}')
    print(f'weights_buffer dtype: {cache.weights_buffer.dtype}')

    assert cache.buffer.shape == cache.weights_buffer.shape
    assert cache.buffer.dtype == torch.int32
    assert cache.weights_buffer.dtype == torch.float16
    print('All assertions passed!')
"
```

### 3. Capture Test

```bash
python -c "
import torch
from unittest.mock import MagicMock, patch

mock_args = MagicMock()
mock_args.chunked_prefill_size = 512
mock_args.dp_size = 1

with patch('sglang.srt.layers.moe.routed_experts_capturer.get_global_server_args', return_value=mock_args):
    from sglang.srt.layers.moe.routed_experts_capturer import _RoutedExpertsDeviceCache

    cache = _RoutedExpertsDeviceCache(
        max_running_requests=256,
        num_hidden_layers=28,
        num_experts_per_tok=8,
        num_fused_shared_experts=0,
        weights_dtype=torch.float16,
        device='cpu',
    )

    # Simulate capture
    batch_size = 32
    topk_ids = torch.randint(0, 64, (batch_size, 8), dtype=torch.int32)
    topk_weights = torch.randn(batch_size, 8, dtype=torch.float16)

    cache.capture_fwd_routed_experts(layer_id=5, topk_ids=topk_ids, topk_weights=topk_weights)

    # Verify
    assert torch.equal(cache.buffer[:batch_size, 5, :], topk_ids)
    assert torch.equal(cache.weights_buffer[:batch_size, 5, :], topk_weights)
    print('Capture test passed!')
"
```

### 4. End-to-End Test with SLIME

Run a training job with expert routing capture enabled:

```bash
# In your training script or config
capture_expert_routing: true
```

Check the logs for:
```
Expert routing metrics skipped: rollout_experts=True, train_experts=True
```

Or actual metrics if both are present:
```
expert_top1_agreement: 0.85
expert_topk_overlap: 0.92
expert_weight_kl: 0.023
```

## Metrics Computed

When both rollout (sampler) and train expert routing data are available, these metrics are computed:

| Metric | Description |
|--------|-------------|
| `expert_top1_agreement` | Fraction of tokens where top-1 expert matches |
| `expert_topk_overlap` | Average Jaccard similarity of top-k expert sets |
| `expert_weight_kl` | KL divergence between routing weight distributions |
| `expert_lastk_agreement` | Agreement on last k tokens (most affected by precision) |

## Troubleshooting

### "rollout_experts=False, train_experts=True"

The sampler is not returning expert routing data. Check:
1. `enable_return_routed_experts=True` is set in SGLang server args
2. The model is an MoE model with TopK routing
3. You're using this forked SGLang (not the patched container version)

### Import Errors

Make sure your PYTHONPATH or pip install is correct:
```bash
python -c "import sglang; print(sglang.__file__)"
# Should show: /data/junxiong/slime/sglang/python/sglang/__init__.py
```

### Memory Issues

The weights buffer roughly doubles the memory usage of expert capture. If you see OOM errors, consider:
- Reducing `max_running_requests`
- Using `weights_dtype=torch.bfloat16` instead of `float32`
