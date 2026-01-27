# Encoder Fallback Test Summary

## Overview
Added debug logging and comprehensive tests to verify that the `SequentialTwoStagePredictor` correctly switches between two encoding architectures:
1. **TCN Path**: Sequence-aware temporal convolution (when seq_len divides in_channels evenly)
2. **Fallback Path**: Plain Conv1d stack (when sequence structure cannot be detected)

## Changes Made

### 1. Added Debug Logging to `cnnopacus.py`

**In `__init__` method:**
- Logs `in_channels`, `seq_len`, and inferred `feature_dim`
- Prints `[Encoder] Using TCN encoder (...)` when sequence structure is detected
- Prints `[Encoder] FALLBACK to plain Conv1d (...)` when falling back

**In `_encode_x` method:**
- Logs `[_encode_x] Using FALLBACK Conv1d path` when fallback is triggered
- Shows input shape and intermediate transformation steps

### 2. Test Script: `test_encoder_fallback.py`

5 test cases verify both paths work correctly:

| Test | Condition | Result |
|------|-----------|--------|
| TEST 1 | seq_len=24, in_channels=240 (240÷24=10) | TCN path ✓ |
| TEST 2 | seq_len=24, in_channels=250 (250÷24≠int) | Fallback ✓ |
| TEST 3 | seq_len=None, in_channels=240 | Fallback ✓ |
| TEST 4 | Full forward pass (TCN) | Output (4,128) and (4,2) ✓ |
| TEST 5 | Full forward pass (Fallback) | Output (4,128) and (4,2) ✓ |

## Test Output Highlights

**TCN Activation:**
```
[SequentialTwoStagePredictor] in_channels=240, seq_len=24, feature_dim=10
[Encoder] Using TCN encoder (sequence-aware: [N, 24, 10])
[PASS] TCN path works: input torch.Size([4, 24, 10]) -> hidden torch.Size([4, 128])
```

**Fallback Activation (Bad Divisor):**
```
[SequentialTwoStagePredictor] in_channels=250, seq_len=24, feature_dim=None
[Encoder] FALLBACK to plain Conv1d (flat input: [N, 1, 250])
[_encode_x] Using FALLBACK Conv1d path with input shape torch.Size([4, 250])
[_encode_x] Flattened to torch.Size([4, 250]), adding channel dim -> torch.Size([4, 1, 250])
```

**Fallback Activation (No seq_len):**
```
[SequentialTwoStagePredictor] in_channels=240, seq_len=None, feature_dim=None
[Encoder] FALLBACK to plain Conv1d (flat input: [N, 1, 240])
```

## Fallback Conditions

The fallback to plain Conv1d occurs when **either**:
1. `seq_len` doesn't divide `in_channels` evenly
2. `seq_len` is `None`

When fallback occurs:
- Input is flattened to 2D: `[batch_size, in_channels]`
- A channel dimension is added: `[batch_size, 1, in_channels]`
- Plain Conv1d stack processes: 1→c1→c2→hidden_channels with adaptive pooling

## To Run Tests

```bash
python test_encoder_fallback.py
```

All tests pass with clear debug output showing which path is active.
