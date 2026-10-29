import torch
import triton
import triton.language as tl
import cupy as cp


# ============================================================
# Triton forward kernel
# ============================================================

@triton.jit
def embedding_forward_kernel(
    embeddings_ptr,
    indices_ptr,
    output_ptr,
    n_elements,
    embedding_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    DTYPE_FLAG: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    indices_ptr = tl.cast(indices_ptr, tl.pointer_type(tl.int32))
    if DTYPE_FLAG == 0:
        embeddings_ptr = tl.cast(
            embeddings_ptr, tl.pointer_type(tl.float32)
        )
        output_ptr = tl.cast(output_ptr, tl.pointer_type(tl.float32))
    else:
        embeddings_ptr = tl.cast(
            embeddings_ptr, tl.pointer_type(tl.float16)
        )
        output_ptr = tl.cast(output_ptr, tl.pointer_type(tl.float16))

    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_m = offsets_m < n_elements
    mask_n = offsets_n < embedding_dim
    mask = mask_m[:, None] & mask_n[None, :]

    indices = tl.load(indices_ptr + offsets_m, mask=mask_m, other=0)
    embedding_offsets = (
        indices[:, None] * embedding_dim + offsets_n[None, :]
    )
    values = tl.load(
        embeddings_ptr + embedding_offsets,
        mask=mask,
        other=0.0,
    )

    output_offsets = (
        offsets_m[:, None] * embedding_dim + offsets_n[None, :]
    )
    tl.store(output_ptr + output_offsets, values, mask=mask)


# ============================================================
# Triton backward kernel
# ============================================================

@triton.jit
def embedding_backward_kernel(
    grad_output_ptr,
    grad_weight_ptr,
    indices_ptr,
    n_elements,
    embedding_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    DTYPE_FLAG: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    indices_ptr = tl.cast(indices_ptr, tl.pointer_type(tl.int32))
    if DTYPE_FLAG == 0:
        grad_weight_ptr = tl.cast(
            grad_weight_ptr, tl.pointer_type(tl.float32)
        )
        grad_output_ptr = tl.cast(
            grad_output_ptr, tl.pointer_type(tl.float32)
        )
    else:
        grad_weight_ptr = tl.cast(
            grad_weight_ptr, tl.pointer_type(tl.float16)
        )
        grad_output_ptr = tl.cast(
            grad_output_ptr, tl.pointer_type(tl.float16)
        )

    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_m = offsets_m < n_elements
    mask_n = offsets_n < embedding_dim
    mask = mask_m[:, None] & mask_n[None, :]

    indices = tl.load(indices_ptr + offsets_m, mask=mask_m, other=0)
    grad_output = tl.load(
        grad_output_ptr
        + offsets_m[:, None] * embedding_dim
        + offsets_n[None, :],
        mask=mask,
        other=0.0,
    )

    grad_weight_offsets = (
        indices[:, None] * embedding_dim + offsets_n[None, :]
    )

    tl.atomic_add(
        grad_weight_ptr + grad_weight_offsets,
        grad_output,
        mask=mask,
        sem="relaxed",
    )


# ============================================================
# Helpers and CuPy-facing wrappers
# ============================================================

def _validate_inputs(embedding, indices):
    if embedding.ndim != 2:
        raise ValueError("embedding must have shape (vocab_size, embedding_dim)")
    if indices.size == 0:
        raise ValueError("indices must not be empty")
    if indices.size:
        min_index = int(indices.min().item())
        max_index = int(indices.max().item())
        if min_index < 0 or max_index >= embedding.shape[0]:
            raise IndexError(
                f"indices must be in [0, {embedding.shape[0] - 1}], "
                f"got min={min_index}, max={max_index}"
            )


def fused_embedding_forward(embedding, indices, use_dlpack=True):
    """
    Forward wrapper.

    Inputs may be CUDA PyTorch tensors (use_dlpack=False) or CuPy arrays
    (use_dlpack=True). Returns an object of the same array ecosystem:
    CuPy for CuPy input, PyTorch for PyTorch input.
    """
    if isinstance(embedding, torch.Tensor):
        embedding_t = embedding
        indices_t = indices
        return_cupy = False
    elif isinstance(embedding, cp.ndarray):
        if not use_dlpack:
            raise ValueError("CuPy inputs require use_dlpack=True in this wrapper")
        embedding_t = torch.utils.dlpack.from_dlpack(embedding)
        indices_t = torch.utils.dlpack.from_dlpack(indices)
        return_cupy = True
    else:
        raise TypeError("embedding must be a CUDA torch.Tensor or cupy.ndarray")

    if not embedding_t.is_cuda or not indices_t.is_cuda:
        raise ValueError("embedding and indices must be on CUDA")
    if embedding_t.dtype not in (torch.float32, torch.float16):
        raise TypeError("embedding dtype must be float32 or float16")

    indices_t = indices_t.to(torch.int32).contiguous()
    _validate_inputs(embedding_t, indices_t)

    original_shape = tuple(indices_t.shape)
    flat_indices = indices_t.reshape(-1).contiguous()
    embedding_t = embedding_t.contiguous()

    n_elements = flat_indices.numel()
    embedding_dim = embedding_t.shape[1]
    output = torch.empty(
        (n_elements, embedding_dim),
        device=embedding_t.device,
        dtype=embedding_t.dtype,
    )

    block_m = 32
    block_n = min(128, triton.next_power_of_2(embedding_dim))
    grid = (
        triton.cdiv(n_elements, block_m),
        triton.cdiv(embedding_dim, block_n),
    )

    embedding_forward_kernel[grid](
        embedding_t,
        flat_indices,
        output,
        n_elements,
        embedding_dim,
        block_m,
        block_n,
        0 if embedding_t.dtype == torch.float32 else 1,
    )

    output = output.reshape(*original_shape, embedding_dim).contiguous()
    if return_cupy:
        return cp.from_dlpack(output)
    return output


def fused_embedding_backward(grad_output, embeddings, indices, use_dlpack=True):
    """
    Backward wrapper: returns the accumulated gradient for each embedding row.
    Inputs can be CuPy arrays or CUDA PyTorch tensors.
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings_t = embeddings
        indices_t = indices
        grad_output_t = grad_output
        return_cupy = False
    elif isinstance(embeddings, cp.ndarray):
        if not use_dlpack:
            raise ValueError("CuPy inputs require use_dlpack=True in this wrapper")
        embeddings_t = torch.utils.dlpack.from_dlpack(embeddings)
        indices_t = torch.utils.dlpack.from_dlpack(indices)
        grad_output_t = torch.utils.dlpack.from_dlpack(grad_output)
        return_cupy = True
    else:
        raise TypeError("embeddings must be a CUDA torch.Tensor or cupy.ndarray")

    if not embeddings_t.is_cuda or not indices_t.is_cuda or not grad_output_t.is_cuda:
        raise ValueError("all inputs must be on CUDA")
    if embeddings_t.dtype not in (torch.float32, torch.float16):
        raise TypeError("embedding dtype must be float32 or float16")

    indices_t = indices_t.to(torch.int32).contiguous()
    _validate_inputs(embeddings_t, indices_t)

    original_shape = tuple(indices_t.shape)
    flat_indices = indices_t.reshape(-1).contiguous()
    embedding_dim = embeddings_t.shape[1]
    n_elements = flat_indices.numel()

    grad_output_t = grad_output_t.contiguous().reshape(n_elements, embedding_dim)
    if grad_output_t.dtype != embeddings_t.dtype:
        raise TypeError("grad_output dtype must match embeddings dtype")

    grad_weight = torch.zeros_like(embeddings_t.contiguous())

    block_m = 32
    block_n = min(128, triton.next_power_of_2(embedding_dim))
    grid = (
        triton.cdiv(n_elements, block_m),
        triton.cdiv(embedding_dim, block_n),
    )

    embedding_backward_kernel[grid](
        grad_output_t,
        grad_weight,
        flat_indices,
        n_elements,
        embedding_dim,
        block_m,
        block_n,
        0 if embeddings_t.dtype == torch.float32 else 1,
    )

    if return_cupy:
        return cp.from_dlpack(grad_weight)
    return grad_weight


# ============================================================
# Correctness tests against PyTorch
# ============================================================

def test_forward(dtype):
    torch.manual_seed(123)
    device = "cuda"

    vocab_size, embedding_dim = 257, 64
    indices = torch.randint(
        0, vocab_size, (4, 16), device=device, dtype=torch.int32
    )
    # Include boundary IDs explicitly.
    indices[0, 0] = 0
    indices[0, 1] = vocab_size - 1

    embedding = torch.randn(
        vocab_size, embedding_dim, device=device, dtype=dtype
    )

    actual = fused_embedding_forward(embedding, indices, use_dlpack=False)
    expected = torch.nn.functional.embedding(indices.long(), embedding)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    print(
        f"Forward {dtype}: PASS | shape={tuple(actual.shape)} | "
        f"max_abs_error={(actual - expected).abs().max().item():.6g}"
    )


def test_backward(dtype):
    torch.manual_seed(456)
    device = "cuda"

    vocab_size, embedding_dim = 31, 64
    indices = torch.randint(
        0, vocab_size, (4, 16), device=device, dtype=torch.int32
    )

    # Guarantee repeated IDs, which tests gradient accumulation.
    indices.reshape(-1)[:5] = 7

    embedding = torch.randn(
        vocab_size, embedding_dim, device=device, dtype=dtype
    )
    grad_output = torch.randn(
        *indices.shape, embedding_dim, device=device, dtype=dtype
    )

    actual = fused_embedding_backward(
        grad_output, embedding, indices, use_dlpack=False
    )

    # Reference embedding gradient via PyTorch autograd.
    reference_embedding = embedding.detach().clone().requires_grad_(True)
    reference_output = torch.nn.functional.embedding(
        indices.long(), reference_embedding
    )
    reference_output.backward(grad_output)
    expected = reference_embedding.grad

    torch.cuda.synchronize()
    # Half-precision atomic accumulation can have larger rounding differences.
    tol = 2e-2 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
    print(
        f"Backward {dtype}: PASS | shape={tuple(actual.shape)} | "
        f"max_abs_error={(actual - expected).abs().max().item():.6g}"
    )


def test_cupy_wrappers():
    """Smoke-test the CuPy/DLPack interface, when CuPy is installed."""
    vocab_size, embedding_dim = 64, 32
    embedding_cp = cp.random.standard_normal(
        (vocab_size, embedding_dim), dtype=cp.float32
    )
    indices_cp = cp.asarray([[0, 3, 3, 63], [5, 7, 0, 3]], dtype=cp.int32)

    output_cp = fused_embedding_forward(embedding_cp, indices_cp)
    expected_cp = embedding_cp[indices_cp.astype(cp.int64)]
    cp.cuda.get_current_stream().synchronize()
    cp.testing.assert_allclose(output_cp, expected_cp, rtol=1e-5, atol=1e-6)

    grad_output_cp = cp.ones_like(output_cp)
    grad_weight_cp = fused_embedding_backward(
        grad_output_cp, embedding_cp, indices_cp
    )
    expected_grad_cp = cp.zeros_like(embedding_cp)
    cp.add.at(
        expected_grad_cp,
        indices_cp.get().reshape(-1),
        grad_output_cp.get().reshape(-1, embedding_dim),
    )
    cp.testing.assert_allclose(
        grad_weight_cp, expected_grad_cp, rtol=1e-5, atol=1e-6
    )
    print("CuPy/DLPack wrapper tests: PASS")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this on a CUDA-enabled GPU.")

    print("PyTorch version:", torch.__version__)
    print("Triton version:", triton.__version__)
    print("CUDA device:", torch.cuda.get_device_name())

    for test_dtype in (torch.float32, torch.float16):
        test_forward(test_dtype)
        test_backward(test_dtype)

    test_cupy_wrappers()
    print("\nAll embedding tests passed.")


### OUTPUT ###
#PyTorch version: 2.11.0+cu130
#Triton version: 3.6.0
#CUDA device: Tesla T4
#Forward torch.float32: PASS | shape=(4, 16, 64) | max_abs_error=0
#Backward torch.float32: PASS | shape=(31, 64) | max_abs_error=9.53674e-07
#Forward torch.float16: PASS | shape=(4, 16, 64) | max_abs_error=0
#Backward torch.float16: PASS | shape=(31, 64) | max_abs_error=0.00390625
#CuPy/DLPack wrapper tests: PASS

#All embedding tests passed.