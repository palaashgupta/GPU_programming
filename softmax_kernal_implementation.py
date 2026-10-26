
import torch
import triton
import triton.language as tl
import cupy as cp
import pandas as pd
import statistics


DLPACK_DISABLED = False


# ============================================================
# 1. CuPy reference forward implementation
# ============================================================

def naive_softmax(x):
    row_max = cp.max(x, axis=-1, keepdims=True)
    x_stable = x - row_max
    x_exp = cp.exp(x_stable)
    row_sum = cp.sum(x_exp, axis=-1, keepdims=True)
    return x_exp / row_sum


def naive_softmax_backward(grad, s):
    grad_x_s = grad * s
    return grad_x_s - s * cp.sum(
        grad_x_s, axis=-1, keepdims=True
    )


# ============================================================
# 2. Triton forward kernel
# ============================================================

@triton.jit
def softmax_kernal_forward(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    DTYPE_FLAG: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    pointer_dtype = tl.float32 if DTYPE_FLAG == 0 else tl.float16
    input_ptr = tl.cast(input_ptr, tl.pointer_type(pointer_dtype))
    output_ptr = tl.cast(output_ptr, tl.pointer_type(pointer_dtype))

    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offset = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offset

    mask = col_offset < n_cols

    row = tl.load(
        input_ptrs,
        mask=mask,
        other=-float("inf")
    )

    row_minus_max = row - tl.max(row, axis=0)
    row_exp = tl.exp(row_minus_max)
    row_sum = tl.sum(row_exp, axis=0)

    softmax_out = (row_exp / row_sum).to(
        output_ptr.dtype.element_ty
    )

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offset

    tl.store(output_ptrs, softmax_out, mask=mask)


# ============================================================
# 3. Triton backward kernel
# ============================================================

@triton.jit
def softmax_kernal_backward(
    output_ptr, output_row_stride,
    s_ptr, s_row_stride,
    grad_ptr, grad_row_stride,
    DTYPE_FLAG: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    pointer_dtype = tl.float32 if DTYPE_FLAG == 0 else tl.float16
    s_ptr = tl.cast(s_ptr, tl.pointer_type(pointer_dtype))
    grad_ptr = tl.cast(grad_ptr, tl.pointer_type(pointer_dtype))
    output_ptr = tl.cast(output_ptr, tl.pointer_type(pointer_dtype))

    s_row_start_ptr = s_ptr + row_idx * s_row_stride
    grad_row_start_ptr = grad_ptr + row_idx * grad_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)

    s_ptrs = s_row_start_ptr + col_offsets
    grad_ptrs = grad_row_start_ptr + col_offsets

    mask = col_offsets < n_cols

    s_row = tl.load(s_ptrs, mask=mask, other=0.0).to(tl.float32)
    grad_row = tl.load(
        grad_ptrs, mask=mask, other=0.0
    ).to(tl.float32)

    # dX = S * (dY - sum(dY * S))
    grad_x_s = grad_row * s_row
    softmax_grad_out = (
        grad_x_s - s_row * tl.sum(grad_x_s, axis=0)
    )

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets

    tl.store(
        output_ptrs,
        softmax_grad_out,
        mask=mask
    )


# ============================================================
# 4. Forward wrapper
# ============================================================

def fused_softmax_forward(x, use_dlpack=True):
    n_row, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    if not DLPACK_DISABLED and use_dlpack:
        # CuPy -> PyTorch using DLPack
        x_torch = torch.utils.dlpack.from_dlpack(x)
        y_torch = torch.empty_like(x_torch)

        dtype_flag = 0 if x_torch.dtype == torch.float32 else 1

        input_row_stride = x_torch.stride(0)
        output_row_stride = y_torch.stride(0)

        grid = (n_row,)

        softmax_kernal_forward[grid](
            y_torch,
            x_torch,
            input_row_stride,
            output_row_stride,
            dtype_flag,
            n_cols,
            BLOCK_SIZE,
        )

        # PyTorch -> CuPy using DLPack
        return cp.from_dlpack(y_torch)

    else:
        # Direct CuPy-array path
        y = cp.empty_like(x)

        input_row_stride = x.strides[0] // x.itemsize
        output_row_stride = y.strides[0] // y.itemsize

        dtype_flag = 0 if x.dtype == cp.float32 else 1

        grid = (n_row,)

        softmax_kernal_forward[grid](
            y,
            x,
            input_row_stride,
            output_row_stride,
            dtype_flag,
            n_cols,
            BLOCK_SIZE,
        )

        return y


# ============================================================
# 5. Backward wrapper
# ============================================================

def fused_softmax_backward(grad, s, use_dlpack=True):
    n_rows, n_cols = grad.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    if not DLPACK_DISABLED and use_dlpack:
        grad_torch = torch.utils.dlpack.from_dlpack(grad)
        s_torch = torch.utils.dlpack.from_dlpack(s)

        grad_input = torch.empty_like(grad_torch)

        grad_row_stride = grad_torch.stride(0)
        s_row_stride = s_torch.stride(0)
        output_row_stride = grad_input.stride(0)

        dtype_flag = (
            0 if grad_torch.dtype == torch.float32 else 1
        )

        grid = (n_rows,)

        softmax_kernal_backward[grid](
            grad_input, output_row_stride,
            s_torch, s_row_stride,
            grad_torch, grad_row_stride,
            dtype_flag,
            n_cols,
            BLOCK_SIZE,
        )

        return cp.from_dlpack(grad_input)

    else:
        grad_input = cp.empty_like(grad)

        grad_row_stride = grad.strides[0] // grad.itemsize
        s_row_stride = s.strides[0] // s.itemsize
        output_row_stride = grad_input.strides[0] // grad_input.itemsize

        dtype_flag = 0 if grad.dtype == cp.float32 else 1

        grid = (n_rows,)

        softmax_kernal_backward[grid](
            grad_input, output_row_stride,
            s, s_row_stride,
            grad, grad_row_stride,
            dtype_flag,
            n_cols,
            BLOCK_SIZE,
        )

        return grad_input


# ============================================================
# 6. Repeated median-latency benchmark
# ============================================================

def median_benchmark(
    fn,
    repeats=5,
    warmup=20,
    rep=50,
):
    """
    Median of repeated Triton benchmark measurements.

    Timing includes wrapper overhead and output allocation.
    """
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    timings = []

    for _ in range(repeats):
        ms = triton.testing.do_bench(
            fn,
            warmup=20,
            rep=rep,
        )

        torch.cuda.synchronize()
        timings.append(ms)

    return statistics.median(timings)


# ============================================================
# 7. Correctness and performance test for one shape
# ============================================================

def benchmark_softmax_case(M, N, dtype):
    # Generate input and upstream gradient on the GPU
    x = cp.random.standard_normal((M, N)).astype(dtype)
    grad = cp.random.standard_normal((M, N)).astype(dtype)

    # ---------------- Forward correctness ----------------
    y_ref = naive_softmax(x)
    y_tri = fused_softmax_forward(x)

    torch.cuda.synchronize()

    fwd_error = float(
        cp.max(
            cp.abs(
                y_tri.astype(cp.float32)
                - y_ref.astype(cp.float32)
            )
        ).item()
    )

    tol = 2e-2 if dtype == cp.float16 else 1e-4

    fwd_ok = bool(
        cp.allclose(
            y_tri, y_ref,
            atol=tol,
            rtol=tol,
        ).item()
    )

    # ---------------- Backward correctness ----------------
    dx_ref = naive_softmax_backward(grad, y_ref)
    dx_tri = fused_softmax_backward(grad, y_tri)

    torch.cuda.synchronize()

    bwd_error = float(
        cp.max(
            cp.abs(
                dx_tri.astype(cp.float32)
                - dx_ref.astype(cp.float32)
            )
        ).item()
    )

    bwd_ok = bool(
        cp.allclose(
            dx_tri, dx_ref,
            atol=tol,
            rtol=tol,
        ).item()
    )

    # ---------------- Forward benchmark ----------------
    tri_fwd_ms = median_benchmark(
        lambda: fused_softmax_forward(x)
    )

    ref_fwd_ms = median_benchmark(
        lambda: naive_softmax(x)
    )

    # ---------------- Backward benchmark ----------------
    # Softmax output is precomputed.
    # Only the gradient operation is measured.
    tri_bwd_ms = median_benchmark(
        lambda: fused_softmax_backward(grad, y_tri)
    )

    ref_bwd_ms = median_benchmark(
        lambda: naive_softmax_backward(grad, y_ref)
    )

    # ---------------- Performance metrics ----------------
    element_bytes = x.dtype.itemsize
    elements = M * N

    # Logical bytes:
    # Forward = input read + output write
    # Backward = softmax read + upstream-gradient read + output write
    fwd_gbps = (
        2 * elements * element_bytes
        / (tri_fwd_ms * 1e6)
    )

    bwd_gbps = (
        3 * elements * element_bytes
        / (tri_bwd_ms * 1e6)
    )

    return {
        "M": M,
        "N": N,
        "dtype": str(dtype),
        "forward_correct": fwd_ok,
        "backward_correct": bwd_ok,
        "forward_max_error": fwd_error,
        "backward_max_error": bwd_error,
        "triton_forward_ms": tri_fwd_ms,
        "cupy_forward_ms": ref_fwd_ms,
        "forward_speedup": ref_fwd_ms / tri_fwd_ms,
        "triton_backward_ms": tri_bwd_ms,
        "cupy_backward_ms": ref_bwd_ms,
        "backward_speedup": ref_bwd_ms / tri_bwd_ms,
        "forward_effective_GB_s": fwd_gbps,
        "backward_effective_GB_s": bwd_gbps,
    }


# ============================================================
# 8. Run the full test suite
# ============================================================

if __name__ == "__main__":
    assert torch.cuda.is_available(), (
        "CUDA is required to run the Triton kernels"
    )

    print("PyTorch version:", torch.__version__)
    print("PyTorch CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("CuPy version:", cp.__version__)
    print("Triton version:", triton.__version__)

    cases = [
        (32, 128),
        (256, 512),
        (1024, 1024),
        (512, 2049),   # Non-power-of-two width
        (2048, 4096),
    ]

    results = []

    for dtype in [cp.float32, cp.float16]:
        for M, N in cases:
            result = benchmark_softmax_case(M, N, dtype)
            results.append(result)

            print(
                f"\nShape=({M}, {N}), dtype={dtype}"
            )
            print(
                f"Forward correct: {result['forward_correct']}, "
                f"max error: {result['forward_max_error']:.3e}"
            )
            print(
                f"Backward correct: {result['backward_correct']}, "
                f"max error: {result['backward_max_error']:.3e}"
            )
            print(
                f"Forward: Triton={result['triton_forward_ms']:.4f} ms, "
                f"CuPy={result['cupy_forward_ms']:.4f} ms, "
                f"speedup={result['forward_speedup']:.2f}x"
            )
            print(
                f"Backward: Triton={result['triton_backward_ms']:.4f} ms, "
                f"CuPy={result['cupy_backward_ms']:.4f} ms, "
                f"speedup={result['backward_speedup']:.2f}x"
            )
            print(
                f"Effective bandwidth: "
                f"forward={result['forward_effective_GB_s']:.2f} GB/s, "
                f"backward={result['backward_effective_GB_s']:.2f} GB/s"
            )

    results_df = pd.DataFrame(results)

    display_columns = [
        "M", "N", "dtype",
        "forward_correct", "backward_correct",
        "forward_max_error", "backward_max_error",
        "triton_forward_ms", "cupy_forward_ms",
        "forward_speedup",
        "triton_backward_ms", "cupy_backward_ms",
        "backward_speedup",
        "forward_effective_GB_s",
        "backward_effective_GB_s",
    ]

    print("\n========== SUMMARY ==========")
    display(results_df[display_columns].round(4))
