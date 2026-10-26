import torch
import triton
import triton.language as tl

def blocked_matmul_pseudo(A,B, BLOCK_SIZE_M, BLOCK_SIZE_N):
  M,K = A.shape
  K, N = B.shape
  C = torch.zeros(M,N)

  assert M % BLOCK_SIZE_M == 0
  assert N % BLOCK_SIZE_N == 0

  num_block_rows = M // BLOCK_SIZE_M
  num_block_cols = N // BLOCK_SIZE_N

  total_blocks = num_block_rows*num_block_cols

  num_pids = range(total_blocks)

  for i in num_pids:
    block_m = i // num_block_cols # which block of rows we are on?
    block_n = i % num_block_cols # which block of columns we are on?

    m = block_m * BLOCK_SIZE_M
    n = block_n * BLOCK_SIZE_N

    block_A = A[m:m+BLOCK_SIZE_M, :]
    block_B = B[:,n: n+BLOCK_SIZE_N]

    C[m: m+BLOCK_SIZE_M, n: n+BLOCK_SIZE_N] = block_A @ block_B

  return C




def get_blocked_auto_tune_config():
  return [
      triton.Config({'BLOCK_SIZE_M':128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages = 3, num_warps = 8),
      triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages = 4, num_warps = 4),
      triton.Config({'BLOCK_SIZE_M':128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages = 4, num_warps = 4),
      triton.Config({'BLOCK_SIZE_M':128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages = 4, num_warps = 4),
      triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages = 4, num_warps = 4),
      triton.Config({'BLOCK_SIZE_M':128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages = 4, num_warps = 4),
      triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages = 5, num_warps = 2),
      triton.Config({'BLOCK_SIZE_M':32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages = 5, num_warps = 2)
  ]

@triton.autotune(configs = get_blocked_auto_tune_config(), key = ["M","N","K"])
@triton.jit
def blocked_matmul_kernal(
    A_ptr, B_ptr, C_ptr,
    M,N,K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
  pid = tl.program_id(axis = 0)

  num_blocks_col = tl.cdiv(N,BLOCK_SIZE_N)

  block_m = pid // num_blocks_col
  block_n = pid % num_blocks_col

  offs_m = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
  offs_n = block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
  offs_k = tl.arange(0, BLOCK_SIZE_K)

  a_ptr = A_ptr + (offs_m[:,None]* stride_am + offs_k[None,:]* stride_ak)
  b_ptr = B_ptr + (offs_k[:,None]* stride_bk + offs_n[None,:]* stride_bn)

  accumilator = tl.zeros((BLOCK_SIZE_M,BLOCK_SIZE_N), dtype= tl.float32)

  for k in range(0,tl.cdiv(K, BLOCK_SIZE_K)):
    a_mask = (offs_m[ : ,None]< M) & (offs_k[None, : ]< K - k*BLOCK_SIZE_K)
    a = tl.load(a_ptr, mask = a_mask, other = 0.0)

    b_mask = (offs_k[ : ,None]< K- k*BLOCK_SIZE_K) & (offs_n[None, : ]< N)
    b = tl.load(b_ptr, mask = b_mask, other = 0.0)

    accumilator += tl.dot(a,b, allow_tf32=False) # final block_size_m * block_size_n matrix

    a_ptr += BLOCK_SIZE_K* stride_ak
    b_ptr += BLOCK_SIZE_K* stride_bk

  c = accumilator.to(A_ptr.dtype.element_ty)

  c_offsets = stride_cm*offs_m[:,None] + stride_cn*offs_n[None,:]

  c_mask = (offs_m[:,None] < M) & (offs_n[None,:] < N)

  tl.store(C_ptr + c_offsets, c, mask = c_mask)


def blocked_matmul(A, B):

  M, K = A.shape
  K2, N = B.shape

  assert K == K2, "Inner dimensions must match"

  C = torch.empty(M, N, device=A.device, dtype=A.dtype)

  grid = lambda meta: (
      triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
  )

  blocked_matmul_kernal[grid](
      A, B, C,
      M, N, K,
      A.stride(0), A.stride(1),
      B.stride(0), B.stride(1),
      C.stride(0), C.stride(1),
  )
  return C


if __name__ == "__main__":

  cases = [
      (64, 64, 64),
      (128, 256, 64),
      (256, 128, 192),
      (100, 80, 90),  # not divisible by block sizes — masks should handle this
  ]

  for M, K, N in cases:
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    C = blocked_matmul(A, B)
    C_ref = A @ B
    max_diff = (C - C_ref).abs().max().item()
    ok = torch.allclose(C, C_ref, atol=1e-4, rtol=1e-4)
    print(f"A({M}x{K}) @ B({K}x{N}): match={ok}, max_diff={max_diff:.2e}")


### here is the following output: 
###
#A(64x64) @ B(64x64): match=True, max_diff=8.58e-06
#A(128x256) @ B(256x64): match=True, max_diff=4.20e-05
#A(256x128) @ B(128x192): match=True, max_diff=2.67e-05
#A(100x80) @ B(80x90): match=True, max_diff=1.24e-05
####