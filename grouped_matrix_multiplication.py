import triton 
import triton.language as tl
import torch

def grouped_blocked_malmul_pseudo(A,B,BLOCK_SIZE_M,BLOCK_SIZE_N,GROUP_SIZE):

  M,K = A.shape
  K,N = B.shape

  C = torch.zeros(M,N)

  assert (M% BLOCK_SIZE_M) == 0, "Assuming M is divisible by BLOCK_SIZE_M"
  assert (N% BLOCK_SIZE_N) == 0, "Assuming N is divisible by BLOCK_SIZE_N" 

  assert (BLOCK_SIZE_M % GROUP_SIZE == 0) and (BLOCK_SIZE_N % GROUP_SIZE == 0), "Assuming both the block sizes are divisible by group size"

  num_block_rows = M // BLOCK_SIZE_M
  num_block_cols = N // BLOCK_SIZE_N

  num_pids = list(range(num_block_rows*num_block_cols)) 

  pid_visual = torch.zeros(num_block_rows,num_block_cols) # Just helps visual the pids 

  for i in num_pids:

    num_pid_in_group = GROUP_SIZE * num_block_cols

    group_id = i // num_pid_in_group # Which group are we on?

    ### Starting rows in m dimension
    first_pid_m = group_id* GROUP_SIZE

    ### What is true group size if not divisible 
    group_size_m = min(num_block_rows - first_pid_m, GROUP_SIZE)


    where_in_group = i % num_pid_in_group

    which_row_in_group = where_in_group % group_size_m


    pid_m = first_pid_m + which_row_in_group

    ### to find which column in group we are on
    pid_n = where_in_group // group_size_m


    pid_visual[pid_m,pid_n] = i

    m = pid_m * BLOCK_SIZE_M
    n = pid_n * BLOCK_SIZE_N

    block_A = A[m : m+ BLOCK_SIZE_M, :]
    block_B = B[: , n : n+BLOCK_SIZE_N]

    C[m : m+BLOCK_SIZE_M, n : n + BLOCK_SIZE_N] = block_A @ block_B

  return C, pid_visual


#A = torch.randn(64,128)
#B = torch.randn(128, 64)

#C, pid_visual = grouped_blocked_malmul_pseudo(A,B, 16,16,2)
#display(pid_visual)




def get_grouped_auto_tune_config():
  return [
      triton.Config({'BLOCK_SIZE_M':128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages = 3, num_warps = 8),
      triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages = 4, num_warps = 4),
      triton.Config({'BLOCK_SIZE_M':128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages = 4, num_warps = 4),
      triton.Config({'BLOCK_SIZE_M':128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages = 4, num_warps = 4),
      triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages = 4, num_warps = 4),
      triton.Config({'BLOCK_SIZE_M':128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages = 4, num_warps = 4),
      triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages = 5, num_warps = 2),
      triton.Config({'BLOCK_SIZE_M':32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages = 5, num_warps = 2)
  ]

@triton.autotune(configs = get_grouped_auto_tune_config(), key = ["M","N","K"])
@triton.jit
def grouped_matmul_kernal(
    A_ptr, B_ptr, C_ptr,
    M,N,K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
  pid = tl.program_id(axis = 0)

  num_pid_m = tl.cdiv(M,BLOCK_SIZE_M)
  num_pid_n = tl.cdiv(N,BLOCK_SIZE_N)
  num_pid_in_group = GROUP_SIZE_M * num_pid_n

  group_id = pid //num_pid_in_group
  first_pid_m = group_id * GROUP_SIZE_M
  group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

  pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m


  offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
  offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
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


def grouped_matmul(A, B):

  M, K = A.shape
  K2, N = B.shape

  assert K == K2, "Inner dimensions must match"

  C = torch.empty(M, N, device=A.device, dtype=A.dtype)

  grid = lambda meta: (
      triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
  )

  grouped_matmul_kernal[grid](
      A, B, C,
      M, N, K,
      A.stride(0), A.stride(1),
      B.stride(0), B.stride(1),
      C.stride(0), C.stride(1),
  )
  return C


if __name__ == "__main__":
  # --- CPU pseudo: (M, K, N, BLOCK_M, BLOCK_N, GROUP_SIZE) ---
  # Requires M % BLOCK_M == 0, N % BLOCK_N == 0,
  # and BLOCK_M/BLOCK_N divisible by GROUP_SIZE (per your asserts).
  pseudo_cases = [
      (64, 128, 64, 16, 16, 2),   # small, group=2 — good for printing pid_visual
      (128, 64, 128, 32, 32, 4),
      (256, 128, 192, 64, 32, 8),
  ]

  print("=== grouped_blocked_malmul_pseudo ===")
  for M, K, N, bm, bn, g in pseudo_cases:
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    C, pid_visual = grouped_blocked_malmul_pseudo(A, B, bm, bn, g)
    ok = torch.allclose(C, A @ B)
    print(f"A({M}x{K})@B({K}x{N}) blocks({bm},{bn}) group={g}: match={ok}")
    if (M, K, N) == (64, 128, 64):
      print("pid_visual (row-major group swizzle):\n", pid_visual)

  # --- Triton GPU: (M, K, N) — GROUP_SIZE_M comes from autotune configs ---
  print("\n=== grouped_matmul (Triton) ===")
  assert torch.cuda.is_available(), "CUDA is required to run the Triton kernel"

  triton_cases = [
      (64, 64, 64),
      (128, 256, 64),
      (256, 128, 192),
      (512, 256, 512),
      (100, 80, 90),   # not divisible by blocks — masks should handle this
  ]

  for M, K, N in triton_cases:
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    C = grouped_matmul(A, B)
    C_ref = A @ B
    max_diff = (C - C_ref).abs().max().item()
    ok = torch.allclose(C, C_ref, atol=1e-4, rtol=1e-4)
    print(f"A({M}x{K}) @ B({K}x{N}): match={ok}, max_diff={max_diff:.2e}")


#
#=== grouped_matmul (Triton) ===
#A(64x64) @ B(64x64): match=True, max_diff=7.63e-06
#A(128x256) @ B(256x64): match=True, max_diff=3.43e-05
#A(256x128) @ B(128x192): match=True, max_diff=1.91e-05
#A(512x256) @ B(256x512): match=True, max_diff=0.00e+00
#A(100x80) @ B(80x90): match=True, max_diff=9.54e-06