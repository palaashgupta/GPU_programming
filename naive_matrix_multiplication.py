import torch
import triton
import triton.language as tl

def naive_matmul_pseudo(A,B):

  M,K = A.shape
  K2, N = B.shape

  assert K == K2

  C = torch.zeros(M,N)
  def dot(v1,v2):
    accum = 0
    for p,q in zip(v1,v2):
      accum += p*q

    return accum

  num_pids = list(range(M*N))

  for i in num_pids:
    m = i // N # which row i am on 
    n = i % N # which column i am on
    
    C[m,n] = dot(A[m,:], B[:,n])

  return C


@triton.jit
def naive_matmul_kernal(
    A_ptr, B_ptr, C_ptr,
    M,N,K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn
):
  pid = tl.program_id(axis = 0)

  m = pid // N
  n = pid % N

  ## SANITY CHECK
  if (m >= M) or (n >= N):
    return

  ## DOT PRODUCT

  accum = 0.0
  for k in range(0,K):
    a_val = tl.load(A_ptr + m*stride_am + k*stride_ak)
    b_val = tl.load(B_ptr + k*stride_bk + n*stride_bn)
    
    accum += a_val * b_val

  tl.store(C_ptr + m*stride_cm + n*stride_cn, accum)
  

def naive_matmul(A,B):
  M,K = A.shape
  K2,N = B.shape

  C = torch.zeros(M,N, device = A.device, dtype=A.dtype)

  grid = (M*N,)

  naive_matmul_kernal[grid](
      A,B,C,
      M,N,K,
      A.stride(0), A.stride(1),
      B.stride(0), B.stride(1),
      C.stride(0), C.stride(1)
  )
  return C


A = torch.randn(6,8, device= "cuda")
B = torch.randn(8,4, device= "cuda")

print(naive_matmul(A,B) - A@B)