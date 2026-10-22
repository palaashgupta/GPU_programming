import time
import math
import torch
import triton 
import triton.language as tl


def blocked_vector_add_pseudocode(a, b, output, grid,n_elements, BLOCK_SIZE):
  
  PIDS = torch.arange(grid[0])

  a_ptr = 0
  b_ptr = 0
  output_ptr = 0

  for pid in PIDS:
    block_start = pid* BLOCK_SIZE
    block_index = block_start + torch.arange(BLOCK_SIZE)

    mask = block_index < n_elements

    block_index = block_index[mask]

    a_ptr_offset = a_ptr + block_index
    b_ptr_offset = b_ptr + block_index
    output_ptr_offset = output_ptr + block_index

    output[output_ptr_offset] = a[a_ptr_offset] + b[b_ptr_offset]

def blocked_vector_add_pseudocode_wrapper(a,b):

  BLOCK_SIZE = 1024
  n_elements = len(a)
  grid = (math.ceil(n_elements/BLOCK_SIZE),)

  output = torch.empty_like(a)

  blocked_vector_add_pseudocode(a, b, output, grid,n_elements, BLOCK_SIZE)

  return output





@triton.jit
def blocked_vector_add(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
  pid = tl.program_id(axis = 0)

  block_start = pid* BLOCK_SIZE
  offsets = block_start + tl.arange(0,BLOCK_SIZE)

  mask = (offsets < n_elements)

  a = tl.load(a_ptr + offsets, mask = mask)
  b = tl.load(b_ptr + offsets, mask = mask)

  output = a + b

  tl.store(output_ptr + offsets, output, mask = mask)
  

def blocked_vector_add_wrapper(a,b):
  output = torch.empty_like(a)
  n_elements = len(a)

  grid = lambda args: (triton.cdiv(n_elements, args['BLOCK_SIZE']),)

  blocked_vector_add[grid](a,b,output, n_elements, BLOCK_SIZE = 1024)

  return output





def benchmark_torch(a,b,iters = 100):
    for _ in range(10):
        _ = a + b

    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        out = a + b

    torch.cuda.synchronize()

    end = time.time()

    return (end - start) / iters



def benchmark_triton(a,b,iters = 100):
    for _ in range(10):
        _ = blocked_vector_add_wrapper(a,b)

    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        out = blocked_vector_add_wrapper(a,b)

    torch.cuda.synchronize()

    end = time.time()

    return (end - start) / iters


if __name__ == "__main__":

    a = torch.randn(1000).to("cuda")
    b = torch.randn(1000).to("cuda")
    

    pseudo_output = blocked_vector_add_pseudocode_wrapper(a,b)
    triton_output = blocked_vector_add_wrapper(a,b)

    torch_output = a + b

    assert torch.allclose(pseudo_output, triton_output)
    assert torch.allclose(triton_output, torch_output)

    print("Triton matches Torch")

    N = 50_000_000
    a = torch.randn(N,device = "cuda")
    b = torch.randn(N,device = "cuda")

    torch_time = benchmark_torch(a,b)
    triton_time = benchmark_triton(a,b)

    print(f"Torch time per iteration: {torch_time * 1e3:.3f} ms")
    print(f"Triton time per iteration: {triton_time * 1e3:.3f} ms")


## HERE Triton performed almost same as Torch. i.e. Triton matches Torch
#Torch time per iteration: 2.439 ms
#Triton time per iteration: 2.457 ms