/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
==============================================================================*/

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU

#if GOOGLE_CUDA
#include <cub/cub.cuh>
#include "third_party/gpus/cuda/include/cuda.h"
#endif
#include "fused_instance_norm_op.h"
#include "norm_util.h"
#include "tensorflow/core/kernels/gpu_prim.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

using namespace InNorm;

static const int warp_per_block = kBlockSize / kWarpSize;

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T, typename U, typename Op>
__global__ __launch_bounds__(1024) void InstanceNormToTempWelford(
    const T* __restrict__ x, const int N, const int C, const int D,
    U* __restrict__ temp_mean, U* __restrict__ temp_m2,
    U* __restrict__ temp_count, Op op, const bool is_channel_first = true) {
  if (is_channel_first) {
    const int row_offset = threadIdx.x + blockIdx.x * blockDim.x;

    for (int row_idx = blockIdx.y; row_idx < N * C; row_idx += gridDim.y) {
      WFGeneric<U> wf_partial;

      for (int i = row_offset; i < D; i += gridDim.x * blockDim.x) {
        op.Update(x, row_idx, i, wf_partial);
      }

      WFGeneric<U> wf_block = WelfordBlockAllReduce<U>(wf_partial);
      if (threadIdx.x == 0) {
        temp_mean[row_idx * gridDim.x + blockIdx.x] = wf_block.mean;
        temp_m2[row_idx * gridDim.x + blockIdx.x] = wf_block.m2;
        temp_count[row_idx * gridDim.x + blockIdx.x] = wf_block.n;
      }
    }
  } else {
    const int x_tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int y_tid = threadIdx.y + blockIdx.y * blockDim.y;
    const int z_tid = threadIdx.z + blockIdx.z * blockDim.z;
    if (x_tid >= C) return;
    WFGeneric<U> wf_partial;

    for (int i = y_tid; i < D; i += blockDim.y * gridDim.y) {
      op.Update(x, z_tid, i, x_tid, wf_partial);
    }

    temp_mean[(z_tid * C + x_tid) * blockDim.y * gridDim.y + y_tid] =
        wf_partial.mean;
    temp_m2[(z_tid * C + x_tid) * blockDim.y * gridDim.y + y_tid] =
        wf_partial.m2;
    temp_count[(z_tid * C + x_tid) * blockDim.y * gridDim.y + y_tid] =
        wf_partial.n;
  }
}

template <typename U, typename Op>
__global__ __launch_bounds__(1024) void InstanceNormTempToOutWelford(
    const U* __restrict__ temp_mean, const U* __restrict__ temp_m2,
    const U* __restrict__ temp_count, const int N, const int C, const int cols,
    U* __restrict__ cache_mean, U* __restrict__ cache_ivar, Op op) {
  for (int k = blockIdx.x; k < N * C; k += gridDim.x) {
    WFGeneric<U> wf_partial;
    for (int i = threadIdx.x; i < cols; i += kBlockSize) {
      int idx = k * cols + i;
      WFGeneric<U> wf_local{temp_mean[idx], temp_m2[idx], temp_count[idx]};
      wf_partial = WFGeneric<U>()(wf_local, wf_partial);
    }

    WFGeneric<U> wf_block = WelfordBlockAllReduce<U>(wf_partial);
    if (threadIdx.x == 0) {
      cache_mean[k] = wf_block.mean;
      cache_ivar[k] = op.Finalize(wf_block);
    }
  }
}

template <typename U, typename Op>
__global__ __launch_bounds__(1024) void InstanceNormRowReduceTempToOutFused(
    const U* __restrict__ temp1, const U* __restrict__ temp2,
    const U* __restrict__ temp3, const U* __restrict__ temp4, const int N,
    const int C, const int cols, U* __restrict__ cache1, U* __restrict__ cache2,
    U* __restrict__ cache3, U* __restrict__ cache4, Op op) {
  typedef cub::BlockReduce<U, kBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  for (int k = blockIdx.x; k < N * C; k += gridDim.x) {
    U partial_sum1 = 0;
    U partial_sum2 = 0;
    for (int i = threadIdx.x; i < cols; i += kBlockSize) {
      partial_sum1 += temp1[k * cols + i];
      partial_sum2 += temp2[k * cols + i];
    }

    U sum1 = BlockReduce(temp_storage).Sum(partial_sum1);
    U sum2 = BlockReduce(temp_storage).Sum(partial_sum2);

    if (threadIdx.x == 0) {
      cache1[k] = op.Finalize(sum1);
      cache2[k] = op.Finalize(sum2);
    }
  }
  if (blockIdx.x < C) {  // only first C block participate
    U partial_sum3 = 0;
    U partial_sum4 = 0;
    for (int k = blockIdx.x; k < N * C; k += C) {
      for (int i = threadIdx.x; i < cols; i += kBlockSize) {
        partial_sum3 += temp3[k * cols + i];
        partial_sum4 += temp4[k * cols + i];
      }
    }
    U sum3 = BlockReduce(temp_storage).Sum(partial_sum3);
    U sum4 = BlockReduce(temp_storage).Sum(partial_sum4);
    if (threadIdx.x == 0) {
      cache3[blockIdx.x] = op.Finalize(sum3);
      cache4[blockIdx.x] = op.Finalize(sum4);
    }
  }
}

template <typename U, typename Op>
__global__ __launch_bounds__(1024) void InstanceNormRowReduceTempToOut(
    const U* __restrict__ temp, const int N, const int C, const int cols,
    U* __restrict__ cache, Op op) {
  typedef cub::BlockReduce<U, kBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  for (int k = blockIdx.x; k < N * C; k += gridDim.x) {
    U partial_sum = 0;

    for (int i = threadIdx.x; i < cols; i += kBlockSize) {
      partial_sum += temp[k * cols + i];
    }

    U sum = BlockReduce(temp_storage).Sum(partial_sum);

    if (threadIdx.x == 0) {
      cache[k] = op.Finalize(sum);
    }
  }
}

template <typename T, typename U, typename Op>
__global__ __launch_bounds__(1024) void InstanceNormRowReduceInToOutFused(
    const T* __restrict__ in, const int N, const int D, U* out1, U* out2,
    U* out3, U* out4, Op op) {
  const int tid = threadIdx.x;

  typedef cub::BlockReduce<U, kBlockSize> BlockReduce;
  __shared__ union { typename BlockReduce::TempStorage reduce; } temp_storage;

  for (int k = blockIdx.x; k < N; k += gridDim.x) {
    U partial_sum1 = 0;
    U partial_sum2 = 0;
    U partial_sum3 = 0;
    U partial_sum4 = 0;
    U ret1, ret2, ret3, ret4;

    for (int i = tid; i < D; i += kBlockSize) {
      op.Compute(in, k, i, &ret1, &ret2, &ret3, &ret4);
      partial_sum1 += ret1;
      partial_sum2 += ret2;
      partial_sum3 += ret3;
      partial_sum4 += ret4;
    }

    U sum1 = BlockReduce(temp_storage.reduce).Sum(partial_sum1);
    U sum2 = BlockReduce(temp_storage.reduce).Sum(partial_sum2);
    U sum3 = BlockReduce(temp_storage.reduce).Sum(partial_sum3);
    U sum4 = BlockReduce(temp_storage.reduce).Sum(partial_sum4);
    if (tid == 0) {
      out1[k] = op.Finalize(sum1);
      out2[k] = op.Finalize(sum2);
      out3[k] = op.Finalize(sum3);
      out4[k] = op.Finalize(sum4);
    }
    __syncthreads();
  }
}

template <typename T, typename U, typename Op>
__global__ __launch_bounds__(1024) void InstanceNormRowReduceInToOutWF(
    const T* __restrict__ in, const int N, const int D, U* out1, U* out2,
    Op op) {
  const int tid = threadIdx.x;

  U block_mean, block_m2, block_count;
  for (int k = blockIdx.x; k < N; k += gridDim.x) {
    WFGeneric<U> wf_thread;

    for (int i = tid; i < D; i += kBlockSize) {
      op.Update(in, k, i, wf_thread);
    }

    WFGeneric<U> wf_row = WelfordBlockAllReduce<U>(wf_thread);

    if (tid == 0) {
      out1[k] = wf_row.mean;
      out2[k] = op.Finalize(wf_row);
    }
  }
}

template <typename T, typename U, typename Op1, typename Op2>
__global__ __launch_bounds__(1024) void InstanceNormRowReduceInToOut(
    const T* __restrict__ in, const int N, const int D, U* out1, U* out2,
    Op1 op1, Op2 op2) {
  const int tid = threadIdx.x;

  typedef cub::BlockReduce<U, kBlockSize> BlockReduce;
  __shared__ union {
    typename BlockReduce::TempStorage reduce;
    U broadcast[1];
  } temp_storage;

  for (int k = blockIdx.x; k < N; k += gridDim.x) {
    U partial_sum = 0;

    for (int i = tid; i < D; i += kBlockSize) {
      partial_sum += op1.Compute(in, k, i);
    }

    U sum = BlockReduce(temp_storage.reduce).Sum(partial_sum);

    if (tid == 0) {
      temp_storage.broadcast[0] = op1.Finalize(sum);
      out1[k] = op1.Finalize(sum);
    }
    __syncthreads();
    sum = temp_storage.broadcast[0];

    partial_sum = 0;

    for (int i = tid; i < D; i += kBlockSize) {
      partial_sum += op2.Compute(in, k, i, sum);
    }

    sum = BlockReduce(temp_storage.reduce).Sum(partial_sum);

    if (tid == 0) {
      out2[k] = op2.Finalize(sum);
    }
  }
}

template <typename T, typename U>
__global__ __launch_bounds__(1024) void InstanceNormBetaGammaRowReduceInToTemp(
    const T* __restrict__ x, const T* __restrict__ dy,
    const U* __restrict__ cache_mean, const U* __restrict__ cache_ivar,
    const int N, const int C, const int D, U* __restrict__ temp_dbeta,
    U* __restrict__ temp_dgamma, const bool is_channel_first = true) {
  if (is_channel_first) {
    typedef cub::BlockReduce<U, kBlockSize> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int row_offset = threadIdx.x + blockIdx.x * blockDim.x;
    const int col_offset = threadIdx.y + blockIdx.y * blockDim.y;

    int NxD = N * D;
    int CxD = C * D;
    int glb_id, cache_idx;
    for (int col_idx = col_offset; col_idx < C;
         col_idx += gridDim.y * blockDim.y) {
      U partial_sum_dbeta = 0;
      U partial_sum_dgamma = 0;
      for (int i = row_offset; i < NxD; i += gridDim.x * blockDim.x) {
        glb_id = (i / D) * CxD + col_idx * D + i % D;
        cache_idx = i / D * C + col_idx;
        U curr = dy[glb_id];
        partial_sum_dbeta += curr;
        partial_sum_dgamma +=
            curr * (x[glb_id] - cache_mean[cache_idx]) * cache_ivar[cache_idx];
      }
      U sum_dbeta = BlockReduce(temp_storage).Sum(partial_sum_dbeta);
      U sum_dgamma = BlockReduce(temp_storage).Sum(partial_sum_dgamma);
      if (threadIdx.x == 0) {
        temp_dbeta[blockIdx.x * C + col_idx] = sum_dbeta;
        temp_dgamma[blockIdx.x * C + col_idx] = sum_dgamma;
      }
    }
  } else {
    typedef cub::BlockReduce<U, 64> BlockReduce;

    const int row_offset = threadIdx.x + blockIdx.x * blockDim.x;
    const int col_offset = threadIdx.y + blockIdx.y * blockDim.y;
    if (row_offset >= C) return;
    if (col_offset >= N * D) return;
    int NxD = N * D;
    int glb_id, cache_idx;

    for (int row_idx = row_offset; row_idx < C;
         row_idx += blockDim.x * gridDim.x) {
      U partial_sum_dbeta = 0;
      U partial_sum_dgamma = 0;
      for (int i = col_offset; i < NxD; i += gridDim.y * blockDim.y) {
        int row_idx = row_offset;
        glb_id = i * C + row_idx;
        cache_idx = i / D * C + row_idx;
        U curr = dy[glb_id];
        partial_sum_dbeta += curr;
        partial_sum_dgamma +=
            curr * (x[glb_id] - cache_mean[cache_idx]) * cache_ivar[cache_idx];
      }
      temp_dbeta[col_offset * C + row_idx] = partial_sum_dbeta;
      temp_dgamma[col_offset * C + row_idx] = partial_sum_dgamma;
    }
  }
}

template <typename U>
__global__ __launch_bounds__(1024) void InstanceNormGradBetaGammaTempToOut(
    const U* __restrict__ tg, const U* __restrict__ tb, const int C,
    const int N, U* __restrict__ dgamma, U* __restrict__ dbeta) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= C) return;

  U sum_dgamma = 0;
  U sum_dbeta = 0;
  for (int i = 0; i < N; i++) {
    U tg_curr = tg[i * C + tid];
    U tb_curr = tb[i * C + tid];

    sum_dgamma += tg_curr;
    sum_dbeta += tb_curr;
  }

  dgamma[tid] = sum_dgamma;
  dbeta[tid] = sum_dbeta;
}

template <typename T, typename U>
__global__ __launch_bounds__(1024) void InstanceNormGradBetaGamma(
    const T* __restrict__ dy, const T* __restrict__ x,
    const U* __restrict__ cache_mean, const U* __restrict__ cache_ivar,
    const int N, const int C, const int D, U* __restrict__ dgamma,
    U* __restrict__ dbeta) {

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= C) return;

  U sum_dgamma = 0;
  U sum_dbeta = 0;
  int NxD = N * D;
  for (int i = 0; i < NxD; i++) {
    int in = i / D;
    int id = i % D;
    U dy_curr = GetAs<T, U>(dy, in * C * D + tid * D + id);
    sum_dgamma += dy_curr *
                  (x[in * C * D + tid * D + id] - cache_mean[in * C + tid]) *
                  cache_ivar[in * C + tid];
    sum_dbeta += dy_curr;
  }
  dgamma[tid] = sum_dgamma;
  dbeta[tid] = sum_dbeta;
}

template <typename U>
__global__ __launch_bounds__(1024) void ReduceNCtoC(const int N, const int C,
                                                    const U* __restrict__ in1,
                                                    const U* __restrict__ in2,
                                                    U* out1, U* out2) {
  int NxC = N * C;
  const int x_tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (x_tid >= C) return;
  U sum = static_cast<U>(0);
  for (int k = x_tid; k < NxC; k += C) {
    sum += in1[k];
  }
  out1[x_tid] = sum;
  sum = static_cast<U>(0);
  for (int k = x_tid; k < NxC; k += C) {
    sum += in2[k];
  }
  out2[x_tid] = sum;
}

template <typename T, typename U, typename Op>
__global__ __launch_bounds__(1024) void InstanceNormRowReduceInToOutWarpFused(
    const T* __restrict__ in, const int N, const int C, const int D, U* out1,
    U* out2, U* out3, U* out4, Op op, const bool is_channel_first) {
  const int tid = threadIdx.x % kWarpSize;
  const int local_warp_id = threadIdx.x / kWarpSize;
  const int warp_id = blockIdx.x * warp_per_block + local_warp_id;

  typedef cub::WarpReduce<U> WarpReduce;
  typename WarpReduce::TempStorage temp_storage[warp_per_block];

  int NxC = N * C;
  for (int k = warp_id; k < NxC; k += gridDim.x * warp_per_block) {
    U partial_sum1 = static_cast<U>(0);
    U partial_sum2 = static_cast<U>(0);
    U partial_sum3 = static_cast<U>(0);
    U partial_sum4 = static_cast<U>(0);
    U ret1, ret2, ret3, ret4;

    for (int i = tid; i < D; i += kWarpSize) {
      if (is_channel_first) {
        op.Compute(in, k, i, &ret1, &ret2, &ret3, &ret4);
      } else {
        op.Compute(in, k / C, i, k % C, &ret1, &ret2, &ret3, &ret4);
      }
      partial_sum1 += ret1;
      partial_sum2 += ret2;
      partial_sum3 += ret3;
      partial_sum4 += ret4;
    }

    U sum1 = WarpReduce(temp_storage[local_warp_id]).Sum(partial_sum1);
    U sum2 = WarpReduce(temp_storage[local_warp_id]).Sum(partial_sum2);
    U sum3 = WarpReduce(temp_storage[local_warp_id]).Sum(partial_sum3);
    U sum4 = WarpReduce(temp_storage[local_warp_id]).Sum(partial_sum4);

    sum1 = cub::ShuffleIndex<kWarpSize>(sum1, 0, 0xffffffff);
    sum2 = cub::ShuffleIndex<kWarpSize>(sum2, 0, 0xffffffff);
    sum3 = cub::ShuffleIndex<kWarpSize>(sum3, 0, 0xffffffff);
    sum4 = cub::ShuffleIndex<kWarpSize>(sum4, 0, 0xffffffff);
    sum1 = op.Finalize(sum1);
    sum2 = op.Finalize(sum2);
    sum3 = op.Finalize(sum3);
    sum4 = op.Finalize(sum4);
    if (tid == 0) {
      out1[k] = sum1;
      out2[k] = sum2;
      out3[k] = sum3;
      out4[k] = sum4;
    }
  }
}

template <typename T, typename U, typename Op1, typename Op2>
__global__ __launch_bounds__(1024) void InstanceNormRowReduceInToOutWarp(
    const T* __restrict__ in, const int N, const int C, const int D, U* out1,
    U* out2, Op1 op1, Op2 op2, const bool is_channel_first) {
  const int tid = threadIdx.x % kWarpSize;
  const int local_warp_id = threadIdx.x / kWarpSize;
  const int warp_id = blockIdx.x * warp_per_block + local_warp_id;

  typedef cub::WarpReduce<U> WarpReduce;
  typename WarpReduce::TempStorage temp_storage[warp_per_block];
  int NxC = N * C;
  for (int k = warp_id; k < NxC; k += gridDim.x * warp_per_block) {
    U partial_sum = 0;

    for (int i = tid; i < D; i += kWarpSize) {
      if (is_channel_first) {
        partial_sum += op1.Compute(in, k, i);
      } else {
        partial_sum += op1.Compute(in, k / C, i, k % C);
      }
    }

    U sum = WarpReduce(temp_storage[local_warp_id]).Sum(partial_sum);

    sum = cub::ShuffleIndex<kWarpSize>(sum, 0, 0xffffffff);
    sum = op1.Finalize(sum);
    if (tid == 0) {
      out1[k] = sum;
    }

    partial_sum = 0;

    for (int i = tid; i < D; i += kWarpSize) {
      if (is_channel_first) {
        partial_sum += op2.Compute(in, k, i, sum);
      } else {
        partial_sum += op2.Compute(in, k / C, i, k % C, sum);
      }
    }

    sum = WarpReduce(temp_storage[local_warp_id]).Sum(partial_sum);

    if (tid == 0) {
      out2[k] = op2.Finalize(sum);
    }
  }
}

template <typename T, typename U, typename Op>
__global__ __launch_bounds__(1024) void InstanceNormRowReduceInToTemp(
    const T* __restrict__ x, const int N, const int C, const int D,
    U* __restrict__ temp, Op op, const bool is_channel_first = true) {
  if (is_channel_first) {
    typedef cub::BlockReduce<U, kBlockSize> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int row_offset = threadIdx.x + blockIdx.x * blockDim.x;

    for (int row_idx = blockIdx.y; row_idx < N * C; row_idx += gridDim.y) {
      U partial_sum = 0;

      for (int i = row_offset; i < D; i += gridDim.x * blockDim.x) {
        partial_sum += op.Compute(x, row_idx, i);
      }
      U sum = BlockReduce(temp_storage).Sum(partial_sum);
      if (threadIdx.x == 0) {
        temp[row_idx * gridDim.x + blockIdx.x] = sum;
      }
    }
  } else {
    const int x_tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int y_tid = threadIdx.y + blockIdx.y * blockDim.y;
    const int z_tid = threadIdx.z + blockIdx.z * blockDim.z;
    if (x_tid >= C) return;
    U partial_sum = 0;

    for (int i = y_tid; i < D; i += blockDim.y * gridDim.y) {
      partial_sum += op.Compute(x, z_tid, i, x_tid);
    }

    temp[(z_tid * C + x_tid) * blockDim.y * gridDim.y + y_tid] = partial_sum;
  }
}

template <typename T, typename U, typename Op>
__global__ __launch_bounds__(1024) void InstanceNormRowReduceInToTempFused(
    const T* __restrict__ x, const int N, const int C, const int D,
    U* __restrict__ temp1, U* __restrict__ temp2, U* __restrict__ temp3,
    U* __restrict__ temp4, Op op, const bool is_channel_first = true) {
  if (is_channel_first) {
    typedef cub::BlockReduce<U, kBlockSize> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    const int row_offset = threadIdx.x + blockIdx.x * blockDim.x;

    for (int row_idx = blockIdx.y; row_idx < N * C; row_idx += gridDim.y) {
      U partial_sum1 = 0;
      U partial_sum2 = 0;
      U partial_sum3 = 0;
      U partial_sum4 = 0;
      U ret1, ret2, ret3, ret4;

      for (int i = row_offset; i < D; i += gridDim.x * blockDim.x) {
        op.Compute(x, row_idx, i, &ret1, &ret2, &ret3, &ret4);
        partial_sum1 += ret1;
        partial_sum2 += ret2;
        partial_sum3 += ret3;
        partial_sum4 += ret4;
      }
      U sum1 = BlockReduce(temp_storage).Sum(partial_sum1);
      U sum2 = BlockReduce(temp_storage).Sum(partial_sum2);
      U sum3 = BlockReduce(temp_storage).Sum(partial_sum3);
      U sum4 = BlockReduce(temp_storage).Sum(partial_sum4);

      if (threadIdx.x == 0) {
        int idx = row_idx * gridDim.x + blockIdx.x;
        temp1[idx] = sum1;
        temp2[idx] = sum2;
        temp3[idx] = sum3;
        temp4[idx] = sum4;
      }
    }
  } else {
    const int x_tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int y_tid = threadIdx.y + blockIdx.y * blockDim.y;
    const int z_tid = threadIdx.z + blockIdx.z * blockDim.z;
    if (x_tid >= C) return;
    U partial_sum1 = 0;
    U partial_sum2 = 0;
    U partial_sum3 = 0;
    U partial_sum4 = 0;
    U ret1, ret2, ret3, ret4;

    for (int i = y_tid; i < D; i += blockDim.y * gridDim.y) {
      op.Compute(x, z_tid, i, x_tid, &ret1, &ret2, &ret3, &ret4);
      partial_sum1 += ret1;
      partial_sum2 += ret2;
      partial_sum3 += ret3;
      partial_sum4 += ret4;
    }
    int idx = (z_tid * C + x_tid) * blockDim.y * gridDim.y + y_tid;
    temp1[idx] = partial_sum1;
    temp2[idx] = partial_sum2;
    temp3[idx] = partial_sum3;
    temp4[idx] = partial_sum4;
  }
}

template <typename T, typename Op>
__global__ __launch_bounds__(1024) void InstanceNormUpdateFused(
    const T* __restrict__ in, const int N, const int D, T* out, Op op,
    const bool is_channel_first = true) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= N) return;
  for (int row_idx = tid; row_idx < N; row_idx += gridDim.x * blockDim.x) {
    out[row_idx] = op.Compute(in, row_idx, is_channel_first);
  }
}

template <typename T, typename Op>
__global__ __launch_bounds__(1024) void InstanceNormUpdate(
    const T* __restrict__ in, const int N, const int D, T* out, Op op,
    const bool is_channel_first = true) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= N) return;
  for (int row_idx = tid; row_idx < N; row_idx += gridDim.x * blockDim.x) {
    out[row_idx] = op.Compute(in, row_idx, is_channel_first);
  }
}

template <typename T, typename U, typename Op>
void InstanceNormReductionsChannelFirstFused(OpKernelContext* context,
                                             const int N, const int C,
                                             const int D, const T* x, U* temp_1,
                                             U* temp_2, U* temp_3, U* temp_4,
                                             Op op) {
  const bool is_channel_first = true;
  const GPUDevice& d = context->eigen_device<GPUDevice>();
  auto NxC = N * C;
  const int min_num_blocks = kWarpSize;
  const int min_workload_per_thread = 100;

  bool use_single_block =
      (D <= min_num_blocks * kBlockSize * min_workload_per_thread);
  if (use_single_block) {
    Tensor scratch3, scratch4;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                   {N * C}, &scratch3));
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                   {N * C}, &scratch4));
    U* temp_buffer_3 = scratch3.flat<U>().data();
    U* temp_buffer_4 = scratch4.flat<U>().data();

    TF_CHECK_OK(GpuLaunchKernel(InstanceNormRowReduceInToOutFused<T, U, Op>,
                                NxC, kBlockSize, 0, d.stream(), x, NxC, D,
                                temp_1, temp_2, temp_buffer_3, temp_buffer_4,
                                op));
    TF_CHECK_OK(GpuLaunchKernel(ReduceNCtoC<U>, DivUp(C, kBlockSize),
                                kBlockSize, 0, d.stream(), N, C, temp_buffer_3,
                                temp_buffer_4, temp_3, temp_4));
  } else {
    const int blocks_per_row = DivUp(D, kBlockSize * min_workload_per_thread);

    Tensor scratch1, scratch2, scratch3, scratch4;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<U>::value,
                                          {NxC * blocks_per_row}, &scratch1));
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<U>::value,
                                          {NxC * blocks_per_row}, &scratch2));
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<U>::value,
                                          {NxC * blocks_per_row}, &scratch3));
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<U>::value,
                                          {NxC * blocks_per_row}, &scratch4));
    U* temp_buffer_1 = scratch1.flat<U>().data();
    U* temp_buffer_2 = scratch2.flat<U>().data();
    U* temp_buffer_3 = scratch3.flat<U>().data();
    U* temp_buffer_4 = scratch4.flat<U>().data();

    dim3 threads(kBlockSize);
    dim3 blocks(blocks_per_row, N);
    int temp_total_rows = blocks_per_row;
    // For long rows, we launch n blocks to process each row. The intermediate
    // results are stored in a temp memory with the size of N*n. Then, we
    // launch single block to handle each row of the temp memory.
    TF_CHECK_OK(GpuLaunchKernel(InstanceNormRowReduceInToTempFused<T, U, Op>,
                                blocks, threads, 0, d.stream(), x, N, C, D,
                                temp_buffer_1, temp_buffer_2, temp_buffer_3,
                                temp_buffer_4, op, is_channel_first));
    TF_CHECK_OK(GpuLaunchKernel(
        InstanceNormRowReduceTempToOutFused<U, Op>, N * C, kBlockSize, 0,
        d.stream(), temp_buffer_1, temp_buffer_2, temp_buffer_3, temp_buffer_4,
        N, C, temp_total_rows, temp_1, temp_2, temp_3, temp_4, op));
  }
}

template <typename T, typename U, typename Op1, typename Op2>
void InstanceNormReductionsChannelFirst(OpKernelContext* context, const int N,
                                        const int C, const int D, const T* x,
                                        U* temp_1, U* temp_2, Op1 op1,
                                        Op2 op2) {
  const bool is_channel_first = true;
  const GPUDevice& d = context->eigen_device<GPUDevice>();
  auto NxC = N * C;
  const int min_num_blocks = kWarpSize;
  const int min_workload_per_thread = 100;

  bool use_single_block =
      (D <= min_num_blocks * kBlockSize * min_workload_per_thread);
  if (use_single_block) {
    TF_CHECK_OK(GpuLaunchKernel(InstanceNormRowReduceInToOut<T, U, Op1, Op2>,
                                NxC, kBlockSize, 0, d.stream(), x, NxC, D,
                                temp_1, temp_2, op1, op2));
  } else {
    const int blocks_per_row = DivUp(D, kBlockSize * min_workload_per_thread);

    Tensor scratch1, scratch2;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<U>::value,
                                          {NxC * blocks_per_row}, &scratch1));
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<U>::value,
                                          {NxC * blocks_per_row}, &scratch2));
    U* temp_buffer_1 = scratch1.flat<U>().data();
    U* temp_buffer_2 = scratch2.flat<U>().data();

    dim3 threads(kBlockSize);
    dim3 blocks(blocks_per_row, N);
    int temp_total_rows = blocks_per_row;

    // For long rows, we launch n blocks to process each row. The intermediate
    // results are stored in a temp memory with the size of N*n. Then, we
    // launch single block to handle each row of the temp memory.
    TF_CHECK_OK(GpuLaunchKernel(InstanceNormRowReduceInToTemp<T, U, Op1>,
                                blocks, threads, 0, d.stream(), x, N, C, D,
                                temp_buffer_1, op1, is_channel_first));
    TF_CHECK_OK(GpuLaunchKernel(InstanceNormRowReduceTempToOut<U, Op1>, NxC,
                                kBlockSize, 0, d.stream(), temp_buffer_1, N, C,
                                temp_total_rows, temp_1, op1));

    TF_CHECK_OK(GpuLaunchKernel(InstanceNormRowReduceInToTemp<T, U, Op2>,
                                blocks, threads, 0, d.stream(), x, N, C, D,
                                temp_buffer_2, op2, is_channel_first));
    TF_CHECK_OK(GpuLaunchKernel(InstanceNormRowReduceTempToOut<U, Op2>, NxC,
                                kBlockSize, 0, d.stream(), temp_buffer_2, N, C,
                                temp_total_rows, temp_2, op2));
  }
}

template <typename T, typename U, typename Op>
void InstanceNormReductionsChannelFirstWelford(OpKernelContext* context,
                                               const int N, const int C,
                                               const int D, const T* x,
                                               U* temp_1, U* temp_2, Op op) {
  const bool is_channel_first = true;
  const GPUDevice& d = context->eigen_device<GPUDevice>();
  auto NxC = N * C;
  const int min_num_blocks = kWarpSize;
  const int min_workload_per_thread = 100;

  bool use_single_block =
      (D <= min_num_blocks * kBlockSize * min_workload_per_thread);
  if (use_single_block) {
    TF_CHECK_OK(GpuLaunchKernel(GeneralNormRowReduceInToOutWelford<T, U, Op>,
                                NxC, kBlockSize, 0, d.stream(), x, NxC, D,
                                temp_1, temp_2, op));
  } else {
    const int blocks_per_row = DivUp(D, kBlockSize * min_workload_per_thread);

    Tensor scratch1, scratch2, scratch3;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<U>::value,
                                          {NxC * blocks_per_row}, &scratch1));
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<U>::value,
                                          {NxC * blocks_per_row}, &scratch2));
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<U>::value,
                                          {NxC * blocks_per_row}, &scratch3));
    U* temp_buffer_1 = scratch1.flat<U>().data();
    U* temp_buffer_2 = scratch2.flat<U>().data();
    U* temp_buffer_3 = scratch3.flat<U>().data();

    dim3 threads(kBlockSize);
    dim3 blocks(blocks_per_row, N);
    int temp_total_rows = blocks_per_row;

    // For long rows, we launch n blocks to process each row. The intermediate
    // results are stored in a temp memory with the size of N*n. Then, we
    // launch single block to handle each row of the temp memory.
    TF_CHECK_OK(GpuLaunchKernel(InstanceNormToTempWelford<T, U, Op>, blocks,
                                threads, 0, d.stream(), x, N, C, D,
                                temp_buffer_1, temp_buffer_2, temp_buffer_3, op,
                                is_channel_first));
    TF_CHECK_OK(GpuLaunchKernel(InstanceNormTempToOutWelford<U, Op>, NxC,
                                kBlockSize, 0, d.stream(), temp_buffer_1,
                                temp_buffer_2, temp_buffer_3, N, C,
                                temp_total_rows, temp_1, temp_2, op));
  }
}

template <typename T, typename U, typename Op1, typename Op2>
void InstanceNormReductionsChannelLast(OpKernelContext* context, const int N,
                                       const int C, const int D, const T* x,
                                       U* temp_1, U* temp_2, Op1 op1, Op2 op2) {
  const GPUDevice& d = context->eigen_device<GPUDevice>();

  dim3 threads(kWarpSize, warp_per_block);
  const int local_min_workload_per_thread = 3200;

  int ppr = DivUp(D, int(threads.y * local_min_workload_per_thread));

  dim3 blocks(DivUp(C, kWarpSize), ppr, N);

  Tensor scratch1, scratch2;
  OP_REQUIRES_OK(context,
                 context->allocate_temp(DataTypeToEnum<U>::value,
                                        {N * C * threads.y * ppr}, &scratch1));
  OP_REQUIRES_OK(context,
                 context->allocate_temp(DataTypeToEnum<U>::value,
                                        {N * C * threads.y * ppr}, &scratch2));
  U* temp_buffer_1 = scratch1.flat<U>().data();
  U* temp_buffer_2 = scratch2.flat<U>().data();

  bool is_channel_first = false;
  TF_CHECK_OK(GpuLaunchKernel(InstanceNormRowReduceInToTemp<T, U, Op1>, blocks,
                              threads, 0, d.stream(), x, N, C, D, temp_buffer_1,
                              op1, is_channel_first));
  TF_CHECK_OK(GpuLaunchKernel(InstanceNormRowReduceTempToOut<U, Op1>, N * C,
                              kBlockSize, 0, d.stream(), temp_buffer_1, N, C,
                              threads.y * ppr, temp_1, op1));

  TF_CHECK_OK(GpuLaunchKernel(InstanceNormRowReduceInToTemp<T, U, Op2>, blocks,
                              threads, 0, d.stream(), x, N, C, D, temp_buffer_2,
                              op2, is_channel_first));
  TF_CHECK_OK(GpuLaunchKernel(InstanceNormRowReduceTempToOut<U, Op2>, N * C,
                              kBlockSize, 0, d.stream(), temp_buffer_2, N, C,
                              threads.y * ppr, temp_2, op2));
}

template <typename T, typename U, typename Op>
void InstanceNormReductionsChannelLastWelford(OpKernelContext* context,
                                              const int N, const int C,
                                              const int D, const T* x,
                                              U* temp_1, U* temp_2, Op op) {
  const GPUDevice& d = context->eigen_device<GPUDevice>();

  dim3 threads(kWarpSize, warp_per_block);
  const int local_min_workload_per_thread = 3200;

  int ppr = DivUp(D, int(threads.y * local_min_workload_per_thread));
  dim3 blocks(DivUp(C, kWarpSize), ppr, N);

  Tensor scratch1, scratch2, scratch3;
  OP_REQUIRES_OK(context,
                 context->allocate_temp(DataTypeToEnum<U>::value,
                                        {N * C * threads.y * ppr}, &scratch1));
  OP_REQUIRES_OK(context,
                 context->allocate_temp(DataTypeToEnum<U>::value,
                                        {N * C * threads.y * ppr}, &scratch2));
  OP_REQUIRES_OK(context,
                 context->allocate_temp(DataTypeToEnum<U>::value,
                                        {N * C * threads.y * ppr}, &scratch3));
  U* temp_buffer_1 = scratch1.flat<U>().data();
  U* temp_buffer_2 = scratch2.flat<U>().data();
  U* temp_buffer_3 = scratch3.flat<U>().data();

  bool is_channel_first = false;
  TF_CHECK_OK(GpuLaunchKernel(
      InstanceNormToTempWelford<T, U, Op>, blocks, threads, 0, d.stream(), x, N,
      C, D, temp_buffer_1, temp_buffer_2, temp_buffer_3, op, is_channel_first));
  TF_CHECK_OK(GpuLaunchKernel(InstanceNormTempToOutWelford<U, Op>, N * C,
                              kBlockSize, 0, d.stream(), temp_buffer_1,
                              temp_buffer_2, temp_buffer_3, N, C,
                              threads.y * ppr, temp_1, temp_2, op));
}

template <typename T, typename U, typename Op>
void InstanceNormReductionsChannelLastFused(OpKernelContext* context,
                                            const int N, const int C,
                                            const int D, const T* x, U* temp_1,
                                            U* temp_2, U* temp_3, U* temp_4,
                                            Op op) {
  const GPUDevice& d = context->eigen_device<GPUDevice>();

  dim3 threads(kWarpSize, warp_per_block);
  const int local_min_workload_per_thread = 3200;

  int ppr = DivUp(D, int(threads.y * local_min_workload_per_thread));

  dim3 blocks(DivUp(C, kWarpSize), ppr, N);

  Tensor scratch1, scratch2, scratch3, scratch4;
  OP_REQUIRES_OK(context,
                 context->allocate_temp(DataTypeToEnum<U>::value,
                                        {N * C * threads.y * ppr}, &scratch1));
  OP_REQUIRES_OK(context,
                 context->allocate_temp(DataTypeToEnum<U>::value,
                                        {N * C * threads.y * ppr}, &scratch2));
  OP_REQUIRES_OK(context,
                 context->allocate_temp(DataTypeToEnum<U>::value,
                                        {N * C * threads.y * ppr}, &scratch3));
  OP_REQUIRES_OK(context,
                 context->allocate_temp(DataTypeToEnum<U>::value,
                                        {N * C * threads.y * ppr}, &scratch4));
  U* temp_buffer_1 = scratch1.flat<U>().data();
  U* temp_buffer_2 = scratch2.flat<U>().data();
  U* temp_buffer_3 = scratch3.flat<U>().data();
  U* temp_buffer_4 = scratch4.flat<U>().data();

  bool is_channel_first = false;
  TF_CHECK_OK(GpuLaunchKernel(InstanceNormRowReduceInToTempFused<T, U, Op>,
                              blocks, threads, 0, d.stream(), x, N, C, D,
                              temp_buffer_1, temp_buffer_2, temp_buffer_3,
                              temp_buffer_4, op, is_channel_first));
  TF_CHECK_OK(GpuLaunchKernel(
      InstanceNormRowReduceTempToOutFused<U, Op>, N * C, kBlockSize, 0,
      d.stream(), temp_buffer_1, temp_buffer_2, temp_buffer_3, temp_buffer_4, N,
      C, threads.y * ppr, temp_1, temp_2, temp_3, temp_4, op));
}

template <typename T, typename U>
void InstanceNormDataWeightsGrad(OpKernelContext* context, const T* dy,
                                 const T* x, const U* cache_mean,
                                 const U* cache_ivar, const U* gamma,
                                 const int N, const int C, const int D,
                                 U* temp_1, U* temp_2, U* temp_3, U* temp_4,
                                 const bool is_channel_first) {
  const GPUDevice& d = context->eigen_device<GPUDevice>();
  bool use_single_warp = (D <= kWarpSize);

  DwStatFusedOp<T, U> dldwstat_ops{gamma, x, cache_ivar, cache_mean, C, D};

  const int NxC = N * C;
  if (use_single_warp) {
    Tensor scratch3, scratch4;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                   {N * C}, &scratch3));
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                   {N * C}, &scratch4));
    U* temp_buffer_3 = scratch3.flat<U>().data();
    U* temp_buffer_4 = scratch4.flat<U>().data();

    TF_CHECK_OK(GpuLaunchKernel(
        InstanceNormRowReduceInToOutWarpFused<T, U, DwStatFusedOp<T, U>>,
        DivUp(NxC, warp_per_block), kBlockSize, 0, d.stream(), dy, N, C, D,
        temp_1, temp_2, temp_buffer_3, temp_buffer_4, dldwstat_ops,
        is_channel_first));
    TF_CHECK_OK(GpuLaunchKernel(ReduceNCtoC<U>, DivUp(C, kBlockSize),
                                kBlockSize, 0, d.stream(), N, C, temp_buffer_3,
                                temp_buffer_4, temp_3, temp_4));
  } else {
    if (is_channel_first) {
      InstanceNormReductionsChannelFirstFused<T, U, DwStatFusedOp<T, U>>(
          context, N, C, D, dy, temp_1, temp_2, temp_3, temp_4, dldwstat_ops);
    } else {
      InstanceNormReductionsChannelLastFused<T, U, DwStatFusedOp<T, U>>(
          context, N, C, D, dy, temp_1, temp_2, temp_3, temp_4, dldwstat_ops);
    }
  }
}

template <typename T, typename U>
void InstanceNormWeightsGrad(OpKernelContext* context, const T* dy, const T* x,
                             const U* cache_mean, const U* cache_ivar,
                             const int N, const int C, const int D, U* dgamma,
                             U* dbeta, const bool is_channel_first) {
  const int min_rows_per_block = 100;
  const int min_cols_per_block = 32;
  const GPUDevice& d = context->eigen_device<GPUDevice>();
  int total_tmp_rows;
  int buffer_size;
  dim3 blocks;
  dim3 threads;
  if (is_channel_first) {
    const int reduced_rows =
        DivUp(N * D, min_rows_per_block * min_rows_per_block);
    const int reduced_cols = DivUp(C, min_cols_per_block);

    blocks.x = reduced_rows;
    blocks.y = reduced_cols;
    threads.x = kBlockSize;

    buffer_size = reduced_rows * C;
    total_tmp_rows = reduced_rows;
  } else {
    int thd_x = min(C, (int)kBlockSize);
    int thd_y = (int)kBlockSize / thd_x;
    int min_workload_per_thread = 3200;
    const int blocks_per_row = DivUp(N * D, thd_y * min_workload_per_thread);
    threads.x = thd_x;
    threads.y = thd_y;

    blocks.x = DivUp(C, thd_x);
    blocks.y = blocks_per_row;
    buffer_size = C * blocks_per_row * thd_y;

    total_tmp_rows = min(N * D, blocks_per_row * thd_y);
  }

  Tensor scratch1, scratch2;
  OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                 {buffer_size}, &scratch1));
  OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                 {buffer_size}, &scratch2));
  U* temp_dgamma = scratch1.flat<U>().data();
  U* temp_dbeta = scratch2.flat<U>().data();

  TF_CHECK_OK(GpuLaunchKernel(InstanceNormBetaGammaRowReduceInToTemp<T, U>,
                              blocks, threads, 0, d.stream(), x, dy, cache_mean,
                              cache_ivar, N, C, D, temp_dbeta, temp_dgamma,
                              is_channel_first));
  TF_CHECK_OK(GpuLaunchKernel(InstanceNormGradBetaGammaTempToOut<U>,
                              DivUp(C, (int)kBlockSize), kBlockSize, 0,
                              d.stream(), temp_dgamma, temp_dbeta, C,
                              total_tmp_rows, dgamma, dbeta));
}

template <typename T, typename U>
struct FusedInstanceNorm<GPUDevice, T, U> {
  void operator()(OpKernelContext* context, const Tensor& x_input,
                  const Tensor& scale_input, const Tensor& offset_input,
                  U epsilon, const bool is_channel_first, Tensor* y_output,
                  Tensor* saved_mean_output, Tensor* saved_inv_var_output) {
    if (x_input.shape().num_elements() == 0) {
      return;
    }
    int channel_dim = is_channel_first ? 1 : 2;
    int feature_dim = is_channel_first ? 2 : 1;
    const auto N = x_input.dim_size(0);
    const auto C = x_input.dim_size(channel_dim);
    const auto D = x_input.dim_size(feature_dim);

    const auto NxC = N * C;

    const T* x = x_input.flat<T>().data();
    const U* gamma = scale_input.flat<U>().data();
    const U* beta = offset_input.flat<U>().data();
    U* cache_mean = saved_mean_output->flat<U>().data();
    U* cache_ivar = saved_inv_var_output->flat<U>().data();
    T* y = y_output->flat<T>().data();

    bool use_single_warp = (D <= kWarpSize);
    const GPUDevice& d = context->eigen_device<GPUDevice>();

    MeanOp<T, U> mean_ops{D};
    IvarOp<T, U> ivar_ops{cache_mean, epsilon, D};
    WFOp<T, U> wf_ops(D, epsilon);
    if (!is_channel_first) {
      mean_ops.SetChannelDim(C);
      ivar_ops.SetChannelDim(C);
      wf_ops.SetChannelDim(C);
    }

    if (use_single_warp) {
      const GPUDevice& d = context->eigen_device<GPUDevice>();
      TF_CHECK_OK(GpuLaunchKernel(
          InstanceNormRowReduceInToOutWarp<T, U, MeanOp<T, U>, IvarOp<T, U>>,
          DivUp(NxC, warp_per_block), kBlockSize, 0, d.stream(), x, N, C, D,
          cache_mean, cache_ivar, mean_ops, ivar_ops, is_channel_first));
    } else {
      if (is_channel_first) {
        InstanceNormReductionsChannelFirstWelford<T, U, WFOp<T, U>>(
            context, N, C, D, x, cache_mean, cache_ivar, wf_ops);

      } else {
        InstanceNormReductionsChannelLastWelford<T, U, WFOp<T, U>>(
            context, N, C, D, x, cache_mean, cache_ivar, wf_ops);
      }
    }

    YOp<T, U> y_ops{cache_mean, cache_ivar, gamma, beta, C, D};
    int min_work_per_thread = 100;
    TF_CHECK_OK(GpuLaunchKernel(
        InstanceNormUpdate<T, YOp<T, U>>,
        DivUp(N * C * D, kBlockSize * min_work_per_thread), kBlockSize, 0,
        d.stream(), x, N * C * D, D, y, y_ops, is_channel_first));
  }
};

template <typename T, typename U>
struct FusedInstanceNormGrad<GPUDevice, T, U> {
  void operator()(OpKernelContext* context, const Tensor& y_backprop_input,
                  const Tensor& x_input, const Tensor& scale_input,
                  const Tensor& saved_mean_input,
                  const Tensor& saved_inv_var_input, U epsilon,
                  const bool is_channel_first, Tensor* x_backprop_output,
                  Tensor* scale_backprop_output,
                  Tensor* offset_backprop_output) {
    if (x_input.shape().num_elements() == 0) {
      return;
    }
    const GPUDevice& d = context->eigen_device<GPUDevice>();
    int channel_dim = is_channel_first ? 1 : 2;
    int feature_dim = is_channel_first ? 2 : 1;
    const auto N = x_input.dim_size(0);
    const auto C = x_input.dim_size(channel_dim);
    const auto D = x_input.dim_size(feature_dim);

    const auto NxC = N * C;

    const T* dy = y_backprop_input.flat<T>().data();
    const T* x = x_input.flat<T>().data();
    const U* gamma = scale_input.flat<U>().data();
    const U* cache_mean = saved_mean_input.flat<U>().data();
    const U* cache_ivar = saved_inv_var_input.flat<U>().data();
    T* dx = x_backprop_output->flat<T>().data();
    U* dgamma = scale_backprop_output->flat<U>().data();
    U* dbeta = offset_backprop_output->flat<U>().data();

    Tensor scratch1, scratch2;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                   {NxC}, &scratch1));
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                   {NxC}, &scratch2));
    U* temp_1 = scratch1.flat<U>().data();
    U* temp_2 = scratch2.flat<U>().data();

    InstanceNormDataWeightsGrad<T, U>(context, dy, x, cache_mean, cache_ivar,
                                      gamma, N, C, D, temp_1, temp_2, dgamma,
                                      dbeta, is_channel_first);
    // The temp_1 and temp_2 are dl_dmu and dl_dvars respectively.
    DxFusedOp<T, U> dx_ops{x, cache_mean, cache_ivar, gamma, temp_2,
                           temp_1, C, D};
    int min_work_per_thread = 100;

    TF_CHECK_OK(GpuLaunchKernel(
        InstanceNormUpdateFused<T, DxFusedOp<T, U>>,
        DivUp(NxC * D, kBlockSize * min_work_per_thread), kBlockSize, 0,
        d.stream(), dy, N * C * D, D, dx, dx_ops, is_channel_first));
  }
};

template struct FusedInstanceNorm<GPUDevice, float, float>;
template struct FusedInstanceNorm<GPUDevice, Eigen::half, float>;
template struct FusedInstanceNormGrad<GPUDevice, float, float>;
template struct FusedInstanceNormGrad<GPUDevice, Eigen::half, float>;

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
