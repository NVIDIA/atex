/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
==============================================================================*/

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU

#if GOOGLE_CUDA
#include <cuda.h>
#endif

#include "fused_instance_norm_op.h"
#include "norm_util.h"
#include "tensorflow/core/kernels/gpu_prim.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

using namespace InNorm;

template <typename T, typename U, int PackSize>
__global__ __launch_bounds__(1024) void InstanceNormToTempWelford(
    const T* __restrict__ x, size_t N, size_t C, size_t D,
    U* __restrict__ temp_mean, U* __restrict__ temp_m2,
    U* __restrict__ temp_count, WFOp<T, U> op, bool is_channel_first = true) {
  if (is_channel_first) {
    U pack[PackSize];
    const auto row_offset = threadIdx.x + blockIdx.x * blockDim.x;
    const auto num_packs = static_cast<size_t>(D / PackSize);
    for (size_t row_idx = blockIdx.y; row_idx < N * C; row_idx += gridDim.y) {
      WFGeneric<U> wf_partial;
      for (size_t pack_id = row_offset; pack_id < num_packs;
           pack_id += gridDim.x * blockDim.x) {
        const auto data_offset = row_idx * D + pack_id * PackSize;
        CopyWithCast<T, U, PackSize>(x, data_offset, pack);

        for (int i = 0; i < PackSize; ++i) {
          op.Update(pack[i], wf_partial);
        }
      }
      WFGeneric<U> wf_block = BlockAllReduce<WFGeneric<U>, WFGeneric<U>>(
          wf_partial, WFGeneric<U>());
      __syncthreads();
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

    for (size_t i = y_tid; i < D; i += blockDim.y * gridDim.y) {
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

template <typename T, typename U>
void (*InstanceNormToTempWelfordCandidates[3])(const T* __restrict__, size_t,
                                               size_t, size_t, U* __restrict__,
                                               U* __restrict__, U* __restrict__,
                                               WFOp<T, U>, bool){
    InstanceNormToTempWelford<T, U, 1>, InstanceNormToTempWelford<T, U, 2>,
    InstanceNormToTempWelford<T, U, 4>};

template <typename U, typename Op>
__global__ __launch_bounds__(1024) void InstanceNormTempToOutWelford(
    const U* __restrict__ temp_mean, const U* __restrict__ temp_m2,
    const U* __restrict__ temp_count, size_t N, size_t C, size_t D,
    U* __restrict__ cache_mean, U* __restrict__ cache_ivar, Op op) {
  for (size_t k = blockIdx.x; k < N * C; k += gridDim.x) {
    WFGeneric<U> wf_partial;
    for (size_t i = threadIdx.x; i < D; i += kBlockSize) {
      auto idx = k * D + i;
      WFGeneric<U> wf_local{temp_mean[idx], temp_m2[idx], temp_count[idx]};
      wf_partial = WFGeneric<U>()(wf_local, wf_partial);
    }

    WFGeneric<U> wf_block =
        BlockAllReduce<WFGeneric<U>, WFGeneric<U>>(wf_partial, WFGeneric<U>());
    __syncthreads();        
    if (threadIdx.x == 0) {
      cache_mean[k] = wf_block.mean;
      cache_ivar[k] = op.Finalize(wf_block);
    }
  }
}

template <typename U, typename Op>
__global__ __launch_bounds__(1024) void InstanceNormRowReduceTempToOutFused(
    const U* __restrict__ temp1, const U* __restrict__ temp2,
    const U* __restrict__ temp3, const U* __restrict__ temp4, size_t N,
    size_t C, size_t D, U* __restrict__ cache1, U* __restrict__ cache2,
    U* __restrict__ cache3, U* __restrict__ cache4, Op op) {
  typedef gpuprim::BlockReduce<U, kBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  for (size_t k = blockIdx.x; k < N * C; k += gridDim.x) {
    U partial_sum1 = 0;
    U partial_sum2 = 0;
    for (size_t i = threadIdx.x; i < D; i += kBlockSize) {
      partial_sum1 += temp1[k * D + i];
      partial_sum2 += temp2[k * D + i];
    }

    U sum1 = BlockReduce(temp_storage).Sum(partial_sum1);
    __syncthreads();
    U sum2 = BlockReduce(temp_storage).Sum(partial_sum2);
    __syncthreads();
    if (threadIdx.x == 0) {
      cache1[k] = op.Finalize(sum1);
      cache2[k] = op.Finalize(sum2);
    }
  }
  if (blockIdx.x < C) {  // only first C block participate
    U partial_sum3 = 0;
    U partial_sum4 = 0;
    for (size_t k = blockIdx.x; k < N * C; k += C) {
      for (size_t i = threadIdx.x; i < D; i += kBlockSize) {
        partial_sum3 += temp3[k * D + i];
        partial_sum4 += temp4[k * D + i];
      }
    }
    U sum3 = BlockReduce(temp_storage).Sum(partial_sum3);
    __syncthreads();
    U sum4 = BlockReduce(temp_storage).Sum(partial_sum4);
    if (threadIdx.x == 0) {
      cache3[blockIdx.x] = op.Finalize(sum3);
      cache4[blockIdx.x] = op.Finalize(sum4);
    }
  }
}

template <typename U, typename Op>
__global__ __launch_bounds__(1024) void InstanceNormRowReduceTempToOut(
    const U* __restrict__ temp, size_t N, size_t C, size_t D,
    U* __restrict__ cache, Op op) {
  typedef gpuprim::BlockReduce<U, kBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  for (size_t k = blockIdx.x; k < N * C; k += gridDim.x) {
    U partial_sum = 0;

    for (size_t i = threadIdx.x; i < D; i += kBlockSize) {
      partial_sum += temp[k * D + i];
    }

    U sum = BlockReduce(temp_storage).Sum(partial_sum);

    if (threadIdx.x == 0) {
      cache[k] = op.Finalize(sum);
    }
  }
}

template <typename T, typename ComputeType, typename Op, int BlockSize,
          int PackSize>
__global__ void InstanceNormGradBlockSMemImpl(
    const T* x, const T* dy, size_t N, size_t C, size_t D,
    const ComputeType* cache_mean, const ComputeType* cache_ivar,
    const ComputeType* gamma, T* dx, ComputeType* out1, ComputeType* out2,
    ComputeType* out3, ComputeType* out4, Op op) {
  extern __shared__ __align__(sizeof(double)) unsigned char grad_shared_buf[];
  auto* buf_x = reinterpret_cast<ComputeType*>(grad_shared_buf);
  auto* buf_dy = buf_x + D;
  const int tid = threadIdx.x;
  assert(D % PackSize == 0);
  // row is N * C actually
  const size_t num_packs = static_cast<size_t>(D / PackSize);
  const ComputeType one_over_cols = static_cast<ComputeType>(1.0f / D);
  ComputeType x_pack[PackSize];
  ComputeType dy_pack[PackSize];
  ComputeType tmp1, tmp2, tmp3, tmp4, partial_sum1, partial_sum2, partial_sum3,
      partial_sum4;

  for (size_t row = blockIdx.x; row < N * C; row += gridDim.x) {
    partial_sum1 = 0;
    partial_sum2 = 0;
    partial_sum3 = 0;
    partial_sum4 = 0;
    const ComputeType row_mean = cache_mean[row];
    const ComputeType row_ivar = cache_ivar[row];
    const ComputeType channel_gamma = gamma[row % C];

    for (size_t pack_id = tid; pack_id < num_packs; pack_id += BlockSize) {
      const int pack_offset = pack_id * PackSize;
      CopyWithCast<T, ComputeType, PackSize>(x, row * D + pack_offset, x_pack);
      CopyWithCast<T, ComputeType, PackSize>(dy, row * D + pack_offset,
                                             dy_pack);

      for (int i = 0; i < PackSize; ++i) {
        const int buf_offset = i * num_packs + pack_id;
        ComputeType x_mean = x_pack[i] - row_mean;
        ComputeType dy = dy_pack[i];
        buf_x[buf_offset] = x_mean;
        buf_dy[buf_offset] = dy;
        op.Compute(x_mean, dy, row_ivar, channel_gamma, &tmp1, &tmp2, &tmp3,
                   &tmp4);
        partial_sum1 += tmp1;
        partial_sum2 += tmp2;
        partial_sum3 += tmp3;
        partial_sum4 += tmp4;
      }
    }

    ComputeType dldmu = BlockAllReduce<ComputeType, gpuprim::Sum, BlockSize>(
        partial_sum1, gpuprim::Sum(), true);
    __syncthreads();
    ComputeType dldvar_x2 =
        BlockAllReduce<ComputeType, gpuprim::Sum, BlockSize>(
            partial_sum2, gpuprim::Sum(), true);
    __syncthreads();
    for (size_t pack_id = tid; pack_id < num_packs; pack_id += BlockSize) {
      for (int i = 0; i < PackSize; ++i) {
        const auto buf_offset = i * num_packs + pack_id;
        x_pack[i] = (buf_x[buf_offset] * dldvar_x2 + dldmu) * one_over_cols +
                    buf_dy[buf_offset] * channel_gamma * row_ivar;
      }
      CopyWithCast<ComputeType, T, PackSize>(x_pack, 0, dx,
                                             row * D + pack_id * PackSize);
    }

    ComputeType sum_gamma =
        BlockAllReduce<ComputeType, gpuprim::Sum, BlockSize>(
            partial_sum3, gpuprim::Sum(), false);
    __syncthreads();
    ComputeType sum_beta = BlockAllReduce<ComputeType, gpuprim::Sum, BlockSize>(
        partial_sum4, gpuprim::Sum(), false);
    __syncthreads();    
    if (tid == 0) {
      out1[row] = op.Finalize(dldmu);
      out2[row] = op.Finalize(dldvar_x2);
      out3[row] = op.Finalize(sum_gamma);
      out4[row] = op.Finalize(sum_beta);
    }
  }
}

template <typename T, typename ComputeType, typename Op>
inline Status TryDispatchInstanceNormGradBlockSMemImpl(
    gpuStream_t stream, const T* x, const T* dy, size_t N, size_t C, size_t D,
    const ComputeType* mean, const ComputeType* inv_variance,
    const ComputeType* gamma, T* dx, ComputeType* out1, ComputeType* out2,
    ComputeType* out3, ComputeType* out4, Op op) {
  const size_t smem = D * sizeof(ComputeType) * 2;
  auto fns = InstanceNormGradBlockSMemImpl<T, ComputeType, Op, kBlockSize, 1>;
  int pack_size = 1;
  if (D % 4 == 0) {
    pack_size = 4;
    fns = InstanceNormGradBlockSMemImpl<T, ComputeType, Op, kBlockSize, 4>;
  } else if (D % 2 == 0) {
    pack_size = 2;
    fns = InstanceNormGradBlockSMemImpl<T, ComputeType, Op, kBlockSize, 2>;
  }

  auto fns_base = fns;
  int blk_cnt_base = 0;

  CUDA_RETURN_IF_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blk_cnt_base, fns, kBlockSize, smem));

  if (blk_cnt_base <= 0) {
    return errors::Internal(
        "Cannot find proper block count for block size 128.");
  }

  int blk_cnt = 0;
#define TRY_AND_LAUNCH(Bsize, Psize)                                         \
  fns = InstanceNormGradBlockSMemImpl<T, ComputeType, Op, Bsize, Psize>;     \
  CUDA_RETURN_IF_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(        \
      &blk_cnt, fns, Bsize, smem));                                          \
  if (blk_cnt == blk_cnt_base) { /*note : grid_dim = N */                    \
    TF_CHECK_OK(GpuLaunchKernel(fns, N, Bsize, smem, stream, x, dy, N, C, D, \
                                mean, inv_variance, gamma, dx, out1, out2,   \
                                out3, out4, op));                            \
    return OkStatus();                                                       \
  }

#define TRY_PACK_SIZE(PSIZE)                       \
  TRY_AND_LAUNCH(kBlockSizeSearchSpace[0], PSIZE); \
  TRY_AND_LAUNCH(kBlockSizeSearchSpace[1], PSIZE); \
  TRY_AND_LAUNCH(kBlockSizeSearchSpace[2], PSIZE);

  // There will be a fixed pack_size to choose from which depends on D.
  // Under each pack_aize, the order of fallback w.r.t. block_size is to try
  // from larger block dim to smaller, e.g. 1024 -> 512 -> 256 -> 128.
  if (pack_size == 4) {
    TRY_PACK_SIZE(4);
  } else if (pack_size == 2) {
    TRY_PACK_SIZE(2);
  }
  TRY_PACK_SIZE(1);
#undef TRY_PACK_SIZE

#undef TRY_AND_LAUNCH
  TF_CHECK_OK(GpuLaunchKernel(fns_base, N, kBlockSize, smem, stream, x, dy, N,
                              C, D, mean, inv_variance, gamma, dx, out1, out2,
                              out3, out4, op));
  return OkStatus();
}

template <typename T, typename U, int PackSize>
__global__ __launch_bounds__(1024) void InstanceNormRowReduceInToOutFused(
    const T* __restrict__ x, const T* __restrict__ dy, const U* cache_mean,
    const U* cache_ivar, const U* gamma, size_t N, size_t C, size_t D, U* out1,
    U* out2, U* out3, U* out4, DwStatFusedOp<T, U> op) {
  const int tid = threadIdx.x;
  const int block_size = blockDim.x;
  assert(D % PackSize == 0);
  const size_t num_packs = D / PackSize;
  typedef gpuprim::BlockReduce<U, kBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  U pack_x[PackSize];
  U pack_dy[PackSize];
  U partial_sum1, partial_sum2, partial_sum3, partial_sum4, ret1, ret2, ret3,
      ret4, mean, ivar, channel_gamma, sum1, sum2, sum3, sum4;
  for (size_t k = blockIdx.x; k < N * C; k += gridDim.x) {
    partial_sum1 = 0;
    partial_sum2 = 0;
    partial_sum3 = 0;
    partial_sum4 = 0;
    mean = cache_mean[k];
    ivar = cache_ivar[k];
    channel_gamma = gamma[k % C];
    for (size_t pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      const size_t data_offset = k * D + pack_id * PackSize;
      CopyWithCast<T, U, PackSize>(x, data_offset, pack_x);
      CopyWithCast<T, U, PackSize>(dy, data_offset, pack_dy);
      for (int i = 0; i < PackSize; ++i) {
        op.Compute(pack_x[i] - mean, pack_dy[i], ivar, channel_gamma, &ret1,
                   &ret2, &ret3, &ret4);
        partial_sum1 += ret1;
        partial_sum2 += ret2;
        partial_sum3 += ret3;
        partial_sum4 += ret4;
      }
    }
    sum1 = BlockReduce(temp_storage).Sum(partial_sum1);
    __syncthreads();
    sum2 = BlockReduce(temp_storage).Sum(partial_sum2);
    __syncthreads();
    sum3 = BlockReduce(temp_storage).Sum(partial_sum3);
    __syncthreads();
    sum4 = BlockReduce(temp_storage).Sum(partial_sum4);
    if (tid == 0) {
      out1[k] = op.Finalize(sum1);
      out2[k] = op.Finalize(sum2);
      out3[k] = op.Finalize(sum3);
      out4[k] = op.Finalize(sum4);
    }
    __syncthreads();
  }
}

template <typename T, typename U>
void (*InstanceNormRowReduceInToOutFusedCandidates[3])(
    const T* __restrict__, const T* __restrict__, const U*, const U*, const U*,
    size_t, size_t, size_t, U*, U*, U*, U*,
    DwStatFusedOp<T, U>){InstanceNormRowReduceInToOutFused<T, U, 1>,
                         InstanceNormRowReduceInToOutFused<T, U, 2>,
                         InstanceNormRowReduceInToOutFused<T, U, 4>};

template <typename T, typename U, typename Op1, typename Op2>
__global__ __launch_bounds__(1024) void InstanceNormRowReduceInToOut(
    const T* __restrict__ in, size_t N, size_t D, U* out1, U* out2, Op1 op1,
    Op2 op2) {
  const int tid = threadIdx.x;

  typedef gpuprim::BlockReduce<U, kBlockSize> BlockReduce;
  __shared__ union {
    typename BlockReduce::TempStorage reduce;
    U broadcast[1];
  } temp_storage;

  for (size_t k = blockIdx.x; k < N; k += gridDim.x) {
    U partial_sum = 0;

    for (size_t i = tid; i < D; i += kBlockSize) {
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

    for (size_t i = tid; i < D; i += kBlockSize) {
      partial_sum += op2.Compute(in, k, i, sum);
    }

    sum = BlockReduce(temp_storage.reduce).Sum(partial_sum);

    if (tid == 0) {
      out2[k] = op2.Finalize(sum);
    }
  }
}

template <typename T, typename U>
__global__ __launch_bounds__(1024) void InstanceNormGradBetaGamma(
    const T* __restrict__ dy, const T* __restrict__ x,
    const U* __restrict__ cache_mean, const U* __restrict__ cache_ivar,
    size_t N, size_t C, size_t D, U* __restrict__ dgamma,
    U* __restrict__ dbeta) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= C) return;

  U sum_dgamma = 0;
  U sum_dbeta = 0;
  for (size_t i = 0; i < N * D; i++) {
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
__global__ __launch_bounds__(1024) void ReduceNCtoC(size_t N, size_t C,
                                                    const U* __restrict__ in1,
                                                    const U* __restrict__ in2,
                                                    U* out1, U* out2) {
  const int x_tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (x_tid >= C) return;
  U sum1 = 0;
  U sum2 = 0;
  for (size_t k = x_tid; k < N * C; k += C) {
    sum1 += in1[k];
    sum2 += in2[k];
  }
  out1[x_tid] = sum1;
  out2[x_tid] = sum2;
}

template <typename T, typename U, typename Op>
__global__ __launch_bounds__(1024) void InstanceNormRowReduceInToOutWarpFused(
    const T* __restrict__ in, size_t N, size_t C, size_t D, U* out1, U* out2,
    U* out3, U* out4, Op op, bool is_channel_first) {
  const int tid = threadIdx.x % kWarpSize;
  const int local_warp_id = threadIdx.x / kWarpSize;
  const int warp_id = blockIdx.x * kWarpPerBlock + local_warp_id;

  typedef gpuprim::WarpReduce<U> WarpReduce;
  typename WarpReduce::TempStorage temp_storage[kWarpPerBlock];

  U partial_sum1, partial_sum2, partial_sum3, partial_sum4, ret1, ret2, ret3,
      ret4;
  for (size_t k = warp_id; k < N * C; k += gridDim.x * kWarpPerBlock) {
    partial_sum1 = 0;
    partial_sum2 = 0;
    partial_sum3 = 0;
    partial_sum4 = 0;
    for (size_t i = tid; i < D; i += kWarpSize) {
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

    sum1 = gpuprim::ShuffleIndex<kWarpSize>(sum1, 0, 0xffffffff);
    sum2 = gpuprim::ShuffleIndex<kWarpSize>(sum2, 0, 0xffffffff);
    sum3 = gpuprim::ShuffleIndex<kWarpSize>(sum3, 0, 0xffffffff);
    sum4 = gpuprim::ShuffleIndex<kWarpSize>(sum4, 0, 0xffffffff);
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
    const T* __restrict__ in, size_t N, size_t C, size_t D, U* out1, U* out2,
    Op1 op1, Op2 op2, bool is_channel_first) {
  const int tid = threadIdx.x % kWarpSize;
  const int local_warp_id = threadIdx.x / kWarpSize;
  const int warp_id = blockIdx.x * kWarpPerBlock + local_warp_id;

  typedef gpuprim::WarpReduce<U> WarpReduce;
  typename WarpReduce::TempStorage temp_storage[kWarpPerBlock];
  U partial_sum;
  for (size_t k = warp_id; k < N * C; k += gridDim.x * kWarpPerBlock) {
    partial_sum = 0;
    for (size_t i = tid; i < D; i += kWarpSize) {
      if (is_channel_first) {
        partial_sum += op1.Compute(in, k, i);
      } else {
        partial_sum += op1.Compute(in, k / C, i, k % C);
      }
    }

    U sum = WarpReduce(temp_storage[local_warp_id]).Sum(partial_sum);

    sum = gpuprim::ShuffleIndex<kWarpSize>(sum, 0, 0xffffffff);
    sum = op1.Finalize(sum);
    if (tid == 0) {
      out1[k] = sum;
    }

    partial_sum = 0;

    for (size_t i = tid; i < D; i += kWarpSize) {
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
    const T* __restrict__ x, size_t N, size_t C, size_t D, U* __restrict__ temp,
    Op op, bool is_channel_first = true) {
  if (is_channel_first) {
    typedef gpuprim::BlockReduce<U, kBlockSize> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int row_offset = threadIdx.x + blockIdx.x * blockDim.x;

    for (size_t row_idx = blockIdx.y; row_idx < N * C; row_idx += gridDim.y) {
      U partial_sum = 0;

      for (size_t i = row_offset; i < D; i += gridDim.x * blockDim.x) {
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

    for (size_t i = y_tid; i < D; i += blockDim.y * gridDim.y) {
      partial_sum += op.Compute(x, z_tid, i, x_tid);
    }

    temp[(z_tid * C + x_tid) * blockDim.y * gridDim.y + y_tid] = partial_sum;
  }
}

template <typename T, typename ComputeType, int PackSize>
__global__
__launch_bounds__(1024) void InstanceNormRowReduceInToTempFusedVectorized(
    const T* x, const T* dy, size_t N, size_t C, size_t D,
    const ComputeType* cache_mean, const ComputeType* cache_ivar,
    const ComputeType* gamma, ComputeType* __restrict__ temp_1,
    ComputeType* __restrict__ temp_2, ComputeType* __restrict__ temp_3,
    ComputeType* __restrict__ temp_4, DwStatFusedOp<T, ComputeType> op) {
  const int row_offset = threadIdx.x + blockIdx.x * blockDim.x;
  const auto num_packs = static_cast<size_t>(D / PackSize);
  ComputeType pack_x[PackSize];
  ComputeType pack_dy[PackSize];
  ComputeType tmp1, tmp2, tmp3, tmp4, partial_sum1, partial_sum2, partial_sum3,
      partial_sum4, row_mean, row_ivar, channel_gamma, sum1, sum2, sum3, sum4;
  for (size_t row_idx = blockIdx.y; row_idx < N * C; row_idx += gridDim.y) {
    partial_sum1 = 0;
    partial_sum2 = 0;
    partial_sum3 = 0;
    partial_sum4 = 0;
    row_mean = cache_mean[row_idx];
    row_ivar = cache_ivar[row_idx];
    channel_gamma = gamma[row_idx % C];

    for (size_t pack_id = row_offset; pack_id < num_packs;
         pack_id += gridDim.x * blockDim.x) {
      const int pack_offset = pack_id * PackSize;
      CopyWithCast<T, ComputeType, PackSize>(x, row_idx * D + pack_offset,
                                             pack_x);
      CopyWithCast<T, ComputeType, PackSize>(dy, row_idx * D + pack_offset,
                                             pack_dy);
      for (int i = 0; i < PackSize; ++i) {
        ComputeType x_mean = pack_x[i] - row_mean;
        ComputeType dy = pack_dy[i];
        op.Compute(x_mean, dy, row_ivar, channel_gamma, &tmp1, &tmp2, &tmp3,
                   &tmp4);
        partial_sum1 += tmp1;
        partial_sum2 += tmp2;
        partial_sum3 += tmp3;
        partial_sum4 += tmp4;
      }
    }

    sum1 =
        BlockAllReduce<ComputeType, gpuprim::Sum>(partial_sum1, gpuprim::Sum());
    __syncthreads();
    sum2 =
        BlockAllReduce<ComputeType, gpuprim::Sum>(partial_sum2, gpuprim::Sum());
    __syncthreads();
    sum3 =
        BlockAllReduce<ComputeType, gpuprim::Sum>(partial_sum3, gpuprim::Sum());
    __syncthreads();
    sum4 =
        BlockAllReduce<ComputeType, gpuprim::Sum>(partial_sum4, gpuprim::Sum());
    __syncthreads();
    if (threadIdx.x == 0) {
      int idx = row_idx * gridDim.x + blockIdx.x;
      temp_1[idx] = sum1;  // dmu
      temp_2[idx] = sum2;  // dvar
      temp_3[idx] = sum3;  // dgamma
      temp_4[idx] = sum4;  // dbeta
    }
  }
}

template <typename T, typename U>
void (*InstanceNormRowReduceInToTempFusedVectorizedCandidates[3])(
    const T*, const T*, size_t, size_t, size_t, const U*, const U*, const U*,
    U* __restrict__, U* __restrict__, U* __restrict__, U* __restrict__,
    DwStatFusedOp<T, U>){InstanceNormRowReduceInToTempFusedVectorized<T, U, 1>,
                         InstanceNormRowReduceInToTempFusedVectorized<T, U, 2>,
                         InstanceNormRowReduceInToTempFusedVectorized<T, U, 4>};

template <typename T, typename U, typename Op>
__global__ __launch_bounds__(1024) void InstanceNormRowReduceInToTempFused(
    const T* __restrict__ x, size_t N, size_t C, size_t D,
    U* __restrict__ temp1, U* __restrict__ temp2, U* __restrict__ temp3,
    U* __restrict__ temp4, Op op, bool is_channel_first = true) {
  if (is_channel_first) {
    typedef gpuprim::BlockReduce<U, kBlockSize> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    const int row_offset = threadIdx.x + blockIdx.x * blockDim.x;

    for (size_t row_idx = blockIdx.y; row_idx < N * C; row_idx += gridDim.y) {
      U partial_sum1 = 0;
      U partial_sum2 = 0;
      U partial_sum3 = 0;
      U partial_sum4 = 0;
      U ret1, ret2, ret3, ret4;

      for (size_t i = row_offset; i < D; i += gridDim.x * blockDim.x) {
        op.Compute(x, row_idx, i, &ret1, &ret2, &ret3, &ret4);
        partial_sum1 += ret1;
        partial_sum2 += ret2;
        partial_sum3 += ret3;
        partial_sum4 += ret4;
      }
      U sum1 = BlockReduce(temp_storage).Sum(partial_sum1);
      __syncthreads();
      U sum2 = BlockReduce(temp_storage).Sum(partial_sum2);
      __syncthreads();
      U sum3 = BlockReduce(temp_storage).Sum(partial_sum3);
      __syncthreads();
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

    for (size_t i = y_tid; i < D; i += blockDim.y * gridDim.y) {
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
    const T* __restrict__ x, const T* __restrict__ dy, size_t N, size_t D,
    T* out, Op op, bool is_channel_first = true) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= N) return;
  for (size_t row_idx = tid; row_idx < N; row_idx += gridDim.x * blockDim.x) {
    out[row_idx] = static_cast<T>(
        op.Compute(x[row_idx], dy[row_idx], row_idx, is_channel_first));
  }
}

template <typename T, typename U, int PackSize>
__global__ __launch_bounds__(1024) void InstanceNormUpdateFusedVectorized(
    const T* __restrict__ x, const T* __restrict__ dy, size_t N, size_t C,
    size_t D, T* dx, DxFusedOp<U> op, bool is_channel_first = true) {
  U pack_x[PackSize];
  U pack_dy[PackSize];
  const size_t num_packs = N * C * D / PackSize;
  const int row_offset = threadIdx.x + blockIdx.x * blockDim.x;
  for (size_t row_idx = row_offset; row_idx < num_packs;
       row_idx += gridDim.x * blockDim.x) {
    const int pack_offset = row_idx * PackSize;
    CopyWithCast<T, U, PackSize>(x, pack_offset, pack_x);
    CopyWithCast<T, U, PackSize>(dy, pack_offset, pack_dy);
    for (int i = 0; i < PackSize; ++i) {
      pack_x[i] =
          op.Compute(pack_x[i], pack_dy[i], pack_offset + i, is_channel_first);
    }
    CopyWithCast<U, T, PackSize>(pack_x, 0, dx, pack_offset);
  }
}

template <typename T, typename U>
void (*InstanceNormUpdateFusedVectorizedCandidates[3])(const T* __restrict__,
                                                       const T* __restrict__,
                                                       size_t, size_t, size_t,
                                                       T*, DxFusedOp<U>, bool){
    InstanceNormUpdateFusedVectorized<T, U, 1>,
    InstanceNormUpdateFusedVectorized<T, U, 2>,
    InstanceNormUpdateFusedVectorized<T, U, 4>};

template <typename T, typename ComputeType, int PackSize>
__global__ __launch_bounds__(1024) void InstanceNormUpdateVectorized(
    const T* x, size_t N, size_t C, size_t D, const ComputeType* gamma,
    const ComputeType* beta, T* y, YOp<T, ComputeType> op) {
  const int row_offset = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t num_packs = D / PackSize;
  for (size_t row_idx = blockIdx.y; row_idx < N * C; row_idx += gridDim.y) {
    size_t c_idx = row_idx % C;
    ComputeType gamma_val = gamma[c_idx];
    ComputeType beta_val = beta[c_idx];
    for (size_t pack_id = row_offset; pack_id < num_packs;
         pack_id += gridDim.x * blockDim.x) {
      ComputeType pack[PackSize];
      const size_t pack_offset = pack_id * PackSize;
      CopyWithCast<T, ComputeType, PackSize>(x, row_idx * D + pack_offset,
                                             pack);

      for (int i = 0; i < PackSize; ++i) {
        pack[i] = op.ComputePartial(pack[i], row_idx);
      }

      CopyWithAffineAndCast<ComputeType, T, PackSize>(
          pack, gamma_val, beta_val, y, row_idx * D + pack_offset);
    }
  }
}

template <typename T, typename U>
void (*InstanceNormUpdateVectorizedCandidates[3])(const T*, size_t, size_t,
                                                  size_t, const U*, const U*,
                                                  T*, YOp<T, U>){
    InstanceNormUpdateVectorized<T, U, 1>,
    InstanceNormUpdateVectorized<T, U, 2>,
    InstanceNormUpdateVectorized<T, U, 4>};

template <typename T, typename Op>
__global__ __launch_bounds__(1024) void InstanceNormUpdate(
    const T* __restrict__ in, size_t N, size_t D, T* out, Op op,
    bool is_channel_first = true) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= N) return;
  for (size_t row_idx = tid; row_idx < N; row_idx += gridDim.x * blockDim.x) {
    out[row_idx] = op.Compute(in, row_idx, is_channel_first);
  }
}

template <typename T, typename U, typename Op>
void InstanceNormReductionsChnlFirstFused(OpKernelContext* context, size_t N,
                                          size_t C, size_t D, const T* x,
                                          const T* dy, const U* cache_mean,
                                          const U* cache_ivar, const U* gamma,
                                          T* dx, U* temp_1, U* temp_2,
                                          U* temp_3, U* temp_4, Op op) {
  const bool is_channel_first = true;
  const GPUDevice& d = context->eigen_device<GPUDevice>();
  const int NxC = static_cast<int>(N * C);
  const int min_num_blocks = kWarpSize;

  Tensor scratch_dgamma, scratch_dbeta;
  OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                 {NxC}, &scratch_dgamma));
  OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                 {NxC}, &scratch_dbeta));
  U* temp_buffer_3 = scratch_dgamma.flat<U>().data();
  U* temp_buffer_4 = scratch_dbeta.flat<U>().data();
  Status status = TryDispatchInstanceNormGradBlockSMemImpl<T, U, Op>(
      d.stream(), x, dy, N, C, D, cache_mean, cache_ivar, gamma, dx, temp_1,
      temp_2, temp_buffer_3, temp_buffer_4, op);

  if (status.ok()) {
    TF_CHECK_OK(GpuLaunchKernel(ReduceNCtoC<U>, DivUp(C, kBlockSize),
                                kBlockSize, 0, d.stream(), N, C, temp_buffer_3,
                                temp_buffer_4, temp_3, temp_4));
  } else {
    bool use_single_block =
        (D <= min_num_blocks * kBlockSize * kMinWorkPerThreadIn);
    if (use_single_block) {
      LaunchVectorizedKernel<T>(
          InstanceNormRowReduceInToOutFusedCandidates<T, U>, NxC, kBlockSize, 0,
          d.stream(), D, x, dy, cache_mean, cache_ivar, gamma, N, C, D, temp_1,
          temp_2, temp_buffer_3, temp_buffer_4, op);

      TF_CHECK_OK(GpuLaunchKernel(
          ReduceNCtoC<U>, DivUp(C, kBlockSize), kBlockSize, 0, d.stream(), N, C,
          temp_buffer_3, temp_buffer_4, temp_3, temp_4));
    } else {
      const auto blocks_per_row = DivUp(D, kBlockSize * kMinWorkPerThreadIn);

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
      size_t temp_total_rows = blocks_per_row;
      // For long rows, we launch n blocks to process each row. The intermediate
      // results are stored in a temp memory with the size of N*n. Then, we
      // launch single block to handle each row of the temp memory.

      LaunchVectorizedKernel<T>(
          InstanceNormRowReduceInToTempFusedVectorizedCandidates<T, U>, blocks,
          threads, 0, d.stream(), D, x, dy, N, C, D, cache_mean, cache_ivar,
          gamma, temp_buffer_1, temp_buffer_2, temp_buffer_3, temp_buffer_4,
          op);

      TF_CHECK_OK(GpuLaunchKernel(InstanceNormRowReduceTempToOutFused<U, Op>,
                                  N * C, kBlockSize, 0, d.stream(),
                                  temp_buffer_1, temp_buffer_2, temp_buffer_3,
                                  temp_buffer_4, N, C, temp_total_rows, temp_1,
                                  temp_2, temp_3, temp_4, op));
    }
    DxFusedOp<U> dx_ops{cache_mean, cache_ivar, gamma, temp_2, temp_1, C, D};

    LaunchVectorizedKernel<T>(InstanceNormUpdateFusedVectorizedCandidates<T, U>,
                              DivUp(NxC * D, kBlockSize * kMinWorkPerThreadIn),
                              kBlockSize, 0, d.stream(), D, x, dy, N, C, D, dx,
                              dx_ops, is_channel_first);
  }
}

template <typename T, typename U, typename Op1, typename Op2>
void InstanceNormReductionsChnlFirst(OpKernelContext* context, size_t N,
                                     size_t C, size_t D, const T* x, U* temp_1,
                                     U* temp_2, Op1 op1, Op2 op2) {
  const bool is_channel_first = true;
  const GPUDevice& d = context->eigen_device<GPUDevice>();
  auto NxC = N * C;
  const int min_num_blocks = kWarpSize;

  bool use_single_block =
      (D <= min_num_blocks * kBlockSize * kMinWorkPerThreadIn);
  if (use_single_block) {
    TF_CHECK_OK(GpuLaunchKernel(InstanceNormRowReduceInToOut<T, U, Op1, Op2>,
                                NxC, kBlockSize, 0, d.stream(), x, NxC, D,
                                temp_1, temp_2, op1, op2));
  } else {
    const auto blocks_per_row = DivUp(D, kBlockSize * kMinWorkPerThreadIn);

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
    size_t temp_total_rows = blocks_per_row;

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

template <typename T, typename ComputeType, typename Op, int BlockSize,
          int PackSize>
__global__ void InstanceNormBlockSMemChnlFirstImpl(
    const T* x, size_t N, size_t C, size_t D, const ComputeType* gamma,
    const ComputeType* beta, T* y, ComputeType* __restrict__ out1,
    ComputeType* __restrict__ out2, Op op) {
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  auto* buf = reinterpret_cast<ComputeType*>(shared_buf);
  const int tid = threadIdx.x;
  assert(D % PackSize == 0);
  const size_t num_packs = D / PackSize;
  ComputeType pack[PackSize];
  for (size_t row = blockIdx.x; row < N * C; row += gridDim.x) {
    int channel_id = row % C;
    ComputeType gamma_val = gamma[channel_id];
    ComputeType beta_val = beta[channel_id];
    WFGeneric<ComputeType> wf_thread;
    for (size_t pack_id = tid; pack_id < num_packs; pack_id += BlockSize) {
      const int data_offset = row * D + pack_id * PackSize;
      CopyWithCast<T, ComputeType, PackSize>(x, data_offset, pack);

      for (int i = 0; i < PackSize; ++i) {
        buf[i * num_packs + pack_id] = pack[i];
        op.Update(pack[i], wf_thread);
      }
    }
    WFGeneric<ComputeType> wf_row =
        BlockAllReduce<WFGeneric<ComputeType>, WFGeneric<ComputeType>,
                       BlockSize>(wf_thread, WFGeneric<ComputeType>(), true);
    ComputeType row_mean = wf_row.mean;
    ComputeType row_inv_var = op.Finalize(wf_row);
    if (threadIdx.x == 0) {
      out1[row] = row_mean;
      out2[row] = row_inv_var;
    }
    for (size_t pack_id = tid; pack_id < num_packs; pack_id += BlockSize) {
      for (int i = 0; i < PackSize; ++i) {
        pack[i] = (buf[i * num_packs + pack_id] - row_mean) * row_inv_var;
      }
      const auto pack_offset = pack_id * PackSize;
      CopyWithAffineAndCast<ComputeType, T, PackSize>(pack, gamma_val, beta_val,
                                                      y, row * D + pack_offset);
    }
  }
}

template <typename T, typename ComputeType, typename Op>
inline Status TryDispatchInstanceNormBlockSMemChnlFirstImpl(
    gpuStream_t stream, const T* x, size_t N, size_t C, size_t D,
    const ComputeType* gamma, const ComputeType* beta, T* y, ComputeType* mean,
    ComputeType* inv_variance, Op op) {
  const size_t smem = D * sizeof(ComputeType);

  auto fns =
      InstanceNormBlockSMemChnlFirstImpl<T, ComputeType, Op, kBlockSize, 1>;
  int pack_size = 1;
  if (D % 4 == 0) {
    pack_size = 4;
    fns = InstanceNormBlockSMemChnlFirstImpl<T, ComputeType, Op, kBlockSize, 4>;
  } else if (D % 2 == 0) {
    pack_size = 2;
    fns = InstanceNormBlockSMemChnlFirstImpl<T, ComputeType, Op, kBlockSize, 2>;
  }

  auto fns_base = fns;
  int blk_cnt_base = 0;
  CUDA_RETURN_IF_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blk_cnt_base, fns, kBlockSize, smem));

  if (blk_cnt_base <= 0) {
    return errors::Internal(
        "Cannot find proper block count for block size 128.");
  }

  int blk_cnt = 0;
#define TRY_AND_LAUNCH(Bsize, Psize)                                          \
  fns = InstanceNormBlockSMemChnlFirstImpl<T, ComputeType, Op, Bsize, Psize>; \
  CUDA_RETURN_IF_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(         \
      &blk_cnt, fns, Bsize, smem));                                           \
  if (blk_cnt == blk_cnt_base) { /*note : grid_dim = N * C*/                  \
    TF_CHECK_OK(GpuLaunchKernel(fns, N* C, Bsize, smem, stream, x, N, C, D,   \
                                gamma, beta, y, mean, inv_variance, op));     \
    return OkStatus();                                                        \
  }
#define TRY_PACK_SIZE(PSIZE)                       \
  TRY_AND_LAUNCH(kBlockSizeSearchSpace[0], PSIZE); \
  TRY_AND_LAUNCH(kBlockSizeSearchSpace[1], PSIZE); \
  TRY_AND_LAUNCH(kBlockSizeSearchSpace[2], PSIZE);

  // There will be a fixed pack_size to choose from which depends on D.
  // Under each pack_aize, the order of fallback w.r.t. block_size is to try
  // from larger block dim to smaller, e.g. 1024 -> 512 -> 256 -> 128.
  if (pack_size == 4) {
    TRY_PACK_SIZE(4);
  } else if (pack_size == 2) {
    TRY_PACK_SIZE(2);
  }
  TRY_PACK_SIZE(1);
#undef TRY_PACK_SIZE

#undef TRY_AND_LAUNCH
  TF_CHECK_OK(GpuLaunchKernel(fns_base, N * C, kBlockSize, smem, stream, x, N,
                              C, D, gamma, beta, y, mean, inv_variance, op));
  return OkStatus();
}

template <typename T, typename ComputeType, typename Op, int BlockSize>
__global__ void InstanceNormBlockSMemChnlLastImpl(
    const T* x, size_t N, size_t C, size_t D, const ComputeType* gamma,
    const ComputeType* beta, T* y, ComputeType* __restrict__ out1,
    ComputeType* __restrict__ out2, Op op) {
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  auto* buf = reinterpret_cast<ComputeType*>(shared_buf);
  auto* buf_for_reduce = reinterpret_cast<WFGeneric<ComputeType>*>(
      shared_buf + D * sizeof(ComputeType) * kWarpSize);
  const int x_tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int y_tid = threadIdx.y + blockIdx.y * blockDim.y;
  const int z_tid = threadIdx.z + blockIdx.z * blockDim.z;
  // use C / 32 blocks in x direction
  if (x_tid >= C) return;
  WFGeneric<ComputeType> wf_partial;
  size_t DC = D * C;
  size_t row_offset = threadIdx.x * D;
  for (size_t i = y_tid; i < D; i += blockDim.y * gridDim.y) {
    ComputeType x_in = x[z_tid * DC + i * C + x_tid];
    buf[row_offset + i] = x_in;
    op.Update(x_in, wf_partial);
  }
  buf_for_reduce[threadIdx.x * BlockSize / kWarpSize + threadIdx.y] =
      wf_partial;
  __syncthreads();
  if (threadIdx.y == 0) {
    WFGeneric<ComputeType> wf_row_sum;
    for (int i = 0; i < BlockSize / kWarpSize; ++i) {
      wf_row_sum = WFGeneric<ComputeType>()(
          wf_row_sum, buf_for_reduce[threadIdx.x * BlockSize / kWarpSize + i]);
    }
    buf_for_reduce[threadIdx.x * BlockSize / kWarpSize] = wf_row_sum;
  }
  __syncthreads();

  WFGeneric<ComputeType> per_row_result =
      buf_for_reduce[threadIdx.x * BlockSize / kWarpSize];
  ComputeType per_row_mean = per_row_result.mean;
  ComputeType per_row_inv_std = op.Finalize(per_row_result);
  if (threadIdx.y == 0) {
    out1[z_tid * C + x_tid] = per_row_mean;
    out2[z_tid * C + x_tid] = per_row_inv_std;
  }

  for (size_t i = y_tid; i < D; i += blockDim.y * gridDim.y) {
    ComputeType x_in = buf[row_offset + i];
    y[z_tid * DC + i * C + x_tid] = static_cast<T>(
        (x_in - per_row_mean) * per_row_inv_std * gamma[x_tid] + beta[x_tid]);
  }
}

template <typename T, typename ComputeType, typename Op>
inline Status TryDispatchInstanceNormBlockSMemChnlLastImpl(
    gpuStream_t stream, const T* x, size_t N, size_t C, size_t D,
    const ComputeType* gamma, const ComputeType* beta, T* y, ComputeType* mean,
    ComputeType* inv_variance, Op op) {
  const size_t smem = D * sizeof(ComputeType) * kWarpSize +
                      kBlockSize * sizeof(WFGeneric<ComputeType>);
  auto fns = InstanceNormBlockSMemChnlLastImpl<T, ComputeType, Op, kBlockSize>;
  auto fns_base = fns;
  int blk_cnt_base = 0;

  CUDA_RETURN_IF_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blk_cnt_base, fns, kBlockSize, smem));

  if (blk_cnt_base <= 0) {
    return errors::Internal(
        "Cannot find proper block count for block size 128.");
  }

  int blk_cnt = 0;
// Since no vectorization load support for NHWC, no need to iterate over Psize
#define TRY_AND_LAUNCH(Bsize)                                                 \
  {                                                                           \
    fns = InstanceNormBlockSMemChnlLastImpl<T, ComputeType, Op, Bsize>;       \
    const size_t smem = D * sizeof(ComputeType) * kWarpSize +                 \
                        Bsize * sizeof(WFGeneric<ComputeType>);               \
    dim3 blk_dim(kWarpSize, Bsize / kWarpSize);                               \
    dim3 grid_dim(DivUp(C, kWarpSize), 1, N);                                 \
    CUDA_RETURN_IF_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(       \
        &blk_cnt, fns, Bsize, smem));                                         \
    if (blk_cnt == blk_cnt_base) {                                            \
      TF_CHECK_OK(GpuLaunchKernel(fns, grid_dim, blk_dim, smem, stream, x, N, \
                                  C, D, gamma, beta, y, mean, inv_variance,   \
                                  op));                                       \
      return OkStatus();                                                      \
    }                                                                         \
  }

  // The order of fallback w.r.t. block_size is to try from larger block dim to
  // smaller, e.g. 1024 -> 512 -> 256 -> 128.
  TRY_AND_LAUNCH(kBlockSizeSearchSpace[0]);
  TRY_AND_LAUNCH(kBlockSizeSearchSpace[1]);
  TRY_AND_LAUNCH(kBlockSizeSearchSpace[2]);
#undef TRY_AND_LAUNCH
  dim3 blk_dim(kWarpSize, kWarpPerBlock);
  dim3 grid_dim(DivUp(C, kWarpSize), 1, N);
  TF_CHECK_OK(GpuLaunchKernel(fns_base, grid_dim, blk_dim, smem, stream, x, N,
                              C, D, gamma, beta, y, mean, inv_variance, op));
  return OkStatus();
}

template <typename T, typename U, typename Op, int PackSize>
__global__ __launch_bounds__(1024) void GeneralNormRowReduceInToOutWelford(
    const T* x, size_t NxC, size_t D, U* out1, U* out2, Op op) {
  const int tid = threadIdx.x;
  assert(D % PackSize == 0);
  const auto num_packs = D / PackSize;
  U pack[PackSize];
  for (size_t k = blockIdx.x; k < NxC; k += gridDim.x) {
    WFGeneric<U> wf_thread;

    for (size_t pack_id = tid; pack_id < num_packs; pack_id += kBlockSize) {
      const int data_offset = k * D + pack_id * PackSize;
      CopyWithCast<T, U, PackSize>(x, data_offset, pack);

      for (int i = 0; i < PackSize; ++i) {
        op.Update(pack[i], wf_thread);
      }
    }

    WFGeneric<U> wf_row =
        BlockAllReduce<WFGeneric<U>, WFGeneric<U>>(wf_thread, WFGeneric<U>());
    __syncthreads();        
    if (tid == 0) {
      out1[k] = wf_row.mean;
      out2[k] = op.Finalize(wf_row);
    }
  }
}

template <typename T, typename U, typename Op>
void (*GeneralNormRowReduceInToOutWelfordCandidates[3])(const T*, size_t,
                                                        size_t, U*, U*, Op){
    GeneralNormRowReduceInToOutWelford<T, U, Op, 1>,
    GeneralNormRowReduceInToOutWelford<T, U, Op, 2>,
    GeneralNormRowReduceInToOutWelford<T, U, Op, 4>};

template <typename T, typename U, typename Op>
void InstanceNormReductionsChnlFirstWelford(OpKernelContext* context, size_t N,
                                            size_t C, size_t D, const T* x,
                                            const U* gamma, const U* beta, T* y,
                                            U* temp_1, U* temp_2, Op op) {
  const bool is_channel_first = true;
  const GPUDevice& d = context->eigen_device<GPUDevice>();
  const int NxC = static_cast<int>(N * C);
  const int min_num_blocks = kWarpSize;

  Status status =
      TryDispatchInstanceNormBlockSMemChnlFirstImpl<T, U, WFOp<T, U>>(
          d.stream(), x, N, C, D, gamma, beta, y, temp_1, temp_2, op);

  if (status.ok()) {
    return;
  }

  bool use_single_block =
      (D <= min_num_blocks * kBlockSize * kMinWorkPerThreadIn);

  const int blocks_per_row = DivUp(D, kBlockSize * kMinWorkPerThreadIn);
  dim3 threads(kBlockSize);
  dim3 blocks(blocks_per_row, N);

  if (use_single_block) {
    LaunchVectorizedKernel<T>(
        GeneralNormRowReduceInToOutWelfordCandidates<T, U, Op>, NxC, kBlockSize,
        0, d.stream(), D, x, NxC, D, temp_1, temp_2, op);

    YOp<T, U> y_op{temp_1, temp_2, gamma, beta, C, D};
    TF_CHECK_OK(GpuLaunchKernel(
        InstanceNormUpdate<T, YOp<T, U>>,
        DivUp(N * C * D, kBlockSize * kMinWorkPerThreadIn), kBlockSize, 0,
        d.stream(), x, N * C * D, D, y, y_op, is_channel_first));
  } else {
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

    size_t temp_total_rows = blocks_per_row;

    // For long rows, we launch n blocks to process each row. The intermediate
    // results are stored in a temp memory with the size of N*n. Then, we
    // launch single block to handle each row of the temp memory.

    LaunchVectorizedKernel<T>(InstanceNormToTempWelfordCandidates<T, U>, blocks,
                              threads, 0, d.stream(), D, x, N, C, D,
                              temp_buffer_1, temp_buffer_2, temp_buffer_3, op,
                              is_channel_first);

    TF_CHECK_OK(GpuLaunchKernel(InstanceNormTempToOutWelford<U, Op>, NxC,
                                kBlockSize, 0, d.stream(), temp_buffer_1,
                                temp_buffer_2, temp_buffer_3, N, C,
                                temp_total_rows, temp_1, temp_2, op));

    YOp<T, U> y_op{temp_1, temp_2, gamma, beta, C, D};
    LaunchVectorizedKernel<T>(InstanceNormUpdateVectorizedCandidates<T, U>,
                              blocks, kBlockSize, 0, d.stream(), D, x, N, C, D,
                              gamma, beta, y, y_op);
  }
}

template <typename T, typename U, typename Op1, typename Op2>
void InstanceNormReductionsChnlLast(OpKernelContext* context, size_t N,
                                    size_t C, size_t D, const T* x, U* temp_1,
                                    U* temp_2, Op1 op1, Op2 op2) {
  const GPUDevice& d = context->eigen_device<GPUDevice>();

  dim3 block_dim(kWarpSize, kWarpPerBlock);

  size_t ppr = DivUp(D, int(block_dim.y * kMinWorkPerThreadCLast));

  dim3 grid_dim(DivUp(C, kWarpSize), ppr, N);

  Tensor scratch1, scratch2;
  OP_REQUIRES_OK(
      context, context->allocate_temp(DataTypeToEnum<U>::value,
                                      {N * C * block_dim.y * ppr}, &scratch1));
  OP_REQUIRES_OK(
      context, context->allocate_temp(DataTypeToEnum<U>::value,
                                      {N * C * block_dim.y * ppr}, &scratch2));
  U* temp_buffer_1 = scratch1.flat<U>().data();
  U* temp_buffer_2 = scratch2.flat<U>().data();

  bool is_channel_first = false;
  TF_CHECK_OK(GpuLaunchKernel(InstanceNormRowReduceInToTemp<T, U, Op1>,
                              grid_dim, block_dim, 0, d.stream(), x, N, C, D,
                              temp_buffer_1, op1, is_channel_first));
  TF_CHECK_OK(GpuLaunchKernel(InstanceNormRowReduceTempToOut<U, Op1>, N * C,
                              kBlockSize, 0, d.stream(), temp_buffer_1, N, C,
                              block_dim.y * ppr, temp_1, op1));

  TF_CHECK_OK(GpuLaunchKernel(InstanceNormRowReduceInToTemp<T, U, Op2>,
                              grid_dim, block_dim, 0, d.stream(), x, N, C, D,
                              temp_buffer_2, op2, is_channel_first));
  TF_CHECK_OK(GpuLaunchKernel(InstanceNormRowReduceTempToOut<U, Op2>, N * C,
                              kBlockSize, 0, d.stream(), temp_buffer_2, N, C,
                              block_dim.y * ppr, temp_2, op2));
}

template <typename T, typename U, typename Op>
void InstanceNormReductionsChnlLastWelford(OpKernelContext* context, size_t N,
                                           size_t C, size_t D, const T* x,
                                           const U* gamma, const U* beta, T* y,
                                           U* temp_1, U* temp_2, Op op) {
  const GPUDevice& d = context->eigen_device<GPUDevice>();
  Status status =
      TryDispatchInstanceNormBlockSMemChnlLastImpl<T, U, WFOp<T, U>>(
          d.stream(), x, N, C, D, gamma, beta, y, temp_1, temp_2, op);
  if (status.ok()) {
    return;
  }
  dim3 block_dim(kWarpSize, kWarpPerBlock);

  size_t ppr = DivUp(D, int(block_dim.y * kMinWorkPerThreadCLast));
  dim3 grid_dim(DivUp(C, kWarpSize), ppr, N);

  TensorShape scratch_shape({static_cast<int>(N * C * block_dim.y * ppr)});

  Tensor scratch1, scratch2, scratch3;
  OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                 scratch_shape, &scratch1));
  OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                 scratch_shape, &scratch2));
  OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                 scratch_shape, &scratch3));
  U* temp_buffer_1 = scratch1.flat<U>().data();
  U* temp_buffer_2 = scratch2.flat<U>().data();
  U* temp_buffer_3 = scratch3.flat<U>().data();

  bool is_channel_first = false;

  LaunchVectorizedKernel<T>(InstanceNormToTempWelfordCandidates<T, U>, grid_dim,
                            block_dim, 0, d.stream(), D, x, N, C, D,
                            temp_buffer_1, temp_buffer_2, temp_buffer_3, op,
                            is_channel_first);

  TF_CHECK_OK(GpuLaunchKernel(InstanceNormTempToOutWelford<U, Op>, N * C,
                              kBlockSize, 0, d.stream(), temp_buffer_1,
                              temp_buffer_2, temp_buffer_3, N, C,
                              block_dim.y * ppr, temp_1, temp_2, op));
  YOp<T, U> y_ops{temp_1, temp_2, gamma, beta, C, D};
  TF_CHECK_OK(GpuLaunchKernel(
      InstanceNormUpdate<T, YOp<T, U>>,
      DivUp(N * C * D, kBlockSize * kMinWorkPerThreadIn), kBlockSize, 0,
      d.stream(), x, N * C * D, D, y, y_ops, is_channel_first));
}

template <typename T, typename U, typename Op>
void InstanceNormReductionsChnlLastFused(OpKernelContext* context, size_t N,
                                         size_t C, size_t D, const T* x,
                                         const T* dy, const U* cache_mean,
                                         const U* cache_ivar, const U* gamma,
                                         T* dx, U* temp_1, U* temp_2, U* temp_3,
                                         U* temp_4, Op op) {
  const GPUDevice& d = context->eigen_device<GPUDevice>();

  dim3 threads(kWarpSize, kWarpPerBlock);

  const size_t ppr = DivUp(D, int(threads.y * kMinWorkPerThreadCLast));

  dim3 blocks(DivUp(C, kWarpSize), ppr, N);
  TensorShape scratch_shape({static_cast<int>(N * C * threads.y * ppr)});
  Tensor scratch1, scratch2, scratch3, scratch4;
  OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                 scratch_shape, &scratch1));
  OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                 scratch_shape, &scratch2));
  OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                 scratch_shape, &scratch3));
  OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                 scratch_shape, &scratch4));
  U* temp_buffer_1 = scratch1.flat<U>().data();
  U* temp_buffer_2 = scratch2.flat<U>().data();
  U* temp_buffer_3 = scratch3.flat<U>().data();
  U* temp_buffer_4 = scratch4.flat<U>().data();

  bool is_channel_first = false;
  TF_CHECK_OK(GpuLaunchKernel(InstanceNormRowReduceInToTempFused<T, U, Op>,
                              blocks, threads, 0, d.stream(), dy, N, C, D,
                              temp_buffer_1, temp_buffer_2, temp_buffer_3,
                              temp_buffer_4, op, is_channel_first));
  TF_CHECK_OK(GpuLaunchKernel(
      InstanceNormRowReduceTempToOutFused<U, Op>, N * C, kBlockSize, 0,
      d.stream(), temp_buffer_1, temp_buffer_2, temp_buffer_3, temp_buffer_4, N,
      C, threads.y * ppr, temp_1, temp_2, temp_3, temp_4, op));

  DxFusedOp<U> dx_ops{cache_mean, cache_ivar, gamma, temp_2, temp_1, C, D};

  LaunchVectorizedKernel<T>(InstanceNormUpdateFusedVectorizedCandidates<T, U>,
                            DivUp(N * C * D, kBlockSize * kMinWorkPerThreadIn),
                            kBlockSize, 0, d.stream(), D, x, dy, N, C, D, dx,
                            dx_ops, is_channel_first);
}

template <typename T, typename U>
void InstanceNormGrad(OpKernelContext* context, const T* x, const T* dy,
                      const U* cache_mean, const U* cache_ivar, const U* gamma,
                      size_t N, size_t C, size_t D, T* dx, U* temp_1, U* temp_2,
                      U* temp_3, U* temp_4, bool is_channel_first) {
  const GPUDevice& d = context->eigen_device<GPUDevice>();
  bool use_single_warp = (D <= kWarpSize);

  DwStatFusedOp<T, U> dldwstat_ops{gamma, x, cache_ivar, cache_mean, C, D};

  const int NxC = static_cast<int>(N * C);
  if (use_single_warp) {
    Tensor scratch3, scratch4;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                   {NxC}, &scratch3));
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                   {NxC}, &scratch4));
    U* temp_buffer_3 = scratch3.flat<U>().data();
    U* temp_buffer_4 = scratch4.flat<U>().data();

    TF_CHECK_OK(GpuLaunchKernel(
        InstanceNormRowReduceInToOutWarpFused<T, U, DwStatFusedOp<T, U>>,
        DivUp(NxC, kWarpPerBlock), kBlockSize, 0, d.stream(), dy, N, C, D,
        temp_1, temp_2, temp_buffer_3, temp_buffer_4, dldwstat_ops,
        is_channel_first));
    TF_CHECK_OK(GpuLaunchKernel(ReduceNCtoC<U>, DivUp(C, kBlockSize),
                                kBlockSize, 0, d.stream(), N, C, temp_buffer_3,
                                temp_buffer_4, temp_3, temp_4));
    // The temp_1 and temp_2 are dl_dmu and dl_dvars respectively.
    DxFusedOp<U> dx_ops{cache_mean, cache_ivar, gamma, temp_2, temp_1, C, D};

    TF_CHECK_OK(GpuLaunchKernel(
        InstanceNormUpdateFused<T, DxFusedOp<U>>,
        DivUp(NxC * D, kBlockSize * kMinWorkPerThreadIn), kBlockSize, 0,
        d.stream(), x, dy, N * C * D, D, dx, dx_ops, is_channel_first));
  } else {
    if (is_channel_first) {
      InstanceNormReductionsChnlFirstFused<T, U, DwStatFusedOp<T, U>>(
          context, N, C, D, x, dy, cache_mean, cache_ivar, gamma, dx, temp_1,
          temp_2, temp_3, temp_4, dldwstat_ops);
    } else {
      InstanceNormReductionsChnlLastFused<T, U, DwStatFusedOp<T, U>>(
          context, N, C, D, x, dy, cache_mean, cache_ivar, gamma, dx, temp_1,
          temp_2, temp_3, temp_4, dldwstat_ops);
    }
  }
}

template <typename T, typename U>
struct FusedInstanceNorm<GPUDevice, T, U> {
  void operator()(OpKernelContext* context, const Tensor& x_input,
                  const Tensor& scale_input, const Tensor& offset_input,
                  U epsilon, bool is_channel_first, Tensor* y_output,
                  Tensor* saved_mean_output, Tensor* saved_inv_var_output) {
    if (x_input.shape().num_elements() == 0) {
      return;
    }
    int channel_dim = is_channel_first ? 1 : 2;
    int feature_dim = is_channel_first ? 2 : 1;
    const size_t N = x_input.dim_size(0);
    const size_t C = x_input.dim_size(channel_dim);
    const size_t D = x_input.dim_size(feature_dim);

    const int NxC = static_cast<int>(N * C);

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
          DivUp(NxC, kWarpPerBlock), kBlockSize, 0, d.stream(), x, N, C, D,
          cache_mean, cache_ivar, mean_ops, ivar_ops, is_channel_first));
      YOp<T, U> y_ops{cache_mean, cache_ivar, gamma, beta, C, D};
      TF_CHECK_OK(GpuLaunchKernel(
          InstanceNormUpdate<T, YOp<T, U>>,
          DivUp(N * C * D, kBlockSize * kMinWorkPerThreadIn), kBlockSize, 0,
          d.stream(), x, N * C * D, D, y, y_ops, is_channel_first));

    } else {
      if (is_channel_first) {
        InstanceNormReductionsChnlFirstWelford<T, U, WFOp<T, U>>(
            context, N, C, D, x, gamma, beta, y, cache_mean, cache_ivar,
            wf_ops);
      } else {
        InstanceNormReductionsChnlLastWelford<T, U, WFOp<T, U>>(
            context, N, C, D, x, gamma, beta, y, cache_mean, cache_ivar,
            wf_ops);
      }
    }
  }
};

template <typename T, typename U>
struct FusedInstanceNormGrad<GPUDevice, T, U> {
  void operator()(OpKernelContext* context, const Tensor& y_backprop_input,
                  const Tensor& x_input, const Tensor& scale_input,
                  const Tensor& saved_mean_input,
                  const Tensor& saved_inv_var_input, U epsilon,
                  bool is_channel_first, Tensor* x_backprop_output,
                  Tensor* scale_backprop_output,
                  Tensor* offset_backprop_output) {
    if (x_input.shape().num_elements() == 0) {
      return;
    }
    const GPUDevice& d = context->eigen_device<GPUDevice>();
    int channel_dim = is_channel_first ? 1 : 2;
    int feature_dim = is_channel_first ? 2 : 1;
    const size_t N = x_input.dim_size(0);
    const size_t C = x_input.dim_size(channel_dim);
    const size_t D = x_input.dim_size(feature_dim);

    const T* dy = y_backprop_input.flat<T>().data();
    const T* x = x_input.flat<T>().data();
    const U* gamma = scale_input.flat<U>().data();
    const U* cache_mean = saved_mean_input.flat<U>().data();
    const U* cache_ivar = saved_inv_var_input.flat<U>().data();
    T* dx = x_backprop_output->flat<T>().data();
    U* dgamma = scale_backprop_output->flat<U>().data();
    U* dbeta = offset_backprop_output->flat<U>().data();

    TensorShape scratch_shape({static_cast<int>(N * C)});
    Tensor scratch1, scratch2;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                   scratch_shape, &scratch1));
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                   scratch_shape, &scratch2));
    U* temp_1 = scratch1.flat<U>().data();
    U* temp_2 = scratch2.flat<U>().data();

    InstanceNormGrad<T, U>(context, x, dy, cache_mean, cache_ivar, gamma, N, C,
                           D, dx, temp_1, temp_2, dgamma, dbeta,
                           is_channel_first);
    // The temp_1 and temp_2 are dl_dmu and dl_dvars respectively.
  }
};

template struct FusedInstanceNorm<GPUDevice, float, float>;
template struct FusedInstanceNorm<GPUDevice, Eigen::half, float>;
template struct FusedInstanceNormGrad<GPUDevice, float, float>;
template struct FusedInstanceNormGrad<GPUDevice, Eigen::half, float>;

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
