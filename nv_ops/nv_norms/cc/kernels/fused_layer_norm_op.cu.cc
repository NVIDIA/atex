/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
==============================================================================*/

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#endif

#include "fused_layer_norm_op.h"
#include "norm_util.h"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

using namespace LnNorm;

template <typename T, typename ComputeType, typename Op1, typename Op2,
          int PackSize, int ColsPerThread>
__global__ void LayerNormGradWarpImpl(const T* x, const T* dy,
                                      const ComputeType* gamma,
                                      const int64_t rows, const int64_t cols,
                                      const ComputeType* cache_mean,
                                      const ComputeType* cache_ivar, Op1 op1,
                                      Op2 op2, T* dx, bool is_padding) {
  constexpr int num_packs = ColsPerThread / PackSize;

  const int lane_id = threadIdx.x % kWarpSize;

  constexpr int num_warps = kBlockSize / kWarpSize;
  typedef gpuprim::WarpReduce<ComputeType> WarpReduce;
  typename WarpReduce::TempStorage temp_storage[num_warps];
  const int local_warp_id = threadIdx.x / kWarpSize;
  const int warp_id = blockIdx.x * num_warps + local_warp_id;

  constexpr int thread_group_width = kWarpSize;
  ComputeType buf_x[ColsPerThread];
  ComputeType buf_dy_dot_gamma[ColsPerThread];
  ComputeType one_over_cols =
      static_cast<ComputeType>(1.0) / static_cast<ComputeType>(cols);
  for (int k = warp_id; k < rows; k += gridDim.x * num_warps) {
    ComputeType row_ivar = cache_ivar[k];
    ComputeType row_mean = cache_mean[k];
    ComputeType partial_sum_1 = 0;
    ComputeType partial_sum_2 = 0;

    for (int pack_id = 0; pack_id < num_packs; ++pack_id) {
      const int col = (pack_id * thread_group_width + lane_id) * PackSize;
      const int pack_offset = pack_id * PackSize;
      const int data_offset = k * cols + col;
      if (!is_padding || col < cols) {
        CopyWithCast<T, ComputeType, PackSize>(x, data_offset,
                                               buf_x + pack_offset);
        CopyWithCast<T, ComputeType, PackSize>(dy, data_offset,
                                               buf_dy_dot_gamma + pack_offset);
        CopyWithDot<ComputeType, PackSize>(gamma, col,
                                           buf_dy_dot_gamma + pack_offset);

        for (int i = 0; i < PackSize; ++i) {
          ComputeType x = buf_x[pack_offset + i];
          ComputeType dy_dot_gamma = buf_dy_dot_gamma[pack_offset + i];
          partial_sum_1 += op1.Compute(x, dy_dot_gamma, k);
          partial_sum_2 += op2.Compute(dy_dot_gamma, k);
        }
      } else {
        for (int i = 0; i < PackSize; ++i) {
          buf_x[pack_offset + i] = 0;
          buf_dy_dot_gamma[pack_offset + i] = 0;
        }
      }
    }

    ComputeType sum =
        WarpReduce(temp_storage[local_warp_id]).Sum(partial_sum_1);
    sum = gpuprim::ShuffleIndex<thread_group_width, ComputeType>(sum, 0,
                                                                 0xffffffff);
    ComputeType dldvar = op1.Finalize(sum);
    sum = 0;
    sum = WarpReduce(temp_storage[local_warp_id]).Sum(partial_sum_2);
    sum = gpuprim::ShuffleIndex<thread_group_width, ComputeType>(sum, 0,
                                                                 0xffffffff);
    ComputeType dldmu = op2.Finalize(sum);

    for (int i = 0; i < ColsPerThread; ++i) {
      buf_x[i] = (2.f * (buf_x[i] - row_mean) * dldvar + dldmu) * one_over_cols;
      buf_dy_dot_gamma[i] = buf_dy_dot_gamma[i] * row_ivar;
      buf_x[i] += buf_dy_dot_gamma[i];
    }

    for (int i = 0; i < num_packs; ++i) {
      const int col = (i * thread_group_width + lane_id) * PackSize;
      if (!is_padding || col < cols) {
        CopyWithCast<ComputeType, T, PackSize>(buf_x + i * PackSize, 0, dx,
                                               k * cols + col);
      }
    }
  }
}

template <typename T, typename U, typename Op1, typename Op2>
__global__ __launch_bounds__(1024) void LayerNormRowReduceInToOutWarp(
    const T* __restrict__ in, int N, int D, U* __restrict__ out1,
    U* __restrict__ out2, Op1 op1, Op2 op2) {
  const int tid = threadIdx.x % kWarpSize;

  const int num_warps = kBlockSize / kWarpSize;
  typedef gpuprim::WarpReduce<U> WarpReduce;
  typename WarpReduce::TempStorage temp_storage[num_warps];

  const int local_warp_id = threadIdx.x / kWarpSize;
  const int warp_id = blockIdx.x * num_warps + local_warp_id;

  for (int k = warp_id; k < N; k += gridDim.x * num_warps) {
    U partial_sum1 = 0;
    U partial_sum2 = 0;
    for (int i = tid; i < D; i += kWarpSize) {
      partial_sum1 += op1.Compute(in, k, i);
      partial_sum2 += op2.Compute(in, k, i);
    }

    U sum = WarpReduce(temp_storage[local_warp_id]).Sum(partial_sum1);

    if (tid == 0) {
      out1[k] = op1.Finalize(sum);
    }

    sum = WarpReduce(temp_storage[local_warp_id]).Sum(partial_sum2);

    if (tid == 0) {
      out2[k] = op2.Finalize(sum);
    }
  }
}

template <typename T, typename ComputeType, typename Op, int BlockSize,
          int PackSize>
__global__ void LayerNormBlockSMemImpl(const T* x, const ComputeType* gamma,
                                       const ComputeType* beta,
                                       const int64_t rows, const int64_t cols,
                                       T* y, ComputeType* __restrict__ out1,
                                       ComputeType* __restrict__ out2, Op op) {
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  auto* buf = reinterpret_cast<ComputeType*>(shared_buf);
  const int tid = threadIdx.x;
  assert(cols % PackSize == 0);
  const int num_packs = cols / PackSize;
  ComputeType pack[PackSize];
  int pack_id = tid;
  int data_offset;

  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    WFGeneric<ComputeType> wf_thread;
    for (pack_id = tid; pack_id < num_packs; pack_id += BlockSize) {
      data_offset = row * cols + pack_id * PackSize;
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
    pack_id = tid;
    int pack_offset;
    for (pack_id = tid; pack_id < num_packs; pack_id += BlockSize) {
      for (int i = 0; i < PackSize; ++i) {
        pack[i] = (buf[i * num_packs + pack_id] - row_mean) * row_inv_var;
      }

      pack_offset = pack_id * PackSize;
      CopyWithAffineAndCast<ComputeType, T, PackSize>(pack, gamma, pack_offset,
                                                      beta, pack_offset, y,
                                                      row * cols + pack_offset);
    }
  }
}

template <typename T, typename ComputeType, typename Op1, typename Op2,
          int BlockSize, int PackSize>
__global__ void LayerNormGradBlockSMemImpl(
    const T* x, const T* dy, const ComputeType* gamma, const int64_t rows,
    const int64_t cols, const ComputeType* cache_mean,
    const ComputeType* cache_ivar, T* dx, Op1 op1, Op2 op2) {
  extern __shared__ __align__(sizeof(double)) unsigned char grad_shared_buf[];
  auto* buf_x = reinterpret_cast<ComputeType*>(grad_shared_buf);
  auto* buf_dy_dot_gamma = buf_x + cols;
  const int tid = threadIdx.x;
  assert(cols % PackSize == 0);
  const int num_packs = cols / PackSize;
  const ComputeType one_over_cols =
      static_cast<ComputeType>(1.0) / static_cast<ComputeType>(cols);
  ComputeType x_pack[PackSize];
  ComputeType dy_pack[PackSize];

  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    ComputeType partial_sum_1 = 0;
    ComputeType partial_sum_2 = 0;
    const ComputeType row_mean = cache_mean[row];
    const ComputeType row_inv_var = cache_ivar[row];

    for (int pack_id = tid; pack_id < num_packs; pack_id += BlockSize) {
      const int data_offset = row * cols + pack_id * PackSize;
      CopyWithCast<T, ComputeType, PackSize>(x, data_offset, x_pack);
      CopyWithCast<T, ComputeType, PackSize>(dy, data_offset, dy_pack);
      CopyWithDot<ComputeType, PackSize>(gamma, pack_id * PackSize, dy_pack);

      for (int i = 0; i < PackSize; ++i) {
        const int buf_offset = i * num_packs + pack_id;
        ComputeType x = x_pack[i];
        ComputeType dy_dot_gamma = dy_pack[i];
        buf_x[buf_offset] = x;
        buf_dy_dot_gamma[buf_offset] = dy_dot_gamma;

        partial_sum_1 += op1.Compute(x, dy_dot_gamma, row);
        partial_sum_2 += op2.Compute(dy_dot_gamma, row);
      }
    }

    ComputeType dldvar = BlockAllReduce<ComputeType, gpuprim::Sum, BlockSize>(
        partial_sum_1, gpuprim::Sum(), true);
    __syncthreads();
    ComputeType dldmu = BlockAllReduce<ComputeType, gpuprim::Sum, BlockSize>(
        partial_sum_2, gpuprim::Sum(), true);

    for (int pack_id = tid; pack_id < num_packs; pack_id += BlockSize) {
      for (int i = 0; i < PackSize; ++i) {
        const int buf_offset = i * num_packs + pack_id;
        x_pack[i] = (2.f * (buf_x[buf_offset] - row_mean) * dldvar + dldmu) *
                        one_over_cols +
                    buf_dy_dot_gamma[buf_offset] * row_inv_var;
      }

      CopyWithCast<ComputeType, T, PackSize>(x_pack, 0, dx,
                                             row * cols + pack_id * PackSize);
    }
  }
}

template <typename T, typename ComputeType, typename Op>
Status TryDispatchLayerNormBlockSMemImpl(gpuStream_t stream, const T* x,
                                         const ComputeType* gamma,
                                         const ComputeType* beta,
                                         const int64_t rows, const int64_t cols,
                                         T* y, ComputeType* mean,
                                         ComputeType* inv_variance, Op op) {
  const size_t smem = cols * sizeof(ComputeType);

  auto fns = LayerNormBlockSMemImpl<T, ComputeType, Op, kBlockSize, 1>;
  int pack_size = 1;
  if (cols % 4 == 0) {
    pack_size = 4;
    fns = LayerNormBlockSMemImpl<T, ComputeType, Op, kBlockSize, 4>;
  } else if (cols % 2 == 0) {
    pack_size = 2;
    fns = LayerNormBlockSMemImpl<T, ComputeType, Op, kBlockSize, 2>;
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
#define TRY_AND_LAUNCH(Bsize, Psize)                                           \
  fns = LayerNormBlockSMemImpl<T, ComputeType, Op, Bsize, Psize>;              \
  CUDA_RETURN_IF_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(          \
      &blk_cnt, fns, Bsize, smem));                                            \
  if (blk_cnt == blk_cnt_base) {                                               \
    TF_CHECK_OK(GpuLaunchKernel(fns, rows, Bsize, smem, stream, x, gamma,      \
                                beta, rows, cols, y, mean, inv_variance, op)); \
    return Status::OK();                                                       \
  }
#define TRY_PACK_SIZE(PSIZE)                       \
  TRY_AND_LAUNCH(kBlockSizeSearchSpace[0], PSIZE); \
  TRY_AND_LAUNCH(kBlockSizeSearchSpace[1], PSIZE); \
  TRY_AND_LAUNCH(kBlockSizeSearchSpace[2], PSIZE);

  if (pack_size == 4) {
    TRY_PACK_SIZE(4);
  } else if (pack_size == 2) {
    TRY_PACK_SIZE(2);
  }
  TRY_PACK_SIZE(1);
#undef TRY_PACK_SIZE

#undef TRY_AND_LAUNCH

  TF_CHECK_OK(GpuLaunchKernel(fns_base, rows, kBlockSize, smem, stream, x,
                              gamma, beta, rows, cols, y, mean, inv_variance,
                              op));
  return Status::OK();
}

template <typename T, typename ComputeType, typename Op1, typename Op2>
Status TryDispatchLayerNormGradBlockSMemImpl(
    gpuStream_t stream, const T* x, const T* dy, const ComputeType* gamma,
    const int64_t rows, const int64_t cols, const ComputeType* mean,
    const ComputeType* inv_variance, T* dx, Op1 op1, Op2 op2) {
  // smem needs to store x and dy
  const size_t smem = cols * sizeof(ComputeType) * 2;

  auto fns =
      LayerNormGradBlockSMemImpl<T, ComputeType, Op1, Op2, kBlockSize, 1>;
  int pack_size = 1;
  if (cols % 4 == 0) {
    pack_size = 4;
    fns = LayerNormGradBlockSMemImpl<T, ComputeType, Op1, Op2, kBlockSize, 4>;
  } else if (cols % 2 == 0) {
    pack_size = 2;
    fns = LayerNormGradBlockSMemImpl<T, ComputeType, Op1, Op2, kBlockSize, 2>;
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
  fns = LayerNormGradBlockSMemImpl<T, ComputeType, Op1, Op2, Bsize, Psize>;   \
  CUDA_RETURN_IF_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(         \
      &blk_cnt, fns, Bsize, smem));                                           \
  if (blk_cnt == blk_cnt_base) {                                              \
    TF_CHECK_OK(GpuLaunchKernel(fns, rows, Bsize, smem, stream, x, dy, gamma, \
                                rows, cols, mean, inv_variance, dx, op1,      \
                                op2));                                        \
    return Status::OK();                                                      \
  }

#define TRY_PACK_SIZE(PSIZE)                       \
  TRY_AND_LAUNCH(kBlockSizeSearchSpace[0], PSIZE); \
  TRY_AND_LAUNCH(kBlockSizeSearchSpace[1], PSIZE); \
  TRY_AND_LAUNCH(kBlockSizeSearchSpace[2], PSIZE);

  if (pack_size == 4) {
    TRY_PACK_SIZE(4);
  } else if (pack_size == 2) {
    TRY_PACK_SIZE(2);
  }
  TRY_PACK_SIZE(1);
#undef TRY_PACK_SIZE
#undef TRY_AND_LAUNCH
  TF_CHECK_OK(GpuLaunchKernel(fns_base, rows, kBlockSize, smem, stream, x, dy,
                              gamma, rows, cols, mean, inv_variance, dx, op1,
                              op2));
  return Status::OK();
}

template <typename T, typename ComputeType, typename Op>
__global__ void LayerNormWarpImplWelford(const T* x, const ComputeType* gamma,
                                         const ComputeType* beta, int rows,
                                         int cols, T* y,
                                         ComputeType* __restrict__ out1,
                                         ComputeType* __restrict__ out2, Op op,
                                         bool is_padding) {
  // Each thread works on kMaxWorkPerThread number of cols.
  constexpr int pack_size = PackSizeTraits<T>::PackSize;
  constexpr int cols_per_thread = kMaxWorkPerThread;
  constexpr int num_packs = cols_per_thread / pack_size;

  const int lane_id = threadIdx.x % kWarpSize;

  constexpr int num_warps = kBlockSize / kWarpSize;
  const int local_warp_id = threadIdx.x / kWarpSize;
  const int warp_id = blockIdx.x * num_warps + local_warp_id;

  constexpr int thread_group_width = kWarpSize;
  ComputeType row_buf[cols_per_thread];

  for (int k = warp_id; k < rows; k += gridDim.x * num_warps) {
    WFGeneric<ComputeType> wf_thread;

    for (int pack_id = 0; pack_id < num_packs; ++pack_id) {
      const int col = (pack_id * thread_group_width + lane_id) * pack_size;
      const int pack_offset = pack_id * pack_size;
      const int data_offset = k * cols + col;
      if (!is_padding || col < cols) {
        CopyWithCast<T, ComputeType, pack_size>(x, data_offset,
                                                row_buf + pack_offset);

        for (int i = 0; i < pack_size; ++i) {
          op.Update(row_buf[pack_offset + i], wf_thread);
        }
      } else {
        for (int i = 0; i < pack_size; ++i) {
          row_buf[pack_offset + i] = 0;
        }
      }
    }
    WFGeneric<ComputeType> wf_row =
        WelfordWarpReduce<ComputeType>(wf_thread, true);
    ComputeType row_mean = wf_row.mean;
    ComputeType row_inv_var = op.Finalize(wf_row);
    if (lane_id == 0) {
      out1[k] = row_mean;
      out2[k] = row_inv_var;
    }

    for (int i = 0; i < cols_per_thread; ++i) {
      row_buf[i] = (row_buf[i] - row_mean) * row_inv_var;
    }

    for (int i = 0; i < num_packs; ++i) {
      const int col = (i * thread_group_width + lane_id) * pack_size;
      if (!is_padding || col < cols) {
        CopyWithAffineAndCast<ComputeType, T, pack_size>(
            row_buf + i * pack_size, gamma, col, beta, col, y, k * cols + col);
      }
    }
  }
}

template <typename T, typename U, typename Op>
__global__ __launch_bounds__(1024) void LayerNormRowReduceInToOutWarpWelford(
    const T* __restrict__ in, const U* __restrict__ gamma,
    const U* __restrict__ beta, int N, int D, U* __restrict__ out1,
    U* __restrict__ out2, T* __restrict__ y, Op op) {
  const int tid = threadIdx.x % kWarpSize;

  const int num_warps = kBlockSize / kWarpSize;
  typedef gpuprim::WarpReduce<U> WarpReduce;

  const int local_warp_id = threadIdx.x / kWarpSize;
  const int warp_id = blockIdx.x * num_warps + local_warp_id;

  for (int k = warp_id; k < N; k += gridDim.x * num_warps) {
    WFGeneric<U> wf_thread;
    for (int i = tid; i < D; i += kWarpSize) {
      op.Update(in, k, i, wf_thread);
    }

    WFGeneric<U> wf_row = WelfordWarpReduce<U>(wf_thread, true);
    U row_mean = wf_row.mean;
    U row_inv_var = op.Finalize(wf_row);

    if (tid == 0) {
      out1[k] = row_mean;
      out2[k] = row_inv_var;
    }

    U curr;
    for (int i = tid; i < D; i += kWarpSize) {
      int idx = k * D + i;
      curr = (static_cast<U>(in[idx]) - row_mean) * row_inv_var * gamma[i] +
             beta[i];
      y[idx] = static_cast<T>(curr);
    }
  }
}

template <typename T, typename U, typename Op1, typename Op2>
__global__ __launch_bounds__(1024) void LayerNormRowReduceInToOut(
    const T* __restrict__ in, int N, int D, U* out1, U* out2, Op1 op1,
    Op2 op2) {
  const int tid = threadIdx.x;

  typedef gpuprim::BlockReduce<U, kBlockSize> BlockReduce;
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
      partial_sum += op2.Compute(in, k, i);
    }

    sum = BlockReduce(temp_storage.reduce).Sum(partial_sum);

    if (tid == 0) {
      out2[k] = op2.Finalize(sum);
    }
  }
}

template <typename T, typename ComputeType, int PackSize>
__global__ __launch_bounds__(1024) void LayerNormRowReduceInToTemp(
    const T* x, const T* dy, const ComputeType* gamma, int N, int D,
    ComputeType* __restrict__ temp_1, ComputeType* __restrict__ temp_2,
    DvarOp<T, ComputeType> op1, DmeanOp<T, ComputeType> op2) {
  const int row_offset = threadIdx.x + blockIdx.x * blockDim.x;
  const int num_packs = static_cast<int>(D) / PackSize;
  for (int row_idx = blockIdx.y; row_idx < N; row_idx += gridDim.y) {
    ComputeType partial_sum1 = 0;
    ComputeType partial_sum2 = 0;
    for (int pack_id = row_offset; pack_id < num_packs;
         pack_id += gridDim.x * blockDim.x) {
      ComputeType pack_x[PackSize];
      ComputeType pack_dy[PackSize];
      const int data_offset = row_idx * D + pack_id * PackSize;
      CopyWithCast<T, ComputeType, PackSize>(x, data_offset, pack_x);
      CopyWithCast<T, ComputeType, PackSize>(dy, data_offset, pack_dy);
      CopyWithDot<ComputeType, PackSize>(gamma, pack_id * PackSize, pack_dy);

      for (int i = 0; i < PackSize; ++i) {
        ComputeType x = pack_x[i];
        ComputeType dy_dot_gamma = pack_dy[i];
        partial_sum1 += op1.Compute(x, dy_dot_gamma, row_idx);
        partial_sum2 += op2.Compute(dy_dot_gamma, row_idx);
      }
    }

    ComputeType sum1 =
        BlockAllReduce<ComputeType, gpuprim::Sum>(partial_sum1, gpuprim::Sum());
    __syncthreads();
    ComputeType sum2 =
        BlockAllReduce<ComputeType, gpuprim::Sum>(partial_sum2, gpuprim::Sum());

    if (threadIdx.x == 0) {
      temp_1[row_idx * gridDim.x + blockIdx.x] = sum1;
      temp_2[row_idx * gridDim.x + blockIdx.x] = sum2;
    }
  }
}

template <typename T, typename ComputeType>
void (*LayerNormRowReduceInToTempCandidates[3])(
    const T*, const T*, const ComputeType*, int N, int D,
    ComputeType* __restrict__, ComputeType* __restrict__,
    DvarOp<T, ComputeType>,
    DmeanOp<T, ComputeType>){LayerNormRowReduceInToTemp<T, ComputeType, 1>,
                             LayerNormRowReduceInToTemp<T, ComputeType, 2>,
                             LayerNormRowReduceInToTemp<T, ComputeType, 4>};

template <typename T, typename ComputeType, int PackSize>
__global__ __launch_bounds__(1024) void LayerNormRowReduceInToTempWelford(
    const T* x, int N, int D, ComputeType* __restrict__ temp_mean,
    ComputeType* __restrict__ temp_m2, ComputeType* __restrict__ temp_count,
    WFOp<T, ComputeType> op) {
  const int row_offset = threadIdx.x + blockIdx.x * blockDim.x;
  const int num_packs = static_cast<int>(D) / PackSize;
  assert(D % PackSize == 0);
  for (int row_idx = blockIdx.y; row_idx < N; row_idx += gridDim.y) {
    WFGeneric<ComputeType> wf_partial;
    for (int pack_id = row_offset; pack_id < num_packs;
         pack_id += gridDim.x * blockDim.x) {
      ComputeType pack[PackSize];

      CopyWithCast<T, ComputeType, PackSize>(
          x, row_idx * D + pack_id * PackSize, pack);
      for (int i = 0; i < PackSize; ++i) {
        op.Update(pack[i], wf_partial);
      }
    }
    WFGeneric<ComputeType> wf_block =
        BlockAllReduce<WFGeneric<ComputeType>, WFGeneric<ComputeType>>(
            wf_partial, WFGeneric<ComputeType>());
    if (threadIdx.x == 0) {
      temp_mean[row_idx * gridDim.x + blockIdx.x] = wf_block.mean;
      temp_m2[row_idx * gridDim.x + blockIdx.x] = wf_block.m2;
      temp_count[row_idx * gridDim.x + blockIdx.x] = wf_block.n;
    }
  }
}

template <typename T, typename ComputeType>
void (*LayerNormRowReduceInToTempWelfordCandidates[3])(
    const T*, int, int, ComputeType* __restrict__, ComputeType* __restrict__,
    ComputeType* __restrict__,
    WFOp<T, ComputeType>){LayerNormRowReduceInToTempWelford<T, ComputeType, 1>,
                          LayerNormRowReduceInToTempWelford<T, ComputeType, 2>,
                          LayerNormRowReduceInToTempWelford<T, ComputeType, 4>};

template <typename U, typename Op>
__global__ __launch_bounds__(1024) void LayerNormRowReduceTempToOutWelford(
    const U* __restrict__ temp_mean, const U* __restrict__ temp_m2,
    const U* __restrict__ temp_count, int N, int cols,
    U* __restrict__ cache_mean, U* __restrict__ cache_ivar, Op op) {
  for (int k = blockIdx.x; k < N; k += gridDim.x) {
    WFGeneric<U> wf_partial;

    for (int i = threadIdx.x; i < cols; i += kBlockSize) {
      int idx = k * cols + i;
      WFGeneric<U> wf_local{temp_mean[idx], temp_m2[idx], temp_count[idx]};
      wf_partial = WFGeneric<U>()(wf_local, wf_partial);
    }
    WFGeneric<U> wf_block =
        BlockAllReduce<WFGeneric<U>, WFGeneric<U>>(wf_partial, WFGeneric<U>());

    if (threadIdx.x == 0) {
      cache_mean[k] = wf_block.mean;
      cache_ivar[k] = op.Finalize(wf_block);
    }
  }
}

template <typename T>
__global__ __launch_bounds__(1024) void LayerNormRowReduceTempToOut(
    int N, int cols, T* __restrict__ temp_1, T* __restrict__ temp_2) {
  // Inplace reduction
  typedef gpuprim::BlockReduce<T, kBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  for (int k = blockIdx.x; k < N; k += gridDim.x) {
    T partial_sum1 = 0;
    T partial_sum2 = 0;
    for (int i = threadIdx.x; i < cols; i += kBlockSize) {
      partial_sum1 += temp_1[k * cols + i];
      partial_sum2 += temp_2[k * cols + i];
    }

    T sum_1 = BlockReduce(temp_storage).Sum(partial_sum1);
    __syncthreads();
    T sum_2 = BlockReduce(temp_storage).Sum(partial_sum2);

    if (threadIdx.x == 0) {
      temp_1[k * cols] = sum_1;
      temp_2[k * cols] = sum_2;
    }
  }
}

template <typename T, typename ComputeType, int PackSize>
__global__ __launch_bounds__(1024) void LayerNormForwardUpdate(
    const T* x, const ComputeType* gamma, const ComputeType* beta, int N, int D,
    T* y, YOp<T, ComputeType> op) {
  const int row_offset = threadIdx.x + blockIdx.x * blockDim.x;
  const int num_packs = static_cast<int>(D) / PackSize;
  for (int row_idx = blockIdx.y; row_idx < N; row_idx += gridDim.y) {
    for (int pack_id = row_offset; pack_id < num_packs;
         pack_id += gridDim.x * blockDim.x) {
      ComputeType pack[PackSize];
      const int pack_offset = pack_id * PackSize;

      CopyWithCast<T, ComputeType, PackSize>(x, row_idx * D + pack_offset,
                                             pack);

      for (int i = 0; i < PackSize; ++i) {
        pack[i] = op.ComputePartial(pack[i], row_idx);
      }

      CopyWithAffineAndCast<ComputeType, T, PackSize>(
          pack, gamma, pack_offset, beta, pack_offset, y,
          row_idx * D + pack_offset);
    }
  }
}

template <typename T, typename ComputeType>
void (*LayerNormForwardUpdateCandidates[3])(const T*, const ComputeType*,
                                            const ComputeType*, int, int, T*,
                                            YOp<T, ComputeType>){
    LayerNormForwardUpdate<T, ComputeType, 1>,
    LayerNormForwardUpdate<T, ComputeType, 2>,
    LayerNormForwardUpdate<T, ComputeType, 4>};

template <typename T, typename ComputeType, int PackSize>
__global__ __launch_bounds__(1024) void LayerNormBackwardUpdate(
    const T* x, const T* dy, const ComputeType* gamma, int N, int D,
    int dl_stride, T* dx, DxOp<T, ComputeType> op) {
  const int row_offset = threadIdx.x + blockIdx.x * blockDim.x;
  const int num_packs = static_cast<int>(D) / PackSize;
  for (int row_idx = blockIdx.y; row_idx < N; row_idx += gridDim.y) {
    for (int pack_id = row_offset; pack_id < num_packs;
         pack_id += gridDim.x * blockDim.x) {
      ComputeType pack_x[PackSize];
      ComputeType pack_dy[PackSize];
      const int pack_offset = pack_id * PackSize;
      const int data_offset = pack_offset + row_idx * D;
      CopyWithCast<T, ComputeType, PackSize>(x, data_offset, pack_x);
      CopyWithCast<T, ComputeType, PackSize>(dy, data_offset, pack_dy);

      for (int i = 0; i < PackSize; ++i) {
        pack_dy[i] = op.ComputePartial0(pack_dy[i], row_idx);
        pack_x[i] = op.ComputePartial1(pack_x[i], row_idx, dl_stride);
      }

      CopyWithAffineAndCast<ComputeType, T, PackSize>(
          pack_dy, gamma, pack_offset, pack_x, 0, dx, data_offset);
    }
  }
}

template <typename T, typename ComputeType>
void (*LayerNormBackwardUpdateCandidates[3])(const T*, const T*,
                                             const ComputeType*, int, int, int,
                                             T*, DxOp<T, ComputeType>){
    LayerNormBackwardUpdate<T, ComputeType, 1>,
    LayerNormBackwardUpdate<T, ComputeType, 2>,
    LayerNormBackwardUpdate<T, ComputeType, 4>};

template <typename T, typename Op>
__global__ __launch_bounds__(1024) void LayerNormUpdate(
    const T* __restrict__ in, int N, int D, T* out, Op op) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= N * D) return;

  const int col = tid % D;
  const int row = tid / D;
  out[tid] = op.Compute(in, row, col);
}

template <typename T, typename U>
__global__ __launch_bounds__(1024) void LayerNormGradBetaGamma(
    const T* __restrict__ dy, const T* __restrict__ x,
    const U* __restrict__ cache_mean, const U* __restrict__ cache_ivar, int N,
    int D, U* __restrict__ dgamma, U* __restrict__ dbeta) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= D) return;

  U sum_dgamma = 0;
  U sum_dbeta = 0;
  for (int i = 0; i < N; i++) {
    U dy_curr = GetAs<T, U>(dy, i * D + tid);
    sum_dgamma += dy_curr * (x[i * D + tid] - cache_mean[i]) * cache_ivar[i];
    sum_dbeta += dy_curr;
  }

  dgamma[tid] = sum_dgamma;
  dbeta[tid] = sum_dbeta;
}

template <typename T, typename U>
__global__ __launch_bounds__(1024) void LayerNormGradBetaGammaInToTemp(
    const T* __restrict__ dy, const T* __restrict__ x,
    const U* __restrict__ cache_mean, const U* __restrict__ cache_ivar, int N,
    int D, int rows, U* __restrict__ tgamma, U* __restrict__ tbeta) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= D) return;

  int j = tid;
  U sum_dgamma = 0;
  U sum_dbeta = 0;
  for (int i = blockIdx.y * rows; i < min(blockIdx.y * rows + rows, N); i++) {
    U dy_curr = GetAs<T, U>(dy, i * D + j);
    sum_dgamma += dy_curr * (x[i * D + j] - cache_mean[i]) * cache_ivar[i];
    sum_dbeta += dy_curr;
  }
  tgamma[blockIdx.y * D + j] = sum_dgamma;
  tbeta[blockIdx.y * D + j] = sum_dbeta;
}

template <typename U>
__global__ __launch_bounds__(1024) void LayerNormGradBetaGammaTempToOut(
    const U* __restrict__ tg, const U* __restrict__ tb, int N, int D,
    U* __restrict__ dgamma, U* __restrict__ dbeta) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= D) return;

  U sum_dgamma = 0;
  U sum_dbeta = 0;
  for (int i = 0; i < N; i++) {
    U tg_curr = tg[i * D + tid];
    U tb_curr = tb[i * D + tid];
    sum_dgamma += tg_curr;
    sum_dbeta += tb_curr;
  }

  dgamma[tid] = sum_dgamma;
  dbeta[tid] = sum_dbeta;
}

template <typename T, typename U>
struct FusedLayerNorm<GPUDevice, T, U> {
  void operator()(OpKernelContext* context, const Tensor& x_input,
                  const Tensor& scale_input, const Tensor& offset_input,
                  U epsilon, int batch_dim, int feature_dim, Tensor* y_output,
                  Tensor* saved_mean_output, Tensor* saved_inv_var_output) {
    if (x_input.shape().num_elements() == 0) {
      return;
    }
    const auto N = x_input.dim_size(batch_dim);
    const auto D = x_input.dim_size(feature_dim);

    const T* x = x_input.flat<T>().data();
    const U* gamma = scale_input.flat<U>().data();
    const U* beta = offset_input.flat<U>().data();
    U* cache_mean = saved_mean_output->flat<U>().data();
    U* cache_ivar = saved_inv_var_output->flat<U>().data();
    T* y = y_output->flat<T>().data();

    bool use_single_warp = (D <= kMaxWorkPerThread * kWarpSize);

    const int min_workload_per_thread = 100;

    WFOp<T, U> wf_ops{D, epsilon};

    const GPUDevice& d = context->eigen_device<GPUDevice>();
    if (use_single_warp) {
      if (D < kWarpSize) {
        TF_CHECK_OK(GpuLaunchKernel(
            LayerNormRowReduceInToOutWarpWelford<T, U, WFOp<T, U>>,
            Eigen::divup(N, kBlockSize / kWarpSize), kBlockSize, 0, d.stream(),
            x, gamma, beta, N, D, cache_mean, cache_ivar, y, wf_ops));
      } else {
        const bool is_padding = (D != kMaxWorkPerThread * kWarpSize);
        TF_CHECK_OK(
            GpuLaunchKernel(LayerNormWarpImplWelford<T, U, decltype(wf_ops)>,
                            Eigen::divup(N, kBlockSize / kWarpSize), kBlockSize,
                            0, d.stream(), x, gamma, beta, N, D, y, cache_mean,
                            cache_ivar, wf_ops, is_padding));
      }
    } else {
      Status status = TryDispatchLayerNormBlockSMemImpl<T, U, WFOp<T, U>>(
          d.stream(), x, gamma, beta, N, D, y, cache_mean, cache_ivar, wf_ops);

      if (status.ok()) {
        return;
      }
      const int blocks_per_row =
          Eigen::divup(D, kBlockSize * min_workload_per_thread);

      Tensor scratch1, scratch2, scratch3;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<U>::value,
                                            {N * blocks_per_row}, &scratch1));
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<U>::value,
                                            {N * blocks_per_row}, &scratch2));
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<U>::value,
                                            {N * blocks_per_row}, &scratch3));
      U* temp_mean = scratch1.flat<U>().data();
      U* temp_m2 = scratch2.flat<U>().data();
      U* temp_count = scratch3.flat<U>().data();

      dim3 threads(kBlockSize);
      dim3 blocks(blocks_per_row, N);

      // For long rows, we launch n blocks to process each row. The
      // intermediate results are stored in a temp memory with the size of
      // N*n. Then, we launch single block to handle each row of the temp
      // memory.

      LaunchVectorizedKernel<T>(
          LayerNormRowReduceInToTempWelfordCandidates<T, U>, blocks, threads, 0,
          d.stream(), D, x, N, D, temp_mean, temp_m2, temp_count, wf_ops);

      TF_CHECK_OK(GpuLaunchKernel(
          LayerNormRowReduceTempToOutWelford<U, WFOp<T, U>>, N, threads, 0,
          d.stream(), temp_mean, temp_m2, temp_count, N, blocks_per_row,
          cache_mean, cache_ivar, wf_ops));

      YOp<T, U> y_op{cache_mean, cache_ivar, gamma, beta, D};

      LaunchVectorizedKernel<T>(LayerNormForwardUpdateCandidates<T, U>, blocks,
                                threads, 0, d.stream(), D, x, gamma, beta, N, D,
                                y, y_op);
    }
  }
};

template <typename T, typename U>
struct FusedLayerNormGrad<GPUDevice, T, U> {
  void operator()(OpKernelContext* context, const Tensor& y_backprop_input,
                  const Tensor& x_input, const Tensor& scale_input,
                  const Tensor& saved_mean_input,
                  const Tensor& saved_inv_var_input, U epsilon, int batch_dim,
                  int feature_dim, Tensor* x_backprop_output,
                  Tensor* scale_backprop_output,
                  Tensor* offset_backprop_output) {
    if (x_input.shape().num_elements() == 0) {
      return;
    }
    const auto N = x_input.dim_size(batch_dim);
    const auto D = x_input.dim_size(feature_dim);

    const T* dy = y_backprop_input.flat<T>().data();
    const T* x = x_input.flat<T>().data();
    const U* gamma = scale_input.flat<U>().data();
    const U* cache_mean = saved_mean_input.flat<U>().data();
    const U* cache_ivar = saved_inv_var_input.flat<U>().data();
    T* dx = x_backprop_output->flat<T>().data();
    U* dgamma = scale_backprop_output->flat<U>().data();
    U* dbeta = offset_backprop_output->flat<U>().data();

    const int64_t min_rows_per_block = 10000;
    bool use_temp_space = (N > min_rows_per_block);
    const GPUDevice& d = context->eigen_device<GPUDevice>();

    if (!use_temp_space) {
      TF_CHECK_OK(GpuLaunchKernel(
          LayerNormGradBetaGamma<T, U>, Eigen::divup(D, kBlockSize), kBlockSize,
          0, d.stream(), dy, x, cache_mean, cache_ivar, N, D, dgamma, dbeta));
    } else {
      const int reduced_rows = Eigen::divup(N, min_rows_per_block);

      Tensor scratch1, scratch2;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<U>::value,
                                            {reduced_rows * D}, &scratch1));
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<U>::value,
                                            {reduced_rows * D}, &scratch2));
      U* temp_dgamma = scratch1.flat<U>().data();
      U* temp_dbeta = scratch2.flat<U>().data();

      dim3 blocks(Eigen::divup(D, kBlockSize), reduced_rows);
      TF_CHECK_OK(GpuLaunchKernel(LayerNormGradBetaGammaInToTemp<T, U>, blocks,
                                  kBlockSize, 0, d.stream(), dy, x, cache_mean,
                                  cache_ivar, N, D, min_rows_per_block,
                                  temp_dgamma, temp_dbeta));
      TF_CHECK_OK(GpuLaunchKernel(
          LayerNormGradBetaGammaTempToOut<U>, Eigen::divup(D, kBlockSize),
          kBlockSize, 0, d.stream(), temp_dgamma, temp_dbeta, reduced_rows,
          static_cast<int>(D), dgamma, dbeta));
    }
    bool use_single_warp = (D <= kMaxWorkPerThread * kWarpSize);
    const int min_workload_per_thread = 50;

    DvarOp<T, U> dl_dvar_ops{gamma, x, cache_ivar, cache_mean, D};
    DmeanOp<T, U> dl_dmu_ops{gamma, x, cache_ivar, cache_mean, D};

    if (use_single_warp) {
      if (D < kWarpSize) {
        Tensor scratch_dl_dvars, scratch_dl_dmus;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                       {N}, &scratch_dl_dvars));
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                       {N}, &scratch_dl_dmus));
        U* temp_1 = scratch_dl_dvars.flat<U>().data();
        U* temp_2 = scratch_dl_dmus.flat<U>().data();

        TF_CHECK_OK(GpuLaunchKernel(
            LayerNormRowReduceInToOutWarp<T, U, DvarOp<T, U>, DmeanOp<T, U>>,
            Eigen::divup(N, kBlockSize / kWarpSize), kBlockSize, 0, d.stream(),
            dy, N, D, temp_1, temp_2, dl_dvar_ops, dl_dmu_ops));

        DxOp<T, U> dx_op{x, cache_mean, cache_ivar, gamma, temp_1, temp_2, D};
        TF_CHECK_OK(GpuLaunchKernel(LayerNormUpdate<T, DxOp<T, U>>,
                                    Eigen::divup(N * D, kBlockSize), kBlockSize,
                                    0, d.stream(), dy, N, D, dx, dx_op));
      } else {
        TF_CHECK_OK(GpuLaunchKernel(
            LayerNormGradWarpImpl<T, U, DvarOp<T, U>, DmeanOp<T, U>,
                                  PackSizeTraits<T>::PackSize,
                                  kMaxWorkPerThread>,
            Eigen::divup(N, kBlockSize / kWarpSize), kBlockSize, 0, d.stream(),
            x, dy, gamma, N, D, cache_mean, cache_ivar, dl_dvar_ops, dl_dmu_ops,
            dx, D != kMaxWorkPerThread * kWarpSize));
      }
    } else {
      Status status = TryDispatchLayerNormGradBlockSMemImpl<T, U, DvarOp<T, U>,
                                                            DmeanOp<T, U>>(
          d.stream(), x, dy, gamma, N, D, cache_mean, cache_ivar, dx,
          dl_dvar_ops, dl_dmu_ops);

      if (status.ok()) {
        return;
      }
      const int blocks_per_row =
          Eigen::divup(D, kBlockSize * min_workload_per_thread);

      Tensor scratch_temp_dl_dvars, scratch_temp_dl_dmus;
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                     {N * blocks_per_row},
                                                     &scratch_temp_dl_dvars));
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                     {N * blocks_per_row},
                                                     &scratch_temp_dl_dmus));
      U* temp_dl_dvars = scratch_temp_dl_dvars.flat<U>().data();
      U* temp_dl_dmus = scratch_temp_dl_dmus.flat<U>().data();

      dim3 threads(kBlockSize, 1, 1);
      dim3 blocks(blocks_per_row, N, 1);

      LaunchVectorizedKernel<T>(LayerNormRowReduceInToTempCandidates<T, U>,
                                blocks, threads, 0, d.stream(), D, x, dy, gamma,
                                N, D, temp_dl_dvars, temp_dl_dmus, dl_dvar_ops,
                                dl_dmu_ops);
      if (blocks_per_row > 1) {
        TF_CHECK_OK(GpuLaunchKernel(LayerNormRowReduceTempToOut<U>, N, threads,
                                    0, d.stream(), N, blocks_per_row,
                                    temp_dl_dvars, temp_dl_dmus));
      }
      DxOp<T, U> dx_op{
          x, cache_mean, cache_ivar, gamma, temp_dl_dvars, temp_dl_dmus, D};

      LaunchVectorizedKernel<T>(LayerNormBackwardUpdateCandidates<T, U>, blocks,
                                threads, 0, d.stream(), D, x, dy, gamma, N, D,
                                blocks_per_row, dx, dx_op);
    }
  }
};

template struct FusedLayerNorm<GPUDevice, float, float>;
template struct FusedLayerNorm<GPUDevice, Eigen::half, float>;
template struct FusedLayerNormGrad<GPUDevice, float, float>;
template struct FusedLayerNormGrad<GPUDevice, Eigen::half, float>;

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
