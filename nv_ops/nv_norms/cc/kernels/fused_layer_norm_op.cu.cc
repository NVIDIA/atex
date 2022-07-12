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

template <typename T, typename U, typename Op1, typename Op2>
__global__ __launch_bounds__(1024) void LayerNormRowReduceInToOutWarp(
    const T* __restrict__ in, const int N, const int D, U* __restrict__ out1,
    U* __restrict__ out2, Op1 op1, Op2 op2) {
  const int tid = threadIdx.x % kWarpSize;

  const int num_warps = kBlockSize / kWarpSize;
  typedef gpuprim::WarpReduce<U> WarpReduce;
  typename WarpReduce::TempStorage temp_storage[num_warps];

  const int local_warp_id = threadIdx.x / kWarpSize;
  const int warp_id = blockIdx.x * num_warps + local_warp_id;

  for (int k = warp_id; k < N; k += gridDim.x * num_warps) {
    U partial_sum = 0;
    for (int i = tid; i < D; i += kWarpSize) {
      partial_sum += op1.Compute(in, k, i);
    }

    U sum = WarpReduce(temp_storage[local_warp_id]).Sum(partial_sum);

    sum = gpuprim::ShuffleIndex<kWarpSize>(sum, 0, 0xffffffff);
    sum = op1.Finalize(sum);
    if (tid == 0) {
      out1[k] = sum;
    }

    partial_sum = 0;
    for (int i = tid; i < D; i += kWarpSize) {
      partial_sum += op2.Compute(in, k, i);
    }

    sum = WarpReduce(temp_storage[local_warp_id]).Sum(partial_sum);

    if (tid == 0) {
      out2[k] = op2.Finalize(sum);
    }
  }
}

template <typename T, typename U, typename Op>
__global__ __launch_bounds__(1024) void LayerNormRowReduceInToOutWarpWelford(
    const T* __restrict__ in, const int N, const int D, U* __restrict__ out1,
    U* __restrict__ out2, Op op) {
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
    WFGeneric<U> wf_row = WelfordWarpReduce<U>(wf_thread);

    if (tid == 0) {
      out1[k] = wf_row.mean;
      out2[k] = op.Finalize(wf_row);
    }
  }
}

template <typename T, typename U, typename Op1, typename Op2>
__global__ __launch_bounds__(1024) void LayerNormRowReduceInToOut(
    const T* __restrict__ in, const int N, const int D, U* out1, U* out2,
    Op1 op1, Op2 op2) {
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

template <typename T, typename U, typename Op>
__global__ __launch_bounds__(1024) void LayerNormRowReduceInToTemp(
    const T* __restrict__ x, const int N, const int D, U* __restrict__ temp,
    Op op) {
  typedef gpuprim::BlockReduce<U, kBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const int row_offset = threadIdx.x + blockIdx.x * blockDim.x;

  for (int row_idx = blockIdx.y; row_idx < N; row_idx += gridDim.y) {
    U partial_sum = 0;
    for (int i = row_offset; i < D; i += gridDim.x * blockDim.x) {
      partial_sum += op.Compute(x, row_idx, i);
    }
    U sum = BlockReduce(temp_storage).Sum(partial_sum);
    if (threadIdx.x == 0) {
      temp[row_idx * gridDim.x + blockIdx.x] = sum;
    }
  }
}

template <typename T, typename U, typename Op>
__global__ __launch_bounds__(1024) void LayerNormRowReduceInToTempWelford(
    const T* __restrict__ x, const int N, const int D,
    U* __restrict__ temp_mean, U* __restrict__ temp_m2,
    U* __restrict__ temp_count, Op op) {
  const int row_offset = threadIdx.x + blockIdx.x * blockDim.x;

  for (int row_idx = blockIdx.y; row_idx < N; row_idx += gridDim.y) {
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
}

template <typename U, typename Op>
__global__ __launch_bounds__(1024) void LayerNormRowReduceTempToOutWelford(
    const U* __restrict__ temp_mean, const U* __restrict__ temp_m2,
    const U* __restrict__ temp_count, const int N, const int cols,
    U* __restrict__ cache_mean, U* __restrict__ cache_ivar, Op op) {
  for (int k = blockIdx.x; k < N; k += gridDim.x) {
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
__global__ __launch_bounds__(1024) void LayerNormRowReduceTempToOut(
    const U* __restrict__ temp, const int N, const int cols,
    U* __restrict__ cache, Op op) {
  typedef gpuprim::BlockReduce<U, kBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  for (int k = blockIdx.x; k < N; k += gridDim.x) {
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

template <typename T, typename Op>
__global__ __launch_bounds__(1024) void LayerNormUpdate(
    const T* __restrict__ in, const int N, const int D, T* out, Op op) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= N * D) return;

  const int col = tid % D;
  const int row = tid / D;
  out[tid] = op.Compute(in, row, col);
}

template <typename T, typename U>
__global__ __launch_bounds__(1024) void LayerNormGradBetaGamma(
    const T* __restrict__ dy, const T* __restrict__ x,
    const U* __restrict__ cache_mean, const U* __restrict__ cache_ivar,
    const int N, const int D, U* __restrict__ dgamma, U* __restrict__ dbeta) {
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
    const U* __restrict__ cache_mean, const U* __restrict__ cache_ivar,
    const int N, const int D, const int rows, U* __restrict__ tgamma,
    U* __restrict__ tbeta) {
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
    const U* __restrict__ tg, const U* __restrict__ tb, const int N,
    const int D, U* __restrict__ dgamma, U* __restrict__ dbeta) {
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

    bool use_single_warp = (D <= kMaxWorkPerWarp * kWarpSize);

    const int min_num_blocks = kWarpSize;
    const int min_workload_per_thread = 100;
    bool use_single_block =
        (D <= min_num_blocks * kBlockSize * min_workload_per_thread);

    WFOp<T, U> wf_ops{D, epsilon};
    const GPUDevice& d = context->eigen_device<GPUDevice>();

    if (use_single_warp) {
      TF_CHECK_OK(GpuLaunchKernel(
          LayerNormRowReduceInToOutWarpWelford<T, U, WFOp<T, U>>,
          Eigen::divup(N, kBlockSize / kWarpSize), kBlockSize, 0, d.stream(), x,
          N, D, cache_mean, cache_ivar, wf_ops));
    } else if (use_single_block) {
      TF_CHECK_OK(GpuLaunchKernel(
          GeneralNormRowReduceInToOutWelford<T, U, WFOp<T, U>>, N, kBlockSize,
          0, d.stream(), x, N, D, cache_mean, cache_ivar, wf_ops));
    } else {
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

      dim3 threads(kBlockSize, 1, 1);
      dim3 blocks(blocks_per_row, N, 1);

      // For long rows, we launch n blocks to process each row. The intermediate
      // results are stored in a temp memory with the size of N*n. Then, we
      // launch single block to handle each row of the temp memory.
      TF_CHECK_OK(GpuLaunchKernel(
          LayerNormRowReduceInToTempWelford<T, U, WFOp<T, U>>, blocks, threads,
          0, d.stream(), x, N, D, temp_mean, temp_m2, temp_count, wf_ops));
      TF_CHECK_OK(GpuLaunchKernel(
          LayerNormRowReduceTempToOutWelford<U, WFOp<T, U>>, N, threads, 0,
          d.stream(), temp_mean, temp_m2, temp_count, N, blocks_per_row,
          cache_mean, cache_ivar, wf_ops));
    }

    YOp<T, U> y_ops{cache_mean, cache_ivar, gamma, beta, D};

    TF_CHECK_OK(GpuLaunchKernel(LayerNormUpdate<T, YOp<T, U>>,
                                Eigen::divup(N * D, kBlockSize), kBlockSize, 0,
                                d.stream(), x, N, D, y, y_ops));
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

    Tensor scratch_dl_dvars, scratch_dl_dmus;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                   {N}, &scratch_dl_dvars));
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                   {N}, &scratch_dl_dmus));
    U* temp_1 = scratch_dl_dvars.flat<U>().data();
    U* temp_2 = scratch_dl_dmus.flat<U>().data();

    bool use_single_warp = (D <= kMaxWorkPerWarp * kWarpSize);

    const int min_num_blocks = kWarpSize;
    const int min_workload_per_thread = 50;
    bool use_single_block =
        (D <= min_num_blocks * kBlockSize * min_workload_per_thread);

    DvarOp<T, U> dl_dvar_ops{gamma, x, cache_ivar, cache_mean, D};
    DmeanOp<T, U> dl_dmu_ops{gamma, x, cache_ivar, cache_mean, D};

    if (use_single_warp) {
      TF_CHECK_OK(GpuLaunchKernel(
          LayerNormRowReduceInToOutWarp<T, U, DvarOp<T, U>, DmeanOp<T, U>>,
          Eigen::divup(N, kBlockSize / kWarpSize), kBlockSize, 0, d.stream(),
          dy, N, D, temp_1, temp_2, dl_dvar_ops, dl_dmu_ops));
    } else if (use_single_block) {
      TF_CHECK_OK(GpuLaunchKernel(
          LayerNormRowReduceInToOut<T, U, DvarOp<T, U>, DmeanOp<T, U>>, N,
          kBlockSize, 0, d.stream(), dy, N, D, temp_1, temp_2, dl_dvar_ops,
          dl_dmu_ops));

    } else {
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

      TF_CHECK_OK(GpuLaunchKernel(
          LayerNormRowReduceInToTemp<T, U, DvarOp<T, U>>, blocks, threads, 0,
          d.stream(), dy, N, D, temp_dl_dvars, dl_dvar_ops));

      TF_CHECK_OK(GpuLaunchKernel(LayerNormRowReduceTempToOut<U, DvarOp<T, U>>,
                                  N, threads, 0, d.stream(), temp_dl_dvars, N,
                                  blocks_per_row, temp_1, dl_dvar_ops));

      TF_CHECK_OK(GpuLaunchKernel(
          LayerNormRowReduceInToTemp<T, U, DmeanOp<T, U>>, blocks, threads, 0,
          d.stream(), dy, N, D, temp_dl_dmus, dl_dmu_ops));
      TF_CHECK_OK(GpuLaunchKernel(LayerNormRowReduceTempToOut<U, DmeanOp<T, U>>,
                                  N, threads, 0, d.stream(), temp_dl_dmus, N,
                                  blocks_per_row, temp_2, dl_dmu_ops));
    }

    DxOp<T, U> dx_ops{x, cache_mean, cache_ivar, gamma, temp_1, temp_2, D};
    TF_CHECK_OK(GpuLaunchKernel(LayerNormUpdate<T, DxOp<T, U>>,
                                Eigen::divup(N * D, kBlockSize), kBlockSize, 0,
                                d.stream(), dy, N, D, dx, dx_ops));
  }
};

template struct FusedLayerNorm<GPUDevice, float, float>;
template struct FusedLayerNorm<GPUDevice, Eigen::half, float>;
template struct FusedLayerNormGrad<GPUDevice, float, float>;
template struct FusedLayerNormGrad<GPUDevice, Eigen::half, float>;

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
