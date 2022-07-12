/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
==============================================================================*/

#include "absl/types/optional.h"
#include "tensorflow/core/kernels/gpu_prim.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {

static const int64_t kBlockSize = 128;
static const int64_t kWarpSize = 32;
static const int kMaxWorkPerWarp = 32;

template <typename T, typename U>
__device__ inline U GetAs(const T* __restrict__ in, int offset) {
  return static_cast<U>(in[offset]);
}

template <typename T>
struct WFGeneric {
  T mean = 0;
  T m2 = 0;
  T n = 0;
  __device__ WFGeneric(T _mean, T _m2, T _n) : mean(_mean), m2(_m2), n(_n) {}
  __device__ WFGeneric() : WFGeneric(0, 0, 0) {}
  __device__ WFGeneric(T val) : WFGeneric(val, 0, 1) {}
  inline __device__ WFGeneric<T> operator()(const WFGeneric<T>& lhs,
                                            const WFGeneric<T>& rhs) {
    // Use Welford Online algorithm to compute mean and variance
    // For more details you can refer to:
    // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    if (lhs.n == 0) {
      return rhs;
    }
    if (rhs.n == 0) {
      return lhs;
    }
    // By updating the mean and m2, this algorithm combines two sets(one of
    // which can be a scalar) of WFGeneric data into 1 set.
    T n = lhs.n + rhs.n;
    T rhs_n_over_new_n = rhs.n / n;
    T lhs_n_over_new_n = lhs.n / n;
    T mean = lhs.mean * lhs_n_over_new_n + rhs.mean * rhs_n_over_new_n;
    T delta = rhs.mean - lhs.mean;
    T m2 = lhs.m2 + rhs.m2 + delta * delta * lhs.n * rhs_n_over_new_n;
    return WFGeneric<T>{mean, m2, n};
  }
};

template <typename T, typename U>
struct WFOp {
  int64_t D;
  U epsilon;
  absl::optional<int64_t> C_;
  WFOp(int64_t d, U e) : D(d), epsilon(e) {}
  WFOp(int64_t d, U e, int64_t c) : D(d), epsilon(e), C_(c) {}
  inline void SetChannelDim(const int64_t c) { C_ = c; }
  inline __device__ int GetIndex(const int row, const int col) const {
    if (C_.has_value()) {
      // Channel last layout
      int64_t C = C_.value();
      return (row / C) * C * D + col * C + row % C;
    } else {
      return row * D + col;
    }
  }

  inline __device__ int GetIndex(const int row, const int col,
                                 const int z) const {
    // Channel last layout
    int64_t C = C_.value();
    return row * C * D + col * C + z;
  }

  inline __device__ void Update(const T* x, const int row, const int col,
                                WFGeneric<U>& wf_data) {
    // Channel first layout
    wf_data = WFGeneric<U>()(GetAs<T, U>(x, GetIndex(row, col)), wf_data);
  }

  inline __device__ void Update(const T* x, const int row, const int col,
                                const int z, WFGeneric<U>& wf_data) {
    // Channel last layout
    wf_data = WFGeneric<U>()(GetAs<T, U>(x, GetIndex(row, col, z)), wf_data);
  }

  inline __device__ U Finalize(const WFGeneric<U> wf_data) {
    // Finalize computing the shifted inverse of std from m2 and n.
    return static_cast<U>(
        rsqrt(static_cast<float>(wf_data.m2) / static_cast<float>(wf_data.n) +
              epsilon));
  }
};

template <typename T, typename U>
struct MeanOp {
  int64_t D;
  absl::optional<int64_t> C_;
  MeanOp(int64_t d) : D(d) {}
  inline void SetChannelDim(const int64_t c) { C_ = c; }
  inline __device__ int GetIndex(const int row, const int col) const {
    if (C_.has_value()) {
      // Channel last layout
      int64_t C = C_.value();
      return (row / C) * C * D + col * C + row % C;
    } else {
      return row * D + col;
    }
  }

  inline __device__ int GetIndex(const int row, const int col,
                                 const int z) const {
    // Channel last layout
    int64_t C = C_.value();
    return row * C * D + col * C + z;
  }

  __device__ U Compute(const T* x, const int row, const int col) const {
    return GetAs<T, U>(x, GetIndex(row, col));
  }
  __device__ U Compute(const T* x, const int row, const int col,
                       const int z) const {
    // Channel last layout
    return GetAs<T, U>(x, GetIndex(row, col, z));
  }
  __device__ U Finalize(const U& sum) const { return sum / D; }
};

template <typename T, typename U>
struct IvarOp {
  const U* cache_mean;
  U epsilon;
  int64_t D;
  absl::optional<int64_t> C_;
  IvarOp(const U* m_ptr, U e, int64_t d)
      : cache_mean(m_ptr), epsilon(e), D(d) {}
  inline void SetChannelDim(const int64_t c) { C_ = c; }

  inline __device__ int GetIndex(const int row, const int col) const {
    if (C_.has_value()) {
      // Channel last layout
      int64_t C = C_.value();
      return (row / C) * C * D + col * C + row % C;
    } else {
      return row * D + col;
    }
  }

  inline __device__ int GetIndex(const int row, const int col,
                                 const int z) const {
    // Channel last layout
    int64_t C = C_.value();
    return row * C * D + col * C + z;
  }

  __device__ U Compute(const T* x, const int row, const int col,
                       const U mean) const {
    U curr = GetAs<T, U>(x, GetIndex(row, col));
    return (curr - mean) * (curr - mean);
  }
  __device__ U Compute(const T* x, const int row, const int col) const {
    return Compute(x, row, col, cache_mean[row]);
  }
  __device__ U Compute(const T* x, const int row, const int col, const int z,
                       const U mean) const {
    U curr = GetAs<T, U>(x, GetIndex(row, col, z));
    return (curr - mean) * (curr - mean);
  }
  __device__ U Compute(const T* x, const int row, const int col,
                       const int z) const {
    // Channel last layout
    int64_t C = C_.value();
    U m = cache_mean[row * C + z];
    return Compute(x, row, col, z, m);
  }
  __device__ U Finalize(const U sum) const {
    return static_cast<U>(rsqrt(static_cast<float>(sum) / static_cast<float>(D) +
                    static_cast<float>(epsilon)));
  }
};

template <typename T, typename U>
struct DvarOp {
  const U* gamma;
  const T* x;
  const U* cache_ivar;
  const U* cache_mean;
  int64_t D;
  absl::optional<int64_t> C_;
  DvarOp(const U* g_ptr, const T* x_ptr, const U* civar, const U* cmean, int64_t d)
    : gamma(g_ptr), x(x_ptr), cache_ivar(civar), cache_mean(cmean), D(d) {}
  inline void SetChannelDim(const int64_t c) { C_ = c; }

  __device__ inline U Compute(const T* dy, const int row, const int col) const {
    U curr = GetAs<T, U>(dy, row * D + col);
    U g_val;
    if (C_.has_value()) {
      int64_t C = C_.value();
      g_val = gamma[row % C];
    } else {
      g_val = gamma[col];
    }
    return curr * g_val * (x[row * D + col] - cache_mean[row]) *
           (-0.5) * (cache_ivar[row] * cache_ivar[row] * cache_ivar[row]);
  }
  __device__ inline U Compute(const T* dy, const int row, const int col,
                              const int z) const {
    int64_t C = C_.value();
    U curr = GetAs<T, U>(dy, row * D * C + col * C + z);
    U ivar = cache_ivar[row * C + z];
    U me = cache_mean[row * C + z];
    return curr * gamma[z] * (x[row * D * C + col * C + z] - me) * (-0.5) *
           (ivar * ivar * ivar);
  }
  __device__ U Finalize(const U sum) const { return sum; }
};

template <typename T, typename U>
struct DmeanOp {
  const U* gamma;
  const T* x;
  const U* cache_ivar;
  const U* cache_mean;
  int64_t D;
  absl::optional<int64_t> C_;
  DmeanOp(const U* g_ptr, const T* x_ptr, const U* civar, const U* cmean, int64_t d)
    : gamma(g_ptr), x(x_ptr), cache_ivar(civar), cache_mean(cmean), D(d) {}
  inline void SetChannelDim(const int64_t c) { C_ = c; }

  __device__ inline U Compute(const T* dy, const int row, const int col) const {
    U curr = GetAs<T, U>(dy, row * D + col);
    U g_val;
    if (C_.has_value()) {
      int64_t C = C_.value();
      g_val = gamma[row % C];
    } else {
      g_val = gamma[col];
    }
    return -curr * g_val * cache_ivar[row];
  }

  __device__ inline U Compute(const T* dy, const int row, const int col,
                       const int z) const {
    // Channel last layout
    int64_t C = C_.value();
    U curr = GetAs<T, U>(dy, row * D * C + col * C + z);
    U ivar = cache_ivar[row * C + z];
    return -curr * gamma[z] * ivar;
  }

  __device__ inline U Finalize(const U sum) const { return sum; }
};

template <typename T, int thread_group_width = kWarpSize>
__inline__ __device__ WFGeneric<T> WelfordWarpReduce(
    const WFGeneric<T>& wf_thread) {
  const int num_warps = kBlockSize / thread_group_width;
  typedef gpuprim::WarpReduce<WFGeneric<T>> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_storage[num_warps];
  const int local_warp_id = threadIdx.x / thread_group_width;
  return WarpReduce(temp_storage[local_warp_id])
      .Reduce(wf_thread, WFGeneric<T>());
}

template <typename T>
__inline__ __device__ WFGeneric<T> WelfordBlockAllReduce(
    const WFGeneric<T>& wf_thread) {
  typedef gpuprim::BlockReduce<WFGeneric<T>, kBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  return BlockReduce(temp_storage).Reduce(wf_thread, WFGeneric<T>());
}

template <typename T, typename U, typename Op>
__global__ __launch_bounds__(1024) void GeneralNormRowReduceInToOutWelford(
    const T* __restrict__ in, const int N, const int D, U* out1, U* out2,
    Op op) {
  const int tid = threadIdx.x;

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

namespace LnNorm {

template <typename T, typename U>
struct YOp {
  const U* cache_mean;
  const U* cache_ivar;
  const U* gamma;
  const U* beta;
  int64_t D;
  __device__ inline T Compute(const T* x, const int row,
                              const int col) const {
    U mean = cache_mean[row];
    U ivar = cache_ivar[row];
    U curr = GetAs<T, U>(x, row * D + col);
    return static_cast<T>((curr - mean) * ivar * gamma[col] + beta[col]);
  }
};

template <typename T, typename U>
struct DxOp {
  const T* x;
  const U* cache_mean;
  const U* cache_ivar;
  const U* gamma;
  const U* dl_dvars;
  const U* dl_dmus;
  int64_t D;
  __device__ inline T Compute(const T* dy, const int row,
                              const int col) const {
    U curr = GetAs<T, U>(dy, row * D + col);
    U dl_di = curr * gamma[col] * cache_ivar[row];
    U di_dx = 1.;
    U dvar_dx = 2. * (x[row * D + col] - cache_mean[row]) / D;
    U dmu_dx = 1. / D;
    U dl_dx = dl_di * di_dx + dl_dvars[row] * dvar_dx + dl_dmus[row] * dmu_dx;
    return static_cast<T>(dl_dx);
  }
};
}  // namespace LnNorm

namespace InNorm {

template <typename T, typename U>
struct DwStatFusedOp {
  const U* gamma;
  const T* x;
  const U* cache_ivar;
  const U* cache_mean;
  int64_t C;
  int64_t D;
  __device__ inline void Compute(const T* dy, const int row, const int col, U* ret1,
                          U* ret2, U* ret3, U* ret4) const {
    U curr = GetAs<T, U>(dy, row * D + col);
    U ivar = cache_ivar[row];
    U x_mean = x[row * D + col] - cache_mean[row];
    U tmp = -curr * ivar;
    *ret1 = tmp * gamma[row % C];          // dL/dmu
    *ret2 = *ret1 * ivar * ivar * x_mean;  // dL/dvar^2
    *ret3 = -tmp * x_mean;                 // dgamma
    *ret4 = curr;                          // dbeta
  }

  __device__ inline void Compute(const T* dy, const int row, const int col,
                          const int z, U* ret1, U* ret2, U* ret3,
                          U* ret4) const {
    // Channel last layout
    U curr = GetAs<T, U>(dy, row * D * C + col * C + z);
    U ivar = cache_ivar[row * C + z];
    U x_mean = x[row * D * C + col * C + z] - cache_mean[row * C + z];
    U tmp = -curr * ivar;
    *ret1 = gamma[z] * tmp;
    *ret2 = *ret1 * ivar * ivar * x_mean;
    *ret3 = -tmp * x_mean;  // dgamma
    *ret4 = curr;           // dbeta
  }

  __device__ U Finalize(const U sum) const { return sum; }
};

template <typename T, typename U>
struct YOp {
  const U* cache_mean;
  const U* cache_ivar;
  const U* gamma;
  const U* beta;
  int64_t C;
  int64_t D;
  __device__ inline T Compute(const T* x, const int idx,
                       const bool is_channel_first) const {
    U curr = GetAs<T, U>(x, idx);
    int gb_idx, cache_idx;
    if (is_channel_first) {
      gb_idx = static_cast<int>((idx / D) % C);
      cache_idx = idx / D;
    } else {
      gb_idx = idx % C;
      cache_idx = (idx / (C * D)) * C + gb_idx;
    }
    U mean = cache_mean[cache_idx];
    U ivar = cache_ivar[cache_idx];
    return static_cast<T>((curr - mean) * ivar * gamma[gb_idx] + beta[gb_idx]);
  }
};

template <typename T, typename U>
struct DxOp {
  const T* x;
  const U* cache_mean;
  const U* cache_ivar;
  const U* gamma;
  const U* dl_dvars;
  const U* dl_dmus;
  int64_t C;
  int64_t D;
  __device__ inline T Compute(const T* dy, const int idx,
                       const bool is_channel_first) const {
    U curr = GetAs<T, U>(dy, idx);
    U dl_dx;
    if (is_channel_first) {
      int row = idx / D;
      U dl_di = curr * gamma[row % C] * cache_ivar[row];
      U di_dx = 1.;
      U dvar_dx = 2. * (x[idx] - cache_mean[row]) / D;
      U dmu_dx = 1. / D;
      dl_dx = dl_di * di_dx + dl_dvars[row] * dvar_dx + dl_dmus[row] * dmu_dx;
    } else {
      int col = idx % C;
      int cache_idx = idx / (C * D) * C + idx % C;
      U dl_di = curr * gamma[col] * cache_ivar[cache_idx];
      U di_dx = 1.;
      U dvar_dx = 2. * (x[idx] - cache_mean[cache_idx]) / D;
      U dmu_dx = 1. / D;
      dl_dx = dl_di * di_dx + dl_dvars[cache_idx] * dvar_dx +
              dl_dmus[cache_idx] * dmu_dx;
    }
    return static_cast<T>(dl_dx);
  }
};

template <typename T, typename U>
struct DxFusedOp {
  const T* x;
  const U* cache_mean;
  const U* cache_ivar;
  const U* gamma;
  const U* dl_dvars;
  const U* dl_dmus;
  int64_t C;
  int64_t D;
  __device__ inline T Compute(const T* dy, const int idx,
                       const bool is_channel_first) const {
    U curr = GetAs<T, U>(dy, idx);
    U dl_dx;
    if (is_channel_first) {
      int row = idx / D;
      dl_dx =
          (dl_dvars[row] * (x[idx] - cache_mean[row]) + dl_dmus[row]) / static_cast<U>(D) +
          curr * gamma[row % C] * cache_ivar[row];
    } else {
      int col = idx % C;
      int cache_idx = idx / (C * D) * C + idx % C;
      dl_dx = (dl_dvars[cache_idx] * (x[idx] - cache_mean[cache_idx]) +
               dl_dmus[cache_idx]) /
                  static_cast<U>(D) +
              curr * gamma[col] * cache_ivar[cache_idx];
    }
    return static_cast<T>(dl_dx);
  }
};

}  // namespace InNorm

}  // namespace functor
}  // namespace tensorflow
