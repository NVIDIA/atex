/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
==============================================================================*/

#include "absl/types/optional.h"
#include "tensorflow/core/kernels/gpu_prim.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {

#define CUDA_RETURN_IF_ERROR(...)                                         \
  do {                                                                    \
    cudaError_t cuda_status = (__VA_ARGS__);                              \
    if (cuda_status != cudaSuccess) {                                     \
      return errors::Internal("CUDA: ", cudaGetErrorString(cuda_status)); \
    }                                                                     \
  } while (0)

static const int64_t kBlockSize = 128;
static const int64_t kWarpSize = 32;
static const int kMaxWorkPerThread = 32;
constexpr int kBlockSizeSearchSpace[3] = {1024, 512, 256};

template <typename T>
class PackSizeTraits;

template <>
class PackSizeTraits<Eigen::half> {
 public:
  static int const PackSize = 4;
};

template <>
class PackSizeTraits<float> {
 public:
  static int const PackSize = 2;
};

template <typename T, int N>
struct GetPackType {
  using type =
      typename std::aligned_storage<N * sizeof(T), N * sizeof(T)>::type;
};

template <typename T, int N>
using PackType = typename GetPackType<T, N>::type;

template <typename T, int N>
union Pack {
  static_assert(sizeof(PackType<T, N>) == sizeof(T) * N);
  __device__ Pack() {
    // do nothing
  }
  PackType<T, N> storage;
  T elem[N];
};

template <typename T, typename... Ts, typename... Args>
void LaunchVectorizedKernel(void (*func[3])(Ts...), dim3 grid_dim,
                            dim3 block_dim, size_t shared_memory_size_bytes,
                            gpuStream_t stream, int D, Args&&... arguments) {
  // Cases each func will be used.
  // func[2] : half and 4-elem aligned inputs
  // func[1] : float/half and 2-elem aligned inputs
  // func[0] : fallback for everything else
  if (D % 4 == 0 && PackSizeTraits<T>::PackSize == 4) {
    TF_CHECK_OK(GpuLaunchKernel(func[2], grid_dim, block_dim,
                                shared_memory_size_bytes, stream,
                                std::forward<Args>(arguments)...));
  } else if (D % 2 == 0) {
    TF_CHECK_OK(GpuLaunchKernel(func[1], grid_dim, block_dim,
                                shared_memory_size_bytes, stream,
                                std::forward<Args>(arguments)...));
  } else {
    TF_CHECK_OK(GpuLaunchKernel(func[0], grid_dim, block_dim,
                                shared_memory_size_bytes, stream,
                                std::forward<Args>(arguments)...));
  }
}

template <typename T, typename U>
inline __device__ U GetAs(const T* __restrict__ in, int offset) {
  return static_cast<U>(in[offset]);
}

template <typename T, typename U, int N>
inline __device__ Pack<U, N> Cast(const Pack<T, N>& in) {
  Pack<U, N> out;
  for (int i = 0; i < N; ++i) {
    out.elem[i] = static_cast<U>(in.elem[i]);
  }
  return out;
}

template <typename T, int N>
inline __device__ Pack<T, N> Dot(const Pack<T, N>& a, const Pack<T, N>& b) {
  Pack<T, N> pack;
  for (int i = 0; i < N; i++) {
    pack.elem[i] = a.elem[i] * b.elem[i];
  }
  return pack;
}

template <typename T, int N>
inline __device__ Pack<T, N> Add(const Pack<T, N>& a, const Pack<T, N>& b) {
  Pack<T, N> pack;
  for (int i = 0; i < N; i++) {
    pack.elem[i] = a.elem[i] + b.elem[i];
  }
  return pack;
}

template <typename T, int N>
inline __device__ Pack<T, N> Load(const T* src, int offset = 0) {
  Pack<T, N> pack;
  pack.storage = *(reinterpret_cast<const PackType<T, N>*>(src) + offset / N);
  return pack;
}

template <typename T, int N>
inline __device__ void Store(T* dst, const Pack<T, N>& pack, int offset = 0) {
  *(reinterpret_cast<PackType<T, N>*>(dst) + offset / N) = pack.storage;
}

template <typename T, typename U, int N>
inline __device__ void CopyWithCast(const T* src, int src_offset, U* dst,
                                    int dst_offset = 0) {
  Pack<T, N> pack_t = Load<T, N>(src, src_offset);
  Store<U, N>(dst, Cast<T, U, N>(pack_t), dst_offset);
}

template <typename T, int N>
inline __device__ void CopyWithDot(const T* src, int offset, T* dst) {
  // dst = src * dst
  Store<T, N>(dst, Dot(Load<T, N>(src, offset), Load<T, N>(dst)));
}

template <typename T, typename U, int N>
inline __device__ void CopyWithAffineAndCast(T* src, const T* scale,
                                             int offset_s, const T* bias,
                                             int offset_b, U* dst,
                                             int offset_dst) {
  // dst = src * scale + bias
  Pack<T, N> dst_pack =
      Add<T, N>(Dot<T, N>(Load<T, N>(src), Load<T, N>(scale, offset_s)),
                Load<T, N>(bias, offset_b));
  Store<U, N>(dst, Cast<T, U, N>(dst_pack), offset_dst);
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
  inline __device__ int GetIndex(int row, int col) const {
    if (C_.has_value()) {
      // Channel last layout
      int64_t C = C_.value();
      return (row / C) * C * D + col * C + row % C;
    } else {
      return row * D + col;
    }
  }

  inline __device__ int GetIndex(int row, int col, int z) const {
    // Channel last layout
    int64_t C = C_.value();
    return row * C * D + col * C + z;
  }

  inline __device__ void Update(const T* x, int row, int col,
                                WFGeneric<U>& wf_data) {
    // Channel first layout
    wf_data = WFGeneric<U>()(GetAs<T, U>(x, GetIndex(row, col)), wf_data);
  }

  inline __device__ void Update(const T* x, int row, int col, int z,
                                WFGeneric<U>& wf_data) {
    // Channel last layout
    wf_data = WFGeneric<U>()(GetAs<T, U>(x, GetIndex(row, col, z)), wf_data);
  }

  inline __device__ void Update(U val, WFGeneric<U>& wf_data) {
    wf_data = WFGeneric<U>()(val, wf_data);
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
  inline __device__ int GetIndex(int row, int col) const {
    if (C_.has_value()) {
      // Channel last layout
      int64_t C = C_.value();
      return (row / C) * C * D + col * C + row % C;
    } else {
      return row * D + col;
    }
  }

  inline __device__ int GetIndex(int row, int col, int z) const {
    // Channel last layout
    int64_t C = C_.value();
    return row * C * D + col * C + z;
  }

  __device__ U Compute(const T* x, int row, int col) const {
    return GetAs<T, U>(x, GetIndex(row, col));
  }
  __device__ U Compute(const T* x, int row, int col, int z) const {
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

  inline __device__ int GetIndex(int row, int col) const {
    if (C_.has_value()) {
      // Channel last layout
      int64_t C = C_.value();
      return (row / C) * C * D + col * C + row % C;
    } else {
      return row * D + col;
    }
  }

  inline __device__ int GetIndex(int row, int col, int z) const {
    // Channel last layout
    int64_t C = C_.value();
    return row * C * D + col * C + z;
  }

  __device__ U Compute(const T* x, int row, int col, U mean) const {
    U curr = GetAs<T, U>(x, GetIndex(row, col));
    return (curr - mean) * (curr - mean);
  }
  __device__ U Compute(const T* x, int row, int col) const {
    return Compute(x, row, col, cache_mean[row]);
  }
  __device__ U Compute(const T* x, int row, int col, int z,
                       U mean) const {
    U curr = GetAs<T, U>(x, GetIndex(row, col, z));
    return (curr - mean) * (curr - mean);
  }
  __device__ U Compute(const T* x, int row, int col, int z) const {
    // Channel last layout
    int64_t C = C_.value();
    U m = cache_mean[row * C + z];
    return Compute(x, row, col, z, m);
  }
  __device__ U Finalize(U sum) const {
    return static_cast<U>(
        rsqrt(static_cast<float>(sum) / static_cast<float>(D) +
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
  DvarOp(const U* g_ptr, const T* x_ptr, const U* civar, const U* cmean,
         int64_t d)
      : gamma(g_ptr), x(x_ptr), cache_ivar(civar), cache_mean(cmean), D(d) {}
  inline void SetChannelDim(const int64_t c) { C_ = c; }

  __device__ inline U Compute(const T* dy, int row, int col) const {
    U curr = GetAs<T, U>(dy, row * D + col);
    U g_val;
    if (C_.has_value()) {
      int64_t C = C_.value();
      g_val = gamma[row % C];
    } else {
      g_val = gamma[col];
    }
    return curr * g_val * (x[row * D + col] - cache_mean[row]) * (-0.5) *
           (cache_ivar[row] * cache_ivar[row] * cache_ivar[row]);
  }

  __device__ inline U Compute(U x, U dy_dot_gamma, int row) {
    U ivar = cache_ivar[row];
    return dy_dot_gamma * (x - cache_mean[row]) * (-0.5) * ivar * ivar * ivar;
  }

  __device__ inline U Compute(const T* dy, int row, int col, int z) const {
    int64_t C = C_.value();
    U curr = GetAs<T, U>(dy, row * D * C + col * C + z);
    return Compute(x[row * D * C + col * C + z], curr * gamma[z], row * C + z);
  }
  __device__ U Finalize(U sum) const { return sum; }
};

template <typename T, typename U>
struct DmeanOp {
  const U* gamma;
  const T* x;
  const U* cache_ivar;
  const U* cache_mean;
  int64_t D;
  absl::optional<int64_t> C_;
  DmeanOp(const U* g_ptr, const T* x_ptr, const U* civar, const U* cmean,
          int64_t d)
      : gamma(g_ptr), x(x_ptr), cache_ivar(civar), cache_mean(cmean), D(d) {}
  inline void SetChannelDim(const int64_t c) { C_ = c; }

  __device__ inline U Compute(const T* dy, int row, int col) const {
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

  __device__ inline U Compute(U dy_dot_gamma, int row) const {
    return -dy_dot_gamma * cache_ivar[row];
  }

  __device__ inline U Compute(const T* dy, int row, int col, int z) const {
    // Channel last layout
    int64_t C = C_.value();
    U curr = GetAs<T, U>(dy, row * D * C + col * C + z);
    return Compute(curr * gamma[z], row * C + z);
  }

  __device__ inline U Finalize(U sum) const { return sum; }
};

template <typename T, int thread_group_width = kWarpSize>
__inline__ __device__ WFGeneric<T> WelfordWarpReduce(
    const WFGeneric<T>& wf_thread, bool is_broadcast = false) {
  const int num_warps = kBlockSize / thread_group_width;
  typedef gpuprim::WarpReduce<WFGeneric<T>> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_storage[num_warps];
  const int local_warp_id = threadIdx.x / thread_group_width;
  auto wp_reduced =
      WarpReduce(temp_storage[local_warp_id]).Reduce(wf_thread, WFGeneric<T>());
  if (is_broadcast) {
    return gpuprim::ShuffleIndex<thread_group_width, WFGeneric<T>>(
        wp_reduced, 0, 0xffffffff);
  }
  return wp_reduced;
}

template <typename T, typename Op, int BlockSize = kBlockSize>
__inline__ __device__ T BlockAllReduce(const T& val, Op reduce_op,
                                       bool is_broadcast = false) {
  typedef gpuprim::BlockReduce<T, BlockSize> BlockReduce;
  __shared__ union temp_storage {
    typename BlockReduce::TempStorage reduce;
    T broadcast[1];
    temp_storage(){};
    ~temp_storage(){};
  } temp_storage;

  T reduced = BlockReduce(temp_storage.reduce).Reduce(val, reduce_op);
  if (!is_broadcast) {
    return reduced;
  }

  if (threadIdx.x == 0) {
    temp_storage.broadcast[0] = reduced;
  }
  __syncthreads();
  return temp_storage.broadcast[0];
}

template <typename T, typename U, typename Op>
__global__ __launch_bounds__(1024) void GeneralNormRowReduceInToOutWelford(
    const T* __restrict__ in, int N, int D, U* out1, U* out2, Op op) {
  const int tid = threadIdx.x;

  for (int k = blockIdx.x; k < N; k += gridDim.x) {
    WFGeneric<U> wf_thread;

    for (int i = tid; i < D; i += kBlockSize) {
      op.Update(in, k, i, wf_thread);
    }

    WFGeneric<U> wf_row =
        BlockAllReduce<WFGeneric<U>, WFGeneric<U>>(wf_thread, WFGeneric<U>());

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
  __device__ inline U ComputePartial(U x, int row) const {
    // partial = (x - mean) * ivar;
    return (x - cache_mean[row]) * cache_ivar[row];
  }

  __device__ inline T Compute(const T* x, int row, int col) const {
    U curr = GetAs<T, U>(x, row * D + col);
    U partial = ComputePartial(curr, row);
    return static_cast<T>(partial * gamma[col] + beta[col]);
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
  __device__ inline U ComputePartial0(U dy, int row) const {
    return dy * cache_ivar[row];
  }

  __device__ inline U ComputePartial1(U x, int row,
                                      int stride = 1) const {
    // To cover the case that dl_dlvars and dl_dmus have strides
    U dvar_dx = 2 * (x - cache_mean[row]);
    return (dvar_dx * dl_dvars[row * stride] + dl_dmus[row * stride]) /
           static_cast<U>(D);
  }

  __device__ inline T Compute(const T* dy, int row, int col,
                              int stride = 1) const {
    U curr = GetAs<T, U>(dy, row * D + col);
    U partial0 = ComputePartial0(curr, row);
    U partial1 = ComputePartial1(x[row * D + col], row, stride);
    return static_cast<T>(partial0 * gamma[col] + partial1);
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
  __device__ inline void Compute(const T* dy, int row, int col, U* ret1,
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

  __device__ inline void Compute(const T* dy, int row, int col, int z, U* ret1,
                                 U* ret2, U* ret3, U* ret4) const {
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

  __device__ U Finalize(U sum) const { return sum; }
};

template <typename T, typename U>
struct YOp {
  const U* cache_mean;
  const U* cache_ivar;
  const U* gamma;
  const U* beta;
  int64_t C;
  int64_t D;
  __device__ inline T Compute(const T* x, int idx,
                              bool is_channel_first) const {
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
  __device__ inline T Compute(const T* dy, int idx,
                              bool is_channel_first) const {
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
  __device__ inline T Compute(const T* dy, int idx,
                              bool is_channel_first) const {
    U curr = GetAs<T, U>(dy, idx);
    U dl_dx;
    if (is_channel_first) {
      int row = idx / D;
      dl_dx = (dl_dvars[row] * (x[idx] - cache_mean[row]) + dl_dmus[row]) /
                  static_cast<U>(D) +
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
