#include <cuda_runtime.h>
#include <stdio.h>
#include <torch/extension.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
using namespace cute;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      /* Optionally, you could also call cudaDeviceReset here */               \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)


__host__ __device__ inline bool is_aligned_128(const void *ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0;
}

template <typename T, int BLK_M, int BLK_N, typename TiledCopyA,
          typename TiledCopyTrans, typename TiledCopyB, typename SmemLayoutB>
__global__ void mat_transpose_cute_smem_vectorized_optimized_kernel(
    const T *pA, T *pB, int M, int N, TiledCopyA copy_a, TiledCopyTrans copy_trans,TiledCopyB copy_b,
    SmemLayoutB sB_layout) {
  int tx = threadIdx.x;
  int bx = blockIdx.x, by = blockIdx.y;

  auto mA = make_tensor(make_gmem_ptr(pA),
                        make_layout(make_shape(M, N), GenRowMajor{})); // (M, N)
  auto mB = make_tensor(make_gmem_ptr(pB),
                        make_layout(make_shape(N, M), GenRowMajor{})); // (N, N)

  auto gA = local_tile(mA, make_shape(Int<BLK_M>{}, Int<BLK_N>{}),
                       make_coord(bx, by)); // (BM, BN)
  auto gB = local_tile(mB, make_shape(Int<BLK_N>{}, Int<BLK_M>{}),
                       make_coord(by, bx)); // (BN, BM)

  __shared__ T smem[BLK_M * BLK_N];
  auto sB = make_tensor(make_smem_ptr(smem),
                        sB_layout); // (BN, BM)

  auto thr_copy_a = copy_a.get_slice(tx);
  Tensor tAgA = thr_copy_a.partition_S(gA);
  auto tAsA = make_tensor_like(tAgA);
  Tensor tAsA_view = thr_copy_a.retile_D(tAsA);
  copy(copy_a, tAgA, tAsA_view);


  auto thr_copy_trans = copy_trans.get_slice(tx);
  auto tAsB = thr_copy_trans.retile_S(tAsA);
  auto tBsB_trans = thr_copy_trans.partition_D(sB);
  copy(copy_trans, tAsB, tBsB_trans);
  __syncthreads();

  auto thr_copy_b = copy_b.get_slice(tx);
  Tensor tBsB = thr_copy_b.partition_S(sB);
  Tensor tBgB = thr_copy_b.partition_D(gB);

  copy(copy_b, tBsB, tBgB);
}


void mat_transpose_cute_row_rvectorized_swizzled_optimized(torch::Tensor x,
                                                 torch::Tensor y) {
  const int BM = 8;
  const int BN = 16;
  auto ptr_A = x.data_ptr<float>();
  auto ptr_B = y.data_ptr<float>();
  const int M = x.size(0);
  const int N = x.size(1);

  // sanity checks
  assert(M % 4 == 0);
  assert(N % 4 == 0);
  static_assert(BM % 4 == 0);
  static_assert(BN % 4 == 0);
  assert(is_aligned_128(ptr_A));
  assert(is_aligned_128(ptr_B));

  // 一次性加载8*16大小的矩阵
  auto tile_copy_a = make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, float>{},
      make_layout(make_shape(Int<BM>{}, Int<BN / 4>{}), GenRowMajor{}),
      make_layout(make_shape(Int<1>{}, Int<4>{}), GenRowMajor{}));


  // 转换数据
  auto tile_copy_trans =  make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, float>{},
      make_layout(make_shape(Int<BN / 4>{}, Int<BM>{}), GenColMajor{}),
      make_layout(make_shape(Int<4>{}, Int<1>{}), GenRowMajor{}));



  // 一次性存储16*8大小的矩阵
  auto tile_copy_b = make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, float>{},
      make_layout(make_shape(Int<BN>{}, Int<BM / 4>{}), GenRowMajor{}),
      make_layout(make_shape(Int<1>{}, Int<4>{}), GenRowMajor{}));



  auto swizzle_func = Swizzle<2, 3, 2>{};
  auto sB_layout =
      composition(swizzle_func,
                  make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenRowMajor{}));

  static_assert(size(tile_copy_a) == size(tile_copy_b));

  dim3 block(size(tile_copy_a));
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

  mat_transpose_cute_smem_vectorized_optimized_kernel<
      float, BM, BN, decltype(tile_copy_a), decltype(tile_copy_trans), decltype(tile_copy_b),
      decltype(sB_layout)><<<grid, block>>>(
      ptr_A, ptr_B, M, N, tile_copy_a, tile_copy_trans, tile_copy_b, sB_layout);
  CUDA_CHECK(cudaGetLastError());
}


void mat_transpose_cute_row_rvectorized_swizzled_optimized_8warp(torch::Tensor x,
                                                 torch::Tensor y) {
  const int BM = 8;
  const int BN = 16*8;
  auto ptr_A = x.data_ptr<float>();
  auto ptr_B = y.data_ptr<float>();
  const int M = x.size(0);
  const int N = x.size(1);

  // sanity checks
  assert(M % 4 == 0);
  assert(N % 4 == 0);
  static_assert(BM % 4 == 0);
  static_assert(BN % 4 == 0);
  assert(is_aligned_128(ptr_A));
  assert(is_aligned_128(ptr_B));

  // 一次性加载8*16大小的矩阵
  auto tile_copy_a = make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, float>{},
      make_layout(
        make_shape(Int<BM>{}, make_shape(Int<4>{}, Int<BN / 16>{})),
        make_stride(Int<4>{}, make_stride(Int<1>{}, Int<32>{}))),
      make_layout(make_shape(Int<1>{}, Int<4>{}), GenRowMajor{}));


  // 转换数据
  auto tile_copy_trans =  make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, float>{},
      make_layout(
        make_shape(make_shape(Int<4>{}, Int<BN / 16>{}), Int<BM>{}),
        make_stride(make_stride(Int<1>{}, Int<32>{}), Int<4>{})),
      make_layout(make_shape(Int<4>{}, Int<1>{}), GenRowMajor{}));



  // 一次性存储16*8大小的矩阵
  auto tile_copy_b = make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, float>{},
      make_layout(make_shape(Int<BN>{}, Int<BM / 4>{}), GenRowMajor{}),
      make_layout(make_shape(Int<1>{}, Int<4>{}), GenRowMajor{}));



  auto swizzle_func = Swizzle<2, 3, 2>{};
  auto sB_layout =
      composition(swizzle_func,
                  make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenRowMajor{}));

  static_assert(size(tile_copy_a) == size(tile_copy_b));

  dim3 block(size(tile_copy_a));
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

  mat_transpose_cute_smem_vectorized_optimized_kernel<
      float, BM, BN, decltype(tile_copy_a), decltype(tile_copy_trans), decltype(tile_copy_b),
      decltype(sB_layout)><<<grid, block>>>(
      ptr_A, ptr_B, M, N, tile_copy_a, tile_copy_trans, tile_copy_b, sB_layout);
  CUDA_CHECK(cudaGetLastError());
}
