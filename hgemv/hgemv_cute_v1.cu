#include <cublas_v2.h>
#include <cuda.h> // NOLINT

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <stdlib.h>
#include <torch/extension.h>

#include "utils.h"

using namespace cute;

template<int ThreadRowsPerWarp, int ThreadColsPerWarp, int BlockWarp>

template <const int kWarpSize = 32>
__device__ __forceinline__ half warp_reduce_sum_f16(half val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask, kGroupSize);
  }
  return val;
}

template<typename TiledCopy, int BlockM, int BlockK, int WARP_SIZE = 32>
__global__ void hgemv_f16_cute_kernel(half *Aptr, half *Bptr, half *Cptr, const int M, const int K)
{
    using namespace cute;

    int thrid = threadIdx.x + threadIdx.y * blockDim.x;
    int blockid = blockIdx.x;

    int laneid = threadIdx.x % WARP_SIZE;
    int warpid = threadIdx.y;

    auto A = make_tensor(make_gmem_ptr(Aptr), make_layout(make_shape(M, K), make_stride(K, Int<1>{})));
    auto B = make_tensor(make_gmem_ptr(Bptr), make_layout(make_shape(M, K), make_stride(0, Int<1>{})));
    auto C = make_tensor(make_gmem_ptr(Cptr), make_layout(make_shape(M, 1), make_stride(Int<1>{}, 0)));
    
    auto ABPre = make_identity_tensor(shape(A));
    auto CPre  = make_identity_tensor(shape(C));

    auto gA = local_tile(A, make_shape(Int<BlockM>{}, Int<BlockK>{}), make_coord(blockid, _));
    auto gB = local_tile(B, make_shape(Int<BlockM>{}, Int<BlockK>{}), make_coord(blockid, _));
    auto gC = local_tile(C, make_shape(Int<BlockM>{}, Int<1>{}), make_coord(blockid, 0));
    
    auto gABPre = local_tile(ABPre, make_shape(Int<BlockM>{}, Int<BlockK>{}), make_coord(blockid, _));
    auto gCPre  = local_tile(CPre, make_shape(Int<BlockM>{}, Int<1>{}), make_coord(blockid, _));
   
    TiledCopy tiled_copy;
    auto thr_copy = tiled_copy.get_slice(thrid);
    
    auto tAgA = thr_copy.partition_S(gA);
    auto tBgB = thr_copy.partition_S(gB);

    auto rABPre = thr_copy.partition_S(gABPre);   

    int num_tile_k = size<2>(gA);

    auto tArA = make_tensor_like(tAgA(_, _, _, 0));
    auto tBrB = make_tensor_like(tBgB(_, _, _, 0));

    auto sum = make_tensor_like(gC(0, _));
    clear(sum);

#pragma unroll
    for(int num_iter_k = 0; num_iter_k < num_tile_k; num_iter_k++)
    {
        auto pre_ = rABPre(_, _, _, num_iter_k);
        auto pred = [&](auto... coords) { return cute::elem_less(pre_(0), shape(A)); }; 

        clear(tArA); copy_if(tiled_copy, pred, tAgA(_, _, _, num_iter_k), tArA);
        clear(tBrB); copy_if(tiled_copy, pred, tBgB(_, _, _, num_iter_k), tBrB);

        sum(0) += tArA(0) * tBrB(0);
    }
    
    sum(0) = warp_reduce_sum_f16<WARP_SIZE>(sum(0));

    auto stord_pred = [&](auto... coords) { return cute::elem_less(gCPre(warpid), shape(C)) && laneid == 0; }; 
    copy_if(stord_pred, sum, gC(warpid, _));

}

template<typename TiledCopy, int BlockM, int BlockK, int NumElemPerThread, int WARP_SIZE = 32>
__global__ void hgemv_f16x8_cute_kernel(half *Aptr, half *Bptr, half *Cptr, const int M, const int K)
{
    using namespace cute;

    int thrid = threadIdx.x + threadIdx.y * blockDim.x;
    int blockid = blockIdx.x;

    int laneid = threadIdx.x % WARP_SIZE;
    int warpid = threadIdx.y;

    auto A = make_tensor(make_gmem_ptr(Aptr), make_layout(make_shape(M, K), make_stride(K, Int<1>{})));
    auto B = make_tensor(make_gmem_ptr(Bptr), make_layout(make_shape(M, K), make_stride(0, Int<1>{})));
    auto C = make_tensor(make_gmem_ptr(Cptr), make_layout(make_shape(M, 1), make_stride(Int<1>{}, 0)));

    auto ABPre = make_identity_tensor(shape(A));
    auto CPre  = make_identity_tensor(shape(C));


    auto gA = local_tile(A, make_shape(Int<BlockM>{}, Int<BlockK>{}), make_coord(blockid, _));
    auto gB = local_tile(B, make_shape(Int<BlockM>{}, Int<BlockK>{}), make_coord(blockid, _));
    auto gC = local_tile(C, make_shape(Int<BlockM>{}, Int<1>{}), make_coord(blockid, 0));

    auto gABPre = local_tile(ABPre, make_shape(Int<BlockM>{}, Int<BlockK>{}), make_coord(blockid, _));
    auto gCPre  = local_tile(CPre, make_shape(Int<BlockM>{}, Int<1>{}), make_coord(blockid, _));

   
    TiledCopy tiled_copy;
    auto thr_copy = tiled_copy.get_slice(thrid);
    
    auto tAgA = thr_copy.partition_S(gA);
    auto tBgB = thr_copy.partition_S(gB);
    auto rABPre = thr_copy.partition_S(gABPre);   

    int num_tile_k = size<2>(gA);

    auto tArA = make_tensor_like(tAgA(_, _, _, 0));
    auto tBrB = make_tensor_like(tBgB(_, _, _, 0));

    auto sum = make_tensor_like(gC(0, _));
    clear(sum);

#pragma unroll
    for(int iter_k = 0; iter_k < num_tile_k; iter_k++)
    {
        auto pre_ = rABPre(_, _, _, iter_k);
        auto pred = [&](auto... coords) { return cute::elem_less(pre_(NumElemPerThread - 1), shape(A)); }; 

        clear(tArA); copy_if(tiled_copy, pred, tAgA(_, _, _, iter_k), tArA);
        clear(tBrB); copy_if(tiled_copy, pred, tBgB(_, _, _, iter_k), tBrB);

        auto tArA_half2 = recast<half2>(tArA);
        auto tBrB_half2 = recast<half2>(tBrB);
        auto sum_half2 = make_tensor<half2>(make_shape(Int<1>{}));

#pragma unroll
        for(int iter_elem = 0; iter_elem < size(tArA_half2); iter_elem++)
        {
            sum_half2(0) = tArA_half2(iter_elem) * tBrB_half2(iter_elem) + sum_half2(0); 
        }

        sum(0) += sum_half2(0).x + sum_half2(0).y;
    }
    
    sum(0) = warp_reduce_sum_f16<WARP_SIZE>(sum(0));

    auto stord_pred = [&](auto... coords) { return cute::elem_less(gCPre(warpid), shape(C)) && laneid == 0; }; 
    copy_if(stord_pred, sum, gC(warpid, _));
}



void hgemv_f16_cute(torch::Tensor A, torch::Tensor B, torch::Tensor C)
{
    
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(B, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(C, torch::kHalf)
    const int M = A.size(0);
    const int K = A.size(1);
    CHECK_TORCH_TENSOR_SHAPE(A, M, K)
    CHECK_TORCH_TENSOR_SHAPE(B, K, 1)
    CHECK_TORCH_TENSOR_SHAPE(C, M, 1)
    // ASSERT_K_IS_MULTIBLE_OF(8)

    constexpr int NumThreadPerRow = 32;
    constexpr int NumThreadPerBlock = 128;
    constexpr int NumRowPerBlcok = NumThreadPerBlock / 32;

    using LoadType = uint16_t;

    constexpr int NumElemPerThread = sizeof(LoadType) / sizeof(half);

    using CopyAtom = Copy_Atom<UniversalCopy<LoadType>, half>;
    using TiledCopy =  decltype(make_tiled_copy(
      CopyAtom{},
      make_layout(
          Shape<Int<NumRowPerBlcok>, Int<NumThreadPerRow>>{},
          GenRowMajor{}),
      make_layout(Shape<_1, Int<NumElemPerThread>>{}, GenRowMajor{})));

    dim3 blcok(NumThreadPerRow, NumRowPerBlcok);
    dim3 grid(ceil_div(M, NumRowPerBlcok));

    hgemv_f16_cute_kernel<TiledCopy, NumRowPerBlcok, NumThreadPerRow * NumElemPerThread><<<grid, blcok>>>(
        reinterpret_cast<half *>(A.data_ptr()),
        reinterpret_cast<half *>(B.data_ptr()),
        reinterpret_cast<half *>(C.data_ptr()),
        M, K);
}

void hgemv_f16x8_cute(torch::Tensor A, torch::Tensor B, torch::Tensor C)
{
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(B, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(C, torch::kHalf)
    const int M = A.size(0);
    const int K = A.size(1);
    CHECK_TORCH_TENSOR_SHAPE(A, M, K)
    CHECK_TORCH_TENSOR_SHAPE(B, K, 1)
    CHECK_TORCH_TENSOR_SHAPE(C, M, 1)
    ASSERT_K_IS_MULTIBLE_OF(8)

    


    constexpr int NumThreadPerRow = 32;
    constexpr int NumThreadPerBlock = 128;
    constexpr int NumRowPerBlcok = NumThreadPerBlock / 32;

    using LoadType = uint128_t;

    constexpr int NumElemPerThread = sizeof(LoadType) / sizeof(half);

    using CopyAtom = Copy_Atom<UniversalCopy<LoadType>, half>;
    using TiledCopy =  decltype(make_tiled_copy(
      CopyAtom{},
      make_layout(
          Shape<Int<NumRowPerBlcok>, Int<NumThreadPerRow>>{},
          GenRowMajor{}),
      make_layout(Shape<_1, Int<NumElemPerThread>>{}, GenRowMajor{})));

    dim3 blcok(NumThreadPerRow, NumRowPerBlcok);
    dim3 grid(ceil_div(M, NumRowPerBlcok));

    hgemv_f16x8_cute_kernel<TiledCopy, NumRowPerBlcok, NumThreadPerRow * NumElemPerThread, NumElemPerThread><<<grid, blcok>>>(
        reinterpret_cast<half *>(A.data_ptr()),
        reinterpret_cast<half *>(B.data_ptr()),
        reinterpret_cast<half *>(C.data_ptr()),
        M, K);
}





