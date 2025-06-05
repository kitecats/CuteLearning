#include <cublas_v2.h>
#include <cuda.h> // NOLINT

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <stdlib.h>
#include <torch/extension.h>

#include "utils.h"

using namespace cute;

template <typename T_, int NWarpPerBlock_>
struct HgemvConfig
{
    using T = T_;
    static constexpr int NWarpPerBlock = NWarpPerBlock_;
    static constexpr int NumThreads = NWarpPerBlock * 32;

    static constexpr int BlockM = 16 * NWarpPerBlock;
    static constexpr int BlockN = 8;
    static constexpr int BlockK = 16;


    using MMA_Atom = MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>;
    using TiledMMA = decltype(make_tiled_mma(
        MMA_Atom{},
        make_layout(Shape<Int<NWarpPerBlock>, _1, _1>{}, GenColMajor{})
        ));
    
    static_assert(size(TiledMMA{}) == NumThreads && size(TiledMMA{}) <= 1024,
                    "NumThreads must be less than or equal 1024");
    
};


// using tensor core
template<typename HgemvConfig_>
__global__ void hgemv_tensor_core_cute_kernel(typename HgemvConfig_::T *Aptr, 
                                  typename HgemvConfig_::T *Bptr,
                                  typename HgemvConfig_::T *Cptr,
                                  const int M, const int K)
{
    using namespace cute;

    using T = typename HgemvConfig_::T; 
    using TiledMMA = typename HgemvConfig_::TiledMMA;
    constexpr int BlockM = HgemvConfig_::BlockM;
    constexpr int BlockN = HgemvConfig_::BlockN;
    constexpr int BlockK = HgemvConfig_::BlockK;

    int thrid = threadIdx.x;
    int blockid = blockIdx.x;

    int warpid = threadIdx.x / 32;
    int laneid = threadIdx.x % 32;
    
    auto A = make_tensor(make_gmem_ptr(Aptr), make_layout(make_shape(M, K), make_stride(K, Int<1>{})));
    auto B = make_tensor(make_gmem_ptr(Bptr), make_layout(make_shape(M, K), make_stride(0, Int<1>{})));
    auto C = make_tensor(make_gmem_ptr(Cptr), make_layout(make_shape(M, 1), make_stride(Int<1>{}, 0)));

    auto ABPre = make_identity_tensor(shape(A));
    auto CPre  = make_identity_tensor(shape(C));


    auto gA = local_tile(A, make_shape(Int<BlockM>{}, Int<BlockK>{}), make_coord(blockid, _));
    auto gB = local_tile(B, make_shape(Int<BlockN>{}, Int<BlockK>{}), make_coord(blockid, _));
    auto gC = local_tile(C, make_shape(Int<BlockM>{}, Int<1>{}), make_coord(blockid, 0));

    auto gABPre = local_tile(ABPre, make_shape(Int<BlockM>{}, Int<BlockK>{}), make_coord(blockid, _));
    auto gCPre  = local_tile(CPre, make_shape(Int<BlockM>{}, Int<1>{}), make_coord(blockid, _));


    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(thrid);
    auto tAgA = thr_mma.partition_A(gA);
    auto tBgB = thr_mma.partition_B(gB);

    auto rAPre = thr_mma.partition_A(gABPre);   
    auto rBPre = thr_mma.partition_B(gABPre);


    auto tArA = make_tensor_like(tAgA(_, _, _, 0));
    auto tBrB = make_tensor_like(tBgB(_, _, _, 0));


    auto tCrC = partition_fragment_C(tiled_mma, Shape<Int<BlockM>, Int<BlockN>>{});


    clear(tCrC);

    int num_tile_k = size<2>(gA);
#pragma unroll
    for(int itile = 0; itile < num_tile_k; itile++)
    {

        auto pre_A = rAPre(_, _, _, itile);
        auto pre_B = rBPre(_, _, _, itile);
        auto pred_A = [&](auto... coords) { return cute::elem_less(pre_A(coords...), shape(A)); }; 
        auto pred_B = [&](auto... coords) { return cute::elem_less(pre_B(coords...), shape(A)); }; 

        clear(tArA);copy_if(pred_A, tAgA(_, _, _, itile), tArA);
        clear(tBrB);copy_if(pred_B, tBgB(_, _, _, itile), tBrB);

        gemm(tiled_mma, tArA, tBrB, tCrC);
    }

    int elem_index1 = warpid * 16 + laneid / 4;
    int elem_index2 = warpid * 16 + laneid / 4 + 8;

    auto sum = make_tensor_like(gC(0, _));
    sum(0) = tCrC(0);
    auto elem_pred1 = [&](auto... coords) { return (laneid % 4 ==0) && cute::elem_less(gCPre(elem_index1), shape(C)) ; };  
    copy_if(elem_pred1, sum, gC(elem_index1, _));

    sum(0) = tCrC(2);
    auto elem_pred2 = [&](auto... coords) { return (laneid % 4 ==0) && cute::elem_less(gCPre(elem_index2), shape(C)) ; };  
    copy_if(elem_pred2, sum, gC(elem_index2, _));


}

void hgemv_tensor_core_cute(torch::Tensor A, torch::Tensor B, torch::Tensor C)
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

    using config = HgemvConfig<half, 4>;

    dim3 blcok(size(config::NumThreads));
    dim3 grid(ceil_div(M, config::BlockM));

    hgemv_tensor_core_cute_kernel<config><<<grid, blcok>>>(
        reinterpret_cast<half *>(A.data_ptr()),
        reinterpret_cast<half *>(B.data_ptr()),
        reinterpret_cast<half *>(C.data_ptr()),
        M, K);
}