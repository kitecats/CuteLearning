#include <cublas_v2.h>
#include <cuda.h> // NOLINT

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <stdlib.h>
#include <torch/extension.h>

using namespace cute;

// #define Debug

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
        // Tile<Int<16 * NWarpPerBlock>, _8, _16>
        ));
    
    static_assert(size(TiledMMA{}) == NumThreads && size(TiledMMA{}) <= 1024,
                    "NumThreads must be less than or equal 1024");
    
};

template<typename HgemvConfig_>
__global__ void hgemv_cute_kernel(typename HgemvConfig_::T *Aptr, 
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
    

    auto A = make_tensor(make_gmem_ptr(Aptr), make_layout(make_shape(M,K), make_stride(K, 1)));
    auto B = make_tensor(make_gmem_ptr(Bptr), make_layout(make_shape(M,K), make_stride(0, 1)));


    auto gA = local_tile(A, make_shape(Int<BlockM>{}, Int<BlockK>{}), make_coord(blockid, _));
    auto gB = local_tile(B, make_shape(Int<BlockN>{}, Int<BlockK>{}), make_coord(blockid, _));


    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(thrid);
    auto tAgA = thr_mma.partition_A(gA);
    auto tBgB = thr_mma.partition_B(gB);



    // auto tArA = make_tensor_like(tAgA);
    // auto tBrB = make_tensor_like(tBgB);

    auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));
    auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));
    auto tCrC = partition_fragment_C(tiled_mma, Shape<Int<BlockM>, Int<BlockN>>{});


#ifdef Debug
    if(thread(4))
    {
        print("A: ");print(A);print("\n");
        print("B: ");print(B);print("\n");
        print("gA: ");print(gA);print("\n");
        print("gB: ");print(gB);print("\n");
        print("tAgA: ");print(tAgA);print("\n");
        print("tBgB: ");print(tBgB);print("\n");
        print("tArA: ");print(tArA);print("\n");
        print("tBrB: ");print(tBrB);print("\n");
        print("tCrC: ");print(tCrC);print("\n");
    }
#endif

    clear(tCrC);

    int num_tile_k = size<2>(gA);
    for(int itile = 0; itile < num_tile_k; itile++)
    {
        copy(tAgA(_, _, _, itile), tArA);
        copy(tBgB(_, _, _, itile), tBrB);

        gemm(tiled_mma, tArA, tBrB, tCrC);
    }

    if(laneid % 4 ==0)
    {
        Cptr[blockid * BlockM + warpid * 16 + laneid / 4] = tCrC(0);
        Cptr[blockid * BlockM + warpid * 16 + laneid / 4 + 8] = tCrC(2);
    }


}

void forward_hgemv_cute(torch::Tensor A, torch::Tensor B, torch::Tensor C)
{
    
    const int M = A.size(0);
    const int K = A.size(1);

    using config = HgemvConfig<half, 4>;

    dim3 blcok(size(config::NumThreads));
    dim3 grid(ceil_div(M, config::BlockM));

    hgemv_cute_kernel<config><<<grid, blcok>>>(
        reinterpret_cast<half *>(A.data_ptr()),
        reinterpret_cast<half *>(B.data_ptr()),
        reinterpret_cast<half *>(C.data_ptr()),
        M, K);
}