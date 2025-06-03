#include <cublas_v2.h>
#include <cuda.h> // NOLINT

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <stdlib.h>
#include <torch/extension.h>

using namespace cute;



template <const int kWarpSize = 32>
__device__ __forceinline__ half warp_reduce_sum_f16(half val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

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


template<typename TiledCopy, int BlockM, int BlockK, int WARP_SIZE = 32>
__global__ void hgemv_f16_cute_kernel(half *Aptr, half *Bptr, half *Cptr, const int M, const int K)
{
    using namespace cute;

    int thrid = threadIdx.x + threadIdx.y * blockDim.x;
    int blockid = blockIdx.x;

    int laneid = threadIdx.x % WARP_SIZE;

    auto A = make_tensor(make_gmem_ptr(Aptr), make_layout(make_shape(M, K), make_stride(K, Int<1>{})));
    auto B = make_tensor(make_gmem_ptr(Bptr), make_layout(make_shape(M, K), make_stride(0, Int<1>{})));
    auto C = make_tensor(make_gmem_ptr(Cptr), make_layout(make_shape(M, 1), make_stride(Int<1>{}, 0)));
    auto Pre = make_identity_tensor(shape(A));

    auto gA = local_tile(A, make_shape(Int<BlockM>{}, Int<BlockK>{}), make_coord(blockid, _));
    auto gB = local_tile(B, make_shape(Int<BlockM>{}, Int<BlockK>{}), make_coord(blockid, _));
    auto gC = local_tile(C, make_shape(Int<BlockM>{}, Int<1>{}), make_coord(blockid, 0));
    auto gPre = local_tile(Pre, make_shape(Int<BlockM>{}, Int<BlockK>{}), make_coord(blockid, _));

   
    TiledCopy tiled_copy;
    auto thr_copy = tiled_copy.get_slice(thrid);
    
    auto tAgA = thr_copy.partition_S(gA);
    auto tBgB = thr_copy.partition_S(gB);
    auto rPre = thr_copy.partition_S(gPre);   

    int num_tile_k = size<2>(gA);

    // auto tArA = make_tensor<half>(make_shape(make_shape(Int<1>{}, Int<1>{})));
    // auto tBrB = make_tensor<half>(make_shape(make_shape(Int<1>{}, Int<1>{})));
    auto tArA = make_tensor_like(tAgA(_, _, _, 0));
    auto tBrB = make_tensor_like(tBgB(_, _, _, 0));
    auto sum =  make_tensor<half>(make_shape(make_shape(Int<1>{}, Int<1>{})));
    clear(sum);

#pragma unroll
    for(int num_iter_k = 0; num_iter_k < num_tile_k; num_iter_k++)
    {
        auto pre_ = rPre(_, _, _, num_iter_k);
        auto pred = [&](auto... coords) { return cute::elem_less(pre_(0), shape(A)); }; 

        clear(tArA); copy_if(tiled_copy, pred, tAgA(_, _, _, num_iter_k), tArA);
        clear(tBrB); copy_if(tiled_copy, pred, tBgB(_, _, _, num_iter_k), tBrB);

        sum(0) += tArA(0) * tBrB(0);
    }
    
    sum(0) = warp_reduce_sum_f16<WARP_SIZE>(sum(0));

    auto stord_pred = [&](auto... coords) { return cute::elem_less(rPre(0), shape(A)) && laneid == 0; }; 
    copy_if(stord_pred, sum, gC(threadIdx.y, _));

}


template<typename TiledCopy, int BlockM, int BlockK, int NumElemPerThread, int WARP_SIZE = 32>
__global__ void hgemv_f16x8_cute_kernel(half *Aptr, half *Bptr, half *Cptr, const int M, const int K)
{
    using namespace cute;

    int thrid = threadIdx.x + threadIdx.y * blockDim.x;
    int blockid = blockIdx.x;

    int laneid = threadIdx.x % WARP_SIZE;


    auto A = make_tensor(make_gmem_ptr(Aptr), make_layout(make_shape(M, K), make_stride(K, Int<1>{})));
    auto B = make_tensor(make_gmem_ptr(Bptr), make_layout(make_shape(M, K), make_stride(0, Int<1>{})));
    auto C = make_tensor(make_gmem_ptr(Cptr), make_layout(make_shape(M, 1), make_stride(Int<1>{}, 0)));
    auto Pre = make_identity_tensor(shape(A));


    auto gA = local_tile(A, make_shape(Int<BlockM>{}, Int<BlockK>{}), make_coord(blockid, _));
    auto gB = local_tile(B, make_shape(Int<BlockM>{}, Int<BlockK>{}), make_coord(blockid, _));
    auto gC = local_tile(C, make_shape(Int<BlockM>{}, Int<1>{}), make_coord(blockid, 0));
    auto gPre = local_tile(Pre, make_shape(Int<BlockM>{}, Int<BlockK>{}), make_coord(blockid, _));

   
    TiledCopy tiled_copy;
    auto thr_copy = tiled_copy.get_slice(thrid);
    
    auto tAgA = thr_copy.partition_S(gA);
    auto tBgB = thr_copy.partition_S(gB);
    auto rPre = thr_copy.partition_S(gPre);   

    int num_tile_k = size<2>(gA);

    // auto tArA = make_tensor<half>(make_shape(make_shape(Int<NumElemPerThread>{}, Int<1>{})));
    // auto tBrB = make_tensor<half>(make_shape(make_shape(Int<NumElemPerThread>{}, Int<1>{})));
    auto tArA = make_tensor_like(tAgA(_, _, _, 0));
    auto tBrB = make_tensor_like(tBgB(_, _, _, 0));
    auto sum =  make_tensor<half>(make_shape(make_shape(Int<1>{}, Int<1>{})));
    clear(sum);

#pragma unroll
    for(int iter_k = 0; iter_k < num_tile_k; iter_k++)
    {
        auto pre_ = rPre(_, _, _, iter_k);
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

    auto stord_pred = [&](auto... coords) { return cute::elem_less(rPre(0), shape(A)) && laneid == 0; }; 
    copy_if(stord_pred, sum, gC(threadIdx.y, _));

}

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
    

    auto A = make_tensor(make_gmem_ptr(Aptr), make_layout(make_shape(M,K), make_stride(K, 1)));
    auto B = make_tensor(make_gmem_ptr(Bptr), make_layout(make_shape(M,K), make_stride(0, 1)));


    auto gA = local_tile(A, make_shape(Int<BlockM>{}, Int<BlockK>{}), make_coord(blockid, _));
    auto gB = local_tile(B, make_shape(Int<BlockN>{}, Int<BlockK>{}), make_coord(blockid, _));


    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(thrid);
    auto tAgA = thr_mma.partition_A(gA);
    auto tBgB = thr_mma.partition_B(gB);



    auto tArA = make_tensor_like(tAgA(_, _, _, 0));
    auto tBrB = make_tensor_like(tBgB(_, _, _, 0));

    // auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));
    // auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));
    auto tCrC = partition_fragment_C(tiled_mma, Shape<Int<BlockM>, Int<BlockN>>{});


    clear(tCrC);

    int num_tile_k = size<2>(gA);
#pragma unroll
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


#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

#define ASSERT_K_IS_MULTIBLE_OF(V)                                             \
  if (K % (V) != 0) {                                                          \
    throw std::runtime_error("K must be multiple of " #V);                     \
  }

#define ASSERT_K_IS_EQUAL_OF(V)                                                \
  if (K != (V)) {                                                              \
    throw std::runtime_error("K must be " #V);                                 \
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


