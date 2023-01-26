#ifndef CARE_MINHASH_QUERY_FILTER_CUH
#define CARE_MINHASH_QUERY_FILTER_CUH

#include <config.hpp>
#include <hpc_helpers.cuh>

#include <gpu/cubwrappers.cuh>
#include <gpu/cuda_unique.cuh>
#include <gpu/cuda_unique_by_count.cuh>
#include <gpu/cuda_block_select.cuh>

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>


/*
    If toFind[s] exists in segment s, remove it from this segment s 
    by shifting remaining elements to the left.
    segmentSizes[s] will be updated.
    */
    template<class T, int BLOCKSIZE, int ITEMS_PER_THREAD>
    __global__ 
    void findAndRemoveFromSegmentKernel(
        const T* __restrict__ toFind,
        T* items,
        int numSegments,
        int* __restrict__ segmentSizes,
        const int* __restrict__ segmentBeginOffsets
    ){
        constexpr int itemsPerIteration = ITEMS_PER_THREAD * BLOCKSIZE;

        assert(BLOCKSIZE == blockDim.x);

        using BlockLoad = cub::BlockLoad<T, BLOCKSIZE, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
        using MyBlockSelect = BlockSelect<T, BLOCKSIZE>;

        __shared__ union{
            typename BlockLoad::TempStorage load;
            typename MyBlockSelect::TempStorage select;
        } temp_storage;

        for(int s = blockIdx.x; s < numSegments; s += gridDim.x){
            const int segmentsize = segmentSizes[s];
            const int beginOffset = segmentBeginOffsets[s];
            const T idToRemove = toFind[s];

            const int numIterations = SDIV(segmentsize, itemsPerIteration);
            T myitems[ITEMS_PER_THREAD];
            int flags[ITEMS_PER_THREAD];

            int numSelectedTotal = 0;
            int remainingItems = segmentsize;
            const T* inputdata = items + beginOffset;
            T* outputdata = items + beginOffset;

            for(int iter = 0; iter < numIterations; iter++){
                const int validItems = min(remainingItems, itemsPerIteration);
                BlockLoad(temp_storage.load).Load(inputdata, myitems, validItems);

                #pragma unroll
                for(int i = 0; i < ITEMS_PER_THREAD; i++){
                    if(threadIdx.x * ITEMS_PER_THREAD + i < validItems && myitems[i] != idToRemove){
                        flags[i] = 1;
                    }else{
                        flags[i] = 0;
                    }
                }

                __syncthreads();

                const int numSelected = MyBlockSelect(temp_storage.select).ForEachFlagged(myitems, flags, validItems,
                    [&](const auto& item, const int& pos){
                        outputdata[pos] = item;
                    }
                );
                assert(numSelected <= validItems);

                numSelectedTotal += numSelected;
                outputdata += numSelected;
                inputdata += validItems;
                remainingItems -= validItems;

                __syncthreads();
            }

            assert(segmentsize >= numSelectedTotal);

            //update segment size
            if(numSelectedTotal != segmentsize){
                if(threadIdx.x == 0){
                    segmentSizes[s] = numSelectedTotal;
                }
            }
        }
    }

    template<class T, int BLOCKSIZE, int ITEMS_PER_THREAD>
    void callFindAndRemoveFromSegmentKernel(
        const T* d_toFind,
        T* d_items,
        int numSegments,
        int* d_segmentSizes,
        const int* d_segmentBeginOffsets,
        cudaStream_t stream
    ){
        if(numSegments <= 0){
            return;
        }

        dim3 block = BLOCKSIZE;
        dim3 grid = numSegments;

        findAndRemoveFromSegmentKernel<T, BLOCKSIZE, ITEMS_PER_THREAD><<<grid, block, 0, stream>>>
            (d_toFind, d_items, numSegments, d_segmentSizes, d_segmentBeginOffsets);
    }

namespace care{ 
namespace gpu{


    struct GpuMinhashQueryFilter{
        static void keepDistinctAndNotMatching(
            const read_number* d_dontMatchPerSegment,
            cub::DoubleBuffer<read_number>& d_items,
            cub::DoubleBuffer<int>& d_numItemsPerSegment,
            cub::DoubleBuffer<int>& d_numItemsPerSegmentPrefixSum, //numSegments + 1
            int numSegments,
            int numItems,
            cudaStream_t stream,
            rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()
        ){
            if(numItems <= 0) return;
            if(numSegments <= 0) return;

            CubCallWrapper(mr).cubReduceMax(
                d_numItemsPerSegment.Current(),
                d_numItemsPerSegment.Alternate(),
                numSegments,
                stream
            );

            int sizeOfLargestSegment = 0;
            CUDACHECK(cudaMemcpyAsync(
                &sizeOfLargestSegment,
                d_numItemsPerSegment.Alternate(),
                sizeof(int),
                D2H,
                stream
            ));
            CUDACHECK(cudaStreamSynchronize(stream));
            
            GpuSegmentedUnique::unique(
                d_items.Current(),
                numItems,
                d_items.Alternate(),
                d_numItemsPerSegment.Alternate(),
                numSegments,
                sizeOfLargestSegment,
                d_numItemsPerSegmentPrefixSum.Current(),
                d_numItemsPerSegmentPrefixSum.Current() + 1,
                0,
                sizeof(read_number) * 8,
                stream,
                mr
            );

            if(d_dontMatchPerSegment != nullptr){
                //remove self read ids (inplace)
                //--------------------------------------------------------------------
                callFindAndRemoveFromSegmentKernel<read_number,128,4>(
                    d_dontMatchPerSegment,
                    d_items.Alternate(),
                    numSegments,
                    d_numItemsPerSegment.Alternate(),
                    d_numItemsPerSegmentPrefixSum.Current(),
                    stream
                );
            }

            CubCallWrapper(mr).cubInclusiveSum(
                d_numItemsPerSegment.Alternate(),
                d_numItemsPerSegmentPrefixSum.Alternate() + 1,
                numSegments,
                stream
            );
            CUDACHECK(cudaMemsetAsync(d_numItemsPerSegmentPrefixSum.Alternate(), 0, sizeof(int), stream));

            //copy final remaining values into contiguous range
            helpers::lambda_kernel<<<numSegments, 128, 0, stream>>>(
                [
                    d_items_in = d_items.Alternate(),
                    d_items_out = d_items.Current(),
                    numSegments,
                    d_numItemsPerSegment = d_numItemsPerSegment.Alternate(),
                    d_offsets = d_numItemsPerSegmentPrefixSum.Current(),
                    d_newOffsets = d_numItemsPerSegmentPrefixSum.Alternate()
                ] __device__ (){

                    for(int s = blockIdx.x; s < numSegments; s += gridDim.x){
                        const int numValues = d_numItemsPerSegment[s];
                        const int inOffset = d_offsets[s];
                        const int outOffset = d_newOffsets[s];

                        for(int c = threadIdx.x; c < numValues; c += blockDim.x){
                            d_items_out[outOffset + c] = d_items_in[inOffset + c];    
                        }
                    }
                }
            ); CUDACHECKASYNC;

            d_numItemsPerSegment.selector++;
            d_numItemsPerSegmentPrefixSum.selector++;
        }
    
        static void keepDistinct(
            cub::DoubleBuffer<read_number>& d_items,
            cub::DoubleBuffer<int>& d_numItemsPerSegment,
            cub::DoubleBuffer<int>& d_numItemsPerSegmentPrefixSum, //numSegments + 1
            int numSegments,
            int numItems,
            cudaStream_t stream,
            rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()
        ){
            keepDistinctAndNotMatching(
                nullptr,
                d_items,
                d_numItemsPerSegment,
                d_numItemsPerSegmentPrefixSum,
                numSegments,
                numItems,
                stream,
                mr
            );
        }
    
    
        static void keepDistinctByFrequency(
            int minFrequencyToKeep,
            cub::DoubleBuffer<read_number>& d_items,
            cub::DoubleBuffer<int>& d_numItemsPerSegment,
            cub::DoubleBuffer<int>& d_numItemsPerSegmentPrefixSum, //numSegments + 1
            int numSegments,
            int numItems,
            cudaStream_t stream,
            rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()
        ){
            GpuSegmentedUniqueByCount::unique(
                d_items.Current(),
                numItems,
                d_items.Alternate(),
                d_numItemsPerSegment.Alternate(),
                numSegments,
                d_numItemsPerSegmentPrefixSum.Current(),
                minFrequencyToKeep,
                stream,
                mr
            );

            CubCallWrapper(mr).cubInclusiveSum(
                d_numItemsPerSegment.Alternate(),
                d_numItemsPerSegmentPrefixSum.Alternate() + 1,
                numSegments,
                stream
            );

            CUDACHECK(cudaMemsetAsync(
                d_numItemsPerSegmentPrefixSum.Alternate(),
                0,
                sizeof(int),
                stream
            ));

            d_items.selector++;
            d_numItemsPerSegment.selector++;
            d_numItemsPerSegmentPrefixSum.selector++;
        }
    };



}}




#endif