#ifndef CUDA_UNIQUE_BY_COUNT_CUH
#define CUDA_UNIQUE_BY_COUNT_CUH

#include <gpu/rmm_utilities.cuh>
#include <gpu/cudaerrorcheck.cuh>
#include <gpu/cubwrappers.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <cub/cub.cuh>

#include <thrust/scatter.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <thrust/host_vector.h>


/*
    For each segment of items, copy an item to output if it occurs at least n times within the segment
    d_items is left in unspecified state
*/
struct GpuSegmentedUniqueByCount{

    template<class T>
    static void unique(
        T* d_items,
        int numItems,
        T* d_unique_items,
        int* d_unique_lengths,
        int numSegments,
        const int* d_offsets,
        int minimumCount,
        cudaStream_t stream = 0,
        rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()
    ){
        auto thrustpolicy = rmm::exec_policy_nosync(stream, mr);

        cub::DoubleBuffer<T> d_items_dblbuf{d_items, d_unique_items};

        #if 0
        //CUB_VERSION >= 101600
        {
            std::size_t cubTempSize = 0;
            CUDACHECK(cub::DeviceSegmentedSort::SortKeys(
                nullptr,
                cubTempSize,
                d_items_dblbuf,
                numItems,
                numSegments,
                d_offsets,
                d_offsets + 1,
                stream
            ));

            rmm::device_uvector<char> d_cubtemp(cubTempSize, stream, mr);

            CUDACHECK(cub::DeviceSegmentedSort::SortKeys(
                d_cubtemp.data(),
                cubTempSize,
                d_items_dblbuf,
                numItems,
                numSegments,
                d_offsets,
                d_offsets + 1,
                stream
            ));
        }
        #else
        {
            std::size_t cubTempSize = 0;
            CUDACHECK(cub::DeviceSegmentedRadixSort::SortKeys(
                nullptr,
                cubTempSize,
                d_items_dblbuf,
                numItems,
                numSegments,
                d_offsets,
                d_offsets + 1,
                0,
                sizeof(T) * 8,
                stream
            ));

            rmm::device_uvector<char> d_cubtemp(cubTempSize, stream, mr);

            CUDACHECK(cub::DeviceSegmentedRadixSort::SortKeys(
                d_cubtemp.data(),
                cubTempSize,
                d_items_dblbuf,
                numItems,
                numSegments,
                d_offsets,
                d_offsets + 1,
                0,
                sizeof(T) * 8,
                stream
            ));
        }
        #endif

        rmm::device_uvector<int> d_segmentIds(numItems, stream, mr);
        rmm::device_uvector<int> d_segmentIds_tmp(numItems, stream, mr);

        thrust::fill(
            thrustpolicy,
            d_segmentIds.begin(), 
            d_segmentIds.end(), 
            0
        );

        thrust::scatter_if(
            thrustpolicy,
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(0) + numSegments,
            d_offsets,
            thrust::make_transform_iterator(
                thrust::make_counting_iterator<std::size_t>(0),
                [d_offsets] __host__ __device__ (std::size_t i){
                    const auto segmentsize = d_offsets[i+1] - d_offsets[i];
                    return segmentsize > 0;
                }
            ),
            d_segmentIds.begin()
        );

        thrust::inclusive_scan(
            thrustpolicy,
            d_segmentIds.begin(),
            d_segmentIds.end(),
            d_segmentIds.begin(),
            thrust::maximum<int>{}
        );

        auto idDataZipped = thrust::make_zip_iterator(thrust::make_tuple(
            d_segmentIds.begin(), 
            d_items_dblbuf.Current()
        ));        

        auto idDataOutZipped = thrust::make_zip_iterator(thrust::make_tuple(
            d_segmentIds_tmp.begin(), 
            d_items_dblbuf.Alternate()
        ));

        rmm::device_uvector<int> d_counts(numItems, stream, mr);

        auto [idDataOutZipped_end, d_counts_end] = thrust::reduce_by_key(
            thrustpolicy,
            idDataZipped,
            idDataZipped + numItems,
            thrust::make_constant_iterator(1),
            idDataOutZipped,
            d_counts.begin()
        );

        auto idDataZipped_end = thrust::copy_if(
            thrustpolicy,
            idDataOutZipped, 
            idDataOutZipped_end, 
            d_counts.begin(), 
            idDataZipped, 
            [minimumCount] __host__ __device__ (int count){
                return count >= minimumCount;
            }
        );

        std::size_t numTotalUniqueItems = thrust::distance(idDataZipped, idDataZipped_end);

        //d_segmentIds.begin() and d_items_dblbuf.Current() contain the unique items per segment. 
        //compute counts per segment and copy unique items to correct buffer

        if(d_items_dblbuf.Current() != d_unique_items){
            CUDACHECK(cudaMemcpyAsync(
                d_unique_items,
                d_items_dblbuf.Current(),
                sizeof(T) * numTotalUniqueItems,
                cudaMemcpyDeviceToDevice,
                stream
            ));
        }

        //cannot directly reduce into output lengths because empty result segments would be left out
        auto [ids_end, counts_end] = thrust::reduce_by_key(
            thrustpolicy,
            d_segmentIds.begin(),
            d_segmentIds.begin() + numTotalUniqueItems,
            thrust::make_constant_iterator(1),
            d_segmentIds_tmp.begin(),
            d_counts.begin()
        );

        CUDACHECK(cudaMemsetAsync(
            d_unique_lengths,
            0,
            sizeof(int) * numSegments,
            stream
        ));

        auto numNonEmptySegments = thrust::distance(d_counts.begin(), counts_end);

        thrust::scatter(
            thrustpolicy,
            d_counts.begin(),
            d_counts.begin() + numNonEmptySegments,
            d_segmentIds_tmp.begin(),
            d_unique_lengths
        );
    }
};

#endif