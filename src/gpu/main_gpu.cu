
//#define COUNT_WINDOW_HITS
//#define FIX_NSIGHT_COMPUTE_MEMORY_POOLS

#include <hpc_helpers.cuh>
#include <config.hpp>
#include <options.hpp>

#include <genome.hpp>
#include <referencewindows.hpp>
#include <windowhitstatisticcollector.hpp>
#include <sequencehelpers.hpp>
#include <chunkedreadstorageconstruction.hpp>
#include <chunkedreadstorage.hpp>
#include <contiguousreadstorage.hpp>
#include <alignmentorientation.hpp>
#include <threadpool.hpp>

#include <algorithm>
#include <vector>
#include <iostream>
#include <sstream> 
#include <mutex>
#include <thread>
#include <memory>
#include <cassert>
#include <map>
#include <string>
#include <string_view>
#include <omp.h>
#include <future>

#include <gpu/gpuminhasherconstruction.cuh>
#include <gpu/cudaerrorcheck.cuh>
#include <gpu/multigpureadstorage.cuh>
#include <gpu/sequenceconversionkernels.cuh>
#include <gpu/minhashqueryfilter.cuh>
#include <gpu/hammingdistancekernels.cuh>
#include <gpu/windowgenerationkernels.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/logging_resource_adaptor.hpp>
#include <gpu/rmm_utilities.cuh>

#include <thrust/equal.h>
#include <thrust/functional.h>
#include <thrust/scatter.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

// Complete-Striped-Smith-Waterman-Library
#include<ssw_cpp.h>
// BAM Variant Caller and Genomic Analysis
#include "cigar.hpp"
#include "constants.hpp"
#include "varianthandler.hpp"

#include <gpu/mappinghandler.cuh>

#include<gpu/mappedread.cuh>

using namespace care;



template<class T>
void printDataStructureMemoryUsage(const T& datastructure, const std::string& name){
    auto toGB = [](std::size_t bytes){
                double gb = bytes / 1024. / 1024. / 1024.0;
                return gb;
            };

    auto memInfo = datastructure.getMemoryInfo();
    
    std::cout << name << " memory usage: " << toGB(memInfo.host) << " GB on host\n";
    for(const auto& pair : memInfo.device){
        std::cout << name << " memory usage: " << toGB(pair.second) << " GB on device " << pair.first << '\n';
    }
}

struct SimilarReadIdsHost{
    int numSequences{};
    int totalNumReadIds{};
    std::vector<read_number> h_readIds{};
    std::vector<int> h_numReadIdsPerSequence{};
    std::vector<int> h_numReadIdsPerSequencePrefixSum{};
    
};

struct SimilarReadIdsDevice{
    SimilarReadIdsDevice() : SimilarReadIdsDevice(0, cudaStreamPerThread){
        CUDACHECK(cudaStreamSynchronize(cudaStreamPerThread));
    }

    SimilarReadIdsDevice(
        int size, 
        cudaStream_t stream, 
        rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()
    ) : 
        numSequences(size),
        totalNumReadIds(0),
        d_readIds(0, stream, mr),
        d_numReadIdsPerSequence(size, stream, mr),
        d_numReadIdsPerSequencePrefixSum(size + 1, stream, mr)
        {
    }

    SimilarReadIdsHost copyToHost(cudaStream_t stream) const{
        SimilarReadIdsHost result;
        result.numSequences = numSequences;
        result.totalNumReadIds = totalNumReadIds;
        result.h_readIds.resize(totalNumReadIds);
        result.h_numReadIdsPerSequence.resize(numSequences);
        result.h_numReadIdsPerSequencePrefixSum.resize(numSequences + 1);
        
        CUDACHECK(cudaMemcpyAsync(
            result.h_readIds.data(),
            d_readIds.data(),
            sizeof(read_number) * totalNumReadIds,
            D2H,
            stream
        ));
        CUDACHECK(cudaMemcpyAsync(
            result.h_numReadIdsPerSequence.data(),
            d_numReadIdsPerSequence.data(),
            sizeof(int) * numSequences,
            D2H,
            stream
        ));
        CUDACHECK(cudaMemcpyAsync(
            result.h_numReadIdsPerSequencePrefixSum.data(),
            d_numReadIdsPerSequencePrefixSum.data(),
            sizeof(int) * (numSequences + 1),
            D2H,
            stream
        ));
        
        CUDACHECK(cudaStreamSynchronize(stream));
        return result;
    }

    int numSequences;
    int totalNumReadIds;
    rmm::device_uvector<read_number> d_readIds;
    rmm::device_uvector<int> d_numReadIdsPerSequence;
    rmm::device_uvector<int> d_numReadIdsPerSequencePrefixSum;

};

//query hashtables
/*This is a function called findReadIdsOfSimilarSequences which takes several inputs and returns an object of type SimilarReadIdsDevice.

The inputs to the function are:

    gpuMinhasher: a pointer to a gpu::GpuMinhasher object
    minhashHandle: a handle to a Minhasher object
    d_encodedSequences: a pointer to the encoded sequences in device memory
    encodedSequencePitchInInts: the pitch (in number of ints) of the encoded sequence array
    d_sequenceLengths: a pointer to an array of sequence lengths in device memory
    numSequences: the number of sequences
    programOptions: an object of type ProgramOptions
    stream: a CUDA stream to use for the kernel launches
    mr: a memory resource to use for device allocations (default is rmm::mr::get_current_device_resource())

The function computes the number of similar sequences for each input sequence and stores the result in an object of type SimilarReadIdsDevice. The SimilarReadIdsDevice object contains several member variables, including d_numReadIdsPerSequence, which stores the number of similar sequences for each input sequence, and d_readIds, which stores the IDs of the similar sequences.

The function first calls the determineNumValues function of the gpuMinhasher object to compute the number of similar sequences for each input sequence. It then allocates device memory for d_readIds and checks if the total number of similar sequences is 0. If the total number of similar sequences is 0, it sets the d_numReadIdsPerSequence and d_numReadIdsPerSequencePrefixSum arrays to 0. Otherwise, it calls the retrieveValues function of the gpuMinhasher object to retrieve the IDs of the similar sequences. It then uses the cub::DoubleBuffer class to perform prefix sum operations to compute the prefix sum of the number of similar sequences per input sequence. Finally, it copies the total number of similar sequences to the host and returns the SimilarReadIdsDevice object.*/
SimilarReadIdsDevice findReadIdsOfSimilarSequences(
    const gpu::GpuMinhasher* gpuMinhasher,
    MinhasherHandle& minhashHandle,
    const unsigned int* d_encodedSequences,
    std::size_t encodedSequencePitchInInts,
    const int* d_sequenceLengths,
    int numSequences,
    const ProgramOptions& programOptions,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()
){
    SimilarReadIdsDevice result(numSequences, stream, mr);

    gpuMinhasher->determineNumValues(
        minhashHandle,
        d_encodedSequences,
        encodedSequencePitchInInts,
        d_sequenceLengths,
        numSequences,
        result.d_numReadIdsPerSequence.data(),
        result.totalNumReadIds,
        stream,
        mr
    );
    CUDACHECK(cudaStreamSynchronize(stream));

    result.d_readIds.resize(result.totalNumReadIds, stream);

    if(result.totalNumReadIds == 0){
        CUDACHECK(cudaMemsetAsync(
            result.d_numReadIdsPerSequence.data(), 
            0, 
            sizeof(int) * numSequences, 
            stream
        ));
        CUDACHECK(cudaMemsetAsync(
            result.d_numReadIdsPerSequencePrefixSum.data(), 
            0, 
            sizeof(int) * (1 + numSequences), 
            stream
        ));
    }else{
        gpuMinhasher->retrieveValues(
            minhashHandle,
            numSequences,                
            result.totalNumReadIds,
            result.d_readIds.data(),
            result.d_numReadIdsPerSequence.data(),
            result.d_numReadIdsPerSequencePrefixSum.data(),
            stream,
            mr
        );

        rmm::device_uvector<read_number> d_valuesTmp(result.totalNumReadIds, stream, mr);
        rmm::device_uvector<int> d_numValuesPerSequenceTmp(numSequences, stream, mr);
        rmm::device_uvector<int> d_numValuesPerSequencePrefixSumTmp(1 + numSequences, stream, mr);

        cub::DoubleBuffer<read_number> d_items{result.d_readIds.data(), d_valuesTmp.data()};
        cub::DoubleBuffer<int> d_numItemsPerSegment{result.d_numReadIdsPerSequence.data(), d_numValuesPerSequenceTmp.data()};
        cub::DoubleBuffer<int> d_numItemsPerSegmentPrefixSum{result.d_numReadIdsPerSequencePrefixSum.data(), d_numValuesPerSequencePrefixSumTmp.data()};

        if(programOptions.minTableHits > 1){
            gpu::GpuMinhashQueryFilter::keepDistinctByFrequency(
                programOptions.minTableHits,
                d_items,
                d_numItemsPerSegment,
                d_numItemsPerSegmentPrefixSum,
                numSequences,
                result.totalNumReadIds,
                stream,
                mr
            );              
        }else{
            gpu::GpuMinhashQueryFilter::keepDistinct(
                d_items,
                d_numItemsPerSegment,
                d_numItemsPerSegmentPrefixSum,
                numSequences,
                result.totalNumReadIds,
                stream,
                mr
            );
        }
        
        if(d_items.Current() != result.d_readIds.data()){
            std::swap(result.d_readIds, d_valuesTmp);
        }
        if(d_numItemsPerSegment.Current() != result.d_numReadIdsPerSequence.data()){
            std::swap(result.d_numReadIdsPerSequence, d_numValuesPerSequenceTmp);
        }
        if(d_numItemsPerSegmentPrefixSum.Current() != result.d_numReadIdsPerSequencePrefixSum.data()){
            std::swap(result.d_numReadIdsPerSequencePrefixSum, d_numValuesPerSequencePrefixSumTmp);
        }  

        CUDACHECK(cudaMemcpyAsync(
            &result.totalNumReadIds,
            result.d_numReadIdsPerSequencePrefixSum.data() + numSequences,
            sizeof(int),
            D2H,
            stream
        ));
    }

    CUDACHECK(cudaStreamSynchronize(stream));

    return result;
}

/*
    input:
    d_segmentSizes {2,3,4}
    d_segmentOffsets {0,2,5}
    numSegments 3
    numElements 9
    output:
    {0,0,1,1,1,2,2,2,2}
*/
rmm::device_uvector<int> getSegmentIdsPerElement(
    const int* d_segmentSizes, 
    const int* d_segmentOffsets, 
    int numSegments, 
    int numElements,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()
){
    auto thrustpolicy = rmm::exec_policy_nosync(stream, mr);

    rmm::device_uvector<int> result(numElements, stream, mr);

    CUDACHECK(cudaMemsetAsync(result.data(), 0, sizeof(int) * numElements, stream));
    //must not scatter for empty segments
    thrust::scatter_if(
        thrustpolicy,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(0) + numSegments, 
        d_segmentOffsets,
        thrust::make_transform_iterator(
            d_segmentSizes, 
            [] __host__ __device__ (int i){return i > 0;}
        ),
        result.data()
    );

    thrust::inclusive_scan(
        thrustpolicy,
        result.data(), 
        result.data() + numElements, 
        result.data(), 
        thrust::maximum<int>{}
    );

    return result;
}


struct ShiftedHammingDistanceResultHost{
    std::vector<int> h_alignment_shifts;
    std::vector<int> h_alignment_hammingdistance;
    std::vector<AlignmentOrientation> h_alignment_orientation;
};

struct ShiftedHammingDistanceResultDevice{
    ShiftedHammingDistanceResultDevice() : ShiftedHammingDistanceResultDevice(0, cudaStreamPerThread){
        CUDACHECK(cudaStreamSynchronize(cudaStreamPerThread));
    }

    ShiftedHammingDistanceResultDevice(
        int size, 
        cudaStream_t stream, 
        rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()
    ) : d_alignment_shifts(size, stream, mr),
        d_alignment_hammingdistance(size, stream, mr),
        d_alignment_orientation(size, stream, mr){
    }

    ShiftedHammingDistanceResultHost copyToHost(cudaStream_t stream) const{
        const int size = d_alignment_hammingdistance.size();

        ShiftedHammingDistanceResultHost result;
        result.h_alignment_shifts.resize(size);
        result.h_alignment_hammingdistance.resize(size);
        result.h_alignment_orientation.resize(size);

        CUDACHECK(cudaMemcpyAsync(
            result.h_alignment_shifts.data(), 
            d_alignment_shifts.data(),
            sizeof(int) * size,
            D2H,
            stream
        ));
        CUDACHECK(cudaMemcpyAsync(
            result.h_alignment_hammingdistance.data(), 
            d_alignment_hammingdistance.data(),
            sizeof(int) * size,
            D2H,
            stream
        ));
        CUDACHECK(cudaMemcpyAsync(
            result.h_alignment_orientation.data(), 
            d_alignment_orientation.data(),
            sizeof(AlignmentOrientation) * size,
            D2H,
            stream
        ));
        CUDACHECK(cudaStreamSynchronize(stream));
        return result;
    }

    rmm::device_uvector<int> d_alignment_shifts;
    rmm::device_uvector<int> d_alignment_hammingdistance;
    rmm::device_uvector<AlignmentOrientation> d_alignment_orientation;
};


//candidate i is aligned to anchor i. there are numSequences anchors and numSequences candidates
ShiftedHammingDistanceResultDevice computeShiftedHammingDistancesFullOverlap(
    int numSequences,
    const unsigned int* d_anchors2bit,
    const int* d_anchorLengths,
    const int maxAnchorLength,
    const unsigned int* d_candidates2bit,
    const int* d_candidateLengths,
    const int maxCandidateLength,
    std::size_t encodedSequencePitchInIntsAnchors,
    std::size_t encodedSequencePitchInIntsCandidates,
    float maxHammingRatioIncl,
    cudaStream_t stream, 
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()
){
    const int numAnchors = numSequences;
    const int numCandidates = numSequences;

    ShiftedHammingDistanceResultDevice result(numCandidates, stream, mr);

    callShiftedHammingDistanceWithFullOverlapKernelSmem1(
        result.d_alignment_shifts.data(),
        result.d_alignment_hammingdistance.data(),
        result.d_alignment_orientation.data(),
        d_anchors2bit,
        d_anchorLengths,
        encodedSequencePitchInIntsAnchors,
        d_candidates2bit,
        d_candidateLengths,
        encodedSequencePitchInIntsCandidates,
        numAnchors,
        numCandidates,
        maxHammingRatioIncl, //allow only <= (candidateLength * maxHammingRatioIncl) mismatches
        maxAnchorLength,
        maxCandidateLength,
        stream,
        mr
    );

    return result;
}




struct WindowBatchProcessor{
    const gpu::GpuReadStorage* gpuReadStorage;
    const gpu::GpuMinhasher* gpuMinhasher;
    const ProgramOptions* programOptions;
    MappedRead* results;
  //  MappedRead* resultsRC;
    WindowHitStatisticCollector* windowHitStatsAfterHashing;
    WindowHitStatisticCollector* windowHitStatsAfterHammingDistance;

    MinhasherHandle minhashHandle;
    ReadStorageHandle readstorageHandle;

    WindowBatchProcessor(
        const gpu::GpuReadStorage* gpuReadStorage_,
        const gpu::GpuMinhasher* gpuMinhasher_,
        const ProgramOptions* programOptions_,
        MappedRead* results_,
      //  MappedRead* resultsRC_,
        WindowHitStatisticCollector* windowHitStatsAfterHashing_,
        WindowHitStatisticCollector* windowHitStatsAfterHammingDistance_
    ) : gpuReadStorage(gpuReadStorage_),
        gpuMinhasher(gpuMinhasher_),
        programOptions(programOptions_),
        results(results_),
    //    resultsRC(resultsRC_),
        windowHitStatsAfterHashing(windowHitStatsAfterHashing_),
        windowHitStatsAfterHammingDistance(windowHitStatsAfterHammingDistance_),
        minhashHandle(gpuMinhasher->makeMinhasherHandle()),
        readstorageHandle(gpuReadStorage->makeHandle())
    {

    }

    ~WindowBatchProcessor(){
        gpuMinhasher->destroyHandle(minhashHandle);
        gpuReadStorage->destroyHandle(readstorageHandle);
    }



    void operator()(const Genome::BatchOfWindows& batch
   // , bool ReverseComplementBatch
   )
    {

        auto* mr = rmm::mr::get_current_device_resource();
        cudaStream_t stream = cudaStreamPerThread;

        const std::size_t decodedWindowPitchInBytes = SDIV(batch.maxWindowSize, 32) * 32;        
        std::size_t encodedWindowPitchInInts = SequenceHelpers::getEncodedNumInts2Bit(batch.maxWindowSize);
        std::size_t encodedReadPitchInInts = SequenceHelpers::getEncodedNumInts2Bit(gpuReadStorage->getSequenceLengthUpperBound());

        //Next step: transfer windows to gpu
        nvtx::push_range("transfer windows to gpu", 0);

        rmm::device_uvector<int> d_windowLengths(batch.numWindows, stream, mr);
        CUDACHECK(cudaMemcpyAsync(
            d_windowLengths.data(),
            batch.windowLengths.data(),
            sizeof(int) * batch.numWindows,
            H2D,
            stream
        ));
        
        rmm::device_uvector<char> d_windowsDecoded(decodedWindowPitchInBytes * batch.numWindows, stream, mr);
        std::vector<char> h_windowsDecoded(decodedWindowPitchInBytes * batch.numWindows);    

        for(int w = 0; w < batch.numWindows; w++){
            std::copy(
                batch.windowsDecoded[w].begin(), 
                batch.windowsDecoded[w].end(),
                h_windowsDecoded.data() + w * decodedWindowPitchInBytes
            );
      
        }

        CUDACHECK(cudaMemcpyAsync(
            d_windowsDecoded.data(),
            h_windowsDecoded.data(),
            sizeof(char) * decodedWindowPitchInBytes * batch.numWindows,
            H2D,
            stream
        ));

        nvtx::pop_range();
        
        //Next step: 2bit encode windows, and query hash tables
        nvtx::push_range("find candidate reads for windows", 1);
        
        rmm::device_uvector<unsigned int> d_windowsEncoded2Bit(batch.numWindows * encodedWindowPitchInInts, stream, mr);
        
        callEncodeSequencesTo2BitKernel(
            d_windowsEncoded2Bit.data(),
            d_windowsDecoded.data(),
            d_windowLengths.data(),
            decodedWindowPitchInBytes,
            encodedWindowPitchInInts,
            batch.numWindows,
            8, //cg::groupsize per sequence
            stream
        );
       

        SimilarReadIdsDevice similarReadIdsOfWindowsDevice = findReadIdsOfSimilarSequences(
            gpuMinhasher,
            minhashHandle,
            d_windowsEncoded2Bit.data(),
            encodedWindowPitchInInts,
            d_windowLengths.data(),
            batch.numWindows,
            *programOptions,
            stream,
            mr
        );
        
        nvtx::pop_range();

        if( similarReadIdsOfWindowsDevice.totalNumReadIds == 0 ){
            //none of the windows in the batch matched a read
            return;
        }

        SimilarReadIdsHost hostIds = similarReadIdsOfWindowsDevice.copyToHost(stream);
        
        #ifdef COUNT_WINDOW_HITS
        nvtx::push_range("window statistics", 9);
        //count hits
        {
            for(int w = 0; w < batch.numWindows; w++){
                const int offset = hostIds.h_numReadIdsPerSequencePrefixSum[w];
                const int num = hostIds.h_numReadIdsPerSequence[w];
                const int windowChromosomeId = batch.chromosomeIds[w];
                const int windowId = batch.windowIds[w];

                windowHitStatsAfterHashing->addHits(
                    windowChromosomeId, 
                    windowId,
                    hostIds.h_readIds.begin() + offset,
                    hostIds.h_readIds.begin() + offset + num
                );
            }                    
        }
        nvtx::pop_range();
        #endif

        //next step: gather the corresponding sequences of similar read ids
        nvtx::push_range("gather candidate reads", 2);

        rmm::device_uvector<int> d_readLengths(similarReadIdsOfWindowsDevice.totalNumReadIds, stream, mr);
        

        rmm::device_uvector<unsigned int> d_readsEncoded2Bit(
            similarReadIdsOfWindowsDevice.totalNumReadIds * encodedReadPitchInInts, 
            stream, 
            mr
        );
       
        

       
        gpuReadStorage->gatherSequenceLengths(
            readstorageHandle,
            d_readLengths.data(),
            similarReadIdsOfWindowsDevice.d_readIds.data(),
            similarReadIdsOfWindowsDevice.totalNumReadIds,            
            stream
        );
        gpuReadStorage->gatherSequences(
            readstorageHandle,
            d_readsEncoded2Bit.data(),
            encodedReadPitchInInts,
            makeAsyncConstBufferWrapper(hostIds.h_readIds.data()),
            similarReadIdsOfWindowsDevice.d_readIds.data(),
            similarReadIdsOfWindowsDevice.totalNumReadIds,
            stream,
            mr
        );

       
        

        nvtx::pop_range();
        
        //next step: get windows extended to left / right by 50% read length, transfer them to gpu, encode to 2bit

        nvtx::push_range("get extended windows", 3);

        #if 0
        //find maximum read length in batch
        const int maxReadLength = thrust::reduce(
            rmm::exec_policy_nosync(stream, mr),
            d_readLengths.data(),
            d_readLengths.data() + similarReadIdsOfWindowsDevice.totalNumReadIds,
            0,
            [] __device__ (int a, int b){
                return max(a,b);
            }
        );
        #else
        //use global maximum of all reads in dataset
        const int maxReadLength = gpuReadStorage->getSequenceLengthUpperBound();
        
        #endif
       
        const int maxExtendedWindowLength = batch.maxWindowSize + maxReadLength;  
        const std::size_t encodedExtendedWindowPitchInInts = SequenceHelpers::getEncodedNumInts2Bit(maxExtendedWindowLength);    

        assert(std::all_of(batch.chromosomeIds.begin(), batch.chromosomeIds.end(), [&](int id){ return id == batch.chromosomeIds[0];}));
        assert(std::is_sorted(batch.positions.begin(), batch.positions.end()));


        Genome::Section genomicSection = batch.genome->getSectionOfGenome(
            batch.chromosomeIds[0], 
            batch.positions[0] - maxReadLength / 2,
            batch.positions[batch.numWindows-1] + programOptions->windowSize + maxReadLength / 2
        );

        rmm::device_uvector<char> d_genomicSection(genomicSection.sequence.size(), stream, mr);
      
        CUDACHECK(cudaMemcpyAsync(
            d_genomicSection.data(),
            genomicSection.sequence.data(),
            sizeof(char) * genomicSection.sequence.size(),
            H2D,
            stream
        ));
        
        rmm::device_uvector<int> d_windowPositions(batch.numWindows, stream, mr);
        CUDACHECK(cudaMemcpyAsync(
            d_windowPositions.data(),
            batch.positions.data(),
            sizeof(int) * batch.numWindows,
            H2D,
            stream
        ));
       

        rmm::device_uvector<int> d_extensionsLeft(similarReadIdsOfWindowsDevice.totalNumReadIds, stream, mr);
        rmm::device_uvector<int> d_extensionsRight(similarReadIdsOfWindowsDevice.totalNumReadIds, stream, mr);
        rmm::device_uvector<int> d_extendedWindowLengths(similarReadIdsOfWindowsDevice.totalNumReadIds, stream, mr);
        rmm::device_uvector<unsigned int> d_extendedwindowsEncoded2Bit(similarReadIdsOfWindowsDevice.totalNumReadIds * encodedExtendedWindowPitchInInts, stream, mr);

        
        //this is just for testing purpose to set the padding bytes to 0
        CUDACHECK(cudaMemsetAsync(
            d_extendedwindowsEncoded2Bit.data(),
            0, 
            sizeof(unsigned int) * d_extendedwindowsEncoded2Bit.size(),
            stream
        ));

        rmm::device_uvector<int> d_segmentIds = getSegmentIdsPerElement(
            similarReadIdsOfWindowsDevice.d_numReadIdsPerSequence.data(), 
            similarReadIdsOfWindowsDevice.d_numReadIdsPerSequencePrefixSum.data(), 
            batch.numWindows, 
            similarReadIdsOfWindowsDevice.totalNumReadIds,
            stream,
            mr
        );
       
        callGenerateExtendedWindows2BitKernel(
            d_genomicSection.data(), 
            genomicSection.begin, 
            genomicSection.end, 
            d_readLengths.data(),
            similarReadIdsOfWindowsDevice.totalNumReadIds,
            d_segmentIds.data(),
            programOptions->windowSize,
            d_windowPositions.data(),
            maxExtendedWindowLength,
            d_extendedwindowsEncoded2Bit.data(),
            d_extendedWindowLengths.data(),
            encodedExtendedWindowPitchInInts,
            d_extensionsLeft.data(),
            d_extensionsRight.data(),
            stream
        );      
       
        nvtx::pop_range();

        nvtx::push_range("align reads to extended windows", 4);

        //align reads to extended windows
        ShiftedHammingDistanceResultDevice shdResult = computeShiftedHammingDistancesFullOverlap(
            similarReadIdsOfWindowsDevice.totalNumReadIds,
            d_extendedwindowsEncoded2Bit.data(),
            d_extendedWindowLengths.data(),
            maxExtendedWindowLength,
            d_readsEncoded2Bit.data(),
            d_readLengths.data(),
            maxReadLength,
            encodedExtendedWindowPitchInInts,
            encodedReadPitchInInts,
            programOptions->maxHammingPercent,
            stream, 
            mr
        );
        
        nvtx::pop_range();
        
        ShiftedHammingDistanceResultHost shdResultHost = shdResult.copyToHost(stream);
       

        std::vector<int> h_extensionsLeft(similarReadIdsOfWindowsDevice.totalNumReadIds);

        CUDACHECK(cudaMemcpyAsync(
            h_extensionsLeft.data(),
            d_extensionsLeft.data(),
            sizeof(int) * similarReadIdsOfWindowsDevice.totalNumReadIds,
            D2H,
            stream
        ));

        

        CUDACHECK(cudaStreamSynchronize(stream));

        // std::vector<int> h_readLengths(similarReadIdsOfWindowsDevice.totalNumReadIds);
        // CUDACHECK(cudaMemcpyAsync(
        //     h_readLengths.data(),
        //     d_readLengths.data(),
        //     sizeof(int) * similarReadIdsOfWindowsDevice.totalNumReadIds,
        //     D2H,
        //     stream
        // ));
        // CUDACHECK(cudaStreamSynchronize(stream));

        // std::vector<unsigned int > h_readsEncoded2Bit(encodedReadPitchInInts * similarReadIdsOfWindowsDevice.totalNumReadIds);
        // CUDACHECK(cudaMemcpyAsync(
        //     h_readsEncoded2Bit.data(),
        //     d_readsEncoded2Bit.data(),
        //     sizeof(unsigned int) * encodedReadPitchInInts * similarReadIdsOfWindowsDevice.totalNumReadIds,
        //     D2H,
        //     stream
        // ));
        // CUDACHECK(cudaStreamSynchronize(stream));

        // std::vector<std::string> readsStrings(similarReadIdsOfWindowsDevice.totalNumReadIds);
        // for(int i = 0; i < similarReadIdsOfWindowsDevice.totalNumReadIds; i++){
        //     readsStrings[i] = SequenceHelpers::get2BitString(h_readsEncoded2Bit.data() + i * encodedReadPitchInInts, h_readLengths[0]);
        // }

        //update results

        nvtx::push_range("update host results", 5);

        for(int wid = 0, offset = 0; wid < batch.numWindows; wid++){
            
            const int numReadsOfWindow = hostIds.h_numReadIdsPerSequence[wid];

            for(int r = 0; r < numReadsOfWindow; r++){
                MappedRead currentResult;
                currentResult.orientation = shdResultHost.h_alignment_orientation[offset + r];
                currentResult.chromosomeId = batch.chromosomeIds[wid];
                currentResult.position = batch.positions[wid];
                currentResult.hammingDistance = shdResultHost.h_alignment_hammingdistance[offset + r];
                const int shift = shdResultHost.h_alignment_shifts[offset + r];

                //translate shift in extended window to shift in original window
                currentResult.shift = shift - h_extensionsLeft[offset + r];

                    //i have to define currentBestResult and then i redefine it depending if RC or not
                    auto& currentBestResult = results[hostIds.h_readIds[offset + r]];
            //    if(ReverseComplementBatch){
            //        auto& currentBestResult = resultsRC[hostIds.h_readIds[offset + r]];
            //    }
            //                auto& currentBestResult = results[hostIds.h_readIds[offset + r]];

                //if the computed alignment is "good"
                if(currentResult.orientation != AlignmentOrientation::None){

                    //if it's the first window for this read, accept alignment
                    if(currentBestResult.orientation == AlignmentOrientation::None){
                        currentBestResult = currentResult;
                    
                    }else{
                        //else accept alignment if hamming distance is smaller
                        if(currentBestResult.hammingDistance > currentResult.hammingDistance){
                            currentBestResult = currentResult;
                        
                        }
                    }      
                    
                }else{
                    // read does not align to window well
                    //std::cout<<"read does not align to window well: "<<static_cast<int>(currentResult.orientation)<<"\n";
                }
            }

            offset += numReadsOfWindow;
        }
        nvtx::pop_range();

        #ifdef COUNT_WINDOW_HITS
        nvtx::push_range("window statistics", 9);
        //count hits
        {
            for(int wid = 0, offset = 0; wid < batch.numWindows; wid++){
                const int numReadsOfWindow = hostIds.h_numReadIdsPerSequence[wid];
                std::vector<read_number> idsOfReadsWithGoodAlignment;

                for(int r = 0; r < numReadsOfWindow; r++){
                    if(shdResultHost.h_alignment_orientation[offset + r] != AlignmentOrientation::None){
                        idsOfReadsWithGoodAlignment.push_back(hostIds.h_readIds[offset + r]);
                    }
                }
            
                const int windowChromosomeId = batch.chromosomeIds[wid];
                const int windowId = batch.windowIds[wid];

                windowHitStatsAfterHammingDistance->addHits(
                    windowChromosomeId, 
                    windowId,
                    idsOfReadsWithGoodAlignment.begin(),
                    idsOfReadsWithGoodAlignment.end()
                );

                offset += numReadsOfWindow;
            }              
        }
        nvtx::pop_range();
        #endif
   //     arschlochgottlelag.print();
    }

};


void performMappingGpu(const ProgramOptions& programOptions){

    if(programOptions.deviceIds.size() == 0){
        std::cout << "No device ids found. Abort!" << std::endl;
        return;
    }

    CUDACHECK(cudaSetDevice(programOptions.deviceIds[0]));

    helpers::PeerAccessDebug peerAccess(programOptions.deviceIds, true);
    peerAccess.enableAllPeerAccesses();

    //Set up memory pool
    #ifdef FIX_NSIGHT_COMPUTE_MEMORY_POOLS
    //cudaErrorCallRequiresNewerDriver API call is not supported in the installed CUDA driver
    //nsight compute 2019.5 (latest for Pascal) cannot use memory pools. Use cudaMalloc instead.
    std::vector<std::unique_ptr<rmm::mr::cuda_memory_resource>> rmmCudaResources;
    for(auto id : programOptions.deviceIds){
        cub::SwitchDevice sd(id);

        auto resource = std::make_unique<rmm::mr::cuda_memory_resource>();
        rmm::mr::set_per_device_resource(rmm::cuda_device_id(id), resource.get());

        rmmCudaResources.push_back(std::move(resource));
    }

    auto trimPools = [&](){

    };
    #else
    std::vector<std::unique_ptr<rmm::mr::cuda_async_memory_resource>> rmmCudaAsyncResources;
    for(auto id : programOptions.deviceIds){
        cub::SwitchDevice sd(id);

        auto resource = std::make_unique<rmm::mr::cuda_async_memory_resource>(0);
        rmm::mr::set_per_device_resource(rmm::cuda_device_id(id), resource.get());

        for(auto otherId : programOptions.deviceIds){
            if(otherId != id){
                if(peerAccess.canAccessPeer(id, otherId)){
                    cudaMemAccessDesc accessDesc = {};
                    accessDesc.location.type = cudaMemLocationTypeDevice;
                    accessDesc.location.id = otherId;
                    accessDesc.flags = cudaMemAccessFlagsProtReadWrite;

                    CUDACHECK(cudaMemPoolSetAccess(resource->pool_handle(), &accessDesc, 1));
                }
            }
        }

        CUDACHECK(cudaDeviceSetMemPool(id, resource->pool_handle()));
        rmmCudaAsyncResources.push_back(std::move(resource));
    }

    auto trimPools = [&](){
        for(size_t i = 0; i < rmmCudaAsyncResources.size(); i++){
            cub::SwitchDevice sd(programOptions.deviceIds[i]);
            CUDACHECK(cudaDeviceSynchronize());
            CUDACHECK(cudaMemPoolTrimTo(rmmCudaAsyncResources[i]->pool_handle(), 0));
        }
    };
    #endif

    helpers::CpuTimer step1Timer("STEP1");

    std::cout << "STEP 1: Database construction" << std::endl;

    helpers::CpuTimer buildReadStorageTimer("build_readstorage");

    std::unique_ptr<ChunkedReadStorage> cpuReadStorage = constructChunkedReadStorageFromFiles(programOptions);

    buildReadStorageTimer.print();

    std::cout << "Determined the following read properties:\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Total number of reads: " << cpuReadStorage->getNumberOfReads() << "\n";
    std::cout << "Minimum sequence length: " << cpuReadStorage->getSequenceLengthLowerBound() << "\n";
    std::cout << "Maximum sequence length: " << cpuReadStorage->getSequenceLengthUpperBound() << "\n";
    std::cout << "----------------------------------------\n";

    if(programOptions.save_binary_reads_to != ""){
        std::cout << "Saving reads to file " << programOptions.save_binary_reads_to << std::endl;
        helpers::CpuTimer timer("save_to_file");
        cpuReadStorage->saveToFile(programOptions.save_binary_reads_to);
        timer.print();
        std::cout << "Saved reads" << std::endl;
    }
    
    //std::cout << "Reads with ambiguous bases: " << cpuReadStorage->getNumberOfReadsWithN() << std::endl;        
   
    trimPools();

    std::vector<std::size_t> gpumemorylimits(programOptions.deviceIds.size(), 0);
    gpu::MultiGpuReadStorage multiGpuReadStorage(
        *cpuReadStorage, 
        programOptions.deviceIds,
        gpumemorylimits,
        0,
        programOptions.qualityScoreBits
    );

    helpers::CpuTimer buildMinhasherTimer("build_minhasher");


    auto minhasherAndType = gpu::constructGpuMinhasherFromGpuReadStorage(
        programOptions,
        multiGpuReadStorage,
        gpu::GpuMinhasherType::Multi
    );


    gpu::GpuMinhasher* gpuMinhasher = minhasherAndType.first.get();

    buildMinhasherTimer.print();

    trimPools();

    std::cout << "Using minhasher type: " << to_string(minhasherAndType.second) << "\n";
    std::cout << "GpuMinhasher can use " << gpuMinhasher->getNumberOfMaps() << " maps\n";

    if(gpuMinhasher->getNumberOfMaps() <= 0){
        std::cout << "Cannot construct a single gpu hashtable. Abort!" << std::endl;
        return;
    }

    if(programOptions.mustUseAllHashfunctions 
        && programOptions.numHashFunctions != gpuMinhasher->getNumberOfMaps()){
        std::cout << "Cannot use specified number of hash functions (" 
            << programOptions.numHashFunctions <<")\n";
        std::cout << "Abort!\n";
        return;
    }

    if(programOptions.save_hashtables_to != "" && gpuMinhasher->canWriteToStream()) {
        std::cout << "Saving minhasher to file " << programOptions.save_hashtables_to << std::endl;
        std::ofstream os(programOptions.save_hashtables_to);
        assert((bool)os);
        helpers::CpuTimer timer("save_to_file");
        gpuMinhasher->writeToStream(os);
        timer.print();

        std::cout << "Saved minhasher" << std::endl;
    }

    printDataStructureMemoryUsage(*gpuMinhasher, "hash tables");

    step1Timer.print();
    //TODO create repeat index here...
    
    //... .
    std::cout << "Loading genome\n";
    helpers::CpuTimer genometimer("genometimer");
    Genome genome(programOptions.genomefile);
    Genome genomeRC(genome);
    genometimer.print();
//    genomeRC.printInfo();
//    genome.printInfo();
    std::cout << "Loading finished\n";

    //After minhasher is constructed, remaining gpu memory can be used to store reads

    std::fill(gpumemorylimits.begin(), gpumemorylimits.end(), 0);
    for(int i = 0; i < int(programOptions.deviceIds.size()); i++){
        std::size_t total = 0;
        cudaMemGetInfo(&gpumemorylimits[i], &total);

        std::size_t safety = 1 << 30; //leave 1 GB for correction algorithm
        if(gpumemorylimits[i] > safety){
            gpumemorylimits[i] -= safety;
        }else{
            gpumemorylimits[i] = 0;
        }
    }

    //TODO account for genome
    std::size_t memoryLimitHost = programOptions.memoryTotalLimit 
        - cpuReadStorage->getMemoryInfo().host
        - gpuMinhasher->getMemoryInfo().host;

    helpers::CpuTimer cpugputimer("cpu->gpu reads");
    multiGpuReadStorage.rebuild(
        *cpuReadStorage,
        programOptions.deviceIds, 
        gpumemorylimits,
        memoryLimitHost,
        programOptions.qualityScoreBits
    );
    cpugputimer.print();

    printDataStructureMemoryUsage(multiGpuReadStorage, "reads");

    // if(multiGpuReadStorage.isStandalone()){
    //     cpuReadStorage.reset();
    // }

    gpu::GpuReadStorage* gpuReadStorage = &multiGpuReadStorage;
    WindowHitStatisticCollector* windowHitStatsAfterHashingPtr = nullptr;
    WindowHitStatisticCollector* windowHitStatsAfterHammingDistancePtr = nullptr;
    #ifdef COUNT_WINDOW_HITS
    
    std::string readwindowsfilename = "/home/fekallen/storage/datasets/artpairedelegans/elegans30cov_1000_30_readwindows_w";
    readwindowsfilename += std::to_string(programOptions.windowSize) + "_k" + std::to_string(programOptions.kmerlength) + ".txt";
    std::shared_ptr<ReferenceWindows> refWindowsPerRead = std::make_shared<ReferenceWindows>(
        ReferenceWindows::fromReferenceWindowsFile(
            genome, 
            readwindowsfilename, 
            programOptions.kmerlength, 
            programOptions.windowSize
        )
    );
    assert(refWindowsPerRead->size() == gpuReadStorage->getNumberOfReads());

    WindowHitStatisticCollector windowHitStatsAfterHashing(refWindowsPerRead);
    WindowHitStatisticCollector windowHitStatsAfterHammingDistance(refWindowsPerRead);
    windowHitStatsAfterHashingPtr = &windowHitStatsAfterHashing;
    windowHitStatsAfterHammingDistancePtr = &windowHitStatsAfterHammingDistance;
    #endif

    #ifdef FAKEGPUMINHASHER_USE_OMP_FOR_QUERY
    omp_set_num_threads(std::max(1, programOptions.threads - 1));
    #endif



    std::cout << "Processing...\n";

    helpers::CpuTimer timerprocessgenome("process genome");

    const std::size_t totalWindowCount = genome.getTotalNumWindows(programOptions.kmerlength, programOptions.windowSize);

    

    //results for 3n genome
    std::vector<MappedRead> results(multiGpuReadStorage.getNumberOfReads());
    //results for rc 3n genome
   // std::vector<MappedRead> resultsRC(multiGpuReadStorage.getNumberOfReads());

ThreadPool threadPool(programOptions.threads);//out of VRAM if i use the 2 threads as intended. -->  Back to single threaded
       ThreadPool::ParallelForHandle pforHandle;
       
        auto mapfk=[&](auto begin, auto end, int /*threadid*/){
           // std::cout<<"i am doing my job!\n";
                for(auto i=begin; i< end; i++){
                std::size_t processedWindowCount = 0;
                std::size_t processedWindowCountProgress = 0;

                    WindowBatchProcessor windowBatchProcessor(
                    gpuReadStorage,
                    gpuMinhasher,
                    &programOptions,
                    results.data(),
                   // resultsRC.data(),
                    windowHitStatsAfterHashingPtr,
                    windowHitStatsAfterHammingDistancePtr
                     );

                    auto processWithProgress = [&](const Genome::BatchOfWindows& batch){
                            windowBatchProcessor(batch);

                            processedWindowCount += batch.numWindows;
                            processedWindowCountProgress += batch.numWindows;
                            if(programOptions.showProgress){
                                if(processedWindowCountProgress >= 100000){
                                    std::cout << "processed " << processedWindowCount << " / " << totalWindowCount << "\n";
                                    processedWindowCountProgress -= 100000;
                                }
                            }       
                    };

                if(i){
       // genomeRC.forEachBatchOfWindows(
       //     programOptions.kmerlength,
       //     programOptions.windowSize,
        //    programOptions.batchsize,
       //     processWithProgress
       // );
                }
                else{
        genome.forEachBatchOfWindows(
            programOptions.kmerlength,
            programOptions.windowSize,
            programOptions.batchsize,
            processWithProgress
        );
            }


                    
                }
                //std::cout << "processed " << totalWindowCount << " / " << totalWindowCount << " windows.\n";
        };
        
       
        std::size_t start=0;
        std::size_t end=2;
        threadPool.parallelFor(pforHandle, start , end ,mapfk);
   
    timerprocessgenome.print();
    std::cout<<"STEP 2: Mapping: \n";
    Mappinghandler mapper(&programOptions, &genome, &genomeRC, &results

    );

    helpers::CpuTimer timermapping("process mapping");

    mapper.go(cpuReadStorage);  

    timermapping.print();
  

    helpers::CpuTimer timerVC("process variant calling");

    std::cout<<"STEP 3: Variant Calling: \n";

    mapper.doVC();

    timerVC.print();



//--------------------------------------------------------------------------------------------------------------
    std::cout<<"done. \n";
    // output results
 //   std::cout<<"writing to file...skipped\n";
    
    #if 0
    std::ofstream outputstream(programOptions.outputfile);

    const int maximumSequenceLength = cpuReadStorage->getSequenceLengthUpperBound();

    for(std::size_t r = 0, processedResults = 0; r < cpuReadStorage->getNumberOfReads(); r++){
        const auto& result = results[r];
        read_number readId = r;

        std::vector<int> readLengths(1);
        cpuReadStorage->gatherSequenceLengths(
            readLengths.data(),
            &readId,
            1
        );

        const int encodedReadNumInts2Bit = SequenceHelpers::getEncodedNumInts2Bit(maximumSequenceLength);
        std::vector<unsigned int> encodedReads(encodedReadNumInts2Bit * 1);

        cpuReadStorage->gatherSequences(
            encodedReads.data(),
            encodedReadNumInts2Bit,
            &readId,
            1
        );


        if(result.orientation == AlignmentOrientation::ReverseComplement){
            SequenceHelpers::reverseComplementSequenceInplace2Bit(encodedReads.data(), readLengths[0]);
        }
        auto readsequence = SequenceHelpers::get2BitString(encodedReads.data(), readLengths[0]);

        if(result.orientation != AlignmentOrientation::None){

            const auto& genomesequence = genome.data[result.chromosomeId];
            const std::size_t windowlength = result.position + 
            programOptions.windowSize < genomesequence.size() ? programOptions.windowSize : genomesequence.size() 
            - result.position;
            std::string_view window(genomesequence.data() + result.position, windowlength);

            #if 0
            
            outputstream << "read " << r << ": orientation " 
            << static_cast<int>(result.orientation) 
            << ", chromosome: " << genome.names[result.chromosomeId]
            << ", position: " << result.position << ", shift: " 
            << result.shift << ", mismatches: " << result.hammingDistance << "\n";
            outputstream << "window: " << window << "\n" << "read: " << readsequence << "\n";

            #else

            outputstream << r << '\t' 
                << static_cast<int>(result.orientation) << '\t' 
                << genome.names[result.chromosomeId] << '\t'
                << result.position << '\t'
                << result.shift << '\t'
                << readLengths[0] << '\t'
                << window << '\t'
                << readsequence << '\n';

            #endif

            processedResults++;
            //if(processedResults == 10) break;
        }else{
            // output unmapped
          
            #if 0

            outputstream << "read " << r << ": unmapped\n";

            #else

            outputstream << r << '\t' 
                << static_cast<int>(result.orientation) << '\t' 
                << "NONE" << '\t'
                << -1 << '\t'
                << 0 << '\t'
                << readLengths[0] << '\t'
                << "NONE" << '\t'
                << readsequence << '\n';

            #endif
        }
    }

    #ifdef COUNT_WINDOW_HITS
    std::string hitstatsAfterHashingFilename = "windowhitstats_afterhashing_k" + std::to_string(programOptions.kmerlength)
        + "_h" + std::to_string(programOptions.numHashFunctions)
        + "_w" + std::to_string(programOptions.windowSize)
        + "_minhits" + std::to_string(programOptions.minTableHits)
        + ".txt";
    std::ofstream ws1(hitstatsAfterHashingFilename);
    assert(bool(ws1));

    windowHitStatsAfterHashing.forEachWindow(
        [&](const auto& stat){
            ws1 << stat.chromosomeId << " " << stat.windowId << " " << stat.truehits << " " << stat.falsehits << "\n";
        }
    );

    std::string hitstatsAfterHammingDistanceFilename = "windowhitstats_afterhammingdistance_k" + std::to_string(programOptions.kmerlength)
        + "_h" + std::to_string(programOptions.numHashFunctions)
        + "_w" + std::to_string(programOptions.windowSize)
        + "_minhits" + std::to_string(programOptions.minTableHits)
        + ".txt";
    std::ofstream ws2(hitstatsAfterHammingDistanceFilename);

    windowHitStatsAfterHammingDistance.forEachWindow(
        [&](const auto& stat){
            ws2 << stat.chromosomeId << " " << stat.windowId << " " << stat.truehits << " " << stat.falsehits << "\n";
        }
    );
    assert(bool(ws2));
    
    #endif
#endif
}


int main(int argc, char** argv){

    cxxopts::Options commandLineOptions(argv[0], "Hash Read Mapper");
    addOptions(commandLineOptions);

    if(argc == 1){
		std::cout << commandLineOptions.help({"Options"}) << std::endl;
		std::exit(0);
    }

	auto parseresults = commandLineOptions.parse(argc, argv);

    ProgramOptions programOptions(parseresults);

    std::cout << programOptions << "\n";
   helpers::CpuTimer totaltime("Total runtime");
    assert(programOptions.kmerlength <= max_k<kmer_type>::value);
    assert(std::size_t(programOptions.maxResultsPerMap) <= std::size_t(std::numeric_limits<BucketSize>::max()));
   //std::cout<<"maxnummaps: "<<maximum_number_of_maps<<" and numhashflt: "<<programOptions.numHashFunctions<<"\n";
    assert(programOptions.numHashFunctions <= maximum_number_of_maps);
    assert(programOptions.numHashFunctions >= programOptions.minTableHits);


    performMappingGpu(programOptions);
   totaltime.print();
}
