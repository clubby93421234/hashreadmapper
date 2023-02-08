
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




using namespace care;



struct MappedRead{
    AlignmentOrientation orientation = AlignmentOrientation::None;
    int hammingDistance;
    int shift;
    std::size_t chromosomeId = 0;
    std::size_t position = 0;
};



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
    MappedRead* resultsRC;
    WindowHitStatisticCollector* windowHitStatsAfterHashing;
    WindowHitStatisticCollector* windowHitStatsAfterHammingDistance;

    MinhasherHandle minhashHandle;
    ReadStorageHandle readstorageHandle;

    WindowBatchProcessor(
        const gpu::GpuReadStorage* gpuReadStorage_,
        const gpu::GpuMinhasher* gpuMinhasher_,
        const ProgramOptions* programOptions_,
        MappedRead* results_,
        MappedRead* resultsRC_,
        WindowHitStatisticCollector* windowHitStatsAfterHashing_,
        WindowHitStatisticCollector* windowHitStatsAfterHammingDistance_
    ) : gpuReadStorage(gpuReadStorage_),
        gpuMinhasher(gpuMinhasher_),
        programOptions(programOptions_),
        results(results_),
        resultsRC(resultsRC_),
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



    void operator()(const Genome::BatchOfWindows& batch, bool ReverseComplementBatch){

        auto* mr = rmm::mr::get_current_device_resource();
        cudaStream_t stream = cudaStreamPerThread;

        const std::size_t decodedWindowPitchInBytes = SDIV(batch.maxWindowSize, 32) * 32;        
        std::size_t encodedWindowPitchInInts = SequenceHelpers::getEncodedNumInts2Bit(batch.maxWindowSize);
        std::size_t encodedReadPitchInInts = SequenceHelpers::getEncodedNumInts2Bit(gpuReadStorage->getSequenceLengthUpperBound());

        //Next step: transfer windows to gpu
        nvtx::push_range("transfer windows to gpu", 0);
        //for 3N-Genome
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
        if(ReverseComplementBatch)
            SequenceHelpers::reverseComplementSequenceDecodedInplaceVector(&h_windowsDecoded, h_windowsDecoded.size());
        //---------------------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!--------------------------------------
        // do Nucleotide conversion here
        SequenceHelpers::NucleotideConverterVectorInplace_CtoT(&h_windowsDecoded, h_windowsDecoded.size());
        //---------------------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!--------------------------------------

        CUDACHECK(cudaMemcpyAsync(
            d_windowsDecoded.data(),
            h_windowsDecoded.data(),
            sizeof(char) * decodedWindowPitchInBytes * batch.numWindows,
            H2D,
            stream
        ));

        //---------------------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!--------------------------------------
        // do RC and then NC
        
        //SequenceHelpers::NucleotideConverterVectorInplace_CtoT(&h_windowsDecoded_RC, h_windowsDecoded.size());
        //---------------------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!--------------------------------------

        

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

//TODO muss ich hier 3-N machen?
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
                if(ReverseComplementBatch){
                    auto& currentBestResult = resultsRC[hostIds.h_readIds[offset + r]];
                }
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
        
    }

};

/*This handler selects and starts the chosen mapping algorithm
*/
struct Mappinghandler
{
    public:

        /// @brief 
        /// @param programOptions_ 
        /// @param genome_ 
        /// @param results_ 
        /// @param resultsRC_
        Mappinghandler(
            const ProgramOptions* programOptions_,
            const Genome* genome_,
             std::vector<MappedRead>* results_
             std::vector<MappedRead>* resultsRC_
             ):
        programOptions(programOptions_),
        genome(genome_),
        results(results_),
        resultsRC(resultsRC_)
        
        {
           
        }

        ~Mappinghandler(){
            
       }

     

      /// @brief Used to slect a mapper and stars it
      /// @param cpuReadStorage_ The storage of reads
      void go(std::unique_ptr<ChunkedReadStorage>& cpuReadStorage_){
        //mappertype=(int) programOptions.mappType;
         
            switch(programOptions->mappType){
                case MapperType::primitiveSW : 
                    std::cout<<"primitive SW selected but should not be used!\n"; 
                    primitiveSW(cpuReadStorage_);
                    break;
                case MapperType::SW : 
                    //std::cout<<"SW selected \n"; 
                    CSSW(cpuReadStorage_);
                break;
                case MapperType::sthelse : 
                    std::cout<<"please implement your personal mapper\n"; 
                break;
                default: 
                    std::cout<<"something went wrong while selecting mappertype\n"; 
                    examplewrapper(cpuReadStorage_);
                break;
            }
       }
       
       /// @brief TODO
    void doVC(){
      /*  
        auto test=(programOptions->outputfile)+".VCF";
        VariantHandler vhandler(test);
        vhandler.VCFFileHeader();
        

        uint32_t mapq=0;
      
        for (long unsigned int i=0; i<mappingout.size();i++){
         
         //https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library/blob/master/src/main.c
         //Line 167 to  169
            mapq = -4.343 * log(1 - (double)abs(mappingout.at(i).alignment.sw_score - 
                mappingout.at(i).alignment.sw_score_next_best)
                /(double)mappingout.at(i).alignment.sw_score);
			mapq = (uint32_t) (mapq + 4.99);
			mapq = mapq < 254 ? mapq : 254;

            
           if(mapq < MAP_QUALITY_THRESHOLD){
                continue;
            }
            

            Cigar marlboro{mappingout.at(i).alignment.cigar_string};
          
            std::string prefix=mappingout.at(i).ref.substr(0,mappingout.at(i).alignment.query_begin);
            
           vhandler.call(
            mappingout.at(i).result.position + mappingout.at(i).alignment.query_begin, //seq position
            prefix,
            mappingout.at(i).ref,
            mappingout.at(i).query, 
            marlboro.getEntries(),
            genome->names.at(mappingout.at(i).result.chromosomeId),
            mappingout.at(i).readId,
            mapq
            );
            
        }
        
        //vhandler.forceFlush();
        */

     } 
    private:
        const ProgramOptions* programOptions;
        const Genome* genome;
        int mappertype=1;
        std::vector<MappedRead>* results;
        std::vector<MappedRead>* resultsRC;

    INLINEQUALIFIER
    void NucleoideConverer(char* output, const char* input, int length){
        if(strlen(input)!=0)
        {

                    for(int i = 0; i < length; ++i){
                switch(input[i]){
                    case 'C': output[i] = 'T'; break;
                    default : break;
                }
            }
        }else
        assert("shit");
     }
        //Helper struct for CSSW-Mapping
struct AlignerArguments{           
           std::string query;
            std::string three_n_query;   //3 nucleotide query
          std::string rc_query;
            std::string three_n_rc_query;
         std::string ref;//the window of the reference genome
            std::string three_n_ref;
         std::string rc_ref;
            std::string three_n_rc_ref;

        std::string sam_tag;
            int ref_len;
        StripedSmithWaterman::Filter filter;
            
            int32_t maskLen;
        MappedRead result;
            read_number readId;
            std::string readsequence;
        std::vector<StripedSmithWaterman::Alignment> alignments;
            AlignerArguments():alignments({StripedSmithWaterman::Alignment(),StripedSmithWaterman::Alignment(),
                                            StripedSmithWaterman::Alignment(),StripedSmithWaterman::Alignment()})
            {
            }

        };

std::vector<AlignerArguments> mappingout;


/// @brief simple print methode for the vector mappingout into the "CSSW_SAM.SAM" file. Used for CSSW-Mapping
void printtoSAM(){
    std::cout<<"muss noch...\n";
    /*
    std::ofstream outputstream("CSSW_SAM.SAM");

    outputstream << "@HD\n"<<
                    "@Coloums: QNAME\tFLAG\tRNAME\tPOS\tMAPQ\tCIGAR\tRNEXT\tPNEXT\tTLEN\tSEQ\tQUAL\n";
      
    for(std::size_t i =0;i<mappingout.size();i++){


        //TODO MAPQ mussüberarbeitet werden!!! Das ist so nicht richtig
        //MAPQ calculated as in CSSW 
        //https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library/blob/master/src/main.c
         //Line 167 to  169
        uint32_t mapq = -4.343 * log(1 - (double)abs(mappingout.at(i).alignment.sw_score - 
                mappingout.at(i).alignment.sw_score_next_best)
                /(double)mappingout.at(i).alignment.sw_score);
			mapq = (uint32_t) (mapq + 4.99);
			mapq = mapq < 254 ? mapq : 254;

        outputstream << mappingout.at(i).readId<<"\t"                       //QNAME
        << "*" <<"\t"                                                       // FLAG
        << genome->names.at(mappingout.at(i).result.chromosomeId) <<"\t"    // RNAME
        << mappingout.at(i).result.position + mappingout.at(i).alignment.query_begin <<"\t"                   // POS //look up my shenanigans in ssw_cpp.cpp for why its queri_begin
        << mapq <<"\t"                                                      // MAPQ
        << mappingout.at(i).alignment.cigar_string <<"\t"                   // CIGAR
        << "=" <<"\t"                                                // RNEXT
        << "" <<"\t"                                                // PNEXT
        << "0" <<"\t"                                               // TLEN
        << mappingout.at(i).query <<"\t"                               // SEQ
        << "*" <<"\t"                                                // QUAL
        <<"\n";
    } 
    outputstream.close();
    */
}

     //Complete-Striped-Smith-Waterman Mapper.
     //https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library
    void CSSW(std::unique_ptr<ChunkedReadStorage>& cpuReadStorage){

        StripedSmithWaterman::Aligner aligner;
        StripedSmithWaterman::Filter filter;
        
        const int maximumSequenceLength = cpuReadStorage->getSequenceLengthUpperBound();

        std::size_t rundenzaehler=0;

        std::size_t numreads=cpuReadStorage->getNumberOfReads();

        for(std::size_t r = 0, processedResults = 0; r < numreads; r++){
            const auto& result = (*results)[r];
            const auto& resultRC = (*resultsRC)[r];
            //TODO -> get the shit for RC batch and do alignment. then update and ccompare. upload best to sam
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
               
                const auto& genomesequence = (*genome).data.at(result.chromosomeId);
                
                const std::size_t windowlength = result.position + 
                    programOptions->windowSize < genomesequence.size() ? programOptions->windowSize : genomesequence.size() - 
                    result.position;

                std::string_view window(genomesequence.data() + result.position, windowlength);
                processedResults++;

                int32_t maskLen = readLengths[0]/2;
                maskLen = maskLen < 15 ? 15 : maskLen;

                   
                    AlignerArguments ali;

                
                    ali.query=readsequence;
                    ali.three_n_query.resize(readLengths[0]);
                    NucleoideConverer(ali.three_n_query.data(), ali.query.c_str(), readLengths[0]);
                    ali.rc_query=SequenceHelpers::reverseComplementSequenceDecoded(ali.query.data(),readLengths[0]); 

                    ali.three_n_rc_query.resize(readLengths[0]);  
                    NucleoideConverer(ali.three_n_rc_query.data(), ali.rc_query.c_str(), readLengths[0]); 
                    

                    ali.ref=std::string(window).c_str();
                    ali.three_n_ref.resize(windowlength);
                    NucleoideConverer(ali.three_n_ref.data() ,ali.ref.c_str(), windowlength);
                    ali.rc_ref=SequenceHelpers::reverseComplementSequenceDecoded(ali.ref.data(), windowlength); 

                    ali.three_n_rc_ref.resize(windowlength);
                    NucleoideConverer(ali.three_n_rc_ref.data(), ali.rc_ref.c_str(), windowlength);

                    ali.filter=filter;
                    ali.maskLen=maskLen;
                    ali.readId=readId; 
                    ali.ref_len=windowlength;
                   
                    ali.result=result;
                    
                    mappingout.push_back(ali);
                

               rundenzaehler++;
                }else{
                    //no need to do sth. here
                }
            
         

        }//end of big for loop
     
        ThreadPool threadPool(std::max(1, programOptions->threads));
       ThreadPool::ParallelForHandle pforHandle;

       //function that maps all 4 alignments: 3NQuery-3NREF , 3NRC_Query-3NREF , 3NRC_Query - 3NRC_REF and 3NQuery - 3NRC_REF
        auto mapfk=[&](auto begin, auto end, int /*threadid*/){
            //std::cout<<"i am doing my job!\n";
                for(auto i=begin; i< end; i++){

                    // 3NQuery-3NREF
                    StripedSmithWaterman::Alignment* ali;
                    ali=&mappingout.at(i).alignments.at(0);
                    aligner.Align(
                        (mappingout.at(i).three_n_query).c_str(),
                        (mappingout.at(i).three_n_ref).c_str(),
                        mappingout.at(i).ref_len,
                        mappingout.at(i).filter,
                        ali,
                        mappingout.at(i).maskLen);

                        // 3NRC_Query-3NREF
                        StripedSmithWaterman::Alignment* alii;
                    alii=&mappingout.at(i).alignments.at(1);
                    aligner.Align(
                        (mappingout.at(i).three_n_rc_query).c_str(),
                        mappingout.at(i).three_n_ref.c_str(),
                        mappingout.at(i).ref_len,
                        mappingout.at(i).filter,
                        alii,
                        mappingout.at(i).maskLen);

                        // 3NRC_Query - 3NRC_REF
                        StripedSmithWaterman::Alignment* aliii;
                    aliii=&mappingout.at(i).alignments.at(2);
                    aligner.Align(
                        mappingout.at(i).three_n_rc_query.c_str(),
                        mappingout.at(i).three_n_rc_ref.c_str(),
                        mappingout.at(i).ref_len,
                        mappingout.at(i).filter,
                        aliii,
                        mappingout.at(i).maskLen);

                        // 3NQuery - 3NRC_REF
                        StripedSmithWaterman::Alignment* aliv;
                    aliv=&mappingout.at(i).alignments.at(3);
                    aligner.Align(
                        mappingout.at(i).three_n_query.c_str(),
                        mappingout.at(i).three_n_rc_ref.c_str(),
                        mappingout.at(i).ref_len,
                        mappingout.at(i).filter,
                        aliv,
                        mappingout.at(i).maskLen);
                }
        };
        
        //std::size_t tomax = mappingout.size();
        std::size_t start=0;
        //helpers::CpuTimer parallelmapping("parallel mapping time:");
        threadPool.parallelFor(pforHandle, start , mappingout.size() ,mapfk);
       // parallelmapping.print();

    auto recalculateAlignmentScorefk=[&](AlignerArguments& aa, const Cigar::Entries& cig, uint8_t h){

            StripedSmithWaterman::Alignment* ali=&aa.alignments.at(h);
            
        int refPos = 0, altPos = 0;
        for (const auto  & cigarEntry : cig) {
            auto basesLeft = std::min(82 - std::max(refPos, altPos), cigarEntry.second);
        switch (cigarEntry.first) {
        case Cigar::Op::Match:
            for (int i = 0; i < basesLeft; ++i) {
                if (aa.ref[refPos + i] == aa.query[altPos + i] || aa.ref[refPos + i] == WILDCARD_NUCLEOTIDE
                    || aa.query[altPos + i] == WILDCARD_NUCLEOTIDE)
                    continue;
                //TODO
            }
            refPos += basesLeft;
            altPos += basesLeft;
            break;

        case Cigar::Op::Insert:
            altPos += basesLeft;
            break;

        case Cigar::Op::Delete:
            refPos += basesLeft;
            break;

        case Cigar::Op::SoftClip:
            altPos += basesLeft;
            break;

        case Cigar::Op::HardClip:
            break;

        case Cigar::Op::Skipped:
            refPos += basesLeft;
            break;

        case Cigar::Op::Padding:
        //TODO: no idea what to do here
            break;

        case Cigar::Op::Mismatch:
        for (int i = 0; i < basesLeft; ++i) {
                if (aa.ref[refPos + i] == aa.query[altPos + i] || aa.ref[refPos + i] == WILDCARD_NUCLEOTIDE
                    || aa.query[altPos + i] == WILDCARD_NUCLEOTIDE)
                    continue;
                //TODO
            }
            refPos +=basesLeft;
            altPos += basesLeft;
            
            break;  
        case Cigar::Op::Equal:
            refPos += basesLeft;
            altPos += basesLeft;
        break;

        default:
            assert(false && "Unhandled CIGAR operation");
            break;
        }

        }

     };
        auto comparefk=[&](auto begin, auto end, int /*threadid*/){
            for(auto i=begin; i< end; i++){
                
                Cigar cigi{mappingout.at(i).alignments.at(0).cigar_string};
                Cigar cigii{mappingout.at(i).alignments.at(1).cigar_string};
                Cigar cigiii{mappingout.at(i).alignments.at(2).cigar_string};
                Cigar cigiv{mappingout.at(i).alignments.at(3).cigar_string};
                
                recalculateAlignmentScorefk(mappingout.at(i), cigi.getEntries(), 0);
                recalculateAlignmentScorefk(mappingout.at(i), cigii.getEntries(), 1);
                recalculateAlignmentScorefk(mappingout.at(i), cigiii.getEntries(), 2);
                recalculateAlignmentScorefk(mappingout.at(i), cigiv.getEntries(), 3);
            
            }
        };
    threadPool.parallelFor(pforHandle, start , mappingout.size() ,comparefk);
 
 
    printtoSAM();

}//end of CSSW-Mapping

     struct Mappcollection{
        std::string read;
        std::string gen;
        std::string consa;
        std::string consb;

        std::string to_string(){
            return read+"\t"+
                gen+"\t"+
                consa+"\t"+
                consb+"\n";
        }
        void reset(){
            read.clear();
            gen.clear();
            consa.clear();
            consb.clear();
        }
     };

        //Simple Smith-Waterman by https://github.com/ngopal/SimpleSmithWatermanCPP
     void primitiveSW(std::unique_ptr<ChunkedReadStorage>& cpuReadStorage){
        
        std::cout<<"dont use it!\n";
        cpuReadStorage.get();//useless, but i wanted to suppress the "unused cpuReadStorage" warning
        #ifdef UNGABUNGA //to turn on/off this mapper
        std::ofstream outputstream("PSW_out.txt");
        std::vector<std::string> mappingout;
        Mappcollection mc;
        std::cout<<"lets go primiiveSW!!\n";
        const int maximumSequenceLength = cpuReadStorage->getSequenceLengthUpperBound();
        helpers::CpuTimer primitivswtimer("primitiSW");
        std::size_t rundenzaehler=0;
        


        std::size_t numreads=cpuReadStorage->getNumberOfReads();
        //std::cout<<"numreads: "<<numreads<<"\n";
        
        for(std::size_t r = 0, processedResults = 0; r < numreads; r++){
            const auto& result = (*results)[r];
            read_number readId = r;
            rundenzaehler++;
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
                //std::cout<<"is alignment orientation != none? ";
               
                const auto& genomesequence = (*genome).data.at(result.chromosomeId);
                
                const std::size_t windowlength = result.position + 
                    programOptions->windowSize < genomesequence.size() ? programOptions->windowSize : genomesequence.size() - 
                    result.position;

                std::string_view window(genomesequence.data() + result.position, windowlength);
                processedResults++;
                //helpers::CpuTimer mappingstep("a mapping step ");
                /*mapping mit readsequence und windows
                *
                */
                //creating matrix and init it with zero's
                std::size_t* matrix;
                matrix = (std::size_t*) calloc((readLengths[0]+1)*(windowlength+1),sizeof(std::size_t));

                double penalty=-4;
                int ind;

                auto findMax = [&](double array[], int length){
                double max = array[0];
	            ind = 0;

	            for(int i=1; i<length; i++){
		            if(array[i] > max){
			            max = array[i];
			            ind=i;
		            }
	            }
	            return max;
                };
           
                auto similarityScore = [&](char a, char b){
                double result;
	            if(a==b){
		            result=1;
	            }else{
		            result=penalty;
	            }
	            return result;
                };

            double traceback[4];

           // std::cout<<"\n vor init I/J matrixen \n";
            int* I_i =(int*)malloc((readLengths[0]+1)*(windowlength+1)*sizeof(int));
            int* I_j =(int*)malloc((readLengths[0]+1)*(windowlength+1)*sizeof(int));
	        //int I_i[readLengths[0]+1][windowlength+1];
	        //int I_j[readLengths[0]+1][windowlength+1];
            //inimatrixtimer.print();

            //std::cout<<"\n nach init I/J matrixen \n";
            //start populating matrix
	        for (std::size_t i=1;i<=readLengths[0];i++){
		        for(std::size_t j=0;j<=windowlength;j++){
                       // cout << i << " " << j << endl;
			        traceback[0] = matrix[(i-1)*readLengths[0]+(j-1)]+similarityScore(readsequence[i-1],window[j-1]);
                    
			        traceback[1] = matrix[(i-1)*readLengths[0]+j]+penalty;
			        traceback[2] = matrix[i*readLengths[0]+(j-1)]+penalty;
			        traceback[3] = 0;
			        matrix[i*readLengths[0]+j] = findMax(traceback,4);
			        switch(ind){
				        case 0:
					        I_i[i*readLengths[0]+j] = i-1;
					        I_j[i*readLengths[0]+j] = j-1;
					    break;
				        case 1:
					        I_i[i*readLengths[0]+j] = i-1;
                            I_j[i*readLengths[0]+j] = j;
                        break;
				        case 2:
				            I_i[i*readLengths[0]+j] = i;
                            I_j[i*readLengths[0]+j] = j-1;
                        break;
				        case 3:
					        I_i[i*readLengths[0]+j] = i;
                            I_j[i*readLengths[0]+j] = j;
                        break;
			        }
                }
	        }

// find the max score in the matrix
        
	    double matrix_max = 0;
	    int i_max=0, j_max=0;

	    for(std::size_t i=1;i<readLengths[0];i++){
		    for(std::size_t j=1;j<windowlength;j++){
			    if(matrix[i*readLengths[0]+j]>matrix_max){
				    matrix_max = matrix[i*readLengths[0]+j];
				    i_max=i;
				    j_max=j;
			    }
		    }
	    }   
        
	    //std::cout << "Max score in the matrix is " << matrix_max << std::endl;


        // traceback
	
	    int current_i=i_max,current_j=j_max;
	    int next_i=I_i[current_i*readLengths[0]+current_j];
	    int next_j=I_j[current_i*readLengths[0]+current_j];
	    int tick=0;
	    char* consensus_a=(char*)malloc((readLengths[0]+windowlength+2)*sizeof(char));
        char* consensus_b=(char*)malloc((readLengths[0]+windowlength+2)*sizeof(char));

	    while(((current_i!=next_i) || (current_j!=next_j)) && (next_j!=0) && (next_i!=0)){

		    if(next_i==current_i)  consensus_a[tick] = '-';                  // deletion in A
		    else                   consensus_a[tick] = readsequence[current_i-1];   // match/mismatch in A

		    if(next_j==current_j)  consensus_b[tick] = '-';                  // deletion in B
		    else                   consensus_b[tick] = window[current_j-1];   // match/mismatch in B

		    current_i = next_i;
		    current_j = next_j;
		    next_i = I_i[current_i*readLengths[0]+current_j];
		    next_j = I_j[current_i*readLengths[0]+current_j];
		    tick++;
	    }

            //std::cout<<"\n traceback erledigt \n";
	//std::cout<<std::endl<<" "<<std::endl;
	//std::cout<<"Alignment:"<<std::endl<<std::endl;
	for(int i=0;i<readLengths[0];i++){
        mc.read+=readsequence[i];
        } 

       // std::cout<<"  and"<<std::endl;
	for(std::size_t i=0;i<windowlength;i++){
        mc.gen+=window[i];
        }

      //   std::cout<<std::endl<<"consensus a"<<std::endl;  
	for(std::size_t i=tick-1;i>=0;i--) 
        mc.consa+=consensus_a[i]; 

	//std::cout<<"\n consensus b"<<std::endl;
	for(std::size_t j=tick-1;j>=0;j--) 
        mc.consb+=consensus_b[j];
	//std::cout<<std::endl;

            mappingout.emplace_back(mc.to_string());
                mc.reset();
                free(I_i);
                free(I_j);
                free(consensus_a);
                free(consensus_b);
                free(matrix);
            }else{

            }

        }
        //std::cout<<"out of primitivSW loop\n";
        for(int i=0;i<mappingout.size();i++){
            outputstream<<mappingout.at(i);
        }
        #endif //to turn on/off this mapper
      }    

        /// @brief THis is a blank fkt for writing your own mapper
        /// @param cpuReadStorage 
        void examplewrapper(std::unique_ptr<ChunkedReadStorage>& cpuReadStorage){

        std::ofstream outputstream("examplemapper_out.txt");
               
        //std::cout<<"lets go Examplemapper!!\n";
        const int maximumSequenceLength = cpuReadStorage->getSequenceLengthUpperBound();
        helpers::CpuTimer mappertimer("timerformapping");
        std::size_t rundenzaehler=0;

        std::size_t numreads=cpuReadStorage->getNumberOfReads();
        
        for(std::size_t r = 0, processedResults = 0; r < numreads; r++){
            const auto& result = (*results)[r];
            read_number readId = r;
            rundenzaehler++;
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
                              
                const auto& genomesequence = (*genome).data.at(result.chromosomeId);
                
                const std::size_t windowlength = result.position + 
                    programOptions->windowSize < genomesequence.size() ? programOptions->windowSize : genomesequence.size() - 
                    result.position;

                std::string_view window(genomesequence.data() + result.position, windowlength);
                processedResults++;
                

                /*Put your mapping algorithm here
                *....
                *use the variables "window" and "readsequence"
                */
                }else{

                }
    mappertimer.print();
    }
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

    
    std::cout << "Loading genome\n";
    Genome genome(programOptions.genomefile);
    //genome.printInfo();
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

    std::size_t processedWindowCount = 0;
    std::size_t processedWindowCountProgress = 0;

    //results for 3n genome
    std::vector<MappedRead> results(multiGpuReadStorage.getNumberOfReads());
    //results for rc 3n genome
    std::vector<MappedRead> resultsRC(multiGpuReadStorage.getNumberOfReads());
    WindowBatchProcessor windowBatchProcessor(
        gpuReadStorage,
        gpuMinhasher,
        &programOptions,
        results.data(),
        resultsRC.data(),
        windowHitStatsAfterHashingPtr,
        windowHitStatsAfterHammingDistancePtr
    );

    auto processWithProgress = [&](const Genome::BatchOfWindows& batch){
        windowBatchProcessor(batch,false);
        windowBatchProcessor(batch,true);

        processedWindowCount += batch.numWindows;
        processedWindowCountProgress += batch.numWindows;
        if(programOptions.showProgress){
            if(processedWindowCountProgress >= 100000){
                std::cout << "processed " << processedWindowCount << " / " << totalWindowCount << "\n";
                processedWindowCountProgress -= 100000;
            }
        }
       
        
    };
    
    genome.forEachBatchOfWindows(
        programOptions.kmerlength,
        programOptions.windowSize,
        programOptions.batchsize,
        processWithProgress
    );
    
    std::cout << "processed " << totalWindowCount << " / " << totalWindowCount << " windows.\n";
    
   
   
    timerprocessgenome.print();
    std::cout<<"STEP 2: Mapping: \n";
    Mappinghandler mapper(&programOptions, &genome, &results, &resultsRC);

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
