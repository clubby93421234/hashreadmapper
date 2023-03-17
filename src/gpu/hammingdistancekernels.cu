#include <gpu/hammingdistancekernels.cuh>
#include <gpu/sequenceconversionkernels.cuh>
#include <alignmentorientation.hpp>
#include <sequencehelpers.hpp>

#include <cstddef>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <gpu/rmm_utilities.cuh>

#include <iostream>

HD_WARNING_DISABLE
template<class IndexTransformation>
HOSTDEVICEQUALIFIER
void shiftBitArrayLeftBy(unsigned int* array, int size, int shiftamount, IndexTransformation indextrafo){
    if(shiftamount == 0) return;

    const int completeInts = shiftamount / (8 * sizeof(unsigned int));

    for(int i = 0; i < size - completeInts; i += 1) {
        array[indextrafo(i)] = array[indextrafo(completeInts + i)];
    }

    for(int i = size - completeInts; i < size; i += 1) {
        array[indextrafo(i)] = 0;
    }

    shiftamount -= completeInts * 8 * sizeof(unsigned int);

    for(int i = 0; i < size - completeInts - 1; i += 1) {
        const unsigned int a = array[indextrafo(i)];
        const unsigned int b = array[indextrafo(i+1)];

        array[indextrafo(i)] = (a << shiftamount) | (b >> (8 * sizeof(unsigned int) - shiftamount));
    }

    array[indextrafo(size - completeInts - 1)] <<= shiftamount;
    std::cout<<"hello"<<std::endl;
}

HD_WARNING_DISABLE
template<int shiftamount, class IndexTransformation>
HOSTDEVICEQUALIFIER
void shiftBitArrayLeftBy(unsigned int* array, int size, IndexTransformation indextrafo){
    if(shiftamount == 0) return;

    constexpr int completeInts = shiftamount / (8 * sizeof(unsigned int));

    for(int i = 0; i < size - completeInts; i += 1) {
        array[indextrafo(i)] = array[indextrafo(completeInts + i)];
    }

    for(int i = size - completeInts; i < size; i += 1) {
        array[indextrafo(i)] = 0;
    }

    constexpr int remainingShift = shiftamount - completeInts * 8 * sizeof(unsigned int);

    for(int i = 0; i < size - completeInts - 1; i += 1) {
        const unsigned int a = array[indextrafo(i)];
        const unsigned int b = array[indextrafo(i+1)];

        array[indextrafo(i)] = (a << remainingShift) | (b >> (8 * sizeof(unsigned int) - remainingShift));
    }

    array[indextrafo(size - completeInts - 1)] <<= remainingShift;
}



HD_WARNING_DISABLE
template<class IndexTransformation1,
            class IndexTransformation2,
            class PopcountFunc>
HOSTDEVICEQUALIFIER
int hammingdistanceHiLo(const unsigned int* lhi,
                        const unsigned int* llo,
                        const unsigned int* rhi,
                        const unsigned int* rlo,
                        int lhi_bitcount,
                        int rhi_bitcount,
                        int max_errors,
                        IndexTransformation1 indextrafoL,
                        IndexTransformation2 indextrafoR,
                        PopcountFunc popcount){

    const int overlap_bitcount = std::min(lhi_bitcount, rhi_bitcount);

    if(overlap_bitcount == 0)
        return max_errors+1;

    const int partitions = SDIV(overlap_bitcount, (8 * sizeof(unsigned int)));
    const int remaining_bitcount = partitions * sizeof(unsigned int) * 8 - overlap_bitcount;

    int result = 0;

    for(int i = 0; i < partitions - 1 && result <= max_errors; i += 1) {
        const unsigned int hixor = lhi[indextrafoL(i)] ^ rhi[indextrafoR(i)];
        const unsigned int loxor = llo[indextrafoL(i)] ^ rlo[indextrafoR(i)];
        const unsigned int bits = hixor | loxor;
        result += popcount(bits);
    }

    if(result > max_errors)
        return result;

    // i == partitions - 1

    const unsigned int mask = remaining_bitcount == 0 ? 0xFFFFFFFF : 0xFFFFFFFF << (remaining_bitcount);
    const unsigned int hixor = lhi[indextrafoL(partitions - 1)] ^ rhi[indextrafoR(partitions - 1)];
    const unsigned int loxor = llo[indextrafoL(partitions - 1)] ^ rlo[indextrafoR(partitions - 1)];
    const unsigned int bits = hixor | loxor;
    result += popcount(bits & mask);

    return result;
}


/*

    For each candidate, compute the alignment of anchor|candidate and anchor|revc-candidate
    Compares both alignments and keeps the better one, i.e the alignment with smallest hamming distance

    kernel assumes that the anchor is at least as long as the candidate. Only shifts are considered where
    the candidate is fully contained in the anchor, i.e. 100% candidate overlap

    uses 1 thread per alignment
*/

template<int blocksize>
__global__
void shiftedHammingDistanceWithFullOverlapKernelSmem1(
    int* __restrict__ d_bestShifts,
    int* __restrict__ d_bestScores,
    AlignmentOrientation* __restrict__ d_bestOrientations,
    const unsigned int* __restrict__ anchorDataHiLoTransposed,
    const int* __restrict__ anchorSequencesLength,
    size_t encodedSequencePitchInInts2BitHiLoAnchor,
    const unsigned int* __restrict__ candidateDataHiLoTransposed,
    const int* __restrict__ candidateSequencesLength,
    size_t encodedSequencePitchInInts2BitHiLoCandidate,
    int numAnchors,
    int numCandidates,
    float maxErrorRate //allow only less than (candidateLength * maxErrorRate) mismatches
){

    auto block_transposed_index = [](int logical_index) -> int {
        return logical_index * blocksize;
    };

    // each thread stores anchor and candidate
    // sizeof(unsigned int) * encodedSequencePitchInInts2BitHiLoCandidate * blocksize
    //  + sizeof(unsigned int) * encodedSequencePitchInInts2BitHiLoCandidate * blocksize 
    extern __shared__ unsigned int sharedmemory[];

    //set up shared memory pointers
    //data is stored block-transposed to avoid bank conflicts
    unsigned int* const sharedAnchors = sharedmemory;
    unsigned int* const mySharedAnchor = sharedAnchors + threadIdx.x;
    unsigned int* const sharedCandidates = sharedAnchors + encodedSequencePitchInInts2BitHiLoAnchor * blocksize;
    unsigned int* const mySharedCandidate = sharedCandidates + threadIdx.x;

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;

    for(int candidateIndex = tid; candidateIndex < numCandidates; candidateIndex += stride){

        const int anchorIndex = candidateIndex;
        const int anchorLength = anchorSequencesLength[anchorIndex];
        const int anchorints = SequenceHelpers::getEncodedNumInts2BitHiLo(anchorLength);
        assert(anchorints <= int(encodedSequencePitchInInts2BitHiLoAnchor));
        const unsigned int* const anchorptr = anchorDataHiLoTransposed + std::size_t(anchorIndex);

        const int candidateLength = candidateSequencesLength[candidateIndex];
        const int candidateints = SequenceHelpers::getEncodedNumInts2BitHiLo(candidateLength);
        assert(candidateints <= int(encodedSequencePitchInInts2BitHiLoCandidate));
        if(candidateLength <= anchorLength){
            const unsigned int* const candidateptr = candidateDataHiLoTransposed + std::size_t(candidateIndex);

            //save candidate in shared memory
            for(int i = 0; i < candidateints; i++) {
                mySharedCandidate[block_transposed_index(i)] = candidateptr[i * numCandidates];
            }

            //we will shift the anchor to the left, which has the same effect as shifting candidate to the right
            unsigned int* const shiftptr_hi = mySharedAnchor;
            unsigned int* const shiftptr_lo = mySharedAnchor + block_transposed_index(anchorints / 2);
            unsigned int* const otherptr_hi = mySharedCandidate;
            unsigned int* const otherptr_lo = mySharedCandidate + block_transposed_index(candidateints / 2);

            int bestScore = std::numeric_limits<int>::max();
            int bestShift = -1;
            int bestOrientation = -1;

            for(int orientation = 0; orientation < 2; orientation++){
                const bool isReverseComplement = orientation == 1;

                if(isReverseComplement) {
                    SequenceHelpers::reverseComplementSequenceInplace2BitHiLo(mySharedCandidate, candidateLength, block_transposed_index);
                }

         //     SequenceHelpers::NucleotideConverterInplace2Bit_CtoT(mySharedCandidate, candidateLength, block_transposed_index);
                
                //save anchor in shared memory
                for(int i = 0; i < anchorints; i++) {
                    mySharedAnchor[block_transposed_index(i)] = anchorptr[i * numAnchors];
                }

                for(int shift = 0; shift < anchorLength - candidateLength + 1; shift += 1) {
                    //if best so far is ,e.g 5 mismatches, can early exit as soon as 4 mismatches are found in current shift
                    const int max_errors = min(int(float(candidateLength) * maxErrorRate), max(0,bestScore - 1));
                    if(shift != 0){
                        shiftBitArrayLeftBy<1>(shiftptr_hi, anchorints / 2, block_transposed_index);
                        shiftBitArrayLeftBy<1>(shiftptr_lo, anchorints / 2, block_transposed_index);
                    }
        
                    const int score = hammingdistanceHiLo(
                        shiftptr_hi,
                        shiftptr_lo,
                        otherptr_hi,
                        otherptr_lo,
                        candidateLength,
                        candidateLength,
                        max_errors,
                        block_transposed_index,
                        block_transposed_index,
                        [](auto i){return __popc(i);}
                    );

                    if(score < bestScore){
                        bestScore = score;
                        bestShift = shift;
                        bestOrientation = orientation;
                    }
                }
                
                //after the first iteration. Reset the read to do RC and then 3N it
           //     if(orientation==0)
           //     for(int i = 0; i < candidateints; i++) {
           //         mySharedCandidate[block_transposed_index(i)] = candidateptr[i * numCandidates];
           //     }
            }

            d_bestShifts[candidateIndex] = bestShift;
            d_bestScores[candidateIndex] = bestScore;
            if(bestScore > int(float(candidateLength) * maxErrorRate)){
                d_bestOrientations[candidateIndex] = AlignmentOrientation::None;
            }else{
                if(bestOrientation == 0){
                    d_bestOrientations[candidateIndex] = AlignmentOrientation::Forward;
                }else{
                    d_bestOrientations[candidateIndex] = AlignmentOrientation::ReverseComplement;
                }
            }
        }else{
            d_bestShifts[candidateIndex] = 0;
            d_bestScores[candidateIndex] = candidateLength;
            d_bestOrientations[candidateIndex] = AlignmentOrientation::None;
        }
    }
}


void callShiftedHammingDistanceWithFullOverlapKernelSmem1(
    int* d_bestShifts,
    int* d_bestScores,
    AlignmentOrientation* d_bestOrientations,
    const unsigned int* d_anchorData2bit,
    const int* d_anchorSequencesLength,
    size_t encodedSequencePitchInInts2BitAnchor,
    const unsigned int* d_candidateData2Bit,
    const int* d_candidateSequencesLength,
    size_t encodedSequencePitchInInts2BitCandidate,
    int numAnchors,
    int numCandidates,
    float maxErrorRate, //allow only less than (candidateLength * maxErrorRate) mismatches
    int maxAnchorLength,
    int maxCandidateLength,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr
){
    if(numAnchors == 0 || numCandidates == 0) return;

    const std::size_t intsPerSequenceHiLoAnchor = SequenceHelpers::getEncodedNumInts2BitHiLo(maxAnchorLength);
    const std::size_t intsPerSequenceHiLoCandidates = SequenceHelpers::getEncodedNumInts2BitHiLo(maxCandidateLength);

    rmm::device_uvector<unsigned int> d_anchorsHiLo(intsPerSequenceHiLoAnchor * numAnchors, stream, mr);
    rmm::device_uvector<unsigned int> d_candidatesHiLoTransposed(intsPerSequenceHiLoCandidates * numCandidates, stream, mr);               

    //2bit -> 2bit hilo , non-transposed -> transposed
    callConversionKernel2BitTo2BitHiLoNT(
        d_candidateData2Bit,
        encodedSequencePitchInInts2BitCandidate,
        d_candidatesHiLoTransposed.data(),
        intsPerSequenceHiLoCandidates,
        d_candidateSequencesLength,
        numCandidates,
        stream
    );

    //2bit -> 2bit hilo , non-transposed -> transposed
    callConversionKernel2BitTo2BitHiLoNT(
        d_anchorData2bit,
        encodedSequencePitchInInts2BitAnchor,
        d_anchorsHiLo.data(),
        intsPerSequenceHiLoAnchor,
        d_anchorSequencesLength,
        numAnchors,
        stream
    );

    constexpr int blocksize = 128;
    const std::size_t smem = sizeof(unsigned int) * (intsPerSequenceHiLoAnchor * blocksize + intsPerSequenceHiLoCandidates * blocksize);
    // std::cerr << "intsPerSequenceHiLoAnchor: " << intsPerSequenceHiLoAnchor << "\n";
    // std::cerr << "intsPerSequenceHiLoCandidates: " << intsPerSequenceHiLoCandidates << "\n";
    // std::cerr << "smem: " << smem << "\n";

    int deviceId = 0;
    int numSMs = 0;
    int maxBlocksPerSM = 0;
    CUDACHECK(cudaGetDevice(&deviceId));
    CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM,
        shiftedHammingDistanceWithFullOverlapKernelSmem1<blocksize>,
        blocksize, 
        smem
    ));

    const int maxBlocks = maxBlocksPerSM * numSMs;  

    dim3 block(blocksize, 1, 1);
    const int numBlocks = SDIV(numCandidates, blocksize);
    dim3 grid(std::min(numBlocks, maxBlocks), 1, 1);

    shiftedHammingDistanceWithFullOverlapKernelSmem1<blocksize><<<grid, block, smem, stream>>>(
        d_bestShifts,
        d_bestScores,
        d_bestOrientations,
        d_anchorsHiLo.data(),
        d_anchorSequencesLength,
        intsPerSequenceHiLoAnchor,
        d_candidatesHiLoTransposed.data(),
        d_candidateSequencesLength,
        intsPerSequenceHiLoCandidates,
        numAnchors,
        numCandidates,
        maxErrorRate
    );
    CUDACHECKASYNC;
}