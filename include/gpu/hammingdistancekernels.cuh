#ifndef HAMMING_DISTANCE_KERNELS_CUH
#define HAMMING_DISTANCE_KERNELS_CUH

#include <alignmentorientation.hpp>

#include <cstddef>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

void callShiftedHammingDistanceWithFullOverlapKernelSmem1(
    int* d_bestShifts,
    int* d_bestScores,
    AlignmentOrientation* d_bestOrientations,
    const unsigned int* d_anchorData2bit,
    const int* d_anchorSequencesLength,
    std::size_t encodedSequencePitchInInts2BitAnchor,
    const unsigned int* d_candidateData2Bit,
    const int* d_candidateSequencesLength,
    std::size_t encodedSequencePitchInInts2BitCandidate,
    int numAnchors,
    int numCandidates,
    float maxErrorRate, //allow only less than (candidateLength * maxErrorRate) mismatches
    int maxAnchorLength,
    int maxCandidateLength,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()
);



#endif