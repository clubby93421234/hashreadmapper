#ifndef MAPPEDREAD_CUH
#define MAPPEDREAD_CUH

#include <alignmentorientation.hpp>

struct MappedRead{
    AlignmentOrientation orientation = AlignmentOrientation::None;
    int hammingDistance;
    int shift;
    std::size_t chromosomeId = 0;
    std::size_t position = 0;
};

#endif