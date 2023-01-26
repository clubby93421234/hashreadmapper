#ifndef SEQUENCE_CONVERSION_KERNELS_CUH
#define SEQUENCE_CONVERSION_KERNELS_CUH

void callConversionKernel2BitTo2BitHiLoNN(
    const unsigned int* d_inputdata,
    size_t inputpitchInInts,
    unsigned int* d_outputdata,
    size_t outputpitchInInts,
    const int* d_sequenceLengths,
    int numSequences,
    cudaStream_t stream
);

void callConversionKernel2BitTo2BitHiLoNT(
    const unsigned int* d_inputdata,
    size_t inputpitchInInts,
    unsigned int* d_outputdata,
    size_t outputpitchInInts,
    const int* d_sequenceLengths,
    int numSequences,
    cudaStream_t stream
);

void callConversionKernel2BitTo2BitHiLoTT(
    const unsigned int* d_inputdata,
    size_t inputpitchInInts,
    unsigned int* d_outputdata,
    size_t outputpitchInInts,
    const int* d_sequenceLengths,
    int numSequences,
    cudaStream_t stream
);            

void callEncodeSequencesTo2BitKernel(
    unsigned int* d_encodedSequences,
    const char* d_decodedSequences,
    const int* d_sequenceLengths,
    size_t decodedSequencePitchInBytes,
    size_t encodedSequencePitchInInts,
    int numSequences,
    int groupsize,
    cudaStream_t stream
);

void callDecodeSequencesFrom2BitKernel(
    char* d_decodedSequences,
    const unsigned int* d_encodedSequences,
    const int* d_sequenceLengths,
    size_t decodedSequencePitchInBytes,
    size_t encodedSequencePitchInInts,
    int numSequences,
    int groupsize,
    cudaStream_t stream
);

#endif
