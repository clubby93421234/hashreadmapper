#ifndef WINDOW_GENERATION_KERNELS_CUH
#define WINDOW_GENERATION_KERNELS_CUH

#include <gpu/cudaerrorcheck.cuh>

#include <cooperative_groups.h>
#include <cstddef>

namespace detail{
    struct WindowLocationInSequenceSection{
        int left;
        int right;
        int length;
        int startpos;
    };

    __host__ __device__
    WindowLocationInSequenceSection computeWindowLocation(
        int sectionBegin, 
        int sectionEnd, 
        int globalWindowPosition,
        int windowSize,
        int extension = 0
    ){
        WindowLocationInSequenceSection result;
        result.length = windowSize;

        result.left = 0;
        if(extension < globalWindowPosition){
            result.left = extension;
            result.length  += extension;
        }
        result.right = 0;
        if(globalWindowPosition + windowSize <= sectionEnd){
            if(globalWindowPosition + windowSize + extension < sectionEnd){
                result.right = extension;
            }else{
                result.right = sectionEnd - (globalWindowPosition + windowSize);
            }
            result.length  += result.right;
        }else{
            result.length  -= (globalWindowPosition + windowSize) - sectionEnd;
        }

        result.startpos = globalWindowPosition - result.left - sectionBegin;

        return result;
    }
};

template<int blocksize, int groupsize>
__global__
void generateExtendedWindowsDecodedKernel(
    const char* d_genomicSection, 
    int sectionBegin, 
    int sectionEnd, 
    const int* d_readLengths,
    int numReads,
    const int* d_segmentIds,
    int windowSize,
    const int* d_windowPositions,
    char* d_decodedExtendedWindows,
    int* d_extendedWindowLengths,
    std::size_t decodedPitchInBytes,
    int* d_extensionsLeft,
    int* d_extensionsRight
){
    auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());

    const int numGroups = (blocksize * gridDim.x) / groupsize;
    const int groupId = (threadIdx.x + blockIdx.x * blocksize) / groupsize;

    for(int r = groupId; r < numReads; r += numGroups){
        const int segmentId = d_segmentIds[r];
        const int readLength = d_readLengths[r];
        const int windowPosition = d_windowPositions[segmentId];
        assert(windowPosition >= sectionBegin);
        const int extension = readLength / 2;

        detail::WindowLocationInSequenceSection location = detail::computeWindowLocation(
            sectionBegin, 
            sectionEnd, 
            windowPosition,
            windowSize,
            extension
        );

        char* const windowOutput = d_decodedExtendedWindows + decodedPitchInBytes * r;

        for(int i = group.thread_rank(); i < location.length; i += group.size()){
            windowOutput[i] = d_genomicSection[location.startpos + i];
        }
        if(group.thread_rank() == 0){
            d_extensionsLeft[r] = location.left;
            d_extensionsRight[r] = location.right;
            d_extendedWindowLengths[r] = location.length;
        }
    }
}

void callGenerateExtendedWindowsDecodedKernel(
    const char* d_genomicSection, 
    int sectionBegin, 
    int sectionEnd, 
    const int* d_readLengths,
    int numReads,
    const int* d_segmentIds,
    int windowSize,
    const int* d_windowPositions,
    char* d_decodedExtendedWindows,
    int* d_extendedWindowLengths,
    std::size_t decodedPitchInBytes,
    int* d_extensionsLeft,
    int* d_extensionsRight,
    cudaStream_t stream
){
    constexpr int groupsize = 32;
    constexpr int blocksize = 256;
    constexpr int numGroupsPerBlock = blocksize / groupsize;

    const std::size_t smem = 0;

    auto kernel = generateExtendedWindowsDecodedKernel<blocksize, groupsize>;

    int deviceId = 0;
    int numSMs = 0;
    int maxBlocksPerSM = 0;
    CUDACHECK(cudaGetDevice(&deviceId));
    CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM,
        kernel,
        blocksize, 
        smem
    ));

    const int maxBlocks = maxBlocksPerSM * numSMs;  

    dim3 block(blocksize, 1, 1);
    const int numBlocks = SDIV(numReads, numGroupsPerBlock);
    dim3 grid(std::min(numBlocks, maxBlocks), 1, 1);

    kernel<<<grid, block, smem, stream>>>(
        d_genomicSection, 
        sectionBegin, 
        sectionEnd, 
        d_readLengths,
        numReads,
        d_segmentIds,
        windowSize,
        d_windowPositions,
        d_decodedExtendedWindows,
        d_extendedWindowLengths,
        decodedPitchInBytes,
        d_extensionsLeft,
        d_extensionsRight
    );
    CUDACHECKASYNC;
}


template<int blocksize, int groupsize>
__global__
void generateExtendedWindows2BitKernel(
    const char* d_genomicSection, 
    int sectionBegin, 
    int sectionEnd, 
    const int* d_readLengths,
    int numReads,
    const int* d_segmentIds,
    int windowSize,
    const int* d_windowPositions,
    unsigned int* d_extendedWindows2Bit,
    int* d_extendedWindowLengths,
    std::size_t encodedSequencePitchInInts,
    int* d_extensionsLeft,
    int* d_extensionsRight,
    size_t smemPitchPerGroup
){
    auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());

    const int numGroups = (blocksize * gridDim.x) / groupsize;
    const int groupId = (threadIdx.x + blockIdx.x * blocksize) / groupsize;
    const int groupIdInBlock = threadIdx.x / groupsize;

    extern __shared__ char smemDecodedExtendedWindows[];
    char* const smemDecodedExtendedWindow = smemDecodedExtendedWindows + smemPitchPerGroup * groupIdInBlock;

    for(int r = groupId; r < numReads; r += numGroups){
        const int segmentId = d_segmentIds[r];
        const int readLength = d_readLengths[r];
        const int windowPosition = d_windowPositions[segmentId];
        assert(windowPosition >= sectionBegin);
        const int extension = readLength / 2;

        detail::WindowLocationInSequenceSection location = detail::computeWindowLocation(
            sectionBegin, 
            sectionEnd, 
            windowPosition,
            windowSize,
            extension
        );

        const char* const windowInput = &d_genomicSection[location.startpos];
        unsigned int* const windowOutput = d_extendedWindows2Bit + encodedSequencePitchInInts * r;
        const int nInts = SequenceHelpers::getEncodedNumInts2Bit(location.length);
        constexpr int basesPerInt = SequenceHelpers::basesPerInt2Bit();

        //load sequence to smem coalesced
        group.sync();
        for(int i = group.thread_rank(); i < location.length; i += group.size()){
            smemDecodedExtendedWindow[i] = windowInput[i];
        }
        group.sync();

        for(int i = group.thread_rank(); i < nInts; i += group.size()){
            unsigned int data = 0;
            auto encodeNuc = [&](char nuc){
                switch(nuc) {
                case 'A':
                    data = (data << 2) | SequenceHelpers::encodedbaseA();
                    break;
                case 'C':
                    data = (data << 2) | SequenceHelpers::encodedbaseC();
                    break;
                case 'G':
                    data = (data << 2) | SequenceHelpers::encodedbaseG();
                    break;
                case 'T':
                    data = (data << 2) | SequenceHelpers::encodedbaseT();
                    break;
                default:
                    data = (data << 2) | SequenceHelpers::encodedbaseA();
                    break;
                }
            };

            if(i < nInts - 1){
                //not last iteration. int encodes 16 chars
                for(int x = 0; x < 16; x++){
                    const char nuc = smemDecodedExtendedWindow[i * basesPerInt + x];
                    encodeNuc(nuc);
                }
            }else{        
                for(int nucIndex = i * basesPerInt; nucIndex < location.length; nucIndex++){
                    encodeNuc(smemDecodedExtendedWindow[nucIndex]);
                }
                //pack bits of last integer into higher order bits
                int leftoverbits = 2 * (nInts * basesPerInt - location.length);
                if(leftoverbits > 0){
                    data <<= leftoverbits;
                }
            }
            windowOutput[i] = data;
        }
        if(group.thread_rank() == 0){
            d_extensionsLeft[r] = location.left;
            d_extensionsRight[r] = location.right;
            d_extendedWindowLengths[r] = location.length;
        }
    }
}



void callGenerateExtendedWindows2BitKernel(
    const char* d_genomicSection, 
    int sectionBegin, 
    int sectionEnd, 
    const int* d_readLengths,
    int numReads,
    const int* d_segmentIds,
    int windowSize,
    const int* d_windowPositions,
    const int maxExtendedWindowLength,
    unsigned int* d_extendedWindows2Bit,
    int* d_extendedWindowLengths,
    std::size_t encodedSequencePitchInInts,
    int* d_extensionsLeft,
    int* d_extensionsRight,
    cudaStream_t stream
){
    constexpr int groupsize = 16;
    constexpr int blocksize = 256;
    constexpr int numGroupsPerBlock = blocksize / groupsize;

    const std::size_t smemPitch = SDIV(maxExtendedWindowLength, 128) * 128;
    const std::size_t smem = smemPitch * numGroupsPerBlock;

    auto kernel = generateExtendedWindows2BitKernel<blocksize, groupsize>;

    int deviceId = 0;
    int numSMs = 0;
    int maxBlocksPerSM = 0;
    CUDACHECK(cudaGetDevice(&deviceId));
    CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM,
        kernel,
        blocksize, 
        smem
    ));

    const int maxBlocks = maxBlocksPerSM * numSMs;  

    dim3 block(blocksize, 1, 1);
    const int numBlocks = SDIV(numReads, numGroupsPerBlock);
    dim3 grid(std::min(numBlocks, maxBlocks), 1, 1);

    kernel<<<grid, block, smem, stream>>>(
        d_genomicSection, 
        sectionBegin, 
        sectionEnd, 
        d_readLengths,
        numReads,
        d_segmentIds,
        windowSize,
        d_windowPositions,
        d_extendedWindows2Bit,
        d_extendedWindowLengths,
        encodedSequencePitchInInts,
        d_extensionsLeft,
        d_extensionsRight,
        smemPitch
    );
    CUDACHECKASYNC;
}




#endif