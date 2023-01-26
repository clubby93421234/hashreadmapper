#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <map>
#include <string_view>

#include <readlibraryio.hpp>
#include <genome.hpp>
#include <util.hpp>

using namespace care;


template<class Iter1, class Iter2, class Equal>
int hammingDistanceFull(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2, Equal isEqual){
    int result = 0;

    while(first1 != last1 && first2 != last2){
        result += isEqual(*first1, *first2) ? 0 : 1;

        ++first1;
        ++first2;
    }

    //positions which do not overlap count as mismatch.
    //at least one of the remaining ranges is empty
    result += std::distance(first1, last1);
    result += std::distance(first2, last2);

    return result;
}

template<class Iter1, class Iter2>
int hammingDistanceFull(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2){
    auto isEqual = [](const auto& l, const auto& r){
        return l == r;
    };

    return hammingDistanceFull(first1, last1, first2, last2, isEqual);
}



int main(int argc, char** argv){

    if(argc < 4){
        std::cout << "Usage: " << argv[0] << " genomefile referencesamfile hashreadmapperoutput\n";
        return 0;
    }

    std::string genomefilename = argv[1];
    std::string referencefilename = argv[2];
    std::string mapperfilename = argv[3];
    const int maxMismatchesBetweedMappedRegions = 0;

    Genome genome(genomefilename);

    std::ifstream referencefile(referencefilename);
    std::ifstream mapperfile(mapperfilename);
    assert(bool(referencefile));
    assert(bool(mapperfile));

    std::string line1;
    std::string line2;

    int processedlines = 0;

    int oneIsUnmapped = 0;
    int numClipped = 0;
    int numRefClipped = 0;

    std::map<unsigned int, int> statusmap;

    while(bool(std::getline(referencefile, line1)) && bool(std::getline(mapperfile, line2))){
        processedlines++;
        
        /*
        sam flags: bit-wise OR of the following
        0x1     PAIRED        .. paired-end (or multiple-segment) sequencing technology
        0x2     PROPER_PAIR   .. each segment properly aligned according to the aligner
        0x4     UNMAP         .. segment unmapped
        0x8     MUNMAP        .. next segment in the template unmapped
        0x10    REVERSE       .. SEQ is reverse complemented
        0x20    MREVERSE      .. SEQ of the next segment in the template is reversed
        0x40    READ1         .. the first segment in the template
        0x80    READ2         .. the last segment in the template
        0x100   SECONDARY     .. secondary alignment
        0x200   QCFAIL        .. not passing quality controls
        0x400   DUP           .. PCR or optical duplicate
        0x800   SUPPLEMENTARY .. supplementary alignment
        */
        /*   
        expected format line2: read id, alignment orientation, chromosome name, 0-based window start position, read shift, read length, tab-separated
        like 0       2       CHROMOSOME_I    1376566    -4  100
        alignment orientation: 1 = forward, 2 = reverse complement, 3 = unmapped
        */

        auto tokens1 = split(line1, '\t');
        assert(tokens1.size() >= 4);
        auto tokens2 = split(line2, '\t');
        assert(tokens2.size() >= 6);

        unsigned int samflags = std::stoi(tokens1[1]);
        const std::string& chromosome1 = tokens1[2];
        int referencePosition = std::stoi(tokens1[3]) - 1; //tokens1[3] is 1-based

        int readId = std::stoi(tokens2[0]);
        unsigned int mapperOrientation = std::stoi(tokens2[1]);
        const std::string& chromosome2 = tokens2[2];
        int windowPosition = std::stoi(tokens2[3]);
        int readShift = std::stoi(tokens2[4]);
        int mapperPosition = windowPosition + readShift;
        int readLength = std::stoi(tokens2[5]);

        bool isMapped1 = !((samflags & 0x4) == 0x4);
        bool isMapped2 = (mapperOrientation != 3);
        bool isMappedOk = (isMapped1 && isMapped2);


        if(isMapped1 && isMapped2){
            const std::size_t chromosomeIndex1 = genome.getNameIndex(chromosome1);
            const std::size_t chromosomeIndex2 = genome.getNameIndex(chromosome2);

            //check if read is mapped out of bounds
            if(mapperPosition < 0 || mapperPosition + readLength >= int(genome.getSequence(chromosomeIndex2).size())){
                numClipped++;
                continue;
            }
            if(referencePosition < 0 || referencePosition + readLength >= int(genome.getSequence(chromosomeIndex1).size())){
                numRefClipped++;
                continue;
            }

            std::string_view genomerange1(genome.getSequence(chromosomeIndex1).data() + referencePosition, readLength);
            std::string_view genomerange2(genome.getSequence(chromosomeIndex2).data() + mapperPosition, readLength);

            bool isReversed1 = ((samflags & 0x10) == 0x10);
            bool isReversed2 = mapperOrientation == 2;
            bool isSameChromosome = chromosome1 == chromosome2;
            bool isOrientationOk = isReversed1 == isReversed2;
            bool isPositionOk = referencePosition == mapperPosition;

            
            //check if mapped region and reference region have small hamming distance, (i.e. location is either identical, or the read mapped to an (inexact) repeat region)
            int bestHamming = 0;
            if(!isSameChromosome || !isOrientationOk || !isPositionOk){
                int hammingdistanceFwd = hammingDistanceFull(
                    genomerange1.begin(),
                    genomerange1.end(),
                    genomerange2.begin(),
                    genomerange2.end()
                );

                std::string reversecomplementrange1(genomerange1.rbegin(), genomerange1.rend());
                for(auto& c : reversecomplementrange1){
                    switch(c){
                        case 'A': c = 'T'; break;
                        case 'C': c = 'G'; break;
                        case 'G': c = 'C'; break;
                        case 'T': c = 'A'; break;
                        default : break; // don't change N
                    }
                }

                int hammingdistanceRevc = hammingDistanceFull(
                    reversecomplementrange1.begin(),
                    reversecomplementrange1.end(),
                    genomerange2.begin(),
                    genomerange2.end()
                );

                bestHamming = std::min(hammingdistanceFwd, hammingdistanceRevc);
            }
            
            bool isGoodHamming = bestHamming <= maxMismatchesBetweedMappedRegions;                        

            unsigned int status = isSameChromosome ? 1u : 0u;
            status = (status << 1) | (isOrientationOk ? 1u : 0u);
            status = (status << 1) | (isPositionOk ? 1u : 0u);
            status = (status << 1) | (isGoodHamming ? 1u : 0u);
            statusmap[status]++;
        }else{
            oneIsUnmapped++;
        }
    }

    std::cout << "Processed " << processedlines << " lines.\n";
    std::cout << "oneIsUnmapped: " << oneIsUnmapped << "\n";
    std::cout << "numClipped: " << numClipped << "\n";
    std::cout << "numRefClipped: " << numRefClipped << "\n";

    int numWithGoodHamming = 0;
    std::cout << "statusmap:\n";
    for(const auto& pair : statusmap){
        bool isGoodHamming = pair.first & 1;
        bool isPositionOk = (pair.first >> 1) & 1;
        bool isOrientationOk = (pair.first >> 2) & 1;
        bool isSameChromosome = (pair.first >> 3) & 1;
        std::cout << pair.first << " @ " << " isSameChromosome " << isSameChromosome << ", isOrientationOk " << isOrientationOk 
            << ", isPositionOk " << isPositionOk << ", isGoodHamming " << isGoodHamming << " : " << pair.second << "\n";

        numWithGoodHamming += (isGoodHamming ? pair.second : 0);
    }
    std::cout << "\n";
    std::cout << "Total reads with region hamming distance <= " << maxMismatchesBetweedMappedRegions << " : " << numWithGoodHamming << "\n";
}