#ifndef CARE_OPTIONS_HPP
#define CARE_OPTIONS_HPP

#include "cxxopts/cxxopts.hpp"

#include <string>
#include <vector>

namespace care
{

    enum class SequencePairType
    {
        Invalid,
        SingleEnd,
        PairedEnd,
    };

    enum class MapperType{
        primitiveSW = 0,
        SW = 1,
        sthelse = 2 
    };

    struct ProgramOptions{
        bool replicateGpuData = false;
        bool useQualityScores = false;
        bool showProgress = false;
        bool mustUseAllHashfunctions = false;
        int batchsize = 2048;
        int kmerlength = 16;
        int numHashFunctions = 16;
        int maxResultsPerMap = 65535;
        int windowSize = 128;
        int minTableHits = 4;
        int warpcore = 0;
        int threads = 1;
        int qualityScoreBits = 8;
        int minInsertSize = -1;
        int maxInsertSize = -1;
        float hashtableLoadfactor = 0.8f;
        float maxHammingPercent = 0.05f;

        SequencePairType pairType = SequencePairType::SingleEnd;

        MapperType mappType = MapperType::SW;

        std::size_t memoryForHashtables = 0;
        std::size_t memoryTotalLimit = 0;
        std::string save_binary_reads_to = "";
        std::string load_binary_reads_from = "";
        std::string save_hashtables_to = "";
        std::string load_hashtables_from = "";
        std::string tempdirectory = ".";
        std::string genomefile = "genome.fasta";       
        std::string outputfile = "output.txt";
        std::string outputdirectory = ".";
        std::vector<int> deviceIds;
        std::vector<std::string> inputfiles;

        ProgramOptions() = default;
        ProgramOptions(const ProgramOptions&) = default;
        ProgramOptions(ProgramOptions&&) = default;

        ProgramOptions(const cxxopts::ParseResult& pr);
    };

    std::ostream& operator<<(std::ostream&, const ProgramOptions&);

    std::string to_string(SequencePairType s);
    std::string to_string(MapperType s);

    void addOptions(cxxopts::Options& commandLineOptions);



}

#endif
