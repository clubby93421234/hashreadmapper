#include <options.hpp>
#include <memorymanagement.hpp>

#include <string>
#include <vector>
#include <thread>

namespace care{

    std::string to_string(SequencePairType s){
        switch (s)
        {
        case SequencePairType::Invalid:
            return "Invalid";
        case SequencePairType::SingleEnd:
            return "SingleEnd";
        case SequencePairType::PairedEnd:
            return "PairedEnd";
        default:
            return "Error";
        }
    }
    std::string to_string(MapperType s){
        switch (s)
        {
        case MapperType::primitiveSW:
            return "primitiv SW";
        case MapperType::SW:
            return "SW";
        case MapperType::sthelse:
            return "sthelse";
        default:
            return "Error";
        }
    }
    template<class T>
    std::string tostring(const T& t){
        return std::to_string(t);
    }

    template<>
    std::string tostring(const bool& b){
        return b ? "true" : "false";
    }

    ProgramOptions::ProgramOptions(const cxxopts::ParseResult& pr){
        ProgramOptions& result = *this;

        if(pr.count("showProgress")){
            result.showProgress = pr["showProgress"].as<bool>();
        }
        if(pr.count("enforceHashmapCount")){
            result.mustUseAllHashfunctions = pr["enforceHashmapCount"].as<bool>();
        }

        if(pr.count("batchsize")){
            result.batchsize = pr["batchsize"].as<int>();
        }

        if(pr.count("kmerlength")){
            result.kmerlength = pr["kmerlength"].as<int>();
        }

        if(pr.count("hashmaps")){
            result.numHashFunctions = pr["hashmaps"].as<int>();
        }

        if(pr.count("maxResultsPerMap")){
            result.maxResultsPerMap = pr["maxResultsPerMap"].as<int>();
        }

        if(pr.count("windowSize")){
            result.windowSize = pr["windowSize"].as<int>();
        }

        if(pr.count("minTableHits")){
            result.minTableHits = pr["minTableHits"].as<int>();
        }

        if(pr.count("warpcore")){
            result.warpcore = pr["warpcore"].as<int>();
        }

        if(pr.count("threads")){
            result.threads = pr["threads"].as<int>();
        }
        result.threads = std::min(result.threads, (int)std::thread::hardware_concurrency());

        if(pr.count("pairmode")){
            const std::string arg = pr["pairmode"].as<std::string>();

            if(arg == "se" || arg == "SE"){
                result.pairType = SequencePairType::SingleEnd;
            }else if(arg == "pe" || arg == "PE"){
                result.pairType = SequencePairType::PairedEnd;
            }else{
                result.pairType = SequencePairType::Invalid;
            }
        }
        //selection for different mappers
        if(pr.count("mappertype")){
            const std::string arg = pr["mappertype"].as<std::string>();

            if(arg == "SW" || arg == "sw"){
                result.mappType = MapperType::SW;
            }else if(arg == "PSW" || arg == "psw"){
                result.mappType = MapperType::primitiveSW;
            }else{
                result.mappType = MapperType::sthelse;
            }
        }

        auto parseMemoryString = [](const auto& string) -> std::size_t{
            if(string.length() > 0){
                std::size_t factor = 1;
                bool foundSuffix = false;
                switch(string.back()){
                    case 'K':{
                        factor = std::size_t(1) << 10; 
                        foundSuffix = true;
                    }break;
                    case 'M':{
                        factor = std::size_t(1) << 20;
                        foundSuffix = true;
                    }break;
                    case 'G':{
                        factor = std::size_t(1) << 30;
                        foundSuffix = true;
                    }break;
                }
                if(foundSuffix){
                    const auto numberString = string.substr(0, string.size()-1);
                    return factor * std::stoull(numberString);
                }else{
                    return std::stoull(string);
                }
            }else{
                return 0;
            }
        };

        if(pr.count("memTotal")){
            const auto memoryTotalLimitString = pr["memTotal"].as<std::string>();
            const std::size_t parsedMemory = parseMemoryString(memoryTotalLimitString);
            const std::size_t availableMemory = getAvailableMemoryInKB() * 1024;

            // user-provided memory limit could be greater than currently available memory.
            result.memoryTotalLimit = std::min(parsedMemory, availableMemory);
        }else{
            std::size_t availableMemoryInBytes = getAvailableMemoryInKB() * 1024;
            if(availableMemoryInBytes > 2*(std::size_t(1) << 30)){
                availableMemoryInBytes = availableMemoryInBytes - 2*(std::size_t(1) << 30);
            }

            result.memoryTotalLimit = availableMemoryInBytes;
        }

        if(pr.count("memHashtables")){
            const auto memoryForHashtablesString = pr["memHashtables"].as<std::string>();
            result.memoryForHashtables = parseMemoryString(memoryForHashtablesString);
        }else{
            std::size_t availableMemoryInBytes = result.memoryTotalLimit;
            constexpr std::size_t safety = std::size_t(1) << 30;
            if(availableMemoryInBytes > safety){
                availableMemoryInBytes = availableMemoryInBytes - safety;
            }

            result.memoryForHashtables = availableMemoryInBytes;
        }

        result.memoryForHashtables = std::min(result.memoryForHashtables, result.memoryTotalLimit);

        if(pr.count("save-preprocessedreads-to")){
            result.save_binary_reads_to = pr["save-preprocessedreads-to"].as<std::string>();
        }

        if(pr.count("load-preprocessedreads-from")){
            result.load_binary_reads_from = pr["load-preprocessedreads-from"].as<std::string>();
        }

        if(pr.count("save-hashtables-to")){
            result.save_hashtables_to = pr["save-hashtables-to"].as<std::string>();
        }

        if(pr.count("load-hashtables-from")){
            result.load_hashtables_from = pr["load-hashtables-from"].as<std::string>();
        }

        if(pr.count("tempdir")){
            result.tempdirectory = pr["tempdir"].as<std::string>();
        }else{
            result.tempdirectory = result.outputdirectory;
        }

        if(pr.count("genomefile")){
            result.genomefile = pr["genomefile"].as<std::string>();
        }      

        if(pr.count("outdir")){
		    result.outputdirectory = pr["outdir"].as<std::string>();
        }

        if(pr.count("inputfiles")){
            result.inputfiles = pr["inputfiles"].as<std::vector<std::string>>();
        }

        if(pr.count("outputfilename")){
            result.outputfile = pr["outputfilename"].as<std::string>();
        }

        if(pr.count("gpu")){
            result.deviceIds = pr["gpu"].as<std::vector<int>>();
        }

        if(pr.count("maxHammingPercent")){
            result.maxHammingPercent = pr["maxHammingPercent"].as<float>();
        }

        if(pr.count("maxInsertSize")){
            result.maxInsertSize = pr["maxInsertSize"].as<int>();
        }

        if(pr.count("minInsertSize")){
            result.minInsertSize = pr["minInsertSize"].as<int>();
        }

        
    }


    std::ostream& operator<<(std::ostream& os, const ProgramOptions& opts){
        os << "mustUseAllHashfunctions: " << opts.mustUseAllHashfunctions << "\n";
        os << "batchsize: " << opts.batchsize << "\n";
        os << "kmerlength: " << opts.kmerlength << "\n";
        os << "numHashFunctions: " << opts.numHashFunctions << "\n";
        os << "maxResultsPerMap: " << opts.maxResultsPerMap << "\n";
        os << "windowSize: " << opts.windowSize << "\n";
        os << "minTableHits: " << opts.minTableHits << "\n";        
        os << "warpcore: " << opts.warpcore << "\n";
        os << "threads: " << opts.threads << "\n";
        os << "pairType: " << to_string(opts.pairType) << "\n";
        os << "genomefile: " << opts.genomefile << "\n";
        os << "MapperType: " << to_string(opts.mappType) << "\n";
        os << "inputfiles: { ";
        for(const auto& s : opts.inputfiles){
            os << s << " ";
        }
        os << "}\n";
        os << "deviceIds: { ";
        for(const auto& s : opts.deviceIds){
            os << s << " ";
        }
        os << "}\n";
        os << "maxHammingPercent: " << opts.maxHammingPercent << "\n";
        os << "maxInsertSize: " << opts.maxInsertSize << "\n";
        os << "minInsertSize: " << opts.minInsertSize << "\n";

        return os;
    }

    

    void addOptions(cxxopts::Options& commandLineOptions){
        commandLineOptions.add_options("Options")
            ("p,showProgress", "If set, progress bar is shown during correction",
                cxxopts::value<bool>()->implicit_value("true"))
            ("enforceHashmapCount",
                "If the requested number of hash maps cannot be fullfilled, the program terminates without error correction. "
                "Default: " + tostring(ProgramOptions{}.mustUseAllHashfunctions),
                cxxopts::value<bool>()->implicit_value("true")
            )
            ("b,batchsize", "Number of reads to correct in a single batch. Must be greater than 0. "
			    "Default: " + tostring(ProgramOptions{}.batchsize),
                cxxopts::value<int>())		
            ("k,kmerlength", "The kmer length for minhashing. ", cxxopts::value<int>())
            ("h,hashmaps", "The requested number of hash maps. Must be greater than 0. "
                "The actual number of used hash maps may be lower to respect the set memory limit. "
                "Default: " + tostring(ProgramOptions{}.numHashFunctions), 
                cxxopts::value<int>())
            ("maxResultsPerMap", "maxResultsPerMap. "
                "Default: " + tostring(ProgramOptions{}.maxResultsPerMap),
                cxxopts::value<int>())
            ("w,windowSize", "windowSize. "
                "Default: " + tostring(ProgramOptions{}.windowSize),
                cxxopts::value<int>())
            ("minTableHits", "minTableHits. "
                "Default: " + tostring(ProgramOptions{}.minTableHits),
                cxxopts::value<int>())
            ("warpcore", "Enable warpcore hash tables. 0: Disabled, 1: Enabled. "
                "Default: " + tostring(ProgramOptions{}.warpcore),
                cxxopts::value<int>())
            ("t,threads", "Maximum number of thread to use. Must be greater than 0", cxxopts::value<int>())
            ("pairmode", 
                "Type of input reads."
                "SE / se : Single-end reads"
                "PE / pe : Paired-end reads",
                cxxopts::value<std::string>())
           ("mappertype", 
                "Type of mapping algoithm."
                "Others can be implemented."
                "SW / sw : Smith-Waterman",
                cxxopts::value<std::string>())
            ("memHashtables", "Memory limit in bytes for hash tables and hash table construction. Can use suffix K,M,G , e.g. 20G means 20 gigabyte. This option is not a hard limit. Default: A bit less than memTotal.",
            cxxopts::value<std::string>())
            ("m,memTotal", "Total memory limit in bytes. Can use suffix K,M,G , e.g. 20G means 20 gigabyte. This option is not a hard limit. Default: All free memory.",
            cxxopts::value<std::string>())
            ("save-preprocessedreads-to", "Save binary dump of data structure which stores input reads to disk",
            cxxopts::value<std::string>())
            ("load-preprocessedreads-from", "Load binary dump of read data structure from disk",
            cxxopts::value<std::string>())
            ("save-hashtables-to", "Save binary dump of hash tables to disk. Ignored for GPU hashtables.",
            cxxopts::value<std::string>())
            ("load-hashtables-from", "Load binary dump of hash tables from disk. Ignored for GPU hashtables.",
            cxxopts::value<std::string>())
            ("tempdir", "Directory to store temporary files. Default: output directory", cxxopts::value<std::string>())
            ("genomefile", "genome fasta", cxxopts::value<std::string>())
            ("d,outdir", "The output directory. Will be created if it does not exist yet.", 
            cxxopts::value<std::string>())
            ("i,inputfiles", 
                "The file(s) to correct. "
                "Fasta or Fastq format. May be gzip'ed. "
                "Repeat this option for each input file (e.g. -i file1.fastq -i file2.fastq). "
                "Must not mix fasta and fastq files. "
                "The collection of input files is treated as a single read library",
                cxxopts::value<std::vector<std::string>>())
            ("o,outputfilename", 
                "output file name. will be placed in output directory. ", 
                cxxopts::value<std::string>())
            ("g,gpu", "Comma-separated list of GPU device ids to be used. (Example: --gpu 0,1 to use GPU 0 and GPU 1)", cxxopts::value<std::vector<int>>())
            ("maxHammingPercent", "A read of length l is only accepted if hamming distance to window is <= maxHammingPercent / 100 * l", cxxopts::value<float>())
            ("maxInsertSize", "Maximum fragment size to consider for paired end reads", cxxopts::value<int>())
            ("minInsertSize", "Minimum fragment size to consider for paired end reads", cxxopts::value<int>());
 
    }

}
