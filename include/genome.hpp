#ifndef GENOME_HPP
#define GENOME_HPP

#include <readlibraryio.hpp>
#include <util.hpp>

#include <cassert>
#include <cstddef>
#include <vector>
#include <string>
#include <string_view>
#include <map>

#include <sequencehelpers.hpp>

struct FastaIndex{
public:
    struct Entry{
        int lineLength;
        int lineLengthWithNewline;
        std::size_t length;
        std::size_t byteOffset;
    };
    /*
        Loads fasta index file which stores lengths and byte offsets of sequences, for example
        cat c_elegans.WS222.genomic.fa.fai 
        CHROMOSOME_I	15072423	14	50	51
        CHROMOSOME_II	15279345	15373901	50	51
        CHROMOSOME_III	13783700	30958849	50	51
        CHROMOSOME_IV	17493793	45018238	50	51
        CHROMOSOME_V	20924149	62861921	50	51
        CHROMOSOME_X	17718866	84204567	50	51
        CHROMOSOME_MtDNA	13794	102277829	50	51
    */

    FastaIndex() = default;
    FastaIndex(const std::string& filename){
        init(filename);
    }

    void init(const std::string& filename){
        std::ifstream is(filename);
        assert(bool(is));

        std::string line;
        while(std::getline(is, line)){
            auto tokens = split(line, '\t');
            assert(tokens.size() == 5);
            const auto& name = tokens[0];
            Entry entry{
                std::stoi(tokens[3]),
                std::stoi(tokens[4]),
                std::stoull(tokens[1]),
                std::stoull(tokens[2])
            };
            names.push_back(name);
            entryMap[name] = entry;
        }
    }

    std::size_t getLength(const std::string& name) const{
        auto it = entryMap.find(name);        
        assert(it != entryMap.end());
        return it->second.length;
    }

    std::size_t getLength(std::size_t nameIndex) const{
        assert(nameIndex < names.size());
        return getLength(names[nameIndex]);
    }

    std::size_t getNameIndex(const std::string& name) const{
        auto it = std::find(names.begin(), names.end(), name);
        assert(it != names.end());
        return std::distance(names.begin(), it);
    }

private:
    std::map<std::string, Entry> entryMap{};
    std::vector<std::string> names{};
};


struct Genome{
    struct BatchOfWindows{
        int numWindows = 0;
        int maxWindowSize = 128;
        const Genome* genome = nullptr;
        std::vector<int> chromosomeIds{}; // which chromosome (chromosome with name given by genome.names[id])
        std::vector<int> windowIds{}; //window number within chromosome
        std::vector<int> globalWindowIds{}; //window number within genome
        std::vector<int> positions{}; //window begin in chromosome
        std::vector<int> windowLengths{}; //window length
        std::vector<std::string_view> windowsDecoded{}; //window sequence

        void reset(){
            numWindows = 0;
            chromosomeIds.clear();
            windowIds.clear();
            globalWindowIds.clear();
            positions.clear();
            windowLengths.clear();
            windowsDecoded.clear();
        }
    };

    struct ExtendedWindow{
        int extensionLeft;
        int extensionRight;
        std::string_view sequence;
    };

    struct Section{
        int begin;
        int end;
        std::string_view sequence;
    };


    Genome(const std::string& filename){
        kseqpp::KseqPP reader(filename);

        auto getNextSequence = [&](){
            const int status = reader.next();
            if(status >= 0){

            }else {
                if(status < -1){
                    std::cerr << "parser error status " << status << " in file " << filename << '\n';
                }
            }

            bool success = (status >= 0);

            return success;
        };

        bool success = getNextSequence();

        while(success){
            std::string s = str_toupper(reader.getCurrentSequence());
            std::string h = reader.getCurrentHeader();

            data[names.size()] = s;

            names.push_back(h);

            success = getNextSequence();
        }
    }
    //copy Genome and saves as Reversecomplement
    Genome(const Genome& old){
        for(std::size_t chromosomeId = 0; chromosomeId < old.names.size(); chromosomeId++){

            auto mapiter = old.data.find(chromosomeId);
            assert(mapiter != old.data.end());
            const auto& sequence = mapiter->second;

            data.emplace( chromosomeId, SequenceHelpers::reverseComplementSequenceDecoded( &sequence[0], sequence.size()) );
            names.emplace_back( old.names[chromosomeId] );
            
            //std::cout << names[chromosomeId] << ", length " << sequence.size() << "\n";
        }
    }


    void printInfo() const{
        for(std::size_t chromosomeId = 0; chromosomeId < names.size(); chromosomeId++){
            auto mapiter = data.find(chromosomeId);
            assert(mapiter != data.end());
            const auto& sequence = mapiter->second;

            std::cout << names[chromosomeId] << ", length " << sequence.size() << "\n";
        }
    }

    std::size_t getNumWindowsInChromosome(int chromosomeId, int kmerSize, int windowSize) const{
        const std::size_t stride = windowSize - kmerSize + 1;
        auto mapiter = data.find(chromosomeId);
        assert(mapiter != data.end());
        const std::size_t length = mapiter->second.size();
        return (length + stride - 1) / stride;
    }

    std::vector<std::size_t> getNumWindowsPerChromosome(int kmerSize, int windowSize) const{
        std::vector<std::size_t> result;

        for(std::size_t chromosomeId = 0; chromosomeId < names.size(); chromosomeId++){
            result.push_back(getNumWindowsInChromosome(chromosomeId, kmerSize, windowSize));
        }

        return result;
    }

    std::size_t getTotalNumWindows(int kmerSize, int windowSize) const{
        auto nums = getNumWindowsPerChromosome(kmerSize, windowSize);
        return std::reduce(nums.begin(), nums.end());
    }

    //get sequence of specific window. sequence is extended to the left and to the right by extension bases
    std::string_view getWindowSequence(int chromosomeId, int windowId, int kmerSize, int windowSize) const{
        const std::size_t stride = windowSize - kmerSize + 1;
        const std::size_t windowBegin = stride * windowId;
        const auto& sequence = data.find(chromosomeId)->second;
        assert(windowBegin < sequence.size());
        const std::size_t end = std::min(sequence.size(), windowBegin + windowSize);
        const std::size_t length = end - windowBegin;
        std::string_view windowSequence(sequence.data() + windowBegin, length);
        return windowSequence;
    }

    ExtendedWindow getExtendedWindowSequence(int chromosomeId, int windowId, int kmerSize, int windowSize, std::size_t extension) const{
        const std::size_t stride = windowSize - kmerSize + 1;
        const std::size_t windowBegin = stride * windowId;
        const auto& sequence = data.find(chromosomeId)->second;
        assert(windowBegin < sequence.size());

        int length = windowSize;

        ExtendedWindow window;
        if(extension < windowBegin){
            window.extensionLeft = extension;
            length += extension;
        }else{
            window.extensionLeft = 0;
        }
        if(windowBegin + windowSize <= sequence.size()){
            if(windowBegin + windowSize + extension < sequence.size()){
                window.extensionRight = extension;
            }else{
                window.extensionRight = sequence.size() - (windowBegin + windowSize);
            }
            length += window.extensionRight;
        }else{
            window.extensionRight = 0;
            length -= (windowBegin + windowSize) - sequence.size();
        }

        window.sequence = std::string_view(sequence.data() + windowBegin - window.extensionLeft, length);

        return window;
    }

    Section getSectionOfGenome(int chromosomeId, int begin, int end) const{
        const auto& sequence = data.find(chromosomeId)->second;
        int size = sequence.size();
        assert(begin <= end);   
        if(begin < 0){
            begin = 0;
        }
        if(end > size){
            end = size;
        }
        const int length = end - begin;
        return Section{begin, end, std::string_view(sequence.data() + begin, length)};
    }

    template<class Func>
    void forEachWindow(int kmerSize, int windowSize, Func&& callback, std::size_t limit = 0) const {
        const std::size_t stride = windowSize - kmerSize + 1;

        std::size_t globalWindowId = 0;

        for(std::size_t chromosomeId = 0; chromosomeId < names.size(); chromosomeId++){
            const std::size_t numWindows = getNumWindowsInChromosome(chromosomeId, kmerSize, windowSize);
            for(std::size_t pos = 0, windowId = 0; windowId < numWindows; pos += stride, windowId++, globalWindowId++){
                std::string_view windowSequence = getWindowSequence(chromosomeId, windowId, kmerSize, windowSize);

                callback(chromosomeId, windowSequence, pos, windowId, globalWindowId);

                if(limit > 0){
                    if(globalWindowId == limit) break;
                }
            }

            if(limit > 0){
                if(globalWindowId == limit) break;
            }
        }
    }

    template<class Func>
    void forEachWindowInChromosome(std::size_t chromosomeId, int kmerSize, int windowSize, Func&& callback, std::size_t limit = 0) const {
        assert(chromosomeId < names.size());
        const std::size_t stride = windowSize - kmerSize + 1;
        const std::vector<std::size_t> numWindowsPerChromosome = getNumWindowsPerChromosome(kmerSize, windowSize);

        //std::cout << "inside...\n";
        std::size_t globalWindowId = numWindowsPerChromosome[chromosomeId];
        const std::size_t numWindows = getNumWindowsInChromosome(chromosomeId, kmerSize, windowSize);
        for(std::size_t pos = 0, windowId = 0; windowId < numWindows; pos += stride, windowId++, globalWindowId++){
            std::string_view windowSequence = getWindowSequence(chromosomeId, windowId, kmerSize, windowSize);
            
            //std::cout << "Processing.inside  .."<<windowId<<"\n";
            callback(chromosomeId, windowSequence, pos, windowId, globalWindowId);
            
            //std::cout << "Processing...\n";
            if(limit > 0){
                std::cout << "limit...\n";
                if(globalWindowId == limit) break;
            }
        }
    }

    template<class Func>
    void forEachBatchOfWindows(
        int kmerSize,
        int windowSize,
        int batchsize,
        Func&& callback
    ) const {
        int numProcessedBatches = 0;
        BatchOfWindows batch;
        batch.genome = this;
        batch.maxWindowSize = windowSize;

        auto submitBatch = [&](BatchOfWindows& batch){
            callback(batch);
            batch.reset();
        };
        //std::cout << "before addwindowtobatch...\n";
        auto addWindowToBatch = [&](std::size_t chromosomeId, const auto& windowSequence, std::size_t positionInWindow, std::size_t windowId, std::size_t globalWindowId){
            auto mapiter = data.find(chromosomeId);
            assert(mapiter != data.end());
            assert(mapiter->second.size() > std::size_t(positionInWindow));

            batch.chromosomeIds.push_back(chromosomeId);
            batch.windowIds.push_back(windowId);
            batch.globalWindowIds.push_back(globalWindowId);
            batch.positions.push_back(positionInWindow);
            batch.windowLengths.push_back(windowSequence.size());
            batch.windowsDecoded.push_back(windowSequence);
            batch.numWindows++;

            if(batch.numWindows == batchsize){
                submitBatch(batch);
                numProcessedBatches++;
            }
        };

        for(std::size_t chromosomeId = 0; chromosomeId < names.size(); chromosomeId++){
            //std::cout << "im loop"<<chromosomeId<<"\n";
            forEachWindowInChromosome(
                chromosomeId,
                kmerSize,
                windowSize,
                addWindowToBatch
            );
            //std::cout << "almost done...\n";
            //process last incomplete batch of chromosome
            if(batch.numWindows > 0){
                submitBatch(batch);
            }
        }
    }


    //find windows which overlap with read by at least 1 base
    std::vector<int> getWindowIds(
        int windowLength,
        int k,
        const std::string& chrName, 
        int pos, 
        int length
    ) const{
        assert(pos >= 0);

        //check that sequence does not exceed chromosome
        auto nameIndex = getNameIndex(chrName);
        auto iter = data.find(nameIndex);
        const int chrLength = iter->second.size();
        const int seqEnd = std::min(pos + length, chrLength);
        length = seqEnd - pos;

        const int windowStride = windowLength - k + 1;
        const int firstWindowId = pos / windowStride;
        const int lastWindowId = (pos + length - 1) / windowStride;

        std::vector<int> result;
        for(int i = firstWindowId; i <= lastWindowId; i++){
            result.push_back(i);
        }

        return result;
    }

    //find windows which overlap with at least overlap bases of read
    int getWindowIdWithOverlap(
        int windowLength,
        int k,
        const std::string& chrName, 
        int pos, 
        int length,
        int overlap
    ) const{
        assert(pos >= 0);
        assert(windowLength >= overlap);

        //check that sequence does not exceed chromosome
        auto nameIndex = getNameIndex(chrName);
        auto iter = data.find(nameIndex);
        const int chrLength = iter->second.size();
        const int seqEnd = std::min(pos + length, chrLength);
        length = seqEnd - pos;

        const int windowStride = windowLength - k + 1;
        const int firstWindowId = pos / windowStride;
        const int lastWindowId = (pos + length - 1) / windowStride;

        int best = -1;
        for(int i = firstWindowId; i <= lastWindowId; i++){
            const int windowBegin = i * windowStride;
            const int windowEnd = (i+1) * windowStride;
            //find overlapping bases
            const int overlapBegin = std::max(windowBegin, pos);
            const int overlapEnd = std::min(windowEnd, seqEnd);
            if(overlapEnd - overlapBegin >= overlap){
                best = i;
                break;
            }
        }

        assert(best != -1);
        return best;
    }

    const std::string& getSequenceName(std::size_t id) const{
        assert(id < names.size());
        return names[id];
    }

    const std::string& getSequence(std::size_t id) const{
        assert(id < names.size());
        return data.find(id)->second;
    }

    std::size_t getNameIndex(const std::string& name) const{
        auto it = std::find(names.begin(), names.end(), name);
        assert(it != names.end());
        return std::distance(names.begin(), it);
    }

public:
    std::map<int, std::string> data;
    std::vector<std::string> names;

};




#endif