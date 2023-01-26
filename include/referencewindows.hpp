#ifndef REFERENCEWINDOWS_HPP
#define REFERENCEWINDOWS_HPP

#include <genome.hpp>

#include <vector>
#include <fstream>
#include <cassert>
#include <string>
#include <map>
#include <algorithm>
#include <numeric>
struct RefWindow{
    int chromosomeId{};
    int windowId{};
    bool operator==(const RefWindow& rhs) const{
        return chromosomeId == rhs.chromosomeId && windowId == rhs.windowId;
    }
    bool operator!=(const RefWindow& rhs) const{
        return !operator==(rhs);
    }
};

/*
    For each read, stores the window which overlaps by at least 50% of the read
    Assumes windowsize >= readlength / 2, i.e exactly 1 window per read
*/
struct ReferenceWindows{
public:
    //reference windows file produces via evaluation/computeWindowsFromSam
    static ReferenceWindows fromReferenceWindowsFile(const Genome& genome, const std::string& filename, int kmerSize, int windowSize){
        ReferenceWindows result;

        std::map<int, std::size_t> numWindowsMap;

        std::ifstream refwindowsstream(filename);
        assert(bool(refwindowsstream));
        std::size_t readId = 0;
        std::string line;
        while(std::getline(refwindowsstream, line)){
            auto tokens = split(line, ' ');
            assert(tokens.size() == 2);
            const auto& chromosomename = tokens[0];
            const int chromId = genome.getNameIndex(chromosomename);

            RefWindow refWindow;
            refWindow.chromosomeId = chromId;
            refWindow.windowId = std::stoi(tokens[1]);
            result.data.push_back(refWindow);

            numWindowsMap[chromId]++;

            readId++;
        }

        result.numWindowsPerChr = genome.getNumWindowsPerChromosome(kmerSize, windowSize);
        result.numWindowsPerChrPrefixSum.resize(result.numWindowsPerChr.size() + 1);
        result.numWindowsPerChrPrefixSum[0] = 0;
        std::inclusive_scan(result.numWindowsPerChr.begin(), result.numWindowsPerChr.end(), result.numWindowsPerChrPrefixSum.begin() + 1);

        return result;
    }

    std::size_t getNumWindows(int chrId) const{
        return numWindowsPerChr[chrId];
    }

    std::size_t getChromosomeWindowOffset(int chrId) const{
        return numWindowsPerChrPrefixSum[chrId];
    }

    std::size_t getNumChromosomes() const{
        return numWindowsPerChr.size();
    }

    auto& operator[](std::size_t index){
        return data[index];
    }

    const auto& operator[](std::size_t index) const{
        return data[index];
    }

    std::size_t size() const{
        return data.size();
    }
private:
    std::vector<std::size_t> numWindowsPerChr;
    std::vector<std::size_t> numWindowsPerChrPrefixSum;
    std::vector<RefWindow> data;
};



#endif