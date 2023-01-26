#ifndef WINDOW_HIT_STATISTIC_COLLECTOR_HPP
#define WINDOW_HIT_STATISTIC_COLLECTOR_HPP

#include <referencewindows.hpp>

#include <cstddef>
#include <vector>
#include <string>
#include <fstream>

struct WindowHitStatisticCollector{
public:
    struct WindowHits{
        int chromosomeId{};
        int windowId{};
        int truehits{};
        int falsehits{};
    };

    WindowHitStatisticCollector(std::shared_ptr<ReferenceWindows> referenceWindows_)
        : referenceWindows(referenceWindows_)
    {        
        
        const std::size_t numChr = referenceWindows->getNumChromosomes();
        assert(numChr > 0);
        windowHitStats.resize(referenceWindows->getNumWindows(numChr - 1) + referenceWindows->getChromosomeWindowOffset(numChr - 1));

        for(std::size_t i = 0; i < numChr; i++){
            const std::size_t numWindows = referenceWindows->getNumWindows(i);
            const std::size_t offset = referenceWindows->getChromosomeWindowOffset(i);

            for(std::size_t w = 0; w < numWindows; w++){
                const std::size_t index = offset + w;
                windowHitStats[index].chromosomeId = i;
                windowHitStats[index].windowId = w;
                windowHitStats[index].truehits = 0;
                windowHitStats[index].falsehits = 0;
            }
        }
    }

    template<class Iter>
    void addHits(int chromosomeId, int windowId, Iter readIdsBegin, Iter readIdsEnd){
        const RefWindow refWindowToFind{chromosomeId, windowId};

        int truehits = 0;
        int falsehits = 0;

        for(auto it = readIdsBegin; it != readIdsEnd; ++it){
            if((*referenceWindows)[*it] == refWindowToFind){
                truehits++;
            }else{
                falsehits++;
            }
        }

        const std::size_t offset = referenceWindows->getChromosomeWindowOffset(chromosomeId);
        const std::size_t index = offset + windowId;
        windowHitStats[index].truehits += truehits;
        windowHitStats[index].falsehits += falsehits;
    }

    template<class Func>
    void forEachWindow(Func callback){
        std::for_each(
            windowHitStats.begin(),
            windowHitStats.end(),
            callback
        );
    }

private:
    std::shared_ptr<ReferenceWindows> referenceWindows;
    std::vector<WindowHits> windowHitStats;
};



#endif