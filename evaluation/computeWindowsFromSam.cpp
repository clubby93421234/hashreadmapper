#include <genome.hpp>
#include <util.hpp>

#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <vector>
#include <sstream>
#include <cassert>




int main(int argc, char** argv){
    if(argc < 5){
        std::cout << "Usage: " << argv[0] << " samfile genomefile windowlength k\n";
        std::cout << "For each non-secondary alignment, write chromosome and windowIds per read to stdout\n";
        std::cout << "If samfile is - , reads input from stdin\n";
        return 0;
    }

    const std::string samfilename = argv[1];
    const std::string genomefilename = argv[2];
    const int windowLength = std::atoi(argv[3]);
    const int k = std::atoi(argv[4]);

    const int windowStride = windowLength - k + 1;
    const bool useStdin = (samfilename == "-");

    std::ifstream samfile;
    if(!useStdin){
        samfile.open(samfilename);
    }else{
        samfile.open("/dev/stdin");
    }

    Genome genome(genomefilename);
    std::size_t linenr = 0;

    std::string line;
    while(bool(std::getline(samfile, line))){
        auto tokens = split(line, '\t');
        assert(tokens.size() >= 10);

        unsigned int samflags = std::stoi(tokens[1]);
        const bool isSecondary = (samflags & 0x100);
        const bool isSupplementary = (samflags & 0x800);
        if(!isSecondary && !isSupplementary){
            const std::string& chromosome = tokens[2];
            int referencePosition = std::stoi(tokens[3]) - 1; //tokens[3] is 1-based
            const std::string& seq = tokens[9];

            int bestWindow = genome.getWindowIdWithOverlap(
                windowLength,
                k,
                chromosome,
                referencePosition,
                seq.size(),
                seq.size() / 2
            );

            std::cout << chromosome;
            std::cout << ' ' << bestWindow;
            std::cout << '\n';
        }

        linenr++;
    }
}