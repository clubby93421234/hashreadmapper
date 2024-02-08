#ifndef MAPPINGHANDLER_CUH
#define MAPPINGHANDLER_CUH

#include <hpc_helpers.cuh>
#include <config.hpp>
#include <options.hpp>

#include <genome.hpp>
#include <sequencehelpers.hpp>
#include <chunkedreadstorage.hpp>

#include <alignmentorientation.hpp>
#include <threadpool.hpp>

#include <algorithm>
#include <vector>
#include <iostream>
#include <sstream> 
#include <mutex>
#include <thread>
#include <memory>
#include <cassert>
#include <map>
#include <string>
#include <string_view>
#include <omp.h>
#include <future>

// Complete-Striped-Smith-Waterman-Library
#include<ssw_cpp.h>
// BAM Variant Caller and Genomic Analysis
#include "cigar.hpp"
#include "constants.hpp"
#include "varianthandler.hpp"

#include <gpu/mappedread.cuh>
#include "edlib.h"
using namespace care;

class Mappinghandler{
    public:

        Mappinghandler(
            const ProgramOptions* programOptions_,
            const Genome* genome_,
            const Genome* genomeRC_,
             std::vector<MappedRead>* results_
         //    ,             std::vector<MappedRead>* resultsRC_
             );
        ~Mappinghandler();

        void go(std::unique_ptr<ChunkedReadStorage>& cpuReadStorage_);
        void doVC();

    private:
        const ProgramOptions* programOptions;
        const Genome* genome;
        const Genome* genomeRC;
        int mappertype=1;
        std::vector<MappedRead>* results;
       // std::vector<MappedRead>* resultsRC;
        

        INLINEQUALIFIER
        void NucleoideConverer(char* output, const char* input, int length);

struct AlignerArguments{
    uint16_t flag=0;
    uint16_t flag_rc=0;           
           std::string_view query;
            std::string three_n_query;   //3 nucleotide query
          std::string rc_query;
            std::string three_n_rc_query;

         std::string ref;//the window of the reference genome
            std::string three_n_ref;
         std::string rc_ref; //the window of the reference genomeRC
            std::string three_n_rc_ref;

        
            int ref_len;
        StripedSmithWaterman::Filter filter;
            std::size_t windowlength;
     //       std::size_t windowlengthRC;
            int32_t maskLen;
        MappedRead result;
      //  MappedRead resultRC;
            read_number readId;
            std::string readsequence;
        std::string rev;
        std::vector<StripedSmithWaterman::Alignment> alignments;
        std::vector<int> num_conversions;
        
            AlignerArguments():
                            alignments({StripedSmithWaterman::Alignment(),StripedSmithWaterman::Alignment()}),
                            num_conversions({0,0})
            {
            }

        };

        //------------------------------------------------------------------------------
struct Edlibhelper{
    std::string queryOriginal;
    std::string queryOriginal_threen;
    std::string queryOriginal_rc;
    std::string queryOriginal_rc_threen;
    int queryLength;

    int queryStart;
    int queryStart_rc;

    std::string targetOriginal;
    std::string targetOriginal_threen;
    std::string targetOriginal_rc;
    std::string targetOriginal_rc_threen;
    int targetLength;    
    
    uint16_t flag = 0;
    uint16_t flag_rc = 0;
    std::string cigar;
    std::string cigar_rc;

    int score;
    int score_rc;

    int num_conversions;
    int num_conversions_rc;

    MappedRead result;
    read_number readId;
};
//-----------------------------------------------------------------------
        void printtoSAM();
        void printtoedlibSAM();
        void CSSW(std::unique_ptr<ChunkedReadStorage>& cpuReadStorage);

        void edlibAligner(std::unique_ptr<ChunkedReadStorage>& cpuReadStorage);
        void examplewrapper(std::unique_ptr<ChunkedReadStorage>& cpuReadStorage);
        uint32_t mapqfkt(int i, int j);

        std::vector<AlignerArguments> mappingout;
        std::vector<Edlibhelper> edlibout;
};

#endif
