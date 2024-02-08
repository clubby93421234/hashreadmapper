#include "gpu/mappinghandler.cuh"

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
#include <ssw_cpp.h>
// BAM Variant Caller and Genomic Analysis
#include "cigar.hpp"
#include "constants.hpp"
#include "varianthandler.hpp"

#include <gpu/mappedread.cuh>

#include "edlib.h"


/// @brief
/// @param programOptions_
/// @param genome_
/// @param genomeRC_
/// @param results_
/// @param resultsRC_
Mappinghandler::Mappinghandler(
    const ProgramOptions *programOptions_,
    const Genome *genome_,
    const Genome *genomeRC_,
    std::vector<MappedRead> *results_
    //,std::vector<MappedRead>* resultsRC_
    ) : programOptions(programOptions_),
        genomeRC(genomeRC_),
        genome(genome_),
        results(results_)
// ,      resultsRC(resultsRC_)
{
}

/// @brief
Mappinghandler::~Mappinghandler()
{
}

/// @brief Used to select a mapper and starts it
/// @param cpuReadStorage_ The storage of reads
void Mappinghandler::go(std::unique_ptr<ChunkedReadStorage> &cpuReadStorage_)
{
    // mappertype=(int) programOptions.mappType;

    switch (programOptions->mappType)
    {
    case MapperType::edlib:
        std::cout << "edlib selected!\n";
         edlibAligner(cpuReadStorage_);
        break;
    case MapperType::SW:
        // std::cout<<"SSW selected \n";
        CSSW(cpuReadStorage_);
        break;
    case MapperType::sthelse:
        std::cout << "please implement your personal mapper\n";
        break;
    default:
        std::cout << "something went wrong while selecting mappertype\n";
        examplewrapper(cpuReadStorage_);
        break;
    }
}

/// @brief TODO
void Mappinghandler::doVC()
{

    auto test = (programOptions->outputfile) + ".VCF";
    VariantHandler vhandler(test);
    vhandler.VCFFileHeader();

    uint32_t mapq = 0;

    for (long unsigned int i = 0; i < mappingout.size(); i++)
    {

        if (mappingout.at(i).alignments.at(0).sw_score >= mappingout.at(i).alignments.at(1).sw_score)
        {

            mapq = mapqfkt(i, 0);

            if (mapq < MAP_QUALITY_THRESHOLD)
            {
                continue;
            }

            Cigar marlboro{mappingout.at(i).alignments.at(0).cigar_string};

            std::string prefix = mappingout.at(i).ref.substr(0, mappingout.at(i).alignments.at(0).query_begin);

            vhandler.call(
                mappingout.at(i).result.position + mappingout.at(i).alignments.at(0).query_begin, // seq position
                prefix,
                mappingout.at(i).ref,
                mappingout.at(i).query,
                marlboro.getEntries(),
                genome->names.at(mappingout.at(i).result.chromosomeId),
                mappingout.at(i).readId,
                mapq);
        }
        else
        {

            mapq = mapqfkt(i, 1);

            if (mapq < MAP_QUALITY_THRESHOLD)
            {
                continue;
            }
            Cigar marlboro{mappingout.at(i).alignments.at(1).cigar_string};

            std::string prefix = mappingout.at(i).ref.substr(0, mappingout.at(i).alignments.at(1).query_begin);

            vhandler.call(
                mappingout.at(i).result.position + mappingout.at(i).alignments.at(1).query_begin, // seq position
                prefix,
                mappingout.at(i).ref,
                mappingout.at(i).query,
                marlboro.getEntries(),
                genome->names.at(mappingout.at(i).result.chromosomeId),
                mappingout.at(i).readId,
                mapq);
        }
    }
    // vhandler.forceFlush();
}

INLINEQUALIFIER
/// @brief converts every 'C' into a 'T'
/// @param output
/// @param input
/// @param length
void Mappinghandler::NucleoideConverer(char *output, const char *input, int length)
{
    if (strlen(input) != 0)
    {

        for (int i = 0; i < length; ++i)
        {
            switch (input[i])
            {
            case 'C':
                output[i] = 'T';
                break;
            default:
                break;
            }
        }
    }
    else
        assert("Nucleotide Converter Failed");
}

 // MAPQ calculated as in CSSW
    // https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library/blob/master/src/main.c
    // Line 167 to  169
uint32_t Mappinghandler::mapqfkt(int i, int j)
{
   
    uint32_t _mapq = -4.343 * log(1 - (double)abs(mappingout.at(i).alignments.at(j).sw_score -
                                                  mappingout.at(i).alignments.at(j).sw_score_next_best) /
                                          (double)mappingout.at(i).alignments.at(j).sw_score);
    _mapq = (uint32_t)(_mapq + 4.99);
    _mapq = _mapq < 254 ? _mapq : 254;
    return _mapq;
}

/// @brief simple print methode for the vector mappingout into the "CSSW_SAM.SAM" file. Used for CSSW-Mapping
void Mappinghandler::printtoSAM()
{
auto test = (programOptions->outputfile)+".SAM";
    std::ofstream outputstream(test);
    std::cout<<"number of reads:"<<mappingout.size()<<"\n";
    long nootmapped=0;
    long moodmapped=0;
    outputstream << "@HD\tVN:1.4\n";
    //@SQ lines
     for (std::size_t i = 0; i < mappingout.size(); i++)
    {
        outputstream<<"@SQ"<<"\t"
                    <<"SN:"<<mappingout.at(i).readId  <<"\t"
                    <<"LN:"<<mappingout.at(i).windowlength<<"\n";
    }
    outputstream<<"@PG\tHashreadmapper\tID:1.0";
    outputstream<< "@CO: QNAME\tFLAG\tRNAME\tPOS\tMAPQ\tCIGAR\tRNEXT\tPNEXT\tTLEN\tSEQ\tQUAL\tTAG\n";

    for (std::size_t i = 0; i < mappingout.size(); i++)
    {

        std::string samtag = "";
        uint32_t mapq = 0;
        int pos = 0;
        std::string cig = "";
        uint16_t samflag=0;
        if (mappingout.at(i).alignments.at(0).sw_score >= mappingout.at(i).alignments.at(1).sw_score)
        {
            // if(mappingout.at(i).alignments.at(0).sw_score>5)
            // std::cout<<"1: "<<mappingout.at(i).alignments.at(0).sw_score<<" "<<i<<"\n";
            samtag.append("Yf:i:<");
            samtag.append(std::to_string(mappingout.at(i).num_conversions.at(0)));
            samtag.append(">");
            samtag.append("YZ:A:<+>"); // REF-3N
            samflag=mappingout.at(i).flag;
            mapq = mapqfkt(i, 0);
            pos = mappingout.at(i).result.position + mappingout.at(i).alignments.at(0).query_begin;
            cig.append(mappingout.at(i).alignments.at(0).cigar_string);
        }
        else
        {
            // if(mappingout.at(i).alignments.at(1).sw_score>5)
            // std::cout<<"2: "<<mappingout.at(i).alignments.at(1).sw_score<<" "<<i<<"\n";
            samtag.append("Yf:i:<");
            samtag.append(std::to_string(mappingout.at(i).num_conversions.at(1)));
            samtag.append(">");
            samtag.append("YZ:A:<->"); // REF-RC-3N
            samflag=mappingout.at(i).flag_rc;
            mapq = mapqfkt(i, 1);
            pos = mappingout.at(i).result.position + mappingout.at(i).alignments.at(1).query_begin;
            cig.append(mappingout.at(i).alignments.at(1).cigar_string);
        }
        if ((mappingout.at(i).flag & 0x4) == 0) {//check if unmapped bit is not set
            moodmapped++;
            outputstream << mappingout.at(i).readId << "\t" // QNAME
                << samflag << "\t"                                                // FLAG
                << genome->names.at(mappingout.at(i).result.chromosomeId) << "\t" // RNAME
                << pos << "\t"                                                    // POS //look up my shenanigans in ssw_cpp.cpp for why its queri_begin
                << mapq << "\t"                                                   // MAPQ
                << cig << "\t"                                                    // CIGAR
                << "="
                << "\t" // RNEXT
                << ""
                << "\t" // PNEXT
                << "0"
                << "\t"                           // TLEN
                << mappingout.at(i).query << "\t" // SEQ
                << "*"
                << "\t"           // QUAL
                << samtag << "\t" // TAG
                << "\n";
        }
        else {//print unmapped
            nootmapped++;
            outputstream << mappingout.at(i).readId << "\t" // QNAME
                << samflag << "\t"                                                // FLAG
                << genome->names.at(mappingout.at(i).result.chromosomeId) << "\t" // RNAME
                << pos << "\t"                                                    // POS //look up my shenanigans in ssw_cpp.cpp for why its queri_begin
                << mapq << "\t"                                                   // MAPQ
                << cig << "\t"                                                        // CIGAR
                << "="
                << "\t" // RNEXT
                << ""
                << "\t" // PNEXT
                << "0"
                << "\t"                           // TLEN
                << mappingout.at(i).query << "\t" // SEQ
                << "*"
                << "\t"           // QUAL
                << mappingout.at(i).flag << "\t" // TAG
                << "\n";
        }
    }
    outputstream.close();
    std::cout<<"mapped reads: "<<moodmapped<<"\n";

    std::cout<<"unmapped reads: "<<nootmapped<<"\n";
}

void Mappinghandler::printtoedlibSAM()
{
    auto test = (programOptions->outputfile)+".SAM";
    std::ofstream outputstream(test);

    outputstream << "@HD\tVN:1.4\n";
    //@SQ lines
     for (std::size_t i = 0; i < edlibout.size(); i++)
    {
        outputstream<<"@SQ"<<"\t"
                    <<"SN:"<<genome->getSequenceName(edlibout.at(i).readId)<<"\t"
                    <<"LN:"<<edlibout.at(i).targetLength <<"\n";
    }
    outputstream<<"@PG\tHashreadmapper\tID:1.0";
    outputstream << "@CO: QNAME\tFLAG\tRNAME\tPOS\tMAPQ\tCIGAR\tRNEXT\tPNEXT\tTLEN\tSEQ\tQUAL\tTAG\n";

    for (std::size_t i = 0; i < edlibout.size(); i++)
    {

        std::string samtag = "";
        uint32_t mapq = 0;
        int pos = 0;
        std::string cig = "";
        uint16_t samflag=0;
        if (edlibout.at(i).score>= edlibout.at(i).score_rc)
        {
            // if(mappingout.at(i).alignments.at(0).sw_score>5)
            // std::cout<<"1: "<<mappingout.at(i).alignments.at(0).sw_score<<" "<<i<<"\n";
            samtag.append("Yf:i:<");
            samtag.append(std::to_string(edlibout.at(i).num_conversions));
            samtag.append(">");
            samtag.append("YZ:A:<+>"); // REF-3N
            samflag=edlibout.at(i).flag;
            mapq = mapqfkt(i, 0);
            pos = edlibout.at(i).result.position + edlibout.at(i).queryStart;
            cig.append(edlibout.at(i).cigar_rc);
        }
        else
        {
            // if(edlibout.at(i).alignments.at(1).sw_score>5)
            // std::cout<<"2: "<<edlibout.at(i).alignments.at(1).sw_score<<" "<<i<<"\n";
            samtag.append("Yf:i:<");
            samtag.append(std::to_string(edlibout.at(i).num_conversions_rc));
            samtag.append(">");
            samtag.append("YZ:A:<->"); // REF-RC-3N
            samflag=edlibout.at(i).flag_rc;
            mapq = mapqfkt(i, 1);
            pos = edlibout.at(i).result.position + edlibout.at(i).queryStart_rc;
            cig.append(edlibout.at(i).cigar_rc);
        }
        if ((edlibout.at(i).flag & 0x4) == 0) {//check if unmapped bit is not set
            outputstream << edlibout.at(i).readId << "\t" // QNAME
                << samflag << "\t"                                                // FLAG
                << genome->names.at(edlibout.at(i).result.chromosomeId) << "\t" // RNAME
                << pos << "\t"                                                    // POS //look up my shenanigans in ssw_cpp.cpp for why its queri_begin
                << mapq << "\t"                                                   // MAPQ
                << cig << "\t"                                                    // CIGAR
                << "="                << "\t" // RNEXT
                << ""               << "\t" // PNEXT
                << "0"               << "\t"                           // TLEN
                << edlibout.at(i).queryOriginal << "\t" // SEQ
                << "*"
                << "\t"           // QUAL
                << samtag << "\t" // TAG
                << "\n";
        }
        else {//print unmapped
            
            outputstream << edlibout.at(i).readId << "\t" // QNAME
                << samflag << "\t"                                                // FLAG
                << genome->names.at(edlibout.at(i).result.chromosomeId) << "\t" // RNAME
                << pos << "\t"                                                    // POS //look up my shenanigans in ssw_cpp.cpp for why its queri_begin
                << mapq << "\t"                                                   // MAPQ
                << cig << "\t"                                                               // CIGAR
                << "="               << "\t" // RNEXT
                << ""                << "\t" // PNEXT
                << "0"               << "\t"                           // TLEN
                << edlibout.at(i).queryOriginal << "\t" // SEQ
                << "*" << "\t"           // QUAL
                << samtag<< "\t" // TAG
                << "\n";
        }
    }
    outputstream.close();
} 

// Complete-Striped-Smith-Waterman Mapper.
// https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library
void Mappinghandler::CSSW(std::unique_ptr<ChunkedReadStorage> &cpuReadStorage)
{

    StripedSmithWaterman::Aligner aligner;
    StripedSmithWaterman::Filter filter;
long nootmapped = 0;
    long moodmapped = 0;
    const int maximumSequenceLength = cpuReadStorage->getSequenceLengthUpperBound();

    std::size_t numreads = cpuReadStorage->getNumberOfReads();

    std::cout << "lets go bigfor:...\n";

    std::size_t processedResults = 0;
    for (std::size_t r = 0; r < numreads; r++)
    {

        const auto &result = (*results)[r];
        //            const auto& resultRC = (*resultsRC)[r];

        read_number readId = r;
        //std::cout<<"dasd\n";
        std::vector<int> readLengths(1);
        cpuReadStorage->gatherSequenceLengths(
            readLengths.data(),
            &readId,
            1);

        const int encodedReadNumInts2Bit = SequenceHelpers::getEncodedNumInts2Bit(maximumSequenceLength);
        std::vector<unsigned int> encodedReads(encodedReadNumInts2Bit * 1);

        cpuReadStorage->gatherSequences(
            encodedReads.data(),
            encodedReadNumInts2Bit,
            &readId,
            1);
        //std::cout<<"vor if reversecomplemen\n";
        if (result.orientation == AlignmentOrientation::ReverseComplement)
        {
            SequenceHelpers::reverseComplementSequenceInplace2Bit(encodedReads.data(), readLengths[0]);
        }

        auto readsequence = SequenceHelpers::get2BitString(encodedReads.data(), readLengths[0]);

        if (result.orientation != AlignmentOrientation::None)
        {
        // std::cout<<"mapped"<<readId<<"\n";
moodmapped++;
            const auto &genomesequence = (*genome).data.at(result.chromosomeId);
            const auto &genomesequenceRC = (*genomeRC).data.at(result.chromosomeId);

            const std::size_t windowlength = result.position +
                                                         programOptions->windowSize <
                                                     genomesequence.size()
                                                 ? programOptions->windowSize
                                                 : genomesequence.size() -
                                                       result.position;
            const std::size_t windowlengthRC = result.position +
                                                           programOptions->windowSize <
                                                       genomesequenceRC.size()
                                                   ? programOptions->windowSize
                                                   : genomesequenceRC.size() -
                                                         result.position;

            std::size_t aef = genomesequenceRC.size() - result.position - 1;
            std::string_view window(genomesequence.data() + result.position, windowlength);
            std::string_view windowRC(genomesequenceRC.data() + aef, windowlengthRC);

            processedResults++;

            int32_t maskLen = readLengths[0] / 2;
            maskLen = maskLen < 15 ? 15 : maskLen;

            AlignerArguments ali;
   
   ali.query{readsequence,readLengths[0]};
            ali.three_n_query.resize(readLengths[0]);
            NucleoideConverer(ali.three_n_query.data(), ali.query.data(), readLengths[0]);
            ali.rc_query = SequenceHelpers::reverseComplementSequenceDecoded(
               ali.query.data()
                , readLengths[0]);

            //ali.rc_query = SequenceHelpers::reverseComplementSequenceDecoded(ali.query.data(), readLengths[0]);

            ali.three_n_rc_query.resize(readLengths[0]);
            NucleoideConverer(ali.three_n_rc_query.data(), ali.rc_query.c_str(), readLengths[0]);

            ali.ref = std::string(window).c_str();
            ali.three_n_ref.resize(windowlength);
            NucleoideConverer(ali.three_n_ref.data(), ali.ref.c_str(), windowlength);

            ali.rc_ref = std::string(windowRC).c_str();
            ali.three_n_rc_ref.resize(windowlengthRC);
            NucleoideConverer(ali.three_n_rc_ref.data(), ali.rc_ref.c_str(), windowlengthRC);

            ali.filter = filter;

            ali.maskLen = maskLen;
            ali.readId = readId;
            ali.ref_len = windowlength;

            ali.result = result;
            //                    ali.resultRC=resultRC;

            ali.windowlength = windowlength;
            //        ali.windowlengthRC=windowlengthRC;
        
            mappingout.push_back(ali);
        }
        else
        {//read not mapped
nootmapped++;
            const auto &genomesequence = (*genome).data.at(result.chromosomeId);
            const auto &genomesequenceRC = (*genomeRC).data.at(result.chromosomeId);

            const std::size_t windowlength = result.position +
                                                         programOptions->windowSize <
                                                     genomesequence.size()
                                                 ? programOptions->windowSize
                                                 : genomesequence.size() -
                                                       result.position;
            const std::size_t windowlengthRC = result.position +
                                                           programOptions->windowSize <
                                                       genomesequenceRC.size()
                                                   ? programOptions->windowSize
                                                   : genomesequenceRC.size() -
                                                         result.position;

            std::size_t aef = genomesequenceRC.size() - result.position - 1;
            std::string_view window(genomesequence.data() + result.position, windowlength);
            std::string_view windowRC(genomesequenceRC.data() + aef, windowlengthRC);

            processedResults++;

            int32_t maskLen = readLengths[0] / 2;
            maskLen = maskLen < 15 ? 15 : maskLen;

            AlignerArguments ali;

            ali.query = readsequence;
            ali.three_n_query.resize(readLengths[0]);
            NucleoideConverer(ali.three_n_query.data(), ali.query.c_str(), readLengths[0]);
            ali.rc_query = SequenceHelpers::reverseComplementSequenceDecoded(ali.query.data(), readLengths[0]);

            ali.three_n_rc_query.resize(readLengths[0]);
            NucleoideConverer(ali.three_n_rc_query.data(), ali.rc_query.c_str(), readLengths[0]);

            ali.ref = std::string(window).c_str();
            ali.three_n_ref.resize(windowlength);
            NucleoideConverer(ali.three_n_ref.data(), ali.ref.c_str(), windowlength);

            ali.rc_ref = std::string(windowRC).c_str();
            ali.three_n_rc_ref.resize(windowlengthRC);
            NucleoideConverer(ali.three_n_rc_ref.data(), ali.rc_ref.c_str(), windowlengthRC);

            ali.filter = filter;

            ali.maskLen = maskLen;
            ali.readId = readId;
            ali.ref_len = windowlength;

            ali.result = result;
            //                    ali.resultRC=resultRC;

            ali.windowlength = windowlength;
            //        ali.windowlengthRC=windowlengthRC;
        
            ali.flag |= 0x4;
            mappingout.push_back(ali);
           // std::cout<<"unmapped"<<readId<<"\n";
        }

    } // end of big for loop

    std::cout << "big for done, now to mapping:...\n";
    mappingout.size();
    ThreadPool threadPool(std::max(1, programOptions->threads));
    ThreadPool::ParallelForHandle pforHandle;

    // function that maps 2 alignments: 3NQuery-3NREF , 3NRC_Query-3NREF
    auto mapfk = [&](auto begin, auto end, int /*threadid*/)
    {
        for (auto i = begin; i < end; i++)
        {
            if ((mappingout.at(i).flag & 0x4) == 0) {//if unmapped bit is not set --> align it
                // 3NQuery-3NREF
                StripedSmithWaterman::Alignment* ali;
                ali = &mappingout.at(i).alignments.at(0);
                mappingout.at(i).flag = aligner.Align(
                    (mappingout.at(i).three_n_query).c_str(),
                    (mappingout.at(i).three_n_ref).c_str(),
                    mappingout.at(i).ref_len,
                    mappingout.at(i).filter,
                    ali,
                    mappingout.at(i).maskLen);
                    
                // 3NRC_Query-3NREF
                StripedSmithWaterman::Alignment* alii;
                alii = &mappingout.at(i).alignments.at(1);
                mappingout.at(i).flag_rc = aligner.Align(
                    (mappingout.at(i).three_n_rc_query).c_str(),
                    mappingout.at(i).three_n_ref.c_str(),
                    mappingout.at(i).ref_len,
                    mappingout.at(i).filter,
                    alii,
                    mappingout.at(i).maskLen);
            }
            else {//ignore unmapped
                continue;
            }
        }
    };

    std::size_t start = 0;
    threadPool.parallelFor(pforHandle, start, mappingout.size(), mapfk);
    std::cout << "mapped, now to recalculaion of AlignmentScore:...\n";

    auto recalculateAlignmentScorefk = [&](AlignerArguments &aa, const Cigar::Entries &cig, std::size_t h)
    {
        StripedSmithWaterman::Alignment *ali = &aa.alignments.at(h);
        int _num_conversions = 0;
        std::string *_query = &aa.query;
        std::string *_ref = &aa.ref;

        if (!h)
        {
            _query = &aa.rc_query;
        }

        int refPos = 0, altPos = 0;

        for (const auto &cigarEntry : cig)
        {

            auto basesLeft = std::min(82 - std::max(refPos, altPos), cigarEntry.second);

            switch (cigarEntry.first)
            {
            case Cigar::Op::Match:
                for (int i = 0; i < basesLeft; ++i)
                {
                    if (

                        _query->at(altPos + i) == _ref->at(refPos + i) // matching query and ref
                        || _ref->at(refPos + i) == WILDCARD_NUCLEOTIDE // or its N
                        || _query->at(altPos + i) == WILDCARD_NUCLEOTIDE)
                    {
                        continue;
                    }
                    if (_query->at(altPos + i) == 'C')
                    { // if its a mismatch

                        if (('T' == _ref->at(refPos + i) && 'A' == aa.rc_ref.at(refPos + i)) || ('A' == _ref->at(refPos + i) && 'T' == aa.rc_ref.at(refPos + i)))
                        {

                            ali->sw_score -= aligner.getScore('T', _ref->at(refPos + i)); // substract false matching score
                            ali->sw_score += aligner.getScore('C', _ref->at(refPos + i)); // add corrected matching score
                        }
                    }
                    if (_query->at(altPos + i) == 'T')
                    { // if its a conversion

                        if (('C' == _ref->at(refPos + i) && 'G' == aa.rc_ref.at(refPos + i)) || ('G' == _ref->at(refPos + i) && 'C' == aa.rc_ref.at(refPos + i)))
                        {
                            _num_conversions++;

                            ali->sw_score -= aligner.getScore('T', 'T');                  // substract false matching score
                            ali->sw_score += aligner.getScore('T', _ref->at(refPos + i)); // add corrected matching score
                        }
                    }
                }

                refPos += basesLeft;
                altPos += basesLeft;
                break;

            case Cigar::Op::Insert:
                altPos += basesLeft;
                break;

            case Cigar::Op::Delete:
                refPos += basesLeft;
                break;

            case Cigar::Op::SoftClip:
                altPos += basesLeft;
                break;

            case Cigar::Op::HardClip:

                break;

            case Cigar::Op::Skipped:
                refPos += basesLeft;
                break;

            case Cigar::Op::Padding:

                break;

            case Cigar::Op::Mismatch:
                for (int i = 0; i < basesLeft; ++i)
                {
                    if (

                        _query->at(altPos + i) == _ref->at(refPos + i) // matching query and ref
                        || _ref->at(refPos + i) == WILDCARD_NUCLEOTIDE // or its N
                        || _query->at(altPos + i) == WILDCARD_NUCLEOTIDE)
                    {
                        continue;
                    }
                }
                refPos += basesLeft;
                altPos += basesLeft;

                break;

            case Cigar::Op::Equal:
                for (int i = 0; i < basesLeft; ++i)
                {
                    if (

                        _query->at(altPos + i) == _ref->at(refPos + i) // matching query and ref
                        || _ref->at(refPos + i) == WILDCARD_NUCLEOTIDE // or its N
                        || _query->at(altPos + i) == WILDCARD_NUCLEOTIDE)
                    {
                        continue;
                    }
                    if (_query->at(altPos + i) == 'T')
                    { // if its a possible conversion

                        if (('C' == _ref->at(refPos + i) && 'G' == aa.rc_ref.at(refPos + i)) || ('G' == _ref->at(refPos + i) && 'C' == aa.rc_ref.at(refPos + i)))
                        {
                            _num_conversions++;

                            ali->sw_score -= 2;
                            ali->sw_score += aligner.getScore(_query->at(altPos + i), _ref->at(refPos + i));

                            //                 std::cout<<"="<<_query->at(altPos + i)<<_ref->at(refPos + i)<<aa.rc_ref.at(refPos + i)<<"\n";
                        }
                    }
                }
                refPos += basesLeft;
                altPos += basesLeft;
                break;

            default:
                std::cout << "this shouldnt print\n";
                break;
            }
        }

        aa.num_conversions.at(h) = _num_conversions; // update AlignerArguments
    };

    auto comparefk = [&](auto begin, auto end, int /*threadid*/)
    {
        for (auto i = begin; i < end; i++)
        {
            if ((mappingout.at(i).flag & 0x4) == 0) {//if unmapped bit is not set --> align it

                Cigar cigi{ mappingout.at(i).alignments.at(0).cigar_string };
                Cigar cigii{ mappingout.at(i).alignments.at(1).cigar_string };

                recalculateAlignmentScorefk(mappingout.at(i), cigi.getEntries(), 0);
                recalculateAlignmentScorefk(mappingout.at(i), cigii.getEntries(), 1);
            }
            else {//ignore unmapped
                continue;
            }
        }
    };

    threadPool.parallelFor(pforHandle, start, mappingout.size(), comparefk);

    std::cout << "number of reads:" << mappingout.size() << "\n";
    std::cout << "mapped: " << moodmapped << "\n";
    std::cout << "not mapped: " << nootmapped << "\n";
//helpers::CpuTimer sammapping("process sam file writing");
  //  printtoSAM();
   // sammapping.print();
} // end of CSSW-Mapping

void Mappinghandler::examplewrapper(std::unique_ptr<ChunkedReadStorage> &cpuReadStorage)
{

    std::ofstream outputstream("examplemapper_out.txt");

    // std::cout<<"lets go Examplemapper!!\n";
    const int maximumSequenceLength = cpuReadStorage->getSequenceLengthUpperBound();
    helpers::CpuTimer mappertimer("timerformapping");
    std::size_t rundenzaehler = 0;

    std::size_t numreads = cpuReadStorage->getNumberOfReads();

    for (std::size_t r = 0, processedResults = 0; r < numreads; r++)
    {
        const auto &result = (*results)[r];
        read_number readId = r;
        rundenzaehler++;
        std::vector<int> readLengths(1);
        cpuReadStorage->gatherSequenceLengths(
            readLengths.data(),
            &readId,
            1);

        const int encodedReadNumInts2Bit = SequenceHelpers::getEncodedNumInts2Bit(maximumSequenceLength);
        std::vector<unsigned int> encodedReads(encodedReadNumInts2Bit * 1);

        cpuReadStorage->gatherSequences(
            encodedReads.data(),
            encodedReadNumInts2Bit,
            &readId,
            1);

        if (result.orientation == AlignmentOrientation::ReverseComplement)
        {
            SequenceHelpers::reverseComplementSequenceInplace2Bit(encodedReads.data(), readLengths[0]);
        }
        auto readsequence = SequenceHelpers::get2BitString(encodedReads.data(), readLengths[0]);

        if (result.orientation != AlignmentOrientation::None)
        {

            const auto &genomesequence = (*genome).data.at(result.chromosomeId);

            const std::size_t windowlength = result.position +
                                                         programOptions->windowSize <
                                                     genomesequence.size()
                                                 ? programOptions->windowSize
                                                 : genomesequence.size() -
                                                       result.position;

            std::string_view window(genomesequence.data() + result.position, windowlength);
            processedResults++;

            /*Put your mapping algorithm here
             *....
             *use the variables "window" and "readsequence"
             */
        }
        else
        {
        }
        mappertimer.print();
    }
}

void Mappinghandler::edlibAligner(std::unique_ptr<ChunkedReadStorage> &cpuReadStorage)
{

    std::ofstream outputstream("examplemapper_out.txt");

    // std::cout<<"lets go Examplemapper!!\n";
    const int maximumSequenceLength = cpuReadStorage->getSequenceLengthUpperBound();
    helpers::CpuTimer mappertimer("timerformapping");
    std::size_t rundenzaehler = 0;

    std::size_t numreads = cpuReadStorage->getNumberOfReads();

    for (std::size_t r = 0, processedResults = 0; r < numreads; r++)
    {
        const auto &result = (*results)[r];
        read_number readId = r;
        rundenzaehler++;
        std::vector<int> readLengths(1);
        cpuReadStorage->gatherSequenceLengths(
            readLengths.data(),
            &readId,
            1);

        const int encodedReadNumInts2Bit = SequenceHelpers::getEncodedNumInts2Bit(maximumSequenceLength);
        std::vector<unsigned int> encodedReads(encodedReadNumInts2Bit * 1);

        cpuReadStorage->gatherSequences(
            encodedReads.data(),
            encodedReadNumInts2Bit,
            &readId,
            1);

        if (result.orientation == AlignmentOrientation::ReverseComplement)
        {
            SequenceHelpers::reverseComplementSequenceInplace2Bit(encodedReads.data(), readLengths[0]);
        }
        auto readsequence = SequenceHelpers::get2BitString(encodedReads.data(), readLengths[0]);

        if (result.orientation != AlignmentOrientation::None)
        {

            const auto &genomesequence = (*genome).data.at(result.chromosomeId);
            const auto &genomesequenceRC = (*genomeRC).data.at(result.chromosomeId);

            const std::size_t windowlength = result.position +
                                                         programOptions->windowSize <
                                                     genomesequence.size()
                                                 ? programOptions->windowSize
                                                 : genomesequence.size() -
                                                       result.position;
            const std::size_t windowlengthRC = result.position +
                                                           programOptions->windowSize <
                                                       genomesequenceRC.size()
                                                   ? programOptions->windowSize
                                                   : genomesequenceRC.size() -
                                                         result.position;

            
            processedResults++;
            std::size_t aef = genomesequenceRC.size() - result.position - 1;
            std::string_view window(genomesequence.data() + result.position, windowlength);
            std::string_view windowRC(genomesequenceRC.data() + aef, windowlengthRC);

            Edlibhelper eh;
            eh.queryLength = readLengths[0];
            eh.queryOriginal = readsequence;
            eh.queryOriginal.resize(readLengths[0]);
            NucleoideConverer(eh.queryOriginal_threen.data(), eh.queryOriginal.c_str(), readLengths[0]);
            eh.queryOriginal_rc = SequenceHelpers::reverseComplementSequenceDecoded(eh.queryOriginal.data(), readLengths[0]);

            eh.queryOriginal_rc_threen.resize(readLengths[0]);
            NucleoideConverer(eh.queryOriginal_rc_threen.data(), eh.queryOriginal_rc.c_str(), readLengths[0]);

            eh.targetLength=windowlength;
            eh.targetOriginal = std::string(window).c_str();
            eh.targetOriginal_threen.resize(windowlength);
            NucleoideConverer(eh.targetOriginal_threen.data(), eh.targetOriginal.c_str(), windowlength);

            eh.targetOriginal_rc= std::string(windowRC).c_str();
            eh.targetOriginal_rc_threen.resize(windowlengthRC);
            NucleoideConverer(eh.targetOriginal_rc_threen.data(), eh.targetOriginal_rc.c_str(), windowlengthRC);
            eh.result=result;
            edlibout.push_back(eh);
           
        }
        else
        {
            const auto& genomesequence = (*genome).data.at(result.chromosomeId);
            const auto& genomesequenceRC = (*genomeRC).data.at(result.chromosomeId);

            const std::size_t windowlength = result.position +
                programOptions->windowSize <
                genomesequence.size()
                ? programOptions->windowSize
                : genomesequence.size() -
                result.position;
            const std::size_t windowlengthRC = result.position +
                programOptions->windowSize <
                genomesequenceRC.size()
                ? programOptions->windowSize
                : genomesequenceRC.size() -
                result.position;

            std::size_t aef = genomesequenceRC.size() - result.position - 1;
            std::string_view window(genomesequence.data() + result.position, windowlength);
            std::string_view windowRC(genomesequenceRC.data() + aef, windowlengthRC);

            Edlibhelper eh;
            eh.queryLength = readLengths[0];
            eh.queryOriginal = readsequence;
            eh.flag |= 0x4;
            edlibout.push_back(eh);
        }
        mappertimer.print();
    }//end of big for loop
     std::cout << "big for done, now to mapping:...\n";
    std::cout << "we have "<<mappingout.size()<<" to map\n";
    ThreadPool threadPool(std::max(1, programOptions->threads));
    ThreadPool::ParallelForHandle pforHandle;

    // function that maps 2 alignments: 3NQuery-3NREF , 3NRC_Query-3NREF
    auto mapfk = [&](auto begin, auto end, int /*threadid*/)
    {
       for (auto i = begin; i < end; i++)
        {
           if ((edlibout.at(i).flag & 0x4) == 0) {//if unmapped bit is not set --> align it
              // 3NQuery-3NREF
               EdlibAlignResult result = edlibAlign(edlibout.at(i).queryOriginal_threen.c_str(),
                                                    edlibout.at(i).queryLength,
                                                    edlibout.at(i).targetOriginal_threen.c_str(),
                                                    edlibout.at(i).targetLength, 
                                                    edlibDefaultAlignConfig());
               if (result.status == EDLIB_STATUS_OK) {
                   edlibout.at(i).cigar.assign(
                                                edlibAlignmentToCigar(
                                                    result.alignment, result.alignmentLength, EDLIB_CIGAR_STANDARD)
                                                );
                   edlibout.at(i).score = result.editDistance;
               }
               edlibFreeAlignResult(result);

               // 3NRC_Query-3NREF
               EdlibAlignResult resultrc = edlibAlign(edlibout.at(i).queryOriginal_rc_threen.c_str(),
                   edlibout.at(i).queryLength,
                   edlibout.at(i).targetOriginal_rc_threen.c_str(),
                   edlibout.at(i).targetLength,
                   edlibDefaultAlignConfig());
               if (result.status == EDLIB_STATUS_OK) {
                   edlibout.at(i).cigar_rc.assign(
                                            edlibAlignmentToCigar(
                                                resultrc.alignment, resultrc.alignmentLength, EDLIB_CIGAR_STANDARD)
                                            );
                   edlibout.at(i).score_rc = result.editDistance;
               }
               edlibFreeAlignResult(resultrc);

           }
           else {//ignore unmapped
               continue;
           }
        } 
    };
    std::size_t start = 0;
    threadPool.parallelFor(pforHandle, start, edlibout.size(), mapfk);
    std::cout << "mapped, now to recalculaion of AlignmentScore:...\n";

    auto recalculateAlignmentScorefk = [&](Edlibhelper& aa, const Cigar::Entries& cig, std::size_t h)
    {
       
        int _num_conversions = 0;
        std::string* _query = &aa.queryOriginal;
        std::string* _ref = &aa.targetOriginal;
        int temp_score=aa.score;
        if (!h)
        {
            temp_score=aa.score_rc;
            _query = &aa.queryOriginal_rc;
        }

        int refPos = 0, altPos = 0;

        for (const auto& cigarEntry : cig)
        {

            auto basesLeft = std::min(82 - std::max(refPos, altPos), cigarEntry.second);

            switch (cigarEntry.first)
            {
            case Cigar::Op::Match:
                for (int i = 0; i < basesLeft; ++i)
                {
                    if (

                        _query->at(altPos + i) == _ref->at(refPos + i) // matching query and ref
                        || _ref->at(refPos + i) == WILDCARD_NUCLEOTIDE // or its N
                        || _query->at(altPos + i) == WILDCARD_NUCLEOTIDE)
                    {
                        continue;
                    }
                    if (_query->at(altPos + i) == 'C')
                    { // if its a mismatch

                        if (('T' == _ref->at(refPos + i) && 'A' == aa.targetOriginal_rc.at(refPos + i)) 
                        || ('A' == _ref->at(refPos + i) && 'T' == aa.targetOriginal_rc.at(refPos + i)))
                        {

                            //temp_score-= aligner.getScore('T', _ref->at(refPos + i)); // substract false matching score
                           // ali->sw_score += aligner.getScore('C', _ref->at(refPos + i)); // add corrected matching score
                        }
                    }
                    if (_query->at(altPos + i) == 'T')
                    { // if its a conversion

                        if (('C' == _ref->at(refPos + i) && 'G' == aa.targetOriginal_rc.at(refPos + i)) 
                        || ('G' == _ref->at(refPos + i) && 'C' == aa.targetOriginal_rc.at(refPos + i)))
                        {
                            _num_conversions++;

                           // ali->sw_score -= aligner.getScore('T', 'T');                  // substract false matching score
                           // ali->sw_score += aligner.getScore('T', _ref->at(refPos + i)); // add corrected matching score
                        }
                    }
                }

                refPos += basesLeft;
                altPos += basesLeft;
                break;

            case Cigar::Op::Insert:
                altPos += basesLeft;
                break;

            case Cigar::Op::Delete:
                refPos += basesLeft;
                break;

            case Cigar::Op::SoftClip:
                altPos += basesLeft;
                break;

            case Cigar::Op::HardClip:

                break;

            case Cigar::Op::Skipped:
                refPos += basesLeft;
                break;

            case Cigar::Op::Padding:

                break;

            case Cigar::Op::Mismatch:
                for (int i = 0; i < basesLeft; ++i)
                {
                    if (

                        _query->at(altPos + i) == _ref->at(refPos + i) // matching query and ref
                        || _ref->at(refPos + i) == WILDCARD_NUCLEOTIDE // or its N
                        || _query->at(altPos + i) == WILDCARD_NUCLEOTIDE)
                    {
                        continue;
                    }
                }
                refPos += basesLeft;
                altPos += basesLeft;

                break;

            case Cigar::Op::Equal:
                for (int i = 0; i < basesLeft; ++i)
                {
                    if (

                        _query->at(altPos + i) == _ref->at(refPos + i) // matching query and ref
                        || _ref->at(refPos + i) == WILDCARD_NUCLEOTIDE // or its N
                        || _query->at(altPos + i) == WILDCARD_NUCLEOTIDE)
                    {
                        continue;
                    }
                    if (_query->at(altPos + i) == 'T')
                    { // if its a possible conversion

                        if (('C' == _ref->at(refPos + i) && 'G' == aa.targetOriginal_rc.at(refPos + i)) ||
                         ('G' == _ref->at(refPos + i) && 'C' == aa.targetOriginal_rc.at(refPos + i)))
                        {
                            _num_conversions++;

                          //  ali->sw_score -= 2;
                         //   ali->sw_score += aligner.getScore(_query->at(altPos + i), _ref->at(refPos + i));

                            //                 std::cout<<"="<<_query->at(altPos + i)<<_ref->at(refPos + i)<<aa.rc_ref.at(refPos + i)<<"\n";
                        }
                    }
                }
                refPos += basesLeft;
                altPos += basesLeft;
                break;

            default:
                std::cout << "this shouldnt print\n";
                break;
            }
        }

        if(!h){
            aa.num_conversions_rc=_num_conversions;
        }else{
            aa.num_conversions=_num_conversions;
        }
        //aa.num_conversions.at(h) = _num_conversions; // update AlignerArguments
    };
//gzugi
    auto comparefk = [&](auto begin, auto end, int /*threadid*/)
    {
        for (auto i = begin; i < end; i++)
        {
            if ((edlibout.at(i).flag & 0x4) == 0) {//if unmapped bit is not set --> align it

                Cigar cigi{ edlibout.at(i).cigar };
                Cigar cigii{ edlibout.at(i).cigar_rc };

                recalculateAlignmentScorefk(edlibout.at(i), cigi.getEntries(), 0);
                recalculateAlignmentScorefk(edlibout.at(i), cigii.getEntries(), 1);
            }
            else {//ignore unmapped
                continue;
            }
        }
    };

    threadPool.parallelFor(pforHandle, start, mappingout.size(), comparefk);
    //std::cout<<"hello\n";
    //printtoSAM();
    printtoedlibSAM();
}//end of edlib aligner
