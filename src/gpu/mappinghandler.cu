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
#include<ssw_cpp.h>
// BAM Variant Caller and Genomic Analysis
#include "cigar.hpp"
#include "constants.hpp"
#include "varianthandler.hpp"

#include <gpu/mappedread.cuh>

       /// @brief 
       /// @param programOptions_ 
       /// @param genome_ 
       /// @param genomeRC_ 
       /// @param results_ 
       /// @param resultsRC_ 
       Mappinghandler::Mappinghandler(
            const ProgramOptions* programOptions_,
            const Genome* genome_,
            const Genome* genomeRC_,
             std::vector<MappedRead>* results_
             //,std::vector<MappedRead>* resultsRC_
             ):
        programOptions(programOptions_),
        genomeRC(genomeRC_),
        genome(genome_),
        results(results_)
       // ,      resultsRC(resultsRC_)
        {
           
        }

        /// @brief 
        Mappinghandler::~Mappinghandler(){
            
       }

     

      /// @brief Used to select a mapper and starts it
      /// @param cpuReadStorage_ The storage of reads
      void Mappinghandler::go(std::unique_ptr<ChunkedReadStorage>& cpuReadStorage_){
        //mappertype=(int) programOptions.mappType;
         
            switch(programOptions->mappType){
                case MapperType::primitiveSW : 
                    std::cout<<"primitive SW selected but should not be used!\n"; 
                    //primitiveSW(cpuReadStorage_);
                    break;
                case MapperType::SW : 
                    //std::cout<<"SW selected \n"; 
                    CSSW(cpuReadStorage_);
                break;
                case MapperType::sthelse : 
                    std::cout<<"please implement your personal mapper\n"; 
                break;
                default: 
                    std::cout<<"something went wrong while selecting mappertype\n"; 
                    examplewrapper(cpuReadStorage_);
                break;
            }
       }
       
       /// @brief TODO
    void Mappinghandler::doVC(){
      /*  
        auto test=(programOptions->outputfile)+".VCF";
        VariantHandler vhandler(test);
        vhandler.VCFFileHeader();
        

        uint32_t mapq=0;
      
        for (long unsigned int i=0; i<mappingout.size();i++){
         
         //https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library/blob/master/src/main.c
         //Line 167 to  169
            mapq = -4.343 * log(1 - (double)abs(mappingout.at(i).alignment.sw_score - 
                mappingout.at(i).alignment.sw_score_next_best)
                /(double)mappingout.at(i).alignment.sw_score);
			mapq = (uint32_t) (mapq + 4.99);
			mapq = mapq < 254 ? mapq : 254;

            
           if(mapq < MAP_QUALITY_THRESHOLD){
                continue;
            }
            

            Cigar marlboro{mappingout.at(i).alignment.cigar_string};
          
            std::string prefix=mappingout.at(i).ref.substr(0,mappingout.at(i).alignment.query_begin);
            
           vhandler.call(
            mappingout.at(i).result.position + mappingout.at(i).alignment.query_begin, //seq position
            prefix,
            mappingout.at(i).ref,
            mappingout.at(i).query, 
            marlboro.getEntries(),
            genome->names.at(mappingout.at(i).result.chromosomeId),
            mappingout.at(i).readId,
            mapq
            );
            
        }
        
        //vhandler.forceFlush();
        */

     } 
   
    INLINEQUALIFIER
    /// @brief converts every 'C' into a 'T' 
    /// @param output 
    /// @param input 
    /// @param length 
    void Mappinghandler::NucleoideConverer(char* output, const char* input, int length){
        if(strlen(input)!=0)
        {

                    for(int i = 0; i < length; ++i){
                switch(input[i]){
                    case 'C': output[i] = 'T'; break;
                    default : break;
                }
            }
        }else
        assert("Nucleotide Converter Failed");
     }
        
        //Helper struct for CSSW-Mapping




/// @brief simple print methode for the vector mappingout into the "CSSW_SAM.SAM" file. Used for CSSW-Mapping
void Mappinghandler::printtoSAM(){
    //TODO #3 updated saving to SAM file and setting the sam tag
    /*
    std::ofstream outputstream("CSSW_SAM.SAM");

    outputstream << "@HD\n"<<
                    "@Coloums: QNAME\tFLAG\tRNAME\tPOS\tMAPQ\tCIGAR\tRNEXT\tPNEXT\tTLEN\tSEQ\tQUAL\n";
      
    for(std::size_t i =0;i<mappingout.size();i++){


        //TODO MAPQ mussüberarbeitet werden!!! Das ist so nicht richtig
        //MAPQ calculated as in CSSW 
        //https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library/blob/master/src/main.c
         //Line 167 to  169
        uint32_t mapq = -4.343 * log(1 - (double)abs(mappingout.at(i).alignment.sw_score - 
                mappingout.at(i).alignment.sw_score_next_best)
                /(double)mappingout.at(i).alignment.sw_score);
			mapq = (uint32_t) (mapq + 4.99);
			mapq = mapq < 254 ? mapq : 254;

        outputstream << mappingout.at(i).readId<<"\t"                       //QNAME
        << "*" <<"\t"                                                       // FLAG
        << genome->names.at(mappingout.at(i).result.chromosomeId) <<"\t"    // RNAME
        << mappingout.at(i).result.position + mappingout.at(i).alignment.query_begin <<"\t"                   // POS //look up my shenanigans in ssw_cpp.cpp for why its queri_begin
        << mapq <<"\t"                                                      // MAPQ
        << mappingout.at(i).alignment.cigar_string <<"\t"                   // CIGAR
        << "=" <<"\t"                                                // RNEXT
        << "" <<"\t"                                                // PNEXT
        << "0" <<"\t"                                               // TLEN
        << mappingout.at(i).query <<"\t"                               // SEQ
        << "*" <<"\t"                                                // QUAL
        <<"\n";
    } 
    outputstream.close();
    */
}

     //Complete-Striped-Smith-Waterman Mapper.
     //https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library
void Mappinghandler::CSSW(std::unique_ptr<ChunkedReadStorage>& cpuReadStorage){

        StripedSmithWaterman::Aligner aligner;
        StripedSmithWaterman::Filter filter;
     
        const int maximumSequenceLength = cpuReadStorage->getSequenceLengthUpperBound();


        std::size_t numreads=cpuReadStorage->getNumberOfReads();


    
    std::cout<<"lets go bigfor:...\n";

    std::size_t processedResults = 0;
   for(std::size_t r = 0; r < numreads; r++){
    
            const auto& result = (*results)[r];
//            const auto& resultRC = (*resultsRC)[r];
            
            read_number readId = r;
            
            std::vector<int> readLengths(1);
            cpuReadStorage->gatherSequenceLengths(
                readLengths.data(),
                &readId,
                1
            );

            const int encodedReadNumInts2Bit = SequenceHelpers::getEncodedNumInts2Bit(maximumSequenceLength);
            std::vector<unsigned int> encodedReads(encodedReadNumInts2Bit * 1);

            cpuReadStorage->gatherSequences(
                encodedReads.data(),
                encodedReadNumInts2Bit,
                &readId,
                1
            );


           if(result.orientation == AlignmentOrientation::ReverseComplement){
                SequenceHelpers::reverseComplementSequenceInplace2Bit(encodedReads.data(), readLengths[0]);
            }

            auto readsequence = SequenceHelpers::get2BitString(encodedReads.data(), readLengths[0]);

            if(result.orientation != AlignmentOrientation::None ){
               
                const auto& genomesequence = (*genome).data.at(result.chromosomeId);
                const auto& genomesequenceRC = (*genomeRC).data.at(result.chromosomeId);

                const std::size_t windowlength = result.position + 
                    programOptions->windowSize < genomesequence.size() ? programOptions->windowSize : genomesequence.size() - 
                    result.position;
                const std::size_t windowlengthRC = result.position + 
                    programOptions->windowSize < genomesequenceRC.size() ? programOptions->windowSize : genomesequenceRC.size() - 
                    result.position;
                    
                std::size_t aef=genomesequenceRC.size()-result.position-1;
                std::string_view window(genomesequence.data() + result.position, windowlength);            
                std::string_view windowRC(genomesequenceRC.data() + aef, windowlengthRC);

                processedResults++;

                int32_t maskLen = readLengths[0]/2;
                maskLen = maskLen < 15 ? 15 : maskLen;
                   
                AlignerArguments ali;
                
                    ali.query=readsequence;
                    ali.three_n_query.resize(readLengths[0]);
                    NucleoideConverer(ali.three_n_query.data(), ali.query.c_str(), readLengths[0]);
                    ali.rc_query=SequenceHelpers::reverseComplementSequenceDecoded(ali.query.data(),readLengths[0]); 

                    ali.three_n_rc_query.resize(readLengths[0]);  
                    NucleoideConverer(ali.three_n_rc_query.data(), ali.rc_query.c_str(), readLengths[0]); 
                    
                    ali.ref=std::string(window).c_str();
                    ali.three_n_ref.resize(windowlength);
                    NucleoideConverer(ali.three_n_ref.data() ,ali.ref.c_str(), windowlength);
                    
                    ali.rc_ref=std::string(windowRC).c_str();
                    ali.three_n_rc_ref.resize(windowlengthRC);
                    NucleoideConverer(ali.three_n_rc_ref.data(), ali.rc_ref.c_str(), windowlengthRC);

                    ali.filter=filter;
                    
                    ali.maskLen=maskLen;
                    ali.readId=readId; 
                    ali.ref_len=windowlength;
                   
                    ali.result=result;
//                    ali.resultRC=resultRC;
                    
                    ali.windowlength=windowlength;
            //        ali.windowlengthRC=windowlengthRC;


                        mappingout.push_back(ali);
                    
               
                }else{
                    //no need to do sth. here
                }
            
         
        }//end of big for loop

     
//TODO #8 MACH MAPPING CORRECT
    
   
       std::cout<<"big for done, now to mapping:...\n";

        ThreadPool threadPool(std::max(1, programOptions->threads));
       ThreadPool::ParallelForHandle pforHandle;

       //function that maps 2 alignments: 3NQuery-3NREF , 3NRC_Query-3NREF 
        auto mapfk=[&](auto begin, auto end, int /*threadid*/){
          
                for(auto i=begin; i< end; i++){

                    // 3NQuery-3NREF
                    StripedSmithWaterman::Alignment* ali;
                    ali=&mappingout.at(i).alignments.at(0);
                    aligner.Align(
                        (mappingout.at(i).three_n_query).c_str(),
                        (mappingout.at(i).three_n_ref).c_str(),
                        mappingout.at(i).ref_len,
                        mappingout.at(i).filter,
                        ali,
                        mappingout.at(i).maskLen);

                        // 3NRC_Query-3NREF
                        StripedSmithWaterman::Alignment* alii;
                    alii=&mappingout.at(i).alignments.at(1);
                    aligner.Align(
                        (mappingout.at(i).three_n_rc_query).c_str(),
                        mappingout.at(i).three_n_ref.c_str(),
                        mappingout.at(i).ref_len,
                        mappingout.at(i).filter,
                        alii,
                        mappingout.at(i).maskLen);

                   
                }
        };
        
       
        std::size_t start=0;
        threadPool.parallelFor(pforHandle, start , mappingout.size() ,mapfk);
       std::cout<<"mapped, now to recalculaion of AlignmentScore:...\n";

    auto recalculateAlignmentScorefk=[&](AlignerArguments& aa, const Cigar::Entries& cig, std::size_t h){
//TODO #2  lambda recalculateAlignmentScorefk is unfinished: number of conversions is not saved
            StripedSmithWaterman::Alignment* ali=&aa.alignments.at(h);
           int num_conversions=0;
           std::size_t lengthref=aa.ref_len;
           std::string* _query=&aa.query;
           std::string* _ref=&aa.ref;

           if(!h){
            _query=&aa.rc_query;
           }  

        int refPos = 0, altPos = 0;

        for (const auto  & cigarEntry : cig) {

            auto basesLeft = std::min(82 - std::max(refPos, altPos), cigarEntry.second);

        switch (cigarEntry.first) {

        case Cigar::Op::Match:
            for (int i = 0; i < basesLeft; ++i) {
                if (
                    _query->at(altPos + i) == _ref->at(refPos + i) //matching query and ref
                    || _ref->at(refPos + i) == WILDCARD_NUCLEOTIDE // or its N
                    || _query->at(altPos + i) == WILDCARD_NUCLEOTIDE 
                )
                   continue; // not interesed
                
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
            for (int i = 0; i < basesLeft; ++i) {
                if (
                    
                       _query->at(altPos + i) == _ref->at(refPos + i) //matching query and ref
                    || _ref->at(refPos + i) == WILDCARD_NUCLEOTIDE // or its N
                    || _query->at(altPos + i) == WILDCARD_NUCLEOTIDE 
                ){
                    continue;
                }
                    if(_query->at(altPos + i)==SequenceHelpers::complementBaseDecoded(_ref->at(refPos + i))){

                    }
                //TODO find the conversion 
                

            }
            refPos +=basesLeft;
            altPos += basesLeft;
            
        break;

        case Cigar::Op::Equal:

            refPos += basesLeft;
            altPos += basesLeft;
        break;

        default:
            std::cout<<"this shouldnt print\n";
        break;
        }

        }

     };

        auto comparefk=[&](auto begin, auto end, int /*threadid*/){

            for(auto i=begin; i< end; i++){            
                Cigar cigi{mappingout.at(i).alignments.at(0).cigar_string};
                Cigar cigii{mappingout.at(i).alignments.at(1).cigar_string};
                        
                recalculateAlignmentScorefk(mappingout.at(i), cigi.getEntries(), 0);
                recalculateAlignmentScorefk(mappingout.at(i), cigii.getEntries(), 1);
            }
        };

    threadPool.parallelFor(pforHandle, start , mappingout.size() ,comparefk);
 
 
    printtoSAM();

}//end of CSSW-Mapping

       
void Mappinghandler::examplewrapper(std::unique_ptr<ChunkedReadStorage>& cpuReadStorage){

        std::ofstream outputstream("examplemapper_out.txt");
               
        //std::cout<<"lets go Examplemapper!!\n";
        const int maximumSequenceLength = cpuReadStorage->getSequenceLengthUpperBound();
        helpers::CpuTimer mappertimer("timerformapping");
        std::size_t rundenzaehler=0;

        std::size_t numreads=cpuReadStorage->getNumberOfReads();
        
        for(std::size_t r = 0, processedResults = 0; r < numreads; r++){
            const auto& result = (*results)[r];
            read_number readId = r;
            rundenzaehler++;
            std::vector<int> readLengths(1);
            cpuReadStorage->gatherSequenceLengths(
                readLengths.data(),
                &readId,
                1
            );

            const int encodedReadNumInts2Bit = SequenceHelpers::getEncodedNumInts2Bit(maximumSequenceLength);
            std::vector<unsigned int> encodedReads(encodedReadNumInts2Bit * 1);

            cpuReadStorage->gatherSequences(
                encodedReads.data(),
                encodedReadNumInts2Bit,
                &readId,
                1
            );


            if(result.orientation == AlignmentOrientation::ReverseComplement){
                SequenceHelpers::reverseComplementSequenceInplace2Bit(encodedReads.data(), readLengths[0]);
            }
            auto readsequence = SequenceHelpers::get2BitString(encodedReads.data(), readLengths[0]);

            if(result.orientation != AlignmentOrientation::None){
                              
                const auto& genomesequence = (*genome).data.at(result.chromosomeId);
                
                const std::size_t windowlength = result.position + 
                    programOptions->windowSize < genomesequence.size() ? programOptions->windowSize : genomesequence.size() - 
                    result.position;

                std::string_view window(genomesequence.data() + result.position, windowlength);
                processedResults++;
                

                /*Put your mapping algorithm here
                *....
                *use the variables "window" and "readsequence"
                */
                }else{

                }
    mappertimer.print();
    }
}
      

