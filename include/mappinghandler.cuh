#include <hpc_helpers.cuh>
#include <config.hpp>
#include <options.hpp>

#include <genome.hpp>
#include <referencewindows.hpp>
#include <windowhitstatisticcollector.hpp>
#include <sequencehelpers.hpp>
#include <chunkedreadstorageconstruction.hpp>
#include <chunkedreadstorage.hpp>
#include <contiguousreadstorage.hpp>
#include <alignmentorientation.hpp>

#include <algorithm>
#include <vector>
#include <iostream>
#include <mutex>
#include <thread>
#include <memory>
#include <cassert>
#include <map>
#include <string>
#include <string_view>
#include <omp.h>

#include <gpu/gpuminhasherconstruction.cuh>
#include <gpu/cudaerrorcheck.cuh>
#include <gpu/multigpureadstorage.cuh>
#include <gpu/sequenceconversionkernels.cuh>
#include <gpu/minhashqueryfilter.cuh>
#include <gpu/hammingdistancekernels.cuh>
#include <gpu/windowgenerationkernels.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/logging_resource_adaptor.hpp>
#include <gpu/rmm_utilities.cuh>

#include <thrust/equal.h>
#include <thrust/functional.h>
#include <thrust/scatter.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>


struct Mappinghandler
{
    public:

        Mappinghandler(
            const ProgramOptions* programOptions_,
             cons Genome* genome_,
             std::vector<MappedRead>* results_
             ):
        programOptions(programOptions_),
        genome(genome_),
        results(results_)
        {
           
        }

        ~Mappinghandler(){

       }

       reset(){

       }
       go(){
        mappertype=programOptions.mappType;
         
            switch(mappertype){
                case primitiveSW: std::cout<<"primitive SW selected \n" break;
                case SW: std::cout<<"SW selected \n" break;
                case sthelse: std::cout<<"please implement your personal mapper\n" break;
                default: std::cout<<"sth went wrong while selecting mappertype\n" break;
            }
       }
    private:
        const ProgramOptions* programOptions;
        const Genome* genome;
        int mappertype=1;
        std::vector<MappedRead>* results;
};
