#ifndef VARIANT_HANDLER_HPP
#define VARIANT_HANDLER_HPP

#include "cigar.hpp"
#include "filehandler.hpp"
#include "variant.hpp"

#include <set>
#include <string>
#include<climits>
class VariantHandler : public OutFileHandler
{
public:
    VariantHandler(const std::string & path);

    void call(size_t readPos, const std::string & prefix, const std::string & ref,
              const std::string & alt, const Cigar::Entries & cigarEntries,
                const std::string & chrom,
                const uint32_t & rID,
                const uint32_t & qual);

    /* 
    void forceFlush() 
                    { 
                    flush(INT_MAX); 
                    }
*/

    std::string getFile();
    void VCFFileHeader();
protected:
    virtual void write(const VariantEntry & entry, 
                        const std::string & chrom,
                        const uint32_t & rID,
                        const uint32_t & qual);

    void flush(size_t lastPos, 
                        const std::string & chrom,
                        const uint32_t & rID,
                        const uint32_t & qual);

private:
    void save(size_t pos, const std::string & ref, const std::string & alt);

    std::set<VariantEntry, VariantEntryComparator> m_set;
    int m_iterSinceFlush = 0;
   
    std::string m_stringtofile;
};

#endif // VARIANT_HANDLER_HPP
