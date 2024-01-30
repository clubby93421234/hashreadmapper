#include "cigar.hpp"

Cigar::Cigar(const std::string & cigarStr)
{
    int bases = 0;
    for (const char & c : cigarStr) {
        if ((c >= 'A' && c <= 'Z') || c == '=') {
            m_entries.emplace_back(charToOp(c), bases);
            bases = 0;
        }
        else if (c >= '0' && c <= '9') {
            bases = bases * 10 + (c - '0');
        }
    }
}

const Cigar::Entries & Cigar::getEntries()
{
    return m_entries;
}

char Cigar::opToChar(const Cigar::Op & op)
{
    if (op == Cigar::Op::Match)
        return 'M';
    else if (op == Cigar::Op::Insert)
        return 'I';
    else if (op == Cigar::Op::Delete)
        return 'D';
    else if (op == Cigar::Op::Skipped)
        return 'N';
    else if (op == Cigar::Op::SoftClip)
        return 'S';
    else if (op == Cigar::Op::HardClip)
        return 'H';
    else if (op == Cigar::Op::Padding)
        return 'P';
    else if (op == Cigar::Op::Mismatch)
        return 'X';
    else if (op==Cigar::Op::Equal)
        return '=';
    return '?';
}

Cigar::Op Cigar::charToOp(const char & c)
{
    if (c == 'M')
        return Cigar::Op::Match;
    else if (c == 'I')
        return Cigar::Op::Insert;
    else if (c == 'D')
        return Cigar::Op::Delete;
    else if (c == 'S')
        return Cigar::Op::SoftClip;
    else if (c == 'H')
        return Cigar::Op::HardClip;
    else if (c == 'N')
        return Cigar::Op::Skipped;
    else if (c == 'P')
        return Cigar::Op::Padding;
    else if (c == 'X')
        return Cigar::Op::Mismatch;
    else if (c == '=')
        return Cigar::Op::Equal;

    return Cigar::Op::Invalid;
}
