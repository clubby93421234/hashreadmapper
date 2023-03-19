/*Dieser Code definiert die Struktur "MappedRead", die zur Speicherung von Informationen zu einer gelesenen DNA-Sequenz verwendet wird.
Die Struktur enthält mehrere Mitgliedsvariablen, darunter "orientation", "hammingDistance", "shift", "chromosomeId" und "position", 
die jeweils spezifische Informationen über die gelesene Sequenz enthalten. 
Die "AlignmentOrientation" ist eine Enumeration, die den Orientierungstyp der Sequenz angibt.
Der Rest der Variablen speichert Informationen über den Leseprozess, einschließlich der Anzahl der Hamming-Abstände,
der Verschiebung, der Chromosomen-ID und der Position. 
Die Header-Datei dient dazu, die Struktur zu definieren und sie anderen Teilen des Programms zugänglich zu machen.*/

#ifndef MAPPEDREAD_CUH
#define MAPPEDREAD_CUH

#include <alignmentorientation.hpp>

struct MappedRead{
    AlignmentOrientation orientation = AlignmentOrientation::None;
    int hammingDistance;
    int shift;
    std::size_t chromosomeId = 0;
    std::size_t position = 0;
};

#endif
