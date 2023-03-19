/*Dieser Code definiert die Struktur "AsyncConstBufferWrapper" und eine Funktion "makeAsyncConstBufferWrapper", 
die in CUDA-Programmen verwendet wird, um asynchron auf konstante Speicherbereiche zuzugreifen.
Die Struktur "AsyncConstBufferWrapper" enthält einen Zeiger auf einen konstanten Speicherbereich sowie ein Ereignis, das signalisiert,
wenn der Speicherbereich bereit ist. 
Die Funktionen "wait" und "ready" werden verwendet, um auf das Ereignis zu warten oder zu überprüfen,
ob es bereits aufgetreten ist. 
Die Funktion "linkStream" wird verwendet, um sicherzustellen, dass Arbeit, die nach einem bestimmten Stream 
eingereicht wird, erst ausgeführt wird, wenn der konstante Speicherbereich bereit ist.
Die Funktion "makeAsyncConstBufferWrapper" erstellt eine Instanz von "AsyncConstBufferWrapper" und gibt sie zurück.
Es akzeptiert einen Zeiger auf den konstanten Speicherbereich und ein Ereignis, das signalisiert, wann der Speicherbereich bereit ist (optional).
Wenn kein Ereignis angegeben wird, wird der Speicherbereich als sofort bereit betrachtet.*/

#ifndef CARE_ASYNC_RESULT_CUH
#define CARE_ASYNC_RESULT_CUH

#include <hpc_helpers.cuh>
#include <gpu/cudaerrorcheck.cuh>

#include <cassert>

#ifdef __CUDACC__

template<class T>
struct AsyncConstBufferWrapper{
    explicit AsyncConstBufferWrapper(const T* buf) : AsyncConstBufferWrapper(buf, nullptr) {}
    AsyncConstBufferWrapper(const T* buf, cudaEvent_t event) : buffer(buf), readyEvent(event) {}

    void wait() const{
        if(readyEvent != nullptr){
            CUDACHECK(cudaEventSynchronize(readyEvent));
        }
    }

    bool ready() const{
        if(readyEvent != nullptr){
            cudaError_t status = cudaEventQuery(readyEvent);
            assert(status == cudaSuccess || status == cudaErrorNotReady);
            return status == cudaSuccess;
        }else{
            return true;
        }
    }

    //work submitted to stream after linkStream will not be processed until buffer is ready
    void linkStream(cudaStream_t stream) const {
        if(readyEvent != nullptr){
            CUDACHECK(cudaStreamWaitEvent(stream, readyEvent, 0));
        }
    }

    const T* data() const noexcept{
        return buffer;
    }
private:
    const T* buffer = nullptr;
    cudaEvent_t readyEvent = nullptr;
};

template<class T>
AsyncConstBufferWrapper<T> makeAsyncConstBufferWrapper(const T* data, cudaEvent_t event = nullptr){
    return AsyncConstBufferWrapper<T>(data, event);
}





#endif




#endif
