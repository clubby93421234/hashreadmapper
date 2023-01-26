
#CC = gcc
CXX=g++
CUDACC=nvcc
HOSTLINKER=g++

PREFIX = /usr/local

CUB_INCDIR = ./dependencies/cub-1.16.0
THRUST_INCDIR = ./dependencies/thrust-1.16.0
WARPCORE_INCDIR = ./dependencies/warpcore/include
RMM_INCDIR = ./dependencies/rmm/include
SPDLOG_INCDIR = ./dependencies/spdlog/include



WARPCORE_FLAGS = -DCARE_HAS_WARPCORE -I$(WARPCORE_INCDIR)

CXXFLAGS = -std=c++17

COMPILER_WARNINGS = -Wall -Wextra 
COMPILER_DISABLED_WARNING = -Wno-terminate -Wno-deprecated-copy

CFLAGS = $(COMPILER_WARNINGS) $(COMPILER_DISABLED_WARNING) -fopenmp -Iinclude -O3 -g -march=native -I$(THRUST_INCDIR)
CFLAGS += -DFAKEGPUMINHASHER_USE_OMP_FOR_QUERY

NVCCFLAGS = -x cu -lineinfo --expt-extended-lambda --expt-relaxed-constexpr -ccbin $(CXX) -I$(CUB_INCDIR) $(WARPCORE_FLAGS) -I$(RMM_INCDIR) -I$(SPDLOG_INCDIR) 

LDFLAGSGPU = -lpthread -lgomp -lstdc++fs -lnvToolsExt -lz -ldl

SOURCES_GPU_VERSION = \
	src/options.cpp \
	src/readlibraryio.cpp \
	src/gpu/hammingdistancekernels.cu \
    src/gpu/main_gpu.cu \
	src/gpu/gpuminhasherconstruction.cu \
	src/gpu/sequenceconversionkernels.cu \
	src/ssw_cpp.cpp \
	src/ssw.c \
	src/cigar.cpp \
	src/constants.cpp \
	src/filehandler.cpp	\
	src/varianthandler.cpp \
	src/variant.cpp 

#	src/filehandler.cpp 
#	src/referencehandler.cpp
#	src/sequencehandler.cpp 

EXECUTABLE_GPU_VERSION = main-gpu

BUILDDIR_GPU_VERSION = build_gpu_version

SOURCES_GPU_VERSION_NODIR = $(notdir $(SOURCES_GPU_VERSION))

OBJECTS_GPU_VERSION_NODIR_TMP_C = $(SOURCES_GPU_VERSION_NODIR:%.c=%.o)
OBJECTS_GPU_VERSION_NODIR_TMP = $(OBJECTS_GPU_VERSION_NODIR_TMP_C:%.cpp=%.o)
OBJECTS_GPU_VERSION_NODIR = $(OBJECTS_GPU_VERSION_NODIR_TMP:%.cu=%.o)

OBJECTS_GPU_VERSION = $(OBJECTS_GPU_VERSION_NODIR:%=$(BUILDDIR_GPU_VERSION)/%)

findgpus: findgpus.cu
	@$(CUDACC) findgpus.cu -o findgpus

.PHONY: gpuarchs.txt
gpuarchs.txt : findgpus
	$(shell ./findgpus > gpuarchs.txt) 


gpu_version_release: gpuarchs.txt
	@$(MAKE) gpu_version_release_dummy DIR=$(BUILDDIR_GPU_VERSION) CUDA_ARCH="$(shell cat gpuarchs.txt)"

gpu_version_release_dummy: $(BUILDDIR_GPU_VERSION) $(OBJECTS_GPU_VERSION) 
	@echo Linking $(EXECUTABLE_GPU_VERSION)
	@$(CUDACC) $(CUDA_ARCH) $(OBJECTS_GPU_VERSION) $(LDFLAGSGPU) -o $(EXECUTABLE_GPU_VERSION)
	@echo Linked $(EXECUTABLE_GPU_VERSION)

C_OMPILE = @echo "Compiling $< to $@" ; $(CC) $(CFLAGS) -c $< -o $@
COMPILE = @echo "Compiling $< to $@" ; $(CXX) $(CXXFLAGS) $(CFLAGS) -c $< -o $@
CUDA_COMPILE = @echo "Compiling $< to $@" ; $(CUDACC) $(CUDA_ARCH) $(CXXFLAGS) $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -c $< -o $@



.PHONY: gpu install clean
gpu: gpu_version_release

install: 
	@echo "Installing to directory $(PREFIX)/bin"
	mkdir -p $(PREFIX)/bin	
ifneq ("$(wildcard $(EXECUTABLE_GPU_VERSION))","")
	cp $(EXECUTABLE_GPU_VERSION) $(PREFIX)/bin/$(EXECUTABLE_GPU_VERSION)
endif


clean : 
	@rm -rf build_*
	@rm -f $(EXECUTABLE_GPU_VERSION)

$(DIR):
	mkdir $(DIR)


$(DIR)/options.o : src/options.cpp
	$(COMPILE)


#Complete_Striped_Smith_Waterman
$(DIR)/ssw_cpp.o : src/ssw_cpp.cpp
	$(COMPILE)
#Complete_Striped_Smith_Waterman
$(DIR)/ssw.o : src/ssw.c
	$(COMPILE)

#BAM Variant Caller and Genomic Analysis 
$(DIR)/cigar.o : src/cigar.cpp
	$(COMPILE)
$(DIR)/constants.o : src/constants.cpp
	$(COMPILE)
$(DIR)/filehandler.o : src/filehandler.cpp
	$(COMPILE)
$(DIR)/varianthandler.o : src/varianthandler.cpp
	$(COMPILE)
$(DIR)/variant.o : src/variant.cpp
	$(COMPILE)

#$(DIR)/sequencehandler.o : src/sequencehandler.cpp
#	$(COMPILE)
#$(DIR)/referencehandler.o : src/referencehandler.cpp
#	$(COMPILE)



$(DIR)/readlibraryio.o : src/readlibraryio.cpp
	$(COMPILE)

$(DIR)/main_gpu.o : src/gpu/main_gpu.cu
	$(CUDA_COMPILE)

$(DIR)/hammingdistancekernels.o : src/gpu/hammingdistancekernels.cu
	$(CUDA_COMPILE)

$(DIR)/gpuminhasherconstruction.o : src/gpu/gpuminhasherconstruction.cu
	$(CUDA_COMPILE)

$(DIR)/sequenceconversionkernels.o : src/gpu/sequenceconversionkernels.cu
	$(CUDA_COMPILE)






