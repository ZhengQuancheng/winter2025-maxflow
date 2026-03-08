CXX       = g++
NVCC      = nvcc
MXCC      = mxcc
CXXFLAGS  = -O2 -std=c++17
NVCCFLAGS = -O3 -std=c++17 -arch=native -DPLATFORM_NVIDIA
MXCCFLAGS = -O3 -std=c++17 -DPLATFORM_METAX

PLATFORM  ?= NVIDIA
GPUCC  	  ?= $(NVCC)
GPUFLAGS  ?= $(NVCCFLAGS)
ifeq ($(PLATFORM),NVIDIA)
	GPUCC = $(NVCC)
	GPUFLAGS = $(NVCCFLAGS)
else ifeq ($(PLATFORM),METAX)
	GPUCC = $(MXCC)
	GPUFLAGS = $(MXCCFLAGS)
endif

BINS = gen_graph cpu_maxflow gpu_maxflow check_results

.PHONY: all clean test test_small test_medium test_large

all: $(BINS)

gen_graph cpu_maxflow check_results: %: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

gpu_maxflow: gpu_maxflow.cu
	$(GPUCC) $(GPUFLAGS) -o $@ $<

define run_test
	@echo "=== Test: $(1) (N=$(2), M=$(3)) ==="
	./gen_graph $(2) $(3) $(4) $(1).bin $(1)_q.txt $(5)
	./cpu_maxflow $(1).bin $(1)_q.txt $(1)_cpu.txt
	./gpu_maxflow $(1).bin $(1)_q.txt $(1)_gpu.txt
	./check_results $(1)_cpu.txt $(1)_gpu.txt
endef

test_small: all
	$(call run_test,small,100,1000,1,10)

test_medium: all
	$(call run_test,medium,10000,100000,2,10)

test_large: all
	$(call run_test,large,100000,1000000,3,100)

test: test_small test_medium test_large

clean:
	rm -f $(BINS) *.bin *.txt
