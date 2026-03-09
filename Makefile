CXX       = g++
MCC       = mcc
CXXFLAGS  = -O2 -std=c++17

PLATFORM  ?= NVIDIA
GPUCC  	  ?= nvcc
GPUFLAGS  ?= -O3 -std=c++17 -arch=native -DPLATFORM_NVIDIA
ifeq ($(PLATFORM),NVIDIA)
	GPUCC = nvcc
	GPUFLAGS = -O3 -std=c++17 -arch=native -DPLATFORM_NVIDIA
else ifeq ($(PLATFORM),METAX)
	GPUCC = mxcc
	GPUFLAGS = -O3 -std=c++17 -DPLATFORM_METAX
else ifeq ($(PLATFORM),MOORE)
	GPUCC = mcc
	GPUFLAGS = -O3 -std=c++17 -lmusart -DPLATFORM_MOORE
else ifeq ($(PLATFORM),HYGON)
	GPUCC = hipcc
	GPUFLAGS = -O3 -std=c++17 -DPLATFORM_HYGON
endif

BINS = gen_graph cpu_maxflow gpu_maxflow check_results

.PHONY: all clean test test_small test_medium test_large

all: $(BINS)

gen_graph cpu_maxflow check_results: %: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

gpu_maxflow: gpu_maxflow.cu
ifeq ($(PLATFORM),MOORE)
	cp gpu_maxflow.cu gpu_maxflow.mu
	$(GPUCC) $(GPUFLAGS) -o $@ gpu_maxflow.mu
else
	$(GPUCC) $(GPUFLAGS) -o $@ $<
endif

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
