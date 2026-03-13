CXX       = g++
MCC       = mcc
CXXFLAGS  = -O2 -std=c++17

# 平台参数
PLATFORM  ?= NVIDIA
GPUCC     ?= nvcc
GPUFLAGS  ?= -O3 -std=c++17 -arch=native -DPLATFORM_NVIDIA

# 运行参数
N    ?= 1000
M    ?= 10000
SEED ?= 42
Q    ?= 10
# 自动组合文件名
CASE ?= $(N)x$(M)

# 根据平台设置编译器和编译选项
ifeq ($(PLATFORM),NVIDIA)
    GPUCC = nvcc
    GPUFLAGS = -O3 -std=c++17 -arch=native -DPLATFORM_NVIDIA
else ifeq ($(PLATFORM),ILUVATAR)
    GPUCC = clang++
    GPUFLAGS = -O3 -std=c++17 -DPLATFORM_ILUVATAR -lcudart -I/usr/local/corex/include -L/usr/local/corex/lib64 -fPIC
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

.PHONY: all clean test test_small test_medium test_large run-all run-gpu

# 默认编译所有可执行文件
all: $(BINS)

# 编译规则
gen_graph cpu_maxflow check_results: %: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

gpu_maxflow: gpu_maxflow.cu
ifeq ($(PLATFORM),MOORE)
	cp gpu_maxflow.cu gpu_maxflow.mu
	$(GPUCC) $(GPUFLAGS) -o $@ gpu_maxflow.mu
else
	$(GPUCC) $(GPUFLAGS) -o $@ $<
endif

# 若缺少 .bin 和 .txt 图数据文件, 自动调用 gen_graph 生成
$(CASE).bin $(CASE).txt: gen_graph
	@echo "=== [Auto] Generating graph: N=$(N), M=$(M) ==="
	./gen_graph $(N) $(M) $(SEED) $(CASE).bin $(CASE).txt $(Q)

# 若缺少 CPU 结果文件, 自动调用 cpu_maxflow 生成
$(CASE)_cpu.txt: cpu_maxflow $(CASE).bin $(CASE).txt
	@echo "=== [Auto] Running CPU baseline ==="
	./cpu_maxflow $(CASE).bin $(CASE).txt $(CASE)_cpu.txt

# 运行测试目标

run-all: all
	@echo "=== Test All: (N=$(N), M=$(M), SEED=$(SEED), Q=$(Q)) on $(PLATFORM) ==="
	./gen_graph $(N) $(M) $(SEED) $(CASE).bin $(CASE).txt $(Q)
	./cpu_maxflow $(CASE).bin $(CASE).txt $(CASE)_cpu.txt
	./gpu_maxflow $(CASE).bin $(CASE).txt $(PLATFORM)_$(CASE)_gpu.txt
	./check_results $(CASE)_cpu.txt $(PLATFORM)_$(CASE)_gpu.txt

run-gpu: all $(CASE)_cpu.txt $(CASE).bin $(CASE).txt
	@echo "=== Running GPU test: (N=$(N), M=$(M), SEED=$(SEED), Q=$(Q)) on $(PLATFORM) ==="
	./gpu_maxflow $(CASE).bin $(CASE).txt $(PLATFORM)_$(CASE)_gpu.txt
	./check_results $(CASE)_cpu.txt $(PLATFORM)_$(CASE)_gpu.txt

# 预设测试集
test_small:
	$(MAKE) run-all N=2000 M=20000 SEED=1 Q=10

test_medium:
	$(MAKE) run-all N=20000 M=200000 SEED=2 Q=50

test_large:
	$(MAKE) run-all N=200000 M=2000000 SEED=3 Q=100

test: test_small test_medium test_large

# 清理规则
clean:
	rm -f $(BINS) *.bin *.txt *.mu
