// Compile: nvcc -O3 -std=c++17 -arch=native gpu_maxflow.cu -o gpu_maxflow
// Usage: ./gpu_maxflow <graph.bin> <query.txt> <output.txt>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <cassert>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

static constexpr int   INF_LABEL = 1 << 28;
static constexpr float EPS       = 1e-6f;
static constexpr int   BLK       = 256;

// CSR 格式的图结构
struct GraphCSR {
    int V = 0;                  // 节点数
    int E = 0;                  // 有向边数
    // row_ptr[u] 是节点 u 的出边 e 在 col[] 和 cap[] 中的起始索引, row_ptr[u+1] 是下一个节点的起始索引.
    // 对于节点 u, 其所有出边信息存储在 col 和 cap 数组的索引区间 [row_ptr[u], row_ptr[u+1]) 中.
    // row_ptr[u+1] - row_ptr[u] 是节点 u 的出边数 (出度).
    std::vector<int>   row;     // 长度为 N+1
    // 遍历节点 u 的出边, 其索引 e 在区间 [row_ptr[u], row_ptr[u+1]) 中, col[e] 是此边的终点.
    std::vector<int>   col;     // 长度为 M, 存储每条边的终点
    // 遍历节点 u 的出边, 其索引 e 在区间 [row_ptr[u], row_ptr[u+1]) 中, cap[e] 是边 e 的容量.
    std::vector<float> cap;     // 长度为 M, 存储所有边的容量
    // rev[e] 是边 e 的反向边索引, 若 e 表示 u->v 则 rev[e] 表示 v->u.
    // 即 col[rev[e]] 为原边 e 的起点, cap[rev[e]] 为反向边的残余容量.
    // 设 u (col[rev[e]]) 为边 e 的起点, v (col[e]) 为边 e 的终点, 则 col[e] = v, col[rev[e]] = u.
    std::vector<int>   rev;     // 长度为 M
};

/**
 * @brief 从二进制文件加载图数据到 GraphCSR 结构
 * @param path 二进制文件路径
 * @param g 输出参数, 成功时包含加载的图数据
 * @return 成功返回 true, 失败返回 false
 */
bool load_graph_file(const char* path, GraphCSR& g) {
    // 以二进制只读模式打开文件
    FILE* fp = std::fopen(path, "rb");
    if (!fp) { std::fprintf(stderr, "Cannot open '%s': %s\n", path, strerror(errno)); return false; }

    // lambda 函数: 读取指定字节数到 buf 中并检查是否成功
    auto read_file = [&](void* buf, size_t n) { return std::fread(buf, 1, n, fp) == n; };

    // 读取文件元数据 N 和 M, 并进行基本验证
    bool ok = read_file(&g.V, sizeof(int)) && read_file(&g.E, sizeof(int));
    if (!ok || g.V <= 0 || g.E < 0) {
        std::fprintf(stderr, "Invalid header: N=%d M=%d\n", g.V, g.E);
        std::fclose(fp); return false;
    }

    // 读取数组数据到相应的 vector 中
    g.row.resize(g.V + 1);
    g.col.resize(g.E);
    g.cap.resize(g.E);
    ok =  read_file(g.row.data(), (g.V + 1) * sizeof(int))
       && read_file(g.col.data(), g.E * sizeof(int))
       && read_file(g.cap.data(), g.E * sizeof(float));

    // 读取文件完成后关闭文件
    std::fclose(fp);

    if (!ok) { std::fprintf(stderr, "Truncated file '%s'\n", path); return false; }

    // 校验 CSR 格式数据的合法性
    if (g.row[0] != 0 || g.row[g.V] != g.E) {
        std::fprintf(stderr, "Corrupt CSR boundaries\n"); return false;
    }
    for (int u = 0; u < g.V; ++u) {
        if (g.row[u] > g.row[u + 1]) {
            std::fprintf(stderr, "CSR not monotone at u=%d\n", u); return false;
        }
    }
    for (int i = 0; i < g.E; ++i) {
        if (g.col[i] < 0 || g.col[i] >= g.V) {
            std::fprintf(stderr, "col[%d]=%d out of range\n", i, g.col[i]);
            return false;
        }
        if (!std::isfinite(g.cap[i]) || g.cap[i] < 0.0f) {
            std::fprintf(stderr, "Invalid cap[%d]=%g\n", i, (double)g.cap[i]); return false;
        }
    }
    return true;
}

/**
 * @brief 根据原始图构建残量图
 * @param g 输入的原始图 (CSR 格式)
 * @param r 输出参数, 成功时包含构建的残量图 (CSR 格式)
 * @return 成功返回 true, 失败返回 false
 */
bool build_residual_graph(const GraphCSR& g, GraphCSR& r) {
    r.V = g.V;

    // 边数翻倍
    int64_t m2 = (int64_t)g.E * 2;
    if (m2 > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "2*M overflows int32\n"); return false;
    }
    r.E = (int)m2;

    r.row.assign(r.V + 1, 0);
    r.col.resize(r.E);
    r.cap.resize(r.E);
    r.rev.resize(r.E);

    // 统计每个节点的出度
    for (int u = 0; u < g.V; ++u) {
        for (int e = g.row[u]; e < g.row[u + 1]; ++e) {
            int v = g.col[e];
            r.row[u + 1]++; // 正向出边
            r.row[v + 1]++; // 反向出边
        }
    }
    // 此时 row_ptr[u+1] 存储节点 u 在残量图中的出边数(出度).

    // 计算前缀和
    for (int i = 1; i <= r.V; ++i) {
        r.row[i] += r.row[i - 1];
    }
    // 此时 row_ptr[u] 存储节点 u 在残量图中出边的起始位置, row_ptr[u+1]-row_ptr[u] 是出边数.

    // 验证 row_ptr 的正确性
    assert(r.row[r.V] == r.E);

    // cur[u] 是节点 u 下一条待写入边的位置, 每次往节点 u 写入一条边后, cur[u]++ 会向后推进.
    std::vector<int> cur(r.row.begin(), r.row.end() - 1);

    for (int u = 0; u < g.V; ++u)
        for (int e = g.row[u]; e < g.row[u + 1]; ++e) {
            // u -> v
            int v = g.col[e];
            int fwd = cur[u]++; // 正向边 u->v 的位置 fwd
            int bwd = cur[v]++; // 反向边 v->u 的位置 bwd

            // 在位置 fwd 写入正向边 u->v 的信息
            r.col[fwd] = v;
            r.cap[fwd] = (double)g.cap[e];
            // 在位置 bwd 写入反向边 v->u 的信息
            r.col[bwd] = u;
            r.cap[bwd] = 0.0;
            // 记录正反向边的对应关系
            r.rev[fwd] = bwd;
            r.rev[bwd] = fwd;
        }

    return true;
}

// GPU 上存储图的数据结构 DeviceGraph
struct DeviceGraph {
    int   V;
    int   E;
    int   *d_row;
    int   *d_col;
    int   *d_rev;
    float *d_cap;
};

/**
 * @brief 将主机上的图数据传到 GPU 设备内存
 * @param rg 输入的图数据 (CSR 格式)
 * @param dg 输出的 GPU 设备图结构
 */
void upload_graph(const GraphCSR& rg, DeviceGraph& dg) {
    dg.V = rg.V;  dg.E = rg.E;
    CUDA_CHECK(cudaMalloc(&dg.d_row, (rg.V + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dg.d_col, rg.E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dg.d_rev, rg.E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dg.d_cap, rg.E * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dg.d_row, rg.row.data(), (rg.V+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dg.d_col, rg.col.data(), rg.E*sizeof(int),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dg.d_rev, rg.rev.data(), rg.E*sizeof(int),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dg.d_cap, rg.cap.data(), rg.E*sizeof(float),   cudaMemcpyHostToDevice));
}

/**
 * @brief GPU 上的 float 类型的原子减法操作, 但不允许结果变为负数.
 * @param addr 要修改的内存地址, 该地址存储一个 float 类型的值.
 * @param want 想要减去的值
 * @return 实际减去的值
 */
__device__ __forceinline__
float float_atomic_sub(float* addr, float want) {
    // 当 want 非正时, 不执行减法, 直接返回 0
    if (want <= EPS) return 0.0f;
    // 将 float* 转换为 unsigned* 以进行原子操作
    unsigned* ua = reinterpret_cast<unsigned*>(addr);
    // 原子地读出当前值
    unsigned old_u = atomicCAS(ua, 0u, 0u);
    // 自旋循环
    while (true) {
        // 将读出的 unsigned 值转换回 float
        float old_f = __uint_as_float(old_u);
        // 如果当前值已经不大于 EPS 或者不是有限数, 则不执行减法, 直接返回 0
        if (old_f <= EPS || !isfinite(old_f)) return 0.0f;
        // 计算实际要减去的值, 不能超过 old_f, 以避免结果变为负数
        float delta = fminf(old_f, want);
        // 计算执行减法后的新值, 不能小于 0
        float new_f = fmaxf(old_f - delta, 0.0f);
        // 将新值转换为 unsigned 以进行原子 CAS 操作
        unsigned prev = atomicCAS(ua, old_u, __float_as_uint(new_f));
        // 返回值等于期望值 old_u, 表明 CAS 成功
        if (prev == old_u) return delta;
        // 更新 old_u 以进行下一轮尝试 (因为可能有其他线程修改了值)
        old_u = prev;
    }
}

/**
 * @brief 将初始容量 cap0 拷贝到 cap 中
 * @param E 边数
 * @param cap0 输入的初始容量数组 (只读)
 * @param cap 输出的工作容量数组 (可修改)
 */
__global__ void kernel_reset_cap(int E, const float* __restrict__ cap0, float* cap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < E) cap[i] = cap0[i];
}

/**
 * @brief 初始化节点的 excess 和 height 数组
 * @param V 节点数
 * @param s 源节点
 * @param excess excess 数组, 每个节点的初始盈余流量为 0
 * @param height height 数组, 每个节点的初始高度为 0, 但源节点 s 的高度初始化为 V
 */
__global__ void kernel_init_nodes(int V, int s, float* excess, int* height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= V) return;
    excess[i] = 0.0f;
    height[i] = (i == s) ? V : 0;
}

/**
 * @brief 对源点 s 的所有出边做饱和推送, 完成 push-relabel 的预流 preflow 初始化
 * @param s 源点编号
 * @param d_row CSR 格式的 row 数组, 用于定位节点 s 的出边范围
 * @param d_col CSR 格式的 col 数组, 存储边的终点信息
 * @param d_rev CSR 格式的 rev 数组, 存储边的反向边索引
 * @param cap 边的容量数组, 该函数会修改源点 s 的出边容量和反向边容量
 * @param excess 节点盈余流量数组, 该函数会修改源点 s 和其出边终点的 excess 值
 *               excess[u] = 流入 u 的总流量 - 流出 u 的总流量,
 *               在 preflow 初始化后, 源点 s 的 excess 将为负数, 其出边终点的 excess 将为正数.
 */
__global__ void kernel_saturate_source(
    int s,
    const int* __restrict__ d_row,
    const int* __restrict__ d_col,
    const int* __restrict__ d_rev,
    float* cap, float* excess)
{
    // 每个线程处理源点 s 的一部分出边
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // 节点 s 的出边索引区间 [begin, end)
    int begin = d_row[s], end = d_row[s+1];
    // 对每条源点 s 的出边 e = s->v
    for (int e = begin + index; e < end; e += stride) {
        // 若其正向残量 cap[e] > 0, 则执行饱和推送
        float c = cap[e];
        if (c > EPS) {
            // 正向边残量清零
            cap[e] = 0.0f;
            // 反向边残量增加增加同样的流量, d_rev[e] 是边 e 的反向边索引
            atomicAdd(&cap[d_rev[e]], c);
            // 将该流量累加到 excess[d_col[e]], d_col[e] 是边 e 的终点 v
            atomicAdd(&excess[d_col[e]], c);
            // 从 excess[s] 中扣除同样的量
            atomicAdd(&excess[s], -c);
        }
    }
}


/**
 * @brief 并行执行 push-relabel 中单个节点的 fused discharge 操作
 * @param V 节点总数
 * @param s 源点编号
 * @param t 汇点编号
 * @param d_row CSR 格式的 row 数组, 用于定位节点 u 的出边范围
 * @param d_col CSR 格式的 col 数组, 存储边的终点信息
 * @param d_rev CSR 格式的 rev 数组, 存储边的反向边索引
 * @param cap 边的容量数组, 该函数会修改节点 u 的出边容量和反向边容量
 * @param excess 节点盈余流量数组, 该函数会修改节点 u 和其出边终点的 excess 值
 * @param height 节点高度标号数组, 该函数会修改节点 u 的 height 值
 * @param flag 设备内存中的标志位, 如果该函数对节点 u 执行了 push 或 relabel 操作, 则将 flag[0] 置为 1, 否则保持不变
 */
__global__ void kernel_discharge(
    int V, int s, int t,
    const int* __restrict__ d_row,
    const int* __restrict__ d_col,
    const int* __restrict__ d_rev,
    float* cap, float* excess,
    int* height, int* flag)
{
    // 当前线程负责处理的节点编号 u
    int u = blockIdx.x * blockDim.x + threadIdx.x;

    // 越界线程、源点、汇点均不参与此 discharge 操作
    if (u >= V || u == s || u == t) return;

    // 读取节点 u 的当前高度
    int hu = height[u];
    // 若为 INF_LABEL, 表示该点当前不可有效处理, 直接跳过
    if (hu >= INF_LABEL) return;

    // 原子地取走 excess[u] 并将其清零, rem 表示本线程当前要处理的剩余盈余流量
    // atomicExch 确保只有一个线程处理 u 的盈余流量, 避免多个线程同时处理同一节点导致的数据竞争
    float rem = atomicExch(&excess[u], 0.0f);
    // 无效的盈余流量 (非正数或非有限数) 不需要处理, 直接返回
    if (!(rem > EPS) || !isfinite(rem)) return;

    // 标记本线程是否在本次调用中做过有效工作 (push 或 relabel)
    bool work = false;

    // 节点 u 的出边索引区间 [begin, end)
    int begin = d_row[u], end = d_row[u+1];
    // 最多执行两轮
    //     第 1 轮：按旧高度尝试 push, 并收集 relabel 所需信息
    //     第 2 轮：若 relabel 成功, 则按新高度再尝试一次 push,
    //             避免节点在 relabel 后需要等到下一轮迭代才能推送
    // ── Two passes: push -> relabel -> push again with updated height ──
    for (int pass = 0; pass < 2 && rem > EPS; ++pass) {
        // 用于 relabel 操作, 记录所有残余容量的出边中邻居节点的最小高度 + 1
        int new_h = INF_LABEL;

        // 扫描 u 的全部出边
        for (int e = begin; e < end; ++e) {
            // 读取边 e 的残余容量
            float rc = cap[e];
            // 若残余容量过小不能用于 push, 直接跳过
            if (rc <= EPS) continue;

            // 边 e: u -> v, 读取终点 v 及其高度 hv
            int v  = d_col[e];
            int hv = height[v];

            // 若边 (u, v) 是 admissible edge,
            // 则可沿该边尝试推送 rem 中的一部分流量
            // admissible edge 的定义: (1) 残余容量 > 0, (2) 满足高度约束 hu == hv + 1
            if (hv == hu - 1 && rem > EPS) {
                // 尝试从 cap[e] 中原子地减去 rem, 实际成功减去的值为 d
                float d = float_atomic_sub(&cap[e], rem);
                // 如果成功推送了 d > EPS 的流量
                if (d > EPS) {
                    // 正向边推送 d 后, 反向边残量增加 d
                    atomicAdd(&cap[d_rev[e]], d);
                    // 终点 v 收到 d 单位流量, 其盈余增加 d
                    atomicAdd(&excess[v], d);
                    // 从 rem 中扣除已成功推送的 d 单位流量
                    rem -= d;
                    // 记录本线程做过有效工作
                    work = true;
                }
            }

            // 若所有盈余已推送完毕, 提前退出循环
            if (rem <= EPS) break;

            // 收集 relabel 信息:
            // 对于所有有残余容量的出边 e: u->v, 记录邻居节点的最小高度 + 1
            if (cap[e] > EPS && hv < INF_LABEL) {
                int nh = hv + 1;
                if (nh < new_h) new_h = nh;
            }
        }

        // relabel 操作: 提升节点高度
        // relabel 的条件:
        //     1. rem > EPS: 仍有盈余未推送完
        //     2. new_h > hu: 新高度必须严格大于当前高度
        if (rem > EPS && new_h > hu) {
            // 更新局部变量 hu (用于第二轮 push)
            hu = new_h;
            // 更新全局数组 height[u]
            height[u] = hu;
            // 标记执行了有效工作
            work = true;
        } else {
            // 如果没有 relabel, 第二轮 push 不会有新的可行边, 提前退出循环
            break;
        }
    }
    // 如果还有剩余盈余未推送完 (rem > EPS), 将其原子地加回到 excess[u]
    if (rem > EPS) atomicAdd(&excess[u], rem);
    // 如果本线程执行了有效工作, 设置标志位 flag[0] = 1, 以通知主机端本轮迭代有节点被处理过
    if (work)      flag[0] = 1;
}

/**
 * @brief 初始化 two-phase global relabel 所需的高度数组和初始 frontier 队列
 * @param V 节点总数
 * @param s 源点编号
 * @param t 汇点编号
 * @param height 节点高度数组, 该函数会将 height[t] 初始化为 0, height[s] 初始化为 V, 其他节点初始化为 INF_LABEL
 * @param frontier BFS 的前沿队列, 该函数会将 frontier[0] 初始化为 t
 * @param fsize frontier 队列的当前大小, 该函数会将其初始化为 1
 */
__global__ void kernel_bfs_init(
    int V, int s, int t,
    int* height, int* frontier, int* fsize)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= V) return;
    height[u] = (u == t) ? 0 : (u == s) ? V : INF_LABEL;
    if (u == 0) { frontier[0] = t; *fsize = 1; }
}

/**
 * @brief Phase 1: 从汇点 t 反向 BFS, 计算能到达 t 的节点高度
 * @param fsz frontier 队列的当前大小
 * @param frontier 当前的 frontier (输入, 存储当前层的节点)
 * @param next 下一层 frontier (输出, 存储下一层的节点)
 * @param nsz 下一层 frontier 的大小 (输出, 原子计数器)
 * @param d_row CSR 格式的 row 数组
 * @param d_col CSR 格式的 col 数组
 * @param d_rev CSR 格式的 rev 数组 (反向边索引)
 * @param cap 残余容量数组
 * @param height 节点高度数组
 * @param level 当前 BFS 层数
 * @param V 节点总数
 */
__global__ void kernel_bfs_backward(
    int fsz, const int* __restrict__ frontier, int* next, int* nsz,
    const int* __restrict__ d_row, const int* __restrict__ d_col,
    const int* __restrict__ d_rev, const float* __restrict__ cap,
    int* height, int level, int V)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= fsz) return;

    int v = frontier[idx];
    // v 的出边索引区间 [begin, end)
    int begin = d_row[v], end = d_row[v+1];

    // 遍历有向边 v -> w
    for (int i = begin; i < end; ++i) {
        // 如果 cap[rev[i]] > 0, 即反向边 w -> v 有残余容量
        // 则说明在残余网络中 w 可以到达 v, 因此 w 应该被加入下一层前沿队列
        if (cap[d_rev[i]] > EPS) {
            // 若 w 尚未被访问, 则将其高度设置为 level + 1, 并加入下一层前沿队列 next
            int w = d_col[i];
            if (atomicCAS(&height[w], INF_LABEL, level + 1) == INF_LABEL) {
                int pos = atomicAdd(nsz, 1);
                if (pos < V) next[pos] = w;
            }
        }
    }
}

/**
 * @brief Phase 2: 从源点 s 正向 BFS, 计算从 s 可达的节点高度
 * @param fsz frontier 队列的当前大小
 * @param frontier 当前的 frontier (输入, 存储当前层的节点)
 * @param next 下一层 frontier (输出, 存储下一层的节点)
 * @param nsz 下一层 frontier 的大小 (输出, 原子计数器)
 * @param d_row CSR 格式的 row 数组
 * @param d_col CSR 格式的 col 数组
 * @param cap 残余容量数组
 * @param height 节点高度数组
 * @param level 当前 BFS 层数 (从 N 开始)
 * @param V 节点总数
 */
__global__ void kernel_bfs_forward(
    int fsz, const int* __restrict__ frontier, int* next, int* nsz,
    const int* __restrict__ d_row, const int* __restrict__ d_col,
    const float* __restrict__ cap,
    int* height, int level, int V)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= fsz) return;

    int v = frontier[idx];
    // v 的出边索引区间 [begin, end)
    int begin = d_row[v], end = d_row[v+1];

    // 遍历有向边 v -> w
    for (int i = begin; i < end; ++i) {
        // 仅沿当前残量图中仍可走的正向边扩展
        if (cap[i] > EPS) {
            // 若 w 尚未被访问, 则将其高度设置为 level + 1, 并加入下一层前沿队列 next
            int w = d_col[i];
            if (atomicCAS(&height[w], INF_LABEL, level + 1) == INF_LABEL) {
                int pos = atomicAdd(nsz, 1);
                if (pos < V) next[pos] = w;
            }
        }
    }
}

/**
 * @brief 在一个 warp 内执行 float 类型的并行规约求最大值
 * @param value 每个线程提供一个 float 类型的输入值
 * @return 返回 warp 内所有线程输入值的最大值
 */
__device__ __forceinline__ float warp_reduce_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    return value;
}

/**
 * @brief 计算所有活跃节点的最大盈余流量 (用于收敛判断)
 * @param V 节点总数
 * @param s 源点编号
 * @param t 汇点编号
 * @param excess 盈余流量数组
 * @param height 高度数组
 * @param d_max_bits 输出: 最大盈余的位表示 (unsigned 格式)
 */
__global__ void kernel_max_excess(
    int V, int s, int t,
    const float* __restrict__ excess,
    const int*   __restrict__ height,
    unsigned* d_max_bits)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // 当前线程的局部最大盈余值
    float max_val = 0.0f;
    for (int i = index; i < V; i += stride) {
        // 跳过非活跃节点
        if (i != s && i != t && height[i] < INF_LABEL) {
            // 取盈余的绝对值
            float v = fabsf(excess[i]);
            // 只考虑有限的值 (排除 NaN 和 Inf)
            if (isfinite(v)) max_val = fmaxf(max_val, v);
        }
    }
    // Warp 内归约, 计算每个 Warp 内的最大值
    max_val = warp_reduce_max(max_val);
    // 每个 warp 的第一个线程执行原子操作, 将其最大值与全局最大值进行比较并更新
    if ((threadIdx.x & 31) == 0) {
        unsigned bits = __float_as_uint(max_val);
        if (bits) atomicMax(d_max_bits, bits);
    }
}

/**
 * @brief 固定内存 (Pinned Memory) 缓冲区, 存储需要频繁在 CPU 和 GPU 之间传输的小数据
 */
struct PinnedBuffer {
    int *flag;      // 标志位: 是否有节点执行了 push/relabel
    int *fsz;       // BFS 前沿队列大小
    unsigned *max_bits;   // 最大盈余的位表示
};

/**
 * @brief GPU 工作缓冲区, 包含算法运行所需的所有设备内存
 */
struct WorkBuffer {
    float *cap;     // 残余容量数组 (大小 E)
    float *excess;  // 盈余流量数组 (大小 V)
    int   *height;  // 高度数组 (大小 V)
    int   *flag;    // 工作标志
    int   *fa, *fb; // BFS 前沿队列 A 和 B (大小 V, 双缓冲)
    int   *fsz;     // 前沿队列大小
    unsigned *max_bits;   // 最大盈余的位表示
    cudaStream_t stream; // CUDA 流
};

static inline int grid_dim(int n) { return (n + BLK - 1) / BLK; }

/**
 * @brief 在 GPU 上执行两阶段全局重标号操作
 * @param V 节点总数
 * @param s 源点编号
 * @param t 汇点编号
 * @param dg 设备上的图数据结构
 * @param d_cap 残余容量数组
 * @param d_ht 高度数组
 * @param fa 前沿队列 A
 * @param fb 前沿队列 B
 * @param fsz 前沿队列大小
 * @param h_fsz 主机内存中的前沿队列大小 (用于 CPU 和 GPU 之间的同步)
 * @param stream CUDA 流
 */
void global_relabel(int V, int s, int t,
    const DeviceGraph& dg, float* d_cap, int* d_ht,
    int* fa, int* fb, int* fsz, int* h_fsz, cudaStream_t stream)
{
    // 设置高度和初始前沿队列
    kernel_bfs_init<<<grid_dim(V), BLK, 0, stream>>>(V, s, t, d_ht, fa, fsz);
    // 双缓冲指针: cur 指向当前层, nxt 指向下一层
    int *cur = fa, *nxt = fb;

    // Phase 1: 从 t 开始反向 BFS, 设置所有能到达 t 的节点的高度
    for (int lv = 0; lv < V; ++lv) {
        CUDA_CHECK(cudaMemcpyAsync(h_fsz, fsz, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (*h_fsz == 0) break;
        CUDA_CHECK(cudaMemsetAsync(fsz, 0, sizeof(int), stream));
        kernel_bfs_backward<<<grid_dim(*h_fsz), BLK, 0, stream>>>(
            *h_fsz, cur, nxt, fsz,
            dg.d_row, dg.d_col, dg.d_rev, d_cap, d_ht, lv, V);
        // 交换 cur 和 nxt, 为下一轮迭代做准备
        std::swap(cur, nxt);
    }

    // Phase 2: 从 s 开始正向 BFS, 设置"可从 s 到达但无法到达 t"节点的高度
    // 重置前沿队列: 将源点 s 作为起点
    CUDA_CHECK(cudaMemsetAsync(fsz, 0, sizeof(int), stream));
    CUDA_CHECK(cudaMemcpyAsync(cur, &s, sizeof(int), cudaMemcpyHostToDevice, stream));
    int one = 1;
    CUDA_CHECK(cudaMemcpyAsync(fsz, &one, sizeof(int), cudaMemcpyHostToDevice, stream));

    for (int lv = V; lv < 2 * V; ++lv) {
        CUDA_CHECK(cudaMemcpyAsync(h_fsz, fsz, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (*h_fsz == 0) break;
        CUDA_CHECK(cudaMemsetAsync(fsz, 0, sizeof(int), stream));
        kernel_bfs_forward<<<grid_dim(*h_fsz), BLK, 0, stream>>>(
            *h_fsz, cur, nxt, fsz,
            dg.d_row, dg.d_col, d_cap, d_ht, lv, V);
        // 交换 cur 和 nxt, 为下一轮迭代做准备
        std::swap(cur, nxt);
    }
}

/**
 * @brief Push-Relabel 最大流算法的求解函数
 * @param dg 设备图结构
 * @param w 工作缓冲区
 * @param p 固定内存缓冲区
 * @param s 源点编号
 * @param t 汇点编号
 * @param out_iters 输出迭代次数 (可选)
 * @return 最大流量值
 */
float solve(const DeviceGraph& dg,
            WorkBuffer& w, PinnedBuffer& p,
            int s, int t, int* out_iters)
{
    if (s == t) { if (out_iters) *out_iters = 0; return 0.0f; }

    const int V = dg.V, E = dg.E;
    cudaStream_t stream = w.stream;

    // 重置残余容量 (从原始容量拷贝)
    kernel_reset_cap<<<grid_dim(E), BLK, 0, stream>>>(E, dg.d_cap, w.cap);
    // 初始化节点状态 (excess=0; height=0 except s, h[s] = V)
    kernel_init_nodes<<<grid_dim(V), BLK, 0, stream>>>(V, s, w.excess, w.height);
    // 饱和源点的所有出边
    kernel_saturate_source<<<4, BLK, 0, stream>>>(s, dg.d_row, dg.d_col, dg.d_rev, w.cap, w.excess);

    // 首次全局重标号
    global_relabel(V, s, t, dg, w.cap, w.height, w.fa, w.fb, w.fsz, p.fsz, stream);

    constexpr uint64_t  GR_FREQ = 32;   // 全局重标号频率 (每 32 轮 discharge 执行一次)
    constexpr uint64_t  CHECK_FREQ = 8; // 进度检查频率 (每 8 轮 discharge 检查一次收敛)
    const uint64_t MAX_ITERS  = 1000LL * V + 10000; // 最大迭代次数 (防止无限循环), 理论上界: O(V^2 * E), 实际远小于此

    uint64_t iters = 0; // 当前迭代次数
    uint64_t stall = 0; // 连续无有效工作的次数 (用于判断收敛)

    while (iters < MAX_ITERS) {
        // 工作标志清零
        CUDA_CHECK(cudaMemsetAsync(w.flag, 0, sizeof(int), stream));
        // 执行 CHECK 轮 discharge 操作, 期间每 GR_FREQ 轮执行一次全局重标号
        for (int check = 0; check < CHECK_FREQ && iters < MAX_ITERS; ++check, ++iters) {
            // 周期性全局重标号
            if (iters > 0 && iters % GR_FREQ == 0) {
                global_relabel(V, s, t, dg, w.cap, w.height, w.fa, w.fb, w.fsz, p.fsz, stream);
            }
            // 执行 discharge kernel
            kernel_discharge<<<grid_dim(V), BLK, 0, stream>>>(
                V, s, t, dg.d_row, dg.d_col, dg.d_rev,
                w.cap, w.excess, w.height, w.flag);
        }

        // 进度检查: 是否有节点执行了 push/relabel
        CUDA_CHECK(cudaMemcpyAsync(p.flag, w.flag, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // 如果存在有效工作, 重置 stall 计数器并继续
        if (*p.flag) {
            stall = 0;
            continue;
        }

        // 无工作完成, 检查是否真正收敛
        ++stall; // 增加 stall 计数
        // 计算最大盈余 (用于收敛判断)
        CUDA_CHECK(cudaMemsetAsync(w.max_bits, 0, sizeof(unsigned), stream));
        kernel_max_excess<<<grid_dim(V), BLK, 0, stream>>>(
            V, s, t, w.excess, w.height, w.max_bits);
        // 异步拷贝最大盈余到主机
        CUDA_CHECK(cudaMemcpyAsync(p.max_bits, w.max_bits, sizeof(unsigned),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // 将 unsigned 位表示转换回 float
        float max_ex;
        memcpy(&max_ex, p.max_bits, sizeof(float));

        // 收敛判断
        if (stall >= 2 && max_ex <= EPS) break;        // converged

        // 有盈余单无工作, 需要全局重标号
        if (max_ex > EPS) {                             // stuck → relabel
            global_relabel(V, s, t, dg, w.cap, w.height, w.fa, w.fb, w.fsz, p.fsz, stream);
            stall = 0;
        }
    }

    // 读取最终流量并返回
    float flow;
    CUDA_CHECK(cudaMemcpy(&flow, w.excess + t, sizeof(float), cudaMemcpyDeviceToHost));
    // 输出迭代次数
    if (out_iters) *out_iters = iters;

    return flow;
}

/**
 * @brief 从文本文件加载查询对 (s,t) 到 qs 向量中
 * @param path 文本文件路径
 * @param V 图中节点数, 用于验证查询的合法性
 * @param qs 输出参数, 成功时包含加载的查询对
 * @return 成功返回 true, 失败返回 false
 */
bool load_query_file(const char* path, int V, std::vector<std::pair<int,int>>& qs) {
    std::ifstream fin(path);
    if (!fin) { std::fprintf(stderr, "Cannot open '%s': %s\n", path, strerror(errno)); return false; }

    std::string line;
    while (std::getline(fin, line)) {
        // 处理 Windows 风格的换行符
        if (!line.empty() && line.back() == '\r') line.pop_back();
        // 跳过空行和以 '#' 开头的注释行
        auto p = line.find_first_not_of(" \t");
        if (p == std::string::npos || line[p] == '#') continue;

        std::istringstream iss(line);
        long long s, t;
        if (!(iss >> s >> t)) { std::fprintf(stderr, "Bad query line: '%s'\n", line.c_str()); continue; }
        if (s < 0 || s >= V || t < 0 || t >= V) {
            std::fprintf(stderr, "Query (%lld,%lld) out of range [0,%d)\n", s, t, V); continue;
        }
        qs.emplace_back((int)s, (int)t);
    }
    return true;
}

int main(int argc, char** argv) {
    // 检查命令行参数
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <graph.bin> <query.txt> <output.txt>\n", argv[0]);
        return 1;
    }

    // 查询并打印当前 GPU 设备信息
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);

    // 读取原始图, 并构建残量图
    GraphCSR g;
    if (!load_graph_file(argv[1], g)) { fprintf(stderr, "Load failed\n"); return 1; }
    printf("Graph: N=%d  M=%d\n", g.V, g.E);

    GraphCSR rg;
    if (!build_residual_graph(g, rg)) { fprintf(stderr, "Residual graph build failed\n"); return 1; }
    printf("Residual: N=%d  E=%d  rev[]=%d\n", rg.V, rg.E, (int)rg.rev.size());

    // 将残量图上传到 GPU
    DeviceGraph dg;
    upload_graph(rg, dg);

    // 分配 GPU 工作缓冲区
    WorkBuffer w;
    CUDA_CHECK(cudaStreamCreate(&w.stream));
    CUDA_CHECK(cudaMalloc(&w.cap,    rg.E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w.excess, rg.V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w.height, rg.V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&w.flag,   sizeof(int)));
    CUDA_CHECK(cudaMalloc(&w.fa,     rg.V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&w.fb,     rg.V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&w.fsz,    sizeof(int)));
    CUDA_CHECK(cudaMalloc(&w.max_bits,     sizeof(unsigned)));

    // 分配主机侧 pinned memory, 用于和 GPU 异步交换少量控制信息
    PinnedBuffer p;
    CUDA_CHECK(cudaMallocHost(&p.flag, sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&p.fsz,  sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&p.max_bits,   sizeof(unsigned)));

    // 读取所有查询对
    std::vector<std::pair<int,int>> queries;
    if (!load_query_file(argv[2], g.V, queries)) return 1;
    if (queries.empty()) { std::printf("No valid queries.\n"); return 0; }

    // 逐个计算最大流, 并将结果写入输出文件
    FILE* of = fopen(argv[3], "w");
    if (!of) { fprintf(stderr, "Cannot open output file\n"); return 1; }

    auto T0 = std::chrono::high_resolution_clock::now();
    long long total_iters = 0;

    for (size_t i = 0; i < queries.size(); ++i) {
        int it = 0;
        auto t0 = std::chrono::high_resolution_clock::now();

        float flow = solve(dg, w, p, queries[i].first, queries[i].second, &it);

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_iters += it;

        fprintf(of, "%d %d %.6f\n", queries[i].first, queries[i].second, flow);
        printf("  [%zu] %d->%d  flow=%.4f  iters=%d  %.2fms\n",
               i, queries[i].first, queries[i].second, flow, it, ms);
    }
    fclose(of);

    auto T1 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(T1 - T0).count();

    printf("\n========== Performance ==========\n");
    printf("  Query Count  : %zu\n",  queries.size());
    printf("  Total Time   : %.3f ms\n", total_ms);
    printf("  Average Time : %.3f ms\n", queries.empty() ? 0.0 : total_ms / queries.size());
    printf("  Iter Count   : %lld\n", total_iters);

    // 释放 GPU 资源
    cudaFree(w.cap);
    cudaFree(w.excess);
    cudaFree(w.height);
    cudaFree(w.flag);
    cudaFree(w.fa);
    cudaFree(w.fb);
    cudaFree(w.fsz);
    cudaFree(w.max_bits);

    cudaFreeHost(p.flag);
    cudaFreeHost(p.fsz);
    cudaFreeHost(p.max_bits);

    cudaStreamDestroy(w.stream);

    cudaFree(dg.d_row);
    cudaFree(dg.d_col);
    cudaFree(dg.d_rev);
    cudaFree(dg.d_cap);

    return 0;
}