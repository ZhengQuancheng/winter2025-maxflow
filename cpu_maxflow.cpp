// Usage: ./cpu_maxflow_csr <graph.bin> <queries.txt> <output.txt>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <cstdint>
#include <cmath>
#include <vector>
#include <chrono>
#include <limits>
#include <cassert>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>

// CSR 格式的图结构
struct GraphCSR {
    int N = 0;                  // 节点数
    int M = 0;                  // 有向边数
    // row_ptr[u] 是节点 u 的出边 e 在 col[] 和 cap[] 中的起始索引, row_ptr[u+1] 是下一个节点的起始索引.
    // 对于节点 u, 其所有出边信息存储在 col 和 cap 数组的索引区间 [row_ptr[u], row_ptr[u+1]) 中.
    // row_ptr[u+1] - row_ptr[u] 是节点 u 的出边数 (出度).
    std::vector<int>   row_ptr; // 长度为 N+1
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
    bool ok = read_file(&g.N, sizeof(int)) && read_file(&g.M, sizeof(int));
    if (!ok || g.N <= 0 || g.M < 0) {
        std::fprintf(stderr, "Invalid header: N=%d M=%d\n", g.N, g.M);
        std::fclose(fp); return false;
    }

    // 读取数组数据到相应的 vector 中
    g.row_ptr.resize(g.N + 1);
    g.col.resize(g.M);
    g.cap.resize(g.M);
    ok =  read_file(g.row_ptr.data(), (g.N + 1) * sizeof(int))
       && read_file(g.col.data(), g.M * sizeof(int))
       && read_file(g.cap.data(), g.M * sizeof(float));

    // 读取文件完成后关闭文件
    std::fclose(fp);

    if (!ok) { std::fprintf(stderr, "Truncated file '%s'\n", path); return false; }

    // 校验 CSR 格式数据的合法性
    if (g.row_ptr[0] != 0 || g.row_ptr[g.N] != g.M) {
        std::fprintf(stderr, "Corrupt CSR boundaries\n"); return false;
    }
    for (int u = 0; u < g.N; ++u) {
        if (g.row_ptr[u] > g.row_ptr[u + 1]) {
            std::fprintf(stderr, "CSR not monotone at u=%d\n", u); return false;
        }
    }
    for (int i = 0; i < g.M; ++i) {
        if (g.col[i] < 0 || g.col[i] >= g.N) {
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
 * @brief 从文本文件加载查询对 (s,t) 到 qs 向量中
 * @param path 文本文件路径
 * @param N 图中节点数, 用于验证查询的合法性
 * @param qs 输出参数, 成功时包含加载的查询对
 * @return 成功返回 true, 失败返回 false
 */
bool load_query_file(const char* path, int N, std::vector<std::pair<int,int>>& qs) {
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
        if (s < 0 || s >= N || t < 0 || t >= N) {
            std::fprintf(stderr, "Query (%lld,%lld) out of range [0,%d)\n", s, t, N); continue;
        }
        qs.emplace_back((int)s, (int)t);
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
    r.N = g.N;

    // 边数翻倍
    int64_t m2 = (int64_t)g.M * 2;
    if (m2 > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "2*M overflows int32\n"); return false;
    }
    r.M = (int)m2;

    r.row_ptr.assign(r.N + 1, 0);
    r.col.resize(r.M);
    r.cap.resize(r.M);
    r.rev.resize(r.M);

    // 统计每个节点的出度
    for (int u = 0; u < g.N; ++u) {
        for (int e = g.row_ptr[u]; e < g.row_ptr[u + 1]; ++e) {
            int v = g.col[e];
            r.row_ptr[u + 1]++; // 正向出边
            r.row_ptr[v + 1]++; // 反向出边
        }
    }
    // 此时 row_ptr[u+1] 存储节点 u 在残量图中的出边数(出度).

    // 计算前缀和
    for (int i = 1; i <= r.N; ++i) {
        r.row_ptr[i] += r.row_ptr[i - 1];
    }
    // 此时 row_ptr[u] 存储节点 u 在残量图中出边的起始位置, row_ptr[u+1]-row_ptr[u] 是出边数.

    // 验证 row_ptr 的正确性
    assert(r.row_ptr[r.N] == r.M);

    // cur[u] 是节点 u 下一条待写入边的位置, 每次往节点 u 写入一条边后, cur[u]++ 会向后推进.
    std::vector<int> cur(r.row_ptr.begin(), r.row_ptr.end() - 1);

    for (int u = 0; u < g.N; ++u)
        for (int e = g.row_ptr[u]; e < g.row_ptr[u + 1]; ++e) {
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

// Dinic on CSR
class DinicCSR {
public:
    explicit DinicCSR(const GraphCSR& g)
        : g_(g), level_(g.N), it_(g.N), cap_(g.M), bfs_q_(g.N), dfs_s_(g.N) {}

    double max_flow(int s, int t) {
        if (s == t || s < 0 || s >= g_.N || t < 0 || t >= g_.N) return 0.0;

        std::copy(g_.cap.begin(), g_.cap.end(), cap_.begin());

        double flow = 0.0;
        while (bfs(s, t)) {
            for (int v = 0; v < g_.N; ++v) {
                it_[v] = g_.row_ptr[v];
            }
            for (double d; (d = dfs(s, t)) > 0.0; ) {
                flow += d;
            }
        }
        return flow;
    }

private:
    static constexpr double EPS = 1e-12;

    struct Frame {
        int v;              // 当前节点
        int in_edge;        // 到达当前节点的边索引
        double bottleneck;  // 从源点到当前节点的路径上的最小残量(剩余容量)
    };

    const GraphCSR&     g_;     // CRS 格式的残量图
    std::vector<int>    level_; // 每个节点的层次
    std::vector<int>    it_;    // 当前弧, 即有向边
    std::vector<double> cap_;   // 当前残量图中边的容量
    std::vector<int>    bfs_q_; // BFS 队列
    std::vector<Frame>  dfs_s_; // DFS 栈

    // BFS 构建层次图, 返回汇点 t 是否可达
    bool bfs(int s, int t) {
        std::fill(level_.begin(), level_.end(), -1);
        level_[s] = 0;          // 源点 s 的 level 为 0
        int head = 0, tail = 0; // 手动管理队列
        bfs_q_[tail++] = s;     // 入队源点 s

        // 开始 BFS 遍历
        while (head < tail) {
            int u = bfs_q_[head++]; // 节点 u 出队
            for (int e = g_.row_ptr[u], end = g_.row_ptr[u + 1]; e < end; ++e) {
                int v = g_.col[e]; // u -> v
                // u-> v 的边 e 的残量 cap_[e] > 0 且 v 未被访问过 (level_[v] < 0)
                if (cap_[e] > EPS && level_[v] < 0) {
                    level_[v] = level_[u] + 1; // 记录 v 的 level
                    bfs_q_[tail++] = v; // v 入队
                }
            }
        }
        // 汇点 t 的 level >= 0 表示在层次图中可达
        return level_[t] >= 0;
    }

    // 迭代 DFS 寻找增广路径并推送流量, 返回推送的流量值
    double dfs(int s, int t) {
        constexpr double INF = std::numeric_limits<double>::infinity();
        int top = 0;
        dfs_s_[0] = {s, -1, INF}; // 初始化 DFS 栈顶为源点 s, 没有入边, bottleneck 是无穷大

        while (top >= 0) {
            int v = dfs_s_[top].v; // 当前 DFS 栈顶节点 v
            // 如果 v 是汇点 t, 则找到了一个增广路径,
            if (v == t) {
                // 计算该路径的 bottleneck (最小残量)
                double d = dfs_s_[top].bottleneck;
                if (!(d > 0.0)) return 0.0;
                // 沿着栈回溯, 更新残量
                for (int k = top; k >= 1; --k) {
                    int e  = dfs_s_[k].in_edge;
                    int re = g_.rev[e];
                    cap_[e]  -= d;
                    cap_[re] += d;
                    if (cap_[e] < 0.0) cap_[e] = 0.0;   // 防止浮点误差变成负数
                }
                return d;
            }

            // 尝试沿着当前节点 v 的出边继续 DFS
            int& ecur = it_[v];
            const int end = g_.row_ptr[v + 1];
            bool advanced = false;

            // 从当前弧 ecur 开始尝试
            for (; ecur < end; ++ecur) {
                int to = g_.col[ecur];
                // 边 v->to 的残量 cap_[ecur] > 0 且 to 在层次图中是 v 的下一层 (level_[to] == level_[v] + 1)
                if (cap_[ecur] > EPS && level_[to] == level_[v] + 1) {
                    double bn = std::min(dfs_s_[top].bottleneck, cap_[ecur]);
                    dfs_s_[++top] = {to, ecur, bn}; // to 入栈, 记录入边 ecur 和更新后的 bottleneck
                    advanced = true;
                    break;
                }
            }
            // 如果没有找到可行的出边, 则回退
            if (!advanced) {
                // 若当前节点所有出边都无法到达终点, 则将其 level 置为 -1
                // 后续其他分支则不会再访问这个节点
                level_[v] = -1;
                // 弹栈
                if (--top >= 0) {
                    // Advance parent's current-arc past the failed edge
                    it_[dfs_s_[top].v] = dfs_s_[top + 1].in_edge + 1;
                }
            }
        }
        return 0.0; // 找不到增广路径
    }
};

// Edmonds-Karp on CSR
class EdmondsKarpCSR {
public:
    explicit EdmondsKarpCSR(const GraphCSR& g)
        : g_(g), cap_(g.M), parent_edge_(g.N), bfs_q_(g.N), bottleneck_(g.N) {}

    double max_flow(int s, int t) {
        if (s == t || s < 0 || s >= g_.N || t < 0 || t >= g_.N) return 0.0;

        std::copy(g_.cap.begin(), g_.cap.end(), cap_.begin());

        double flow = 0.0;

        while (true) {
            // 使用 BFS 寻找一条最短增广路
            double pushed = bfs(s, t);

            // 若找不到增广路径算法结束
            if (!(pushed > 0.0)) break;

            flow += pushed;

            // 沿着 BFS 记录的边路径回溯, 更新残量网络
            int curr = t;
            while (curr != s) {
                int e = parent_edge_[curr]; // 获取流入 curr 的边索引
                int re = g_.rev[e];         // 获取对应的反向边索引

                cap_[e] -= pushed;
                if (cap_[e] < 0.0) cap_[e] = 0.0;  // 防止浮点误差变成负数
                cap_[re] += pushed;

                // 反向边的目标节点, 就是正向边的源节点 (即前驱节点)
                curr = g_.col[re];
            }
        }
        return flow;
    }

private:
    static constexpr double EPS = 1e-12;

    const GraphCSR&     g_;             // 残量图拓扑
    std::vector<double> cap_;           // 动态残余容量
    std::vector<int>    parent_edge_;   // 记录到达每个节点所经过的边索引
    std::vector<int>    bfs_q_;         // BFS 队列
    std::vector<double> bottleneck_;    // 记录从源点到达该节点的路径瓶颈流量

    // BFS 寻找最短增广路, 返回该路径的瓶颈流量
    double bfs(int s, int t) {
        // -1 表示未访问; 同时重置 bottleneck_ 防止读到上一轮脏数据
        std::fill(parent_edge_.begin(), parent_edge_.end(), -1);
        std::fill(bottleneck_.begin(), bottleneck_.end(), 0.0);

        int head = 0, tail = 0;
        bfs_q_[tail++] = s;
        parent_edge_[s] = -2; // 标记源点已访问
        bottleneck_[s] = std::numeric_limits<double>::infinity();

        while (head < tail) {
            int u = bfs_q_[head++];

            // 一旦 BFS 触达汇点 t, 立刻停止搜索
            // BFS 保证是跳数最少的增广路
            if (u == t) return bottleneck_[t];

            // 遍历节点 u 的所有出边
            for (int e = g_.row_ptr[u], end = g_.row_ptr[u + 1]; e < end; ++e) {
                int v = g_.col[e];

                // 如果边有残余容量, 且目标节点 v 未被访问过
                if (cap_[e] > EPS && parent_edge_[v] == -1) {
                    parent_edge_[v] = e; // 记录是通过哪条边过来的
                    bottleneck_[v] = std::min(bottleneck_[u], cap_[e]); // 更新瓶颈
                    bfs_q_[tail++] = v;  // 入队
                }
            }
        }
        return 0.0;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::fprintf(stderr, "Usage: %s <graph.bin> <queries.txt> <output.txt>\n", argv[0]);
        return 1;
    }

    GraphCSR g;
    if (!load_graph_file(argv[1], g)) return 1;
    std::printf("Graph: N=%d, M=%d\n", g.N, g.M);

    auto t0 = std::chrono::high_resolution_clock::now();
    GraphCSR rg;
    if (!build_residual_graph(g, rg)) return 1;
    auto t1 = std::chrono::high_resolution_clock::now();
    std::printf("Residual CSR: M_expanded=%d  (%.3f ms)\n", rg.M,
                std::chrono::duration<double, std::milli>(t1 - t0).count());

    std::vector<std::pair<int,int>> queries;
    if (!load_query_file(argv[2], g.N, queries)) return 1;
    if (queries.empty()) { std::printf("No valid queries.\n"); return 0; }

    FILE* ofp = std::fopen(argv[3], "w");
    if (!ofp) { std::fprintf(stderr, "Cannot open '%s': %s\n", argv[3], strerror(errno)); return 1; }

    DinicCSR solver1(rg);
    EdmondsKarpCSR solver2(rg);
    int Q = (int)queries.size();
    std::printf("Processing %d queries...\n", Q);
    double total_time_1 = 0.0;
    double total_time_2 = 0.0;
    for (int i = 0; i < Q; ++i) {
        auto [s, t] = queries[i];

        auto qs1 = std::chrono::high_resolution_clock::now();
        double flow1 = solver1.max_flow(s, t);
        auto qe1 = std::chrono::high_resolution_clock::now();
        double ms1 = std::chrono::duration<double, std::milli>(qe1 - qs1).count();
        total_time_1 += ms1;

        auto qs2 = std::chrono::high_resolution_clock::now();
        double flow2 = solver2.max_flow(s, t);
        auto qe2 = std::chrono::high_resolution_clock::now();
        double ms2 = std::chrono::duration<double, std::milli>(qe2 - qs2).count();
        total_time_2 += ms2;

        std::fprintf(ofp, "%d %d %.9f\n", s, t, flow1);
        std::printf("DC [%d] (%d -> %d)  flow=%.6f  %.3f ms\n", i, s, t, flow1, ms1);
        std::printf("EK [%d] (%d -> %d)  flow=%.6f  %.3f ms\n", i, s, t, flow2, ms2);
        if (std::abs(flow1 - flow2) > 0) {
            std::printf("  [!] Mismatch: Dinic=%.6f vs EdmondsKarp=%.6f\n", flow1, flow2);
        }
    }

    std::printf("Total time - Dinic: %.3f ms\n", total_time_1);
    std::printf("Total time - EdmondsKarp: %.3f ms\n", total_time_2);

    std::printf("Average time - Dinic: %.3f ms\n", total_time_1 / Q);
    std::printf("Average time - EdmondsKarp: %.3f ms\n", total_time_2 / Q);

    std::fclose(ofp);

    return 0;
}
