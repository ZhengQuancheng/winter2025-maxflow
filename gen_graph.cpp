// Usage: ./gen_graph <N> <M> <seed> <output.bin> [query_file] [num_queries]

#include <cstdio>
#include <cerrno>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <limits>
#include <optional>
#include <algorithm>

// 解析整数参数
static std::optional<int> parse_int(const char* s) {
    if (!s || *s == '\0') return std::nullopt;
    char* end = nullptr;
    errno = 0;
    long long v = std::strtoll(s, &end, 10);
    if (errno != 0 || *end != '\0') return std::nullopt;
    if (v < std::numeric_limits<int>::min() ||
        v > std::numeric_limits<int>::max()) return std::nullopt;
    return static_cast<int>(v);
}

// 安全 fwrite wrapper
static bool safe_fwrite(const void* ptr, size_t size, size_t count, FILE* fp) {
    return std::fwrite(ptr, size, count, fp) == count;
}

// Edge 结构
struct Edge {
    int    u, v;
    double cap;
};

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::fprintf(stderr,
            "Usage: %s <N> <M> <seed> <output.bin> [query_file] [num_queries]\n"
            "  N           : number of nodes (>= 2)\n"
            "  M           : target edge count (>= N-1 recommended)\n"
            "  seed        : random seed\n"
            "  output.bin  : binary CSR output\n"
            "  query_file  : optional (default: queries.txt)\n"
            "  num_queries : optional (default: 10)\n",
            argv[0]);
        return 1;
    }

    // 参数解析
    auto opt_N   = parse_int(argv[1]);
    auto opt_M   = parse_int(argv[2]);
    auto opt_seed= parse_int(argv[3]);

    // 参数验证
    if (!opt_N)    { std::fprintf(stderr, "Error: invalid N '%s'\n",    argv[1]); return 1; }
    if (!opt_M)    { std::fprintf(stderr, "Error: invalid M '%s'\n",    argv[2]); return 1; }
    if (!opt_seed) { std::fprintf(stderr, "Error: invalid seed '%s'\n", argv[3]); return 1; }

    const int   N       = *opt_N;    // 节点数量
    const int   M       = *opt_M;    // 目标边数
    const int   seed    = *opt_seed; // 随机种子
    const char* outfile = argv[4];   // 输出图数据文件路径

    // 设置默认查询参数
    const char* qfile       = (argc >= 6) ? argv[5] : "queries.txt"; // 查询文件路径
    int         num_queries = 10;                                    // 默认查询次数

    // 解析可选的查询数量参数
    if (argc >= 7) {
        auto opt_q = parse_int(argv[6]);
        if (!opt_q || *opt_q < 1) {
            std::fprintf(stderr, "Error: invalid num_queries '%s'\n", argv[6]);
            return 1;
        }
        num_queries = *opt_q;
    }

    // 校验基本参数范围
    if (N < 2) { std::fprintf(stderr, "Error: N must be >= 2\n"); return 1; }
    if (M < 0) { std::fprintf(stderr, "Error: M must be >= 0\n"); return 1; }

    // 警告: 当 M < N-1 时, 生成的图将不连通
    if (M < static_cast<long long>(N - 1)) {
        std::fprintf(stderr,
            "Warning: M=%d < N-1=%d; actual edge count will be %d (chain only).\n",
            M, N - 1, N - 1);
    }

    // 初始化随机数生成器
    std::mt19937 rng(static_cast<unsigned>(seed));
    std::uniform_int_distribution<int>    node_dist(0, N - 1);
    std::uniform_real_distribution<double> cap_dist(1.0, 100.0);

    // Hamiltonian path: 0 → (shuffled interior) → N-1
    std::vector<int> path_nodes(N);
    path_nodes[0]     = 0;
    path_nodes[N - 1] = N - 1;
    {
        std::vector<int> interior;
        interior.reserve(N - 2);
        for (int i = 1; i < N - 1; ++i) interior.push_back(i);
        // 随机打乱内部节点顺序
        std::shuffle(interior.begin(), interior.end(), rng);
        // 将打乱后的节点插入路径中
        for (int i = 0; i < static_cast<int>(interior.size()); ++i)
            path_nodes[i + 1] = interior[i];
    }

    // 理论最大边数
    const long long max_directed = static_cast<long long>(N) * (N - 1);
    // 基本链路边数
    const long long chain_edges  = N - 1;
    // 需额外添加的边数 (受内存限制和理论上限双重约束)
    long long extra = std::max(0LL, M - chain_edges);

    // 限制总边数（链路边 + 额外边）不超过约 2600 万，以控制内存使用在 ~512 MB 范围内
    // Each Edge = ~20 bytes; 512MB / 20 ≈ 26M edges total
    constexpr long long MAX_EDGES = 26'000'000LL;
    const long long max_extra = std::max(0LL, MAX_EDGES - chain_edges);
    // 边数超过理论最大值时, 进行合理的截断并发出警告
    if (extra > max_directed - chain_edges) {
        extra = max_directed - chain_edges;   // can't exceed complete graph
        std::fprintf(stderr,
            "Warning: M exceeds complete-graph edge count; capped at %lld.\n",
            chain_edges + extra);
    }
    // 边数超过内存限制时, 进行合理的截断并发出警告
    if (extra > max_extra) {
        std::fprintf(stderr,
            "Warning: extra edges capped at %lld to stay within ~512 MB memory budget.\n",
            max_extra);
        extra = max_extra;
    }

    std::vector<Edge> all_edges;
    all_edges.reserve(static_cast<size_t>(chain_edges + extra));

    // 生成基础链路
    for (int i = 0; i < N - 1; ++i)
        all_edges.push_back({path_nodes[i], path_nodes[i + 1], cap_dist(rng)});

    // 生成随机额外边
    for (long long i = 0; i < extra; ++i) {
        int u = node_dist(rng); // 随机起点
        int v = node_dist(rng); // 随机终点
        // 避免生成自环边 (u == v), 尝试重新生成终点, 最多尝试 32 次
        for (int attempt = 0; u == v && attempt < 32; ++attempt)
            v = node_dist(rng);
        // 如果多次尝试后仍然生成了自环边, 则放弃该边 (不添加到图中)
        if (u == v) continue;
        // 添加边
        all_edges.push_back({u, v, cap_dist(rng)});
    }

    // 按 (u, v) 排序便于后续处理
    std::sort(all_edges.begin(), all_edges.end(), [](const Edge& a, const Edge& b) {
        return a.u != b.u ? a.u < b.u : a.v < b.v;
    });

    // 合并平行边
    std::vector<Edge> merged;
    merged.reserve(all_edges.size());
    if (!all_edges.empty()) {
        Edge cur = all_edges[0];
        for (size_t i = 1; i < all_edges.size(); ++i) {
            const Edge& e = all_edges[i];
            if (e.u == cur.u && e.v == cur.v) {
                cur.cap += e.cap;   // 平行边容量累加
            } else {
                merged.push_back(cur);
                cur = e;
            }
        }
        merged.push_back(cur);
    }
    // 释放内存
    all_edges.clear();
    all_edges.shrink_to_fit(); 

    // 检查合并后的边数是否超过 CSR 格式的限制
    const long long actual_M_ll = static_cast<long long>(merged.size());
    if (actual_M_ll > static_cast<long long>(std::numeric_limits<int>::max()) ||
        static_cast<long long>(N) > static_cast<long long>(std::numeric_limits<int>::max())) {
        std::fprintf(stderr,
            "Error: N=%d or M=%lld exceeds 32-bit CSR format limit.\n", N, actual_M_ll);
        return 1;
    }
    const int actual_M = static_cast<int>(actual_M_ll);

    // 构建 CSR 数据结构
    std::vector<int>   row_ptr(N + 1, 0);
    std::vector<int>   col_indices(actual_M);
    std::vector<float> values(actual_M);   // store as float for binary compat

    // 统计每个节点的出边数
    for (const auto& e : merged) row_ptr[e.u + 1]++;
    // 计算边索引的前缀和
    for (int i = 1; i <= N; ++i) row_ptr[i] += row_ptr[i - 1];
    // 填充 col_indices 和 values 数组
    for (int i = 0; i < actual_M; ++i) {
        col_indices[i] = merged[i].v;
        values[i]      = static_cast<float>(merged[i].cap);
    }

    // 写入 CSR 二进制文件
    FILE* fp = std::fopen(outfile, "wb");
    if (!fp) {
        std::fprintf(stderr, "Error: cannot open '%s': %s\n", outfile, std::strerror(errno));
        return 1;
    }
    bool ok = true;
    ok &= safe_fwrite(&N,                sizeof(int),   1,          fp);
    ok &= safe_fwrite(&actual_M,         sizeof(int),   1,          fp);
    ok &= safe_fwrite(row_ptr.data(),    sizeof(int),   N + 1,      fp);
    ok &= safe_fwrite(col_indices.data(),sizeof(int),   actual_M,   fp);
    ok &= safe_fwrite(values.data(),     sizeof(float), actual_M,   fp);
    if (std::fclose(fp) != 0) ok = false;
    if (!ok) {
        std::fprintf(stderr, "Error: write failed for '%s': %s\n", outfile, std::strerror(errno));
        return 1;
    }

    std::printf("Graph written: N=%d, actual_M=%d (merged, sorted) -> %s\n",
                N, actual_M, outfile);

    FILE* qfp = std::fopen(qfile, "w");
    if (!qfp) {
        std::fprintf(stderr, "Error: cannot open '%s': %s\n", qfile, std::strerror(errno));
        return 1;
    }

    std::uniform_int_distribution<int> pos_dist(0, N - 1);
    int written = 0;

    // 生成 num_queries 条查询，随机选择 Hamiltonian path 上的节点对 (s, t)，保证 s < t 以确保 s->t 可达
    for (int i = 0; i < num_queries; ++i) {
        int pos_s = -1, pos_t = -1;
        bool found = false;
        // 尝试随机选择 s 和 t，最多尝试 1024 次以避免死循环
        for (int attempt = 0; attempt < 1024; ++attempt) {
            pos_s = pos_dist(rng);
            pos_t = pos_dist(rng);
            if (pos_s < pos_t) { found = true; break; }
        }
        // 如果多次尝试后仍未找到合法的 (s, t) 对，则使用路径上的第一个和最后一个节点作为 s 和 t
        if (!found) {
            pos_s = 0;
            pos_t = N - 1;
        }
        if (std::fprintf(qfp, "%d %d\n", path_nodes[pos_s], path_nodes[pos_t]) < 0) {
            std::fprintf(stderr, "Error: write failed for '%s'\n", qfile);
            std::fclose(qfp);
            return 1;
        }
        ++written;
    }

    if (std::fclose(qfp) != 0) {
        std::fprintf(stderr, "Error: close failed for '%s': %s\n", qfile, std::strerror(errno));
        return 1;
    }

    std::printf("Queries written: %d pairs (all guaranteed s->t reachable) -> %s\n",
                written, qfile);
    return 0;
}
