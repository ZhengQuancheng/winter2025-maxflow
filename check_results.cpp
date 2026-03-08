// Usage: ./check_results <cpu_output.txt> <gpu_output.txt>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <limits>

struct QueryResult {
    int s, t;
    float flow;
};

// 加载结果
static std::vector<QueryResult> load_results(const char* path) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Cannot open %s\n", path);
        exit(1);
    }

    std::vector<QueryResult> res;
    int s, t;
    float f;

    while (fscanf(fp, "%d %d %f", &s, &t, &f) == 3) {
        res.push_back({s, t, f});
    }

    fclose(fp);
    return res;
}

// 打印分隔线
static void print_separator(char c = '-', int n = 60) {
    for (int i = 0; i < n; i++) putchar(c);
    putchar('\n');
}

// 计算中位数
static float compute_median(const std::vector<float>& sorted_vals) {
    int n = (int)sorted_vals.size();
    if (n == 0) return 0.0f;
    if (n % 2 == 1) return sorted_vals[n / 2];
    return 0.5f * (sorted_vals[n / 2 - 1] + sorted_vals[n / 2]);
}

// 计算分位数
static float compute_percentile(const std::vector<float>& sorted_vals, float p) {
    int n = (int)sorted_vals.size();
    if (n == 0) return 0.0f;
    if (p <= 0.0f) return sorted_vals.front();
    if (p >= 1.0f) return sorted_vals.back();

    int idx = (int)std::ceil(n * p) - 1;
    if (idx < 0) idx = 0;
    if (idx >= n) idx = n - 1;
    return sorted_vals[idx];
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <cpu_output.txt> <gpu_output.txt>\n", argv[0]);
        return 1;
    }

    auto cpu = load_results(argv[1]);
    auto gpu = load_results(argv[2]);

    if (cpu.size() != gpu.size()) {
        fprintf(stderr, "Query count mismatch: cpu=%zu gpu=%zu\n", cpu.size(), gpu.size());
        return 1;
    }

    int Q = (int)cpu.size();
    if (Q == 0) {
        fprintf(stderr, "No valid query results loaded.\n");
        return 1;
    }

    // 校验 query 是否逐项对应
    for (int i = 0; i < Q; i++) {
        if (cpu[i].s != gpu[i].s || cpu[i].t != gpu[i].t) {
            fprintf(stderr,
                    "Query mismatch at index %d: cpu=(%d,%d), gpu=(%d,%d)\n",
                    i, cpu[i].s, cpu[i].t, gpu[i].s, gpu[i].t);
            return 1;
        }
    }

    printf("\n");
    print_separator('=');
    printf("  误差分析报告: %s  vs  %s\n", argv[1], argv[2]);
    print_separator('=');
    printf("  查询总数: %d\n\n", Q);

    // ---- 阈值设置 ----
    // 相对误差阈值（百分比）
    const float tol_rel = 1.0f;   // 1%
    // 绝对误差阈值
    const float tol_abs = 1e-6f;

    // ---- 详细误差 ----
    printf("%-6s %-8s %-8s %-14s %-14s %-16s %-16s %-14s\n",
           "Query", "src", "dst", "CPU flow", "GPU flow",
           "AbsoluteError", "RelativeError(%)", "Status");
    print_separator('-', 96);

    std::vector<float> abs_errs(Q), rel_errs(Q);
    int pass = 0, fail = 0;

    for (int i = 0; i < Q; i++) {
        float ref = cpu[i].flow;
        float got = gpu[i].flow;
        float ae  = fabsf(ref - got);

        // 分母避免为 0, 同时保留相对误差概念
        float denom = std::max(fabsf(ref), 1e-6f);
        float re    = ae / denom * 100.0f;

        abs_errs[i] = ae;
        rel_errs[i] = re;

        // 双阈值判定: 绝对误差足够小 或 相对误差足够小
        bool ok = (ae <= tol_abs) || (re <= tol_rel);
        if (ok) pass++;
        else fail++;

        printf("%-6d %-8d %-8d %-14.6f %-14.6f %-16.6e %-16.6f %-14s\n",
               i, cpu[i].s, cpu[i].t, ref, got, ae, re,
               ok ? "PASS" : "FAIL !!!");
    }

    print_separator('-', 96);

    // ---- 统计汇总 ----
    printf("\n");
    print_separator('=');
    printf("  统计汇总\n");
    print_separator('=');

    // 绝对误差统计
    float ae_max  = *std::max_element(abs_errs.begin(), abs_errs.end());
    float ae_min  = *std::min_element(abs_errs.begin(), abs_errs.end());
    float ae_mean = std::accumulate(abs_errs.begin(), abs_errs.end(), 0.0f) / Q;

    std::vector<float> ae_sorted = abs_errs;
    std::sort(ae_sorted.begin(), ae_sorted.end());
    float ae_median = compute_median(ae_sorted);
    float ae_p95    = compute_percentile(ae_sorted, 0.95f);
    float ae_p99    = compute_percentile(ae_sorted, 0.99f);

    // 相对误差统计
    float re_max  = *std::max_element(rel_errs.begin(), rel_errs.end());
    float re_min  = *std::min_element(rel_errs.begin(), rel_errs.end());
    float re_mean = std::accumulate(rel_errs.begin(), rel_errs.end(), 0.0f) / Q;

    std::vector<float> re_sorted = rel_errs;
    std::sort(re_sorted.begin(), re_sorted.end());
    float re_median = compute_median(re_sorted);
    float re_p95    = compute_percentile(re_sorted, 0.95f);
    float re_p99    = compute_percentile(re_sorted, 0.99f);

    // 流量值统计（以 CPU 结果为参考）
    float flow_max  = cpu[0].flow;
    float flow_min  = cpu[0].flow;
    float flow_mean = 0.0f;
    for (const auto& r : cpu) {
        flow_max  = std::max(flow_max, r.flow);
        flow_min  = std::min(flow_min, r.flow);
        flow_mean += r.flow;
    }
    flow_mean /= Q;

    printf("\n[CPU 参考流量统计]\n");
    printf("  最小值   : %.6e\n", flow_min);
    printf("  最大值   : %.6e\n", flow_max);
    printf("  平均值   : %.6e\n", flow_mean);

    printf("\n[绝对误差 (Absolute Error)]\n");
    printf("  最小值   : %.6e\n", ae_min);
    printf("  最大值   : %.6e\n", ae_max);
    printf("  平均值   : %.6e\n", ae_mean);
    printf("  中位数   : %.6e\n", ae_median);
    printf("  P95      : %.6e\n", ae_p95);
    printf("  P99      : %.6e\n", ae_p99);

    printf("\n[相对误差 (Relative Error, %%)]\n");
    printf("  最小值   : %.6f%%\n", re_min);
    printf("  最大值   : %.6f%%\n", re_max);
    printf("  平均值   : %.6f%%\n", re_mean);
    printf("  中位数   : %.6f%%\n", re_median);
    printf("  P95      : %.6f%%\n", re_p95);
    printf("  P99      : %.6f%%\n", re_p99);

    // ---- 误差分布直方图 ----
    printf("\n[相对误差分布直方图]\n");
    const char* labels[] = {
        "[0,     1e-6%)",
        "[1e-6%, 1e-5%)",
        "[1e-5%, 1e-4%)",
        "[1e-4%, 1e-3%)",
        "[1e-3%, 1e-2%)",
        "[1e-2%, 0.1% )",
        "[0.1%,  1%   )",
        "[>=1%        )"
    };
    float bounds[] = {0.0f, 1e-6f, 1e-5f, 1e-4f, 1e-3f, 1e-2f, 0.1f, 1.0f, std::numeric_limits<float>::max()};
    int bins[8] = {};

    for (float re : rel_errs) {
        for (int b = 0; b < 8; b++) {
            if (re >= bounds[b] && re < bounds[b + 1]) {
                bins[b]++;
                break;
            }
        }
    }

    int bar_max = *std::max_element(bins, bins + 8);
    for (int b = 0; b < 8; b++) {
        int bar_len = (bar_max > 0) ? (bins[b] * 30 / bar_max) : 0;
        std::string bar(bar_len, '#');
        printf("  %s |%-30s %d\n", labels[b], bar.c_str(), bins[b]);
    }

    // ---- 正确性判定 ----
    printf("\n[正确性判定]\n");
    printf("  判定规则: AbsoluteError <= %.3e OR RelativeError <= %.3f%%\n", tol_abs, tol_rel);
    printf("  PASS: %d / %d\n", pass, Q);
    printf("  FAIL: %d / %d\n", fail, Q);

    print_separator('=');
    printf("  结论: %s\n",
           fail == 0 ? "全部通过"
                     : "存在超出容忍范围的误差");
    print_separator('=');
    printf("\n");

    return (fail > 0) ? 1 : 0;
}