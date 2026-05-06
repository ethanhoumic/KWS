#include <bits/stdc++.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
using namespace std;

#define C_OUT      16
#define H_OUT      26      // conv1 output: (101 - 67 + 1) = 35
#define W_OUT      30      // conv1 output: (40  -  8 + 1) = 33
#define N_SPATIAL  H_OUT * W_OUT   // 1155 per channel

// ── 載入工具 ────────────────────────────────────────────────────────────────

void load_1d_int32(const char *path, int32_t *buf, int n) {
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    for (int i = 0; i < n; i++) fscanf(fp, "%d", &buf[i]);
    fclose(fp);
}

void load_1d_int64(const char *path, int64_t *buf, int n) {
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    for (int i = 0; i < n; i++) fscanf(fp, "%lld", &buf[i]);
    fclose(fp);
}

void load_1d_uint32(const char *path, uint32_t *buf, int n) {
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    for (int i = 0; i < n; i++) fscanf(fp, "%u", &buf[i]);
    fclose(fp);
}

// ── Requantization（對應 verify_param 的公式）───────────────────────────────
//
// verify_param:
//   q3 = (((acc * M0) / 2^n) + K * 2^32) / 2^32
//      = (acc * M0 + K * 2^(n+32)) >> (n+32)   [整數版本，含 rounding]
//
// 步驟：
//   1. mult64    = acc * M0                         (int64)
//   2. shifted   = mult64 >> n                      (已做 M0 乘法的 shift)
//   3. with_K    = shifted + K * 2^32              (加 K，K 已 scale 到 2^32)
//   4. result    = with_K >> 32  +  rounding        (最後右移 32，含 round)
//   rounding：看 bit[31] of with_K，若為 1 則 +1

uint8_t requantize(int32_t acc, uint32_t M0, int64_t n, int32_t K, uint8_t zp_out, bool do_relu) {

    // step 1: acc * M0
    int64_t mult64 = (int64_t)acc * (int64_t)M0;

    // step 2: >> n  (M0 已正規化到 [2^31, 2^32)，所以 n 是使 M*2^n 落在此範圍的 shift)
    int64_t shifted = mult64 >> n;

    // step 3: 加 K（K 對應 2^32 scale，等效於 K << 32）
    int64_t with_K = shifted + ((int64_t)K << 32);

    // step 4: rounding（看 bit[31]）+ >> 32
    int64_t rounding = (with_K >> 31) & 1;
    int64_t result   = (with_K >> 32) + rounding;

    // clamp to [0, 255]
    if (result > 255) result = 255;
    if (result < 0)   result = 0;

    // ReLU：clamp min = zp_out
    uint8_t out = (uint8_t)result;
    if (out < zp_out && do_relu) out = zp_out;
    else if (out < 0) out = 0;

    return out;
}

int8_t requantize_sym(int32_t acc, uint32_t M0, int64_t n, int32_t K, uint8_t zp_out) {

    // step 1: acc * M0
    int64_t mult64 = (int64_t)acc * (int64_t)M0;

    // step 2: >> n  (M0 已正規化到 [2^31, 2^32)，所以 n 是使 M*2^n 落在此範圍的 shift)
    int64_t shifted = mult64 >> n;

    // step 3: 加 K（K 對應 2^32 scale，等效於 K << 32）
    int64_t with_K = shifted + ((int64_t)K << 32);

    // step 4: rounding（看 bit[31]）+ >> 32
    int64_t rounding = (with_K >> 31) & 1;
    int64_t result   = (with_K >> 32) + rounding;

    // clamp to [-128, 127]
    if (result > 127) result = 127;
    if (result < -128) result = -128;

    // // ReLU：clamp min = zp_out
    // uint8_t out = (uint8_t)result;
    // if (out < zp_out && do_relu) out = zp_out;
    // else if (out < 0) out = 0;

    return result;
}

// ── Piecewise Linear Sigmoid ─────────────────────────────────────────────────
//
// 輸入：q3（int8，對稱量化，Z3=0）
// 輸出：q4（uint8，非對稱量化）
//
// 分段（依 |q3| 大小）：
//   |q3| >= T1        → q4_pos = B_seg0             （飽和，Y=1）
//   T2 <= |q3| < T1   → q4_pos = (M0_A_seg1 * |q3|) >> n_A_seg1 + B_seg1
//   T3 <= |q3| < T2   → q4_pos = (M0_A_seg2 * |q3|) >> n_A_seg2 + B_seg2
//   |q3| < T3         → q4_pos = (M0_A_seg3 * |q3|) >> n_A_seg3 + B_seg3
//
// 負數對稱：q3 < 0 → q4 = C - q4_pos
//
// 注意：A = alpha * S3 / S4 可能 > 1，所以 M0_A 乘以 |q3|（uint8）後
//       必須右移 n_A bits，和 requantize 的 M0 處理完全一致。

typedef struct {
    int32_t  T1, T2, T3;       // 門檻值（比較 |q3| 用）
    uint32_t  M0_A[4];          // 各段斜率的 fixed-point 表示
    int64_t  n_A[4];           // 各段對應的 shift 量
    int32_t  B[4];             // 各段截距（整數）
    int32_t  C;                // 負數對稱常數：q4_neg = C - q4_pos
} SigmoidParams;

uint8_t sigmoid_pwl(int8_t q3, const SigmoidParams *p) {

    int8_t abs_q3 = (q3 < 0) ? -(int8_t)q3 : (int8_t)q3;
    // cout << "q3=" << (int)q3 << " abs_q3=" << (int)abs_q3 << endl;

    // ── 查找對應段，計算 q4_pos ──────────────────────────────────────────────
    int64_t q4_pos;

    if (abs_q3 >= p->T1) {
        // 飽和段（alpha=0）：q4 = B_seg0
        q4_pos = p->B[0];

    } else if (abs_q3 >= p->T2) {
        // seg1：q4 = (M0_A * |q3|) >> n_A + B
        // M0_A 可能 > 1，所以 shift 後才是真正的斜率乘積
        int64_t prod = (int64_t)p->M0_A[1] * (int64_t)abs_q3;
        int64_t scaled = prod << p->n_A[1];
        int64_t biased = scaled + ((int64_t)p->B[1] << 32);
        int64_t rounding = (biased >> 31) & 1;
        q4_pos = (biased >> 32) + rounding;

    } else if (abs_q3 >= p->T3) {
        // seg2
        int64_t prod = (int64_t)p->M0_A[2] * (int64_t)abs_q3;
        int64_t scaled = prod << p->n_A[2];
        int64_t biased = scaled + ((int64_t)p->B[2] << 32);
        int64_t rounding = (biased >> 31) & 1;
        q4_pos = (biased >> 32) + rounding;

    } else {
        // seg3：|q3| < T3
        int64_t prod = (int64_t)p->M0_A[3] * (int64_t)abs_q3;
        int64_t scaled = prod << p->n_A[3];
        int64_t biased = scaled + ((int64_t)p->B[3] << 32);
        int64_t rounding = (biased >> 31) & 1;
        q4_pos = (biased >> 32) + rounding;
    }

    // ── 負數對稱：q3 < 0 → q4 = C - q4_pos ─────────────────────────────────
    int64_t q4 = (q3 < 0) ? (p->C - q4_pos) : q4_pos;

    // ── clamp to [0, 255] ────────────────────────────────────────────────────
    if (q4 > 255) q4 = 255;
    if (q4 < 0)   q4 = 0;

    return (uint8_t)q4;
}

// ── Sigmoid 參數載入工具 ─────────────────────────────────────────────────────

SigmoidParams load_sigmoid_params(const char *prefix) {
    // prefix 例如 "./sigmoid_params/conv1_sigmoid"
    // 對應檔案：{prefix}_T1.txt, {prefix}_seg0_M0_A.txt, ... 等
    SigmoidParams p;
    char path[512];
    FILE *fp;

    auto load_ui32 = [&](const char *suffix) -> uint32_t {
        snprintf(path, sizeof(path), "%s%s", prefix, suffix);
        fp = fopen(path, "r");
        if (!fp) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
        uint32_t v; fscanf(fp, "%u", &v); fclose(fp);
        return v;
    };

    auto load_i32 = [&](const char *suffix) -> int32_t {
        snprintf(path, sizeof(path), "%s%s", prefix, suffix);
        fp = fopen(path, "r");
        if (!fp) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
        int32_t v; fscanf(fp, "%d", &v); fclose(fp);
        return v;
    };
    auto load_i64 = [&](const char *suffix) -> int64_t {
        snprintf(path, sizeof(path), "%s%s", prefix, suffix);
        fp = fopen(path, "r");
        if (!fp) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
        int64_t v; fscanf(fp, "%lld", &v); fclose(fp);
        return v;
    };

    p.T1 = load_i32("_T1.txt");
    p.T2 = load_i32("_T2.txt");
    p.T3 = load_i32("_T3.txt");
    p.C  = load_i32("_C.txt");

    const char *seg_names[4] = {"_seg0", "_seg1", "_seg2", "_seg3"};
    for (int i = 0; i < 4; i++) {
        char suf_m0[64], suf_n[64], suf_b[64];
        snprintf(suf_m0, sizeof(suf_m0), "%s_M0_A.txt", seg_names[i]);
        snprintf(suf_n,  sizeof(suf_n),  "%s_n_A.txt",  seg_names[i]);
        snprintf(suf_b,  sizeof(suf_b),  "%s_B.txt",    seg_names[i]);
        p.M0_A[i] = load_ui32(suf_m0);
        p.n_A[i]  = load_i64(suf_n);
        p.B[i]    = load_i32(suf_b);
    }

    printf("Loaded sigmoid params from %s\n", prefix);
    printf("  T1=%d, T2=%d, T3=%d, C=%d\n", p.T1, p.T2, p.T3, p.C);
    for (int i = 0; i < 4; i++)
        printf("  seg%d: M0_A=%d, n_A=%lld, B=%d\n",
               i, p.M0_A[i], (long long)p.n_A[i], p.B[i]);

    return p;
}

// ── 主程式 ──────────────────────────────────────────────────────────────────

int main() {

    // 路徑（依你的 outdir 調整）
    const char *acc_path = "./acc_preact_sigmoid/conv2_acc_sigmoid.txt";
    const char *M0_path  = "./layer_outputs_sigmoid/conv2_M0.txt";
    const char *n_path   = "./layer_outputs_sigmoid/conv2_n.txt";
    const char *K_path   = "./layer_outputs_sigmoid/conv2_K.txt";
    const char *out_path = "./sigmoid2_out_c_model_sigmoid.txt";

    // sigmoid 參數 prefix（對應 extract_sigmoid_params.py 的 --layer conv2 輸出）
    const char *sig_prefix = "./sigmoid_params/conv2_sigmoid";

    int total = C_OUT * N_SPATIAL;

    // ── 讀取 acc ──────────────────────────────────────────────────────────────
    int32_t *acc = (int32_t*)malloc(total * sizeof(int32_t));
    load_1d_int32(acc_path, acc, total);

    // ── 讀取 per-channel 離線參數 ─────────────────────────────────────────────
    uint32_t M0[C_OUT];
    int64_t  n_arr[C_OUT];
    int32_t  K[C_OUT];

    load_1d_uint32(M0_path, M0,    C_OUT);
    load_1d_int64 (n_path,  n_arr, C_OUT);
    load_1d_int32 (K_path,  K,     C_OUT);

    // ── 載入 sigmoid 近似參數 ─────────────────────────────────────────────────
    SigmoidParams sig_params = load_sigmoid_params(sig_prefix);

    // zp_out for requantize（conv1 輸出到 sigmoid 輸入，對稱，Z=0）
    uint8_t zp_out = 0;

    // ── 逐元素：requantize → sigmoid ─────────────────────────────────────────
    // conv1 flow：acc → requantize（得到 int8 q3）→ sigmoid_pwl（得到 uint8 q4）
    uint8_t *output = (uint8_t*)malloc(total * sizeof(uint8_t));

    for (int c = 0; c < C_OUT; c++) {
        for (int hw = 0; hw < N_SPATIAL; hw++) {
            int idx = c * N_SPATIAL + hw;

            // step 1：requantize acc → q3（int8，對稱，Z3=0）
            // 注意：conv1 後是 sigmoid，不做 ReLU，do_relu=false
            // requantize 回傳 uint8，但實際上對稱量化 Z=0 所以強轉 int8
            int8_t q3_u8 = requantize_sym(acc[idx], M0[c], n_arr[c], K[c], zp_out);
            int8_t  q3    = (int8_t)q3_u8;   // reinterpret as int8（Z=0，對稱）

            // step 2：piecewise linear sigmoid
            output[idx] = sigmoid_pwl(q3, &sig_params);
        }
    }

    // ── 輸出存檔 ──────────────────────────────────────────────────────────────
    FILE *fp = fopen(out_path, "w");
    if (!fp) { fprintf(stderr, "Cannot open output file\n"); exit(1); }
    for (int i = 0; i < total; i++)
        fprintf(fp, "%d\n", (int)output[i]);
    fclose(fp);

    printf("Saved %s  shape=(%d, %d, %d)\n", out_path, C_OUT, H_OUT, W_OUT);

    free(acc);
    free(output);
    return EXIT_SUCCESS;
}