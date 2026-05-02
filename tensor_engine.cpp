#include <bits/stdc++.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
using namespace std;

#define C_OUT      32
#define H_OUT      35      // conv1 output: (101 - 67 + 1) = 35
#define W_OUT      33      // conv1 output: (40  -  8 + 1) = 33
#define N_SPATIAL  (H_OUT * W_OUT)   // 1155 per channel

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

uint8_t requantize(int32_t acc, uint32_t M0, int64_t n, int32_t K, uint8_t zp_out) {

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
    if (out < zp_out) out = zp_out;

    return out;
}

// ── 主程式 ──────────────────────────────────────────────────────────────────

int main() {

    // 路徑（依你的 outdir 調整）
    const char *acc_path = "./layer_outputs/conv1_acc.txt";   // extract_acc.py 輸出
    const char *M0_path  = "./layer_outputs/conv1_M0.txt";
    const char *n_path   = "./layer_outputs/conv1_n.txt";
    const char *K_path   = "./layer_outputs/conv1_K.txt";
    const char *out_path = "./layer_outputs/conv1_out_c.txt";

    int total = C_OUT * N_SPATIAL;   // 32 * 1155 = 36960

    // ── 讀取 acc（flatten, row-major: channel 優先）──────────────────────────
    // extract_acc.py 存的順序：(1, C_out, H, W) flatten → channel-major
    int32_t *acc = (int32_t*)malloc(total * sizeof(int32_t));
    load_1d_int32(acc_path, acc, total);

    // ── 讀取 per-channel 離線參數 ───────────────────────────────────────────
    uint32_t M0[C_OUT];
    int64_t  n_arr[C_OUT];
    int32_t  K[C_OUT];

    load_1d_uint32(M0_path, M0,    C_OUT);
    load_1d_int64 (n_path,  n_arr, C_OUT);
    load_1d_int32 (K_path,  K,     C_OUT);

    // zp_out 從 pth 取得，先 hardcode（或之後改從檔案讀）
    // 從 verify_param 可知 relu1 的 zp_out = zp('relu1')
    uint8_t zp_out = 0;   // ← 請替換為你的實際值

    // ── 逐元素做 requantization ─────────────────────────────────────────────
    // acc 排列：(C_out, H_out, W_out) flatten，index = c * N_SPATIAL + hw
    uint8_t *output = (uint8_t*)malloc(total * sizeof(uint8_t));

    for (int c = 0; c < C_OUT; c++) {
        for (int hw = 0; hw < N_SPATIAL; hw++) {
            int idx = c * N_SPATIAL + hw;
            output[idx] = requantize(acc[idx], M0[c], n_arr[c], K[c], zp_out);
        }
    }

    // ── 輸出存檔（一行一個整數，對應 python 的 relu1_q.txt 格式）────────────
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