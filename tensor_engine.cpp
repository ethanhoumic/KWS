#include <bits/stdc++.h>
using namespace std;

#define LUT_SIZE 256

int8_t sigmoid_lut[LUT_SIZE];

void load_lut(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { perror("open LUT failed"); exit(1); }
    
    for (int i = 0; i < LUT_SIZE; i++) {
        int val;
        fscanf(fp, "%d", &val);
        sigmoid_lut[i] = (int8_t)val;
    }
    fclose(fp);
}

uint8_t sigmoid(int8_t x) {
    
    return sigmoid_lut[x];
}

uint8_t relu(uint8_t x, uint8_t ZERO_POINT) {
    
    uint8_t relu = max(x, ZERO_POINT);
    
    return relu;
}

int main(){

    load_lut("sigmoid_lut.txt");

    bool is_sigmoid = false;

    int32_t conv_output; 
    int32_t M0;
    int8_t n, K, ZERO_POINT;

    int64_t mult = conv_output * M0;
    int8_t shifted_mult = mult >> n;
    int8_t added_bias = shifted_mult + K;
    uint8_t activated_output = is_sigmoid ? sigmoid(added_bias) : relu(added_bias, ZERO_POINT);

    return EXIT_SUCCESS;
}