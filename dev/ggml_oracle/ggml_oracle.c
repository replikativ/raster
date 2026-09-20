// ggml_oracle — call llama.cpp's own ggml quantization and dot-product code on
// raw float32 input, so Raster's formats can be checked bit-for-bit against it.
//
//   ggml_oracle info
//   ggml_oracle quantize <type> <n_per_row> <nrows> <in.f32> <out.blocks> <out.dequant.f32>
//   ggml_oracle dot <weight-type> <n> <w.f32> <x.f32> <out.act-blocks> <out.act-ref-blocks>
//
// `quantize` runs ggml_quantize_chunk with no importance matrix, the path
// llama-quantize takes, and dequantizes with the type's to_float. `dot`
// quantizes the weight row the same way, quantizes the activation with the CPU
// backend's from_float for the weight type's vec_dot_type (what inference
// does on this CPU) and, separately, with the platform-independent
// from_float_ref. It prints the backend vec_dot over both, and the scalar
// generic vec_dot over the reference activations, as float bit patterns.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ggml.h"
#include "ggml-cpu.h"

// Exported by libggml-base (ggml-quants.h) but absent from the public type
// traits, because q8_K exists only as a dot-product activation format.
void quantize_row_q8_K_ref(const float * x, void * y, int64_t k);

// The scalar vec_dot implementations, exported by libggml-cpu. They are the
// portable definition of each dot product; the CPU backend's vec_dot is a SIMD
// variant that sums in a different order.
typedef void (*vec_dot_fn)(int n, float * s, size_t bs, const void * x, size_t bx,
                           const void * y, size_t by, int nrc);
void ggml_vec_dot_q4_K_q8_K_generic(int, float *, size_t, const void *, size_t, const void *, size_t, int);
void ggml_vec_dot_q6_K_q8_K_generic(int, float *, size_t, const void *, size_t, const void *, size_t, int);
void ggml_vec_dot_q5_0_q8_0_generic(int, float *, size_t, const void *, size_t, const void *, size_t, int);
void ggml_vec_dot_q8_0_q8_0_generic(int, float *, size_t, const void *, size_t, const void *, size_t, int);

static vec_dot_fn generic_dot(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K: return ggml_vec_dot_q4_K_q8_K_generic;
        case GGML_TYPE_Q6_K: return ggml_vec_dot_q6_K_q8_K_generic;
        case GGML_TYPE_Q5_0: return ggml_vec_dot_q5_0_q8_0_generic;
        case GGML_TYPE_Q8_0: return ggml_vec_dot_q8_0_q8_0_generic;
        default: return NULL;
    }
}

static enum ggml_type parse_type(const char * name) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * n = ggml_type_name((enum ggml_type) t);
        if (n && strcmp(n, name) == 0) return (enum ggml_type) t;
    }
    fprintf(stderr, "unknown ggml type %s\n", name);
    exit(2);
}

static float * read_f32(const char * path, size_t n) {
    FILE * f = fopen(path, "rb");
    if (!f) { perror(path); exit(2); }
    float * x = malloc(n * sizeof(float));
    if (fread(x, sizeof(float), n, f) != n) { fprintf(stderr, "short read %s\n", path); exit(2); }
    fclose(f);
    return x;
}

static void write_bytes(const char * path, const void * data, size_t n) {
    FILE * f = fopen(path, "wb");
    if (!f) { perror(path); exit(2); }
    if (fwrite(data, 1, n, f) != n) { fprintf(stderr, "short write %s\n", path); exit(2); }
    fclose(f);
}

int main(int argc, char ** argv) {
    ggml_cpu_init();
    if (argc >= 2 && strcmp(argv[1], "info") == 0) {
        printf("avx2 %d avx512 %d avx_vnni %d f16c %d fma %d\n",
               ggml_cpu_has_avx2(), ggml_cpu_has_avx512(), ggml_cpu_has_avx_vnni(),
               ggml_cpu_has_f16c(), ggml_cpu_has_fma());
        return 0;
    }
    if (argc == 8 && strcmp(argv[1], "quantize") == 0) {
        enum ggml_type type = parse_type(argv[2]);
        int64_t n_per_row = atoll(argv[3]);
        int64_t nrows = atoll(argv[4]);
        float * x = read_f32(argv[5], (size_t) (n_per_row * nrows));
        if (ggml_quantize_requires_imatrix(type)) { fprintf(stderr, "type needs an imatrix\n"); return 2; }
        size_t row_size = ggml_row_size(type, n_per_row);
        void * q = calloc(nrows, row_size);
        size_t written = ggml_quantize_chunk(type, x, q, 0, nrows, n_per_row, NULL);
        if (written != row_size * nrows) { fprintf(stderr, "unexpected size %zu\n", written); return 2; }
        float * y = malloc(sizeof(float) * n_per_row * nrows);
        const struct ggml_type_traits * traits = ggml_get_type_traits(type);
        for (int64_t r = 0; r < nrows; ++r) {
            traits->to_float((const char *) q + r * row_size, y + r * n_per_row, n_per_row);
        }
        write_bytes(argv[6], q, written);
        write_bytes(argv[7], y, sizeof(float) * n_per_row * nrows);
        printf("%zu %zu\n", row_size, written);
        return 0;
    }
    if (argc == 8 && strcmp(argv[1], "dot") == 0) {
        enum ggml_type type = parse_type(argv[2]);
        int64_t n = atoll(argv[3]);
        float * w = read_f32(argv[4], (size_t) n);
        float * x = read_f32(argv[5], (size_t) n);
        const struct ggml_type_traits_cpu * cpu = ggml_get_type_traits_cpu(type);
        enum ggml_type act = cpu->vec_dot_type;
        void * wq = calloc(1, ggml_row_size(type, n));
        ggml_quantize_chunk(type, w, wq, 0, 1, n, NULL);
        size_t act_size = ggml_row_size(act, n);
        void * xq = calloc(1, act_size);
        void * xr = calloc(1, act_size);
        ggml_get_type_traits_cpu(act)->from_float(x, xq, n);
        ggml_from_float_t ref = ggml_get_type_traits(act)->from_float_ref;
        if (ref) {
            ref(x, xr, n);
        } else if (act == GGML_TYPE_Q8_K) {
            quantize_row_q8_K_ref(x, xr, n);
        } else {
            fprintf(stderr, "no reference quantizer for %s\n", ggml_type_name(act));
            return 2;
        }
        float s = 0.0f, sr = 0.0f, sg = 0.0f;
        cpu->vec_dot((int) n, &s, 0, wq, 0, xq, 0, 1);
        cpu->vec_dot((int) n, &sr, 0, wq, 0, xr, 0, 1);
        vec_dot_fn g = generic_dot(type);
        if (!g) { fprintf(stderr, "no generic dot for %s\n", ggml_type_name(type)); return 2; }
        g((int) n, &sg, 0, wq, 0, xr, 0, 1);
        write_bytes(argv[6], xq, act_size);
        write_bytes(argv[7], xr, act_size);
        uint32_t bits, rbits, gbits;
        memcpy(&bits, &s, sizeof bits); memcpy(&rbits, &sr, sizeof rbits); memcpy(&gbits, &sg, sizeof gbits);
        printf("%s %08x %.9g %08x %.9g %08x %.9g\n", ggml_type_name(act), bits, s, rbits, sr, gbits, sg);
        return 0;
    }
    fprintf(stderr, "usage: see header comment\n");
    return 2;
}
