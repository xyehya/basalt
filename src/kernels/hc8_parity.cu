// src/kernels/hc8_parity.cu - the reads and products that take a hyper-connection matrix held only as its 8-bit copy
// (DEC_HC8, hc_fused.hpp's hc8_adopt), real geometry and random inputs (GPU, synthetic, no model):
//
//   1. hc8_gr_read_multi (what fused_gr_read, fused_gr_read_multi and gr_read run for such matrices) against
//      fused_gr_read_multi on the bf16 matrices (the error the copies add) and on the copies dequantized to bf16 (the
//      kernels' own error), 1..8 tokens, with and without the pending write and the injection; every token's outputs
//      bitwise the same whatever the token count and column; gr_read bitwise the one-token read.
//   1b. The read on the copies against frozen hashes of its outputs (taken from the kernels before the early-start
//      change), for several geometries (the model's, no injection, other widths and stream counts), 1..8 tokens, with
//      and without the pending write, launched plain and as a programmatic dependent behind a kernel; inputs come from
//      raw mt19937 words (integers times powers of two), so the hashes do not depend on the C++ library.
//   2. Gemm::bf16_w (the prompt path's product on a copy) against Gemm::bf16 on the bf16 matrix and against a
//      double-precision product with the copy's exact values, on the prompt path's hc shapes, with a scratch that holds
//      the whole matrix and with one that slices it (within fp32 rounding: cuBLAS's kernel can change with the width).
//   3. --bench: the read's time (weights cold in L2) and the products' time against the bf16 ones.
//
//     build/hc8_parity [--bench]
#include "basalt/kernels/fused_gr.hpp"
#include "basalt/kernels/gr.hpp"
#include "basalt/kernels/hc_fused.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/prefill/gemm.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace K = basalt::kernels;

namespace {

uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
float f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
float f16f(uint16_t h) {
    const int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
    float v = e == 0 ? std::ldexp((float) m, -24) : std::ldexp((float) (m | 1024), e - 25);
    return s ? -v : v;
}
std::vector<float> get(const float* d, size_t n) {
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost);
    return h;
}
struct Err {
    double abs = 0, ref = 0;
    void add(float a, float b) { abs = std::max(abs, (double) std::fabs(a - b)); ref = std::max(ref, (double) std::fabs(b)); }
    double rel() const { return ref > 0 ? abs / ref : abs; }
};
int g_fail = 0;
void expect(bool ok, const char* what) {
    if (!ok) { std::printf("FAIL: %s\n", what); ++g_fail; }
}


// ---- 1b. frozen outputs and the early-start dependency
uint64_t fnv(uint64_t h, const void* p, size_t n) {
    const unsigned char* b = static_cast<const unsigned char*>(p);
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}
// a signed 16-bit integer times 2^-shift: exact in float, the same on every standard library
struct Words {
    std::mt19937 g;
    explicit Words(uint32_t seed) : g(seed) {}
    float sym(int shift) { return (float) ((int) (g() >> 16) - 32768) * std::ldexp(1.0f, -shift); }
};
struct Case { int N, HC, LR; bool inject; };
constexpr int kCases = 5;
constexpr Case kCase[kCases] = {
    {2560, 4, 320, true}, {2560, 4, 320, false}, {2304, 4, 320, true}, {2080, 1, 128, true}, {2112, 3, 96, true}};
// FNV-1a of lo, inject (when there is one), mixed, R' and rs for the first T tokens: [case][pending write][T - 1]
const uint64_t kFrozen[kCases][2][8] = {
    {{0x763ee1ef95614510ull, 0xcf024d9e11e68d4aull, 0xbc1fb7cf0d9a2a00ull, 0xa9801e3a77ac7089ull,
      0xd99ec96fa0e8132bull, 0xffc444eacf8c4072ull, 0x85a1b9073412694aull, 0xbe042f9153244957ull},
     {0xede6915c6925c904ull, 0xeba002f3662419f6ull, 0x1bb285dfaa635e9cull, 0xf63c76c8a5ad0bf3ull,
      0xdc23baabc2ce18f4ull, 0x19018c0094bbb9baull, 0xc1bba0dfc68ca0adull, 0xa492f7621d8a6c0full}},
    {{0x5b5dd8daff42cab6ull, 0xd8e85cc894bf5588ull, 0x9661835119d3660bull, 0x0f6b451515eda174ull,
      0xd3072fdaac5444baull, 0x48ef8098e0c73a6cull, 0x7c3773205f00fa3cull, 0x7c9315b19f0197fdull},
     {0x736b26b511120950ull, 0x1cb08143e8aff1e2ull, 0xfad7a4671cf20f9dull, 0xbf897120e47c5a06ull,
      0xb417c4f692aad2b1ull, 0xc8a6cefdc5c3ba86ull, 0x004f5cb1d464d54bull, 0xad13ef673dc4376bull}},
    {{0x59460c455f5ed29dull, 0x13ba7ca385150979ull, 0x3e2f9128fa159352ull, 0x1ed3e48817ccdd6dull,
      0x600ec0653ae6b5dfull, 0x9d5f79c32e59b6cbull, 0x760728981ce282baull, 0x4cffc540edc2d570ull},
     {0xf94983c75077c4f5ull, 0xa15d9f49ef9f0499ull, 0x866e5da61dce86a2ull, 0x3d4be3c223d863a5ull,
      0xa93668f5142372c3ull, 0xb1a576e8a5f7d77aull, 0x957a3a1e4be5ba54ull, 0x18af3730971c4abcull}},
    {{0xe9cec9b66fc1aaf2ull, 0x51d619dc7e1ab2f5ull, 0x3c008c8580733822ull, 0xef9304a276a8678full,
      0x2b10f13d89aa3a96ull, 0xc3d65a1cf0afa92bull, 0x2dc0debbeb7f36e7ull, 0x7bc3fecbbcbde14bull},
     {0x217c58df69439bc6ull, 0x02c1b0c3f9fcb4baull, 0xb0ab0bb0244c8835ull, 0x6e19f31c62ce6f67ull,
      0x43c25aa5627ab417ull, 0xb30f5b9b86612307ull, 0xa8747b2cb77bd477ull, 0xa15047a8c8eac2d7ull}},
    {{0xef87a80c784d33a6ull, 0xe7e079c011d56f02ull, 0x8db3e760656d3dd9ull, 0xfc921207352056faull,
      0xad1b0e9a4e8f785bull, 0x4af45569b10ea122ull, 0xb8cc7bd7bdbf0993ull, 0xf2e91e4a9cf1300eull},
     {0x076392137beed088ull, 0xf878f20bd68f7c6full, 0xed59989ab98e082cull, 0x0443f8e8c67c6156ull,
      0xfa369d17cf3edc9aull, 0xe9eab46a088e8932ull, 0xdd5823e831bb9088ull, 0xf70f09f053410681ull}}};
constexpr int kTok = 8;

template <typename V>
V* upload(const std::vector<V>& h) {
    V* d;
    cudaMalloc((void**) &d, h.size() * sizeof(V));
    cudaMemcpy(d, h.data(), h.size() * sizeof(V), cudaMemcpyHostToDevice);
    return d;
}

struct Rig {
    Case c;
    int D;
    std::vector<float> R;
    float *R0, *bo0, *inj0, *dR, *dbo, *dinj, *dwn, *lo, *rs, *io, *mix;
    const void *kd, *ku, *ki;
    K::Hc8View vi;
    uint16_t* dqi;
    cudaStream_t pst;
    explicit Rig(int idx) : c(kCase[idx]), D(c.N * c.HC) {
        Words w(7 + idx);
        auto fill = [&](size_t n, int shift) { std::vector<float> v(n); for (auto& x : v) x = w.sym(shift); return v; };
        R = fill((size_t) kTok * D, 14);
        const std::vector<float> bo = fill((size_t) kTok * c.N, 15), inj = fill((size_t) kTok * c.HC, 14);
        std::vector<float> wn = fill(D, 17);
        for (auto& x : wn) x += 1.0f;
        auto bf = [&](size_t n, int shift) { std::vector<uint16_t> v(n); for (auto& x : v) x = bf16(w.sym(shift)); return v; };
        std::vector<uint16_t> wd = bf((size_t) c.LR * D, 20);
        const std::vector<uint16_t> wu = bf((size_t) D * c.LR, 19), wi = bf((size_t) c.HC * D, 20);
        for (int k = 0; k < 16; ++k) wd[(size_t) ((k * 5) % c.LR) * D + (k * 101) % D] = bf16(0.6f);   // outliers that set a group's scale
        uint16_t *dwd = upload(wd), *dwu = upload(wu), *dwi = upload(wi);
        kd = K::hc8_adopt(dwd, c.LR, D); ku = K::hc8_adopt(dwu, D, c.LR); ki = K::hc8_adopt(dwi, c.HC, D);
        expect(kd && ku && ki && K::hc8_view(ki, vi), "adopt");
        cudaFree(dwd); cudaFree(dwu); cudaFree(dwi);
        R0 = upload(R); bo0 = upload(bo); inj0 = upload(inj); dwn = upload(wn);
        cudaMalloc((void**) &dR, R.size() * 4); cudaMalloc((void**) &dbo, bo.size() * 4); cudaMalloc((void**) &dinj, inj.size() * 4);
        cudaMalloc((void**) &lo, (size_t) kTok * c.LR * 4); cudaMalloc((void**) &rs, (size_t) kTok * c.HC * 4);
        cudaMalloc((void**) &io, (size_t) kTok * c.HC * 4); cudaMalloc((void**) &mix, (size_t) kTok * c.N * 4);
        cudaMalloc((void**) &dqi, (size_t) c.HC * D * 2);
        cudaStreamCreate(&pst);   // blocking: ordered after the default-stream copies
    }
    // mode 0: plain launch; 1: behind a kernel that does not touch the read's data, started early as a programmatic dependent
    uint64_t read(int T, bool apply, int mode) {
        const size_t nb = (size_t) kTok * c.N, ni = (size_t) kTok * c.HC;
        cudaMemcpy(dR, R0, R.size() * 4, cudaMemcpyDeviceToDevice); cudaMemcpy(dbo, bo0, nb * 4, cudaMemcpyDeviceToDevice);
        cudaMemcpy(dinj, inj0, ni * 4, cudaMemcpyDeviceToDevice);
        cudaMemset(lo, 0xff, (size_t) kTok * c.LR * 4); cudaMemset(rs, 0xff, (size_t) kTok * c.HC * 4);
        cudaMemset(io, 0xff, ni * 4); cudaMemset(mix, 0xff, nb * 4);
        cudaDeviceSynchronize();
        std::unique_ptr<K::PdlScope> scope;
        if (mode == 1) K::hc8_dequant_bf16(vi, 0, c.HC, dqi, pst);
        if (mode == 1) scope.reset(new K::PdlScope(true, false));
        K::FusedGrArgs a[K::kFusedGrMaxT];
        for (int t = 0; t < T; ++t) {
            K::FusedGrArgs& x = a[t];
            x.R = dR + (size_t) t * D; x.R_out = dR + (size_t) t * D; x.apply = apply;
            x.bo_prev = dbo + (size_t) t * c.N; x.inj_prev = dinj + (size_t) t * c.HC; x.w_norm = dwn;
            x.w_down = (const uint16_t*) kd; x.w_up = (const uint16_t*) ku; x.w_inject = c.inject ? (const uint16_t*) ki : nullptr;
            x.eps = 1e-6f; x.lo = lo + (size_t) t * c.LR; x.rs = rs + (size_t) t * c.HC;
            x.inject_out = io + (size_t) t * c.HC; x.mixed = mix + (size_t) t * c.N;
        }
        expect(K::hc8_gr_read_multi(a, T, c.N, c.HC, mode ? (void*) pst : nullptr), "hc8_gr_read_multi");
        scope.reset();
        cudaDeviceSynchronize();
        uint64_t h = 14695981039346656037ull;
        const size_t cnt[5] = {(size_t) T * c.LR, (size_t) T * c.HC, (size_t) T * c.N, (size_t) T * D, (size_t) T * c.HC};
        const float* src[5] = {lo, io, mix, dR, rs};
        for (int k = 0; k < 5; ++k) {
            if (k == 1 && !c.inject) continue;
            const std::vector<float> v = get(src[k], cnt[k]);
            h = fnv(h, v.data(), v.size() * 4);
        }
        return h;
    }
};

void frozen_checks() {
    static const char* const kMode[2] = {"plain launch", "early start behind a kernel"};
    for (int idx = 0; idx < kCases; ++idx) {
        Rig rig(idx);
        const Case& c = rig.c;
        int bad = 0, n = 0;
        for (int mode = 0; mode < 2; ++mode)
            for (int apply = 0; apply < 2; ++apply)
                for (int T = 1; T <= kTok; ++T)
                    {
                        const uint64_t h = rig.read(T, apply != 0, mode);
                        const bool ok = h == kFrozen[idx][apply][T - 1];
                        ++n; bad += !ok;
                        if (!ok || mode == 0)
                            std::printf("  frozen n_embd %d hc %d lr %d %s, %s, %d token%s, %s: %016llx %s\n", c.N, c.HC, c.LR,
                                        c.inject ? "inject" : "no inject", apply ? "pending write" : "no write", T, T == 1 ? "" : "s",
                                        kMode[mode], (unsigned long long) h, ok ? "ok" : "DIFFERS");
                    }
        std::printf("frozen outputs, n_embd %d hc %d lr %d %s: %d reads, %d differ\n", c.N, c.HC, c.LR,
                    c.inject ? "inject" : "no inject", n, bad);
        expect(bad == 0, "the read's outputs against the frozen hashes");
    }
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    const int N = 2560, HC = 4, D = N * HC, LR = 320, TM = 8;
    const float EPS = 1e-6f;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto fill = [&](std::vector<float>& v, float sc) { for (auto& x : v) x = sc * nd(rng); };
    std::vector<float> R((size_t) TM * D), bo((size_t) TM * N), inj((size_t) TM * HC), wn(D);
    fill(R, 1.0f); fill(bo, 0.5f); fill(inj, 1.0f); fill(wn, 0.3f);
    for (auto& x : wn) x += 1.0f;
    std::vector<uint16_t> wd((size_t) LR * D), wu((size_t) D * LR), wi((size_t) HC * D);
    for (auto& x : wd) x = bf16(0.02f * nd(rng));
    for (auto& x : wu) x = bf16(0.05f * nd(rng));
    for (auto& x : wi) x = bf16(0.02f * nd(rng));
    for (int k = 0; k < 64; ++k) wd[(size_t) (k * 5) * D + k * 101] = bf16(0.6f);   // outliers that set a group's scale

    uint16_t *dwd, *dwu, *dwi, *dqd, *dqu, *dqi;   // bf16, and the copies dequantized to bf16
    float *dwn, *dbo, *dinj;
    cudaMalloc((void**) &dwd, wd.size() * 2); cudaMalloc((void**) &dwu, wu.size() * 2); cudaMalloc((void**) &dwi, wi.size() * 2);
    cudaMalloc((void**) &dqd, wd.size() * 2); cudaMalloc((void**) &dqu, wu.size() * 2); cudaMalloc((void**) &dqi, wi.size() * 2);
    cudaMalloc((void**) &dwn, D * 4); cudaMalloc((void**) &dbo, bo.size() * 4); cudaMalloc((void**) &dinj, inj.size() * 4);
    cudaMemcpy(dwd, wd.data(), wd.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dwu, wu.data(), wu.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dwi, wi.data(), wi.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dwn, wn.data(), D * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dbo, bo.data(), bo.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dinj, inj.data(), inj.size() * 4, cudaMemcpyHostToDevice);
    const void* kd = K::hc8_adopt(dwd, LR, D);
    const void* ku = K::hc8_adopt(dwu, D, LR);
    const void* ki = K::hc8_adopt(dwi, HC, D);
    K::Hc8View vd, vu, vi;
    expect(kd && ku && ki && K::hc8_view(kd, vd) && K::hc8_view(ku, vu) && K::hc8_view(ki, vi) && vd.owned, "adopt");
    if (g_fail) return 1;
    K::hc8_dequant_bf16(vd, 0, LR, dqd, nullptr);
    K::hc8_dequant_bf16(vu, 0, D, dqu, nullptr);
    K::hc8_dequant_bf16(vi, 0, HC, dqi, nullptr);
    cudaDeviceSynchronize();
    {   // the copies' quantization error
        std::vector<uint16_t> q(wd.size());
        cudaMemcpy(q.data(), dqd, q.size() * 2, cudaMemcpyDeviceToHost);
        double e2 = 0, w2 = 0;
        for (size_t i = 0; i < q.size(); ++i) {
            const double a = f32(wd[i]), b = f32(q[i]);
            e2 += (a - b) * (a - b); w2 += a * a;
        }
        std::printf("w_down copy (dequantized to bf16): relative rms error %.3e\n", std::sqrt(e2 / w2));
    }

    // ---------------------------------------------------------------- 1. the read
    struct Out { float *R, *xn, *lo, *rs, *io, *mix; };
    auto alloc_out = [&](Out& o) {
        cudaMalloc((void**) &o.R, (size_t) TM * D * 4); cudaMalloc((void**) &o.xn, (size_t) TM * D * 4);
        cudaMalloc((void**) &o.lo, (size_t) TM * LR * 4); cudaMalloc((void**) &o.rs, (size_t) TM * HC * 4);
        cudaMalloc((void**) &o.io, (size_t) TM * HC * 4); cudaMalloc((void**) &o.mix, (size_t) TM * N * 4);
    };
    Out o[3];
    for (auto& x : o) alloc_out(x);
    // run `which` (0: hc8 copies, 1: bf16, 2: dequantized copies through the bf16 read) on tokens `tok` in columns 0..n-1
    auto run = [&](int which, Out& out, const std::vector<int>& tok, bool apply, bool inject) {
        cudaMemcpy(out.R, R.data(), R.size() * 4, cudaMemcpyHostToDevice);
        K::FusedGrArgs a[K::kFusedGrMaxT];
        for (size_t c = 0; c < tok.size(); ++c) {
            const int t = tok[c];
            K::FusedGrArgs& x = a[c];
            x.R = out.R + (size_t) t * D; x.R_out = out.R + (size_t) t * D; x.apply = apply;
            x.bo_prev = dbo + (size_t) t * N; x.inj_prev = dinj + (size_t) t * HC;
            x.w_norm = dwn;
            x.w_down = (const uint16_t*) (which == 0 ? kd : which == 1 ? (const void*) dwd : (const void*) dqd);
            x.w_up = (const uint16_t*) (which == 0 ? ku : which == 1 ? (const void*) dwu : (const void*) dqu);
            x.w_inject = !inject ? nullptr : (const uint16_t*) (which == 0 ? ki : which == 1 ? (const void*) dwi : (const void*) dqi);
            x.eps = EPS; x.lo = out.lo + (size_t) t * LR; x.rs = out.rs + (size_t) t * HC;
            x.inject_out = out.io + (size_t) t * HC;
            x.mixed = out.mix + (size_t) t * N;
        }
        K::fused_gr_read_multi(a, (int) tok.size(), out.xn, nullptr);
        cudaDeviceSynchronize();
    };
    std::vector<int> all(TM);
    for (int t = 0; t < TM; ++t) all[t] = t;
    for (int inject = 1; inject >= 0; --inject)
        for (int apply = 0; apply < 2; ++apply) {
            run(0, o[0], all, apply, inject);
            run(1, o[1], all, apply, inject);
            run(2, o[2], all, apply, inject);
            Err e[2][5];
            const size_t n[5] = {(size_t) TM * N, (size_t) TM * LR, (size_t) TM * HC, (size_t) TM * HC, (size_t) TM * D};
            const char* nm[5] = {"mixed", "lo", "inject", "rs", "R'"};
            std::vector<float> h0[5], h1[5], h2[5];
            for (int k = 0; k < 5; ++k) {
                if (k == 2 && !inject) continue;
                float* p0[5] = {o[0].mix, o[0].lo, o[0].io, o[0].rs, o[0].R};
                float* p1[5] = {o[1].mix, o[1].lo, o[1].io, o[1].rs, o[1].R};
                float* p2[5] = {o[2].mix, o[2].lo, o[2].io, o[2].rs, o[2].R};
                h0[k] = get(p0[k], n[k]); h1[k] = get(p1[k], n[k]); h2[k] = get(p2[k], n[k]);
                for (size_t i = 0; i < n[k]; ++i) { e[0][k].add(h0[k][i], h1[k][i]); e[1][k].add(h0[k][i], h2[k][i]); }
            }
            std::printf("read, 8 tokens, %s, %s:\n", apply ? "pending write" : "no write", inject ? "injection" : "no injection");
            for (int k = 0; k < 5; ++k) {
                if (k == 2 && !inject) continue;
                std::printf("  %-6s vs bf16: max abs %.3e (rel to max %.3e) | vs dequantized copy: max abs %.3e (rel %.3e)\n",
                            nm[k], e[0][k].abs, e[0][k].rel(), e[1][k].abs, e[1][k].rel());
                expect(e[1][k].rel() < 2e-3, "the 8-bit read against the dequantized copy");
                expect(e[0][k].rel() < 5e-2, "the 8-bit read against bf16");
            }
            // every token's outputs the same bits whatever the token count and column
            const std::vector<std::vector<int>> sets = {{0}, {5}, {7, 2, 4}, {3, 1}, {6, 0, 2, 5, 1, 7, 4}};
            bool same = true;
            for (const auto& s : sets) {
                run(0, o[2], s, apply, inject);
                float* p2[5] = {o[2].mix, o[2].lo, o[2].io, o[2].rs, o[2].R};
                for (int k = 0; k < 5; ++k) {
                    if (k == 2 && !inject) continue;
                    const std::vector<float> h = get(p2[k], n[k]);
                    const size_t per = n[k] / TM;
                    for (int t : s) same = same && std::memcmp(h.data() + t * per, h0[k].data() + t * per, per * 4) == 0;
                }
            }
            std::printf("  token count / column independence: %s\n", same ? "bitwise" : "DIFFERS");
            expect(same, "token count / column independence");
        }
    frozen_checks();
    {   // gr_read on the copies: the one-token read
        const K::GrShapes gs{N, HC, LR};
        void* wsb = nullptr;
        cudaMalloc(&wsb, K::gr_workspace_bytes(gs));
        K::GrWorkspace ws;
        K::gr_workspace_init(gs, wsb, ws);
        float *mix, *io;
        cudaMalloc((void**) &mix, N * 4); cudaMalloc((void**) &io, HC * 4);
        cudaMemcpy(o[2].R, R.data(), R.size() * 4, cudaMemcpyHostToDevice);
        K::gr_read(o[2].R + 3 * D, dwn, (const uint16_t*) kd, (const uint16_t*) ku, (const uint16_t*) ki, EPS, gs, ws, mix, io,
                   nullptr);
        run(0, o[0], {3}, false, true);
        const std::vector<float> a = get(mix, N), b = get(o[0].mix + 3 * N, N), c = get(io, HC), d = get(o[0].io + 3 * HC, HC);
        const bool same = std::memcmp(a.data(), b.data(), N * 4) == 0 && std::memcmp(c.data(), d.data(), HC * 4) == 0;
        std::printf("gr_read on the copies vs the one-token read: %s\n", same ? "bitwise" : "DIFFERS");
        expect(same, "gr_read");
    }

    // ---------------------------------------------------------------- 2. the prompt path's product
    cudaStream_t st;
    cudaStreamCreate(&st);
    std::string err;
    basalt::prefill::Gemm big, small;
    if (!big.init(st, 32ll << 20, err) || !small.init(st, 1ll << 20, err)) { std::printf("FAIL: gemm init: %s\n", err.c_str()); return 1; }
    struct Shape { int64_t T, N, K; const void* key; const uint16_t* w; const char* what; };
    const Shape shapes[] = {
        {512, LR, D, kd, dwd, "hc down 512 x 320 x 10240"},
        {512, D, LR, ku, dwu, "hc up   512 x 10240 x 320"},
        {512, HC, D, ki, dwi, "inject  512 x 4 x 10240"},
        {37, LR, D, kd, dwd, "hc down  37 x 320 x 10240"},
        {1, D, LR, ku, dwu, "hc up     1 x 10240 x 320"},
    };
    std::vector<int8_t> hq;
    std::vector<uint16_t> hs;
    for (const Shape& s : shapes) {
        K::Hc8View v;
        K::hc8_view(s.key, v);
        hq.resize((size_t) v.rows * v.cols);
        hs.resize((size_t) v.rows * v.srow);
        cudaMemcpy(hq.data(), v.q, hq.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(hs.data(), v.s, hs.size() * 2, cudaMemcpyDeviceToHost);
        std::vector<uint16_t> X((size_t) s.T * s.K);
        for (size_t i = 0; i < X.size(); ++i) {
            X[i] = bf16(nd(rng));
        }
        uint16_t* dX;
        float *Y0, *Y1, *Y2;
        const int64_t ldy = s.N + 8;
        cudaMalloc((void**) &dX, X.size() * 2);
        cudaMalloc((void**) &Y0, (size_t) s.T * ldy * 4); cudaMalloc((void**) &Y1, (size_t) s.T * ldy * 4);
        cudaMalloc((void**) &Y2, (size_t) s.T * ldy * 4);
        cudaMemcpy(dX, X.data(), X.size() * 2, cudaMemcpyHostToDevice);
        big.bf16_w(dX, s.w, Y0, s.T, s.N, s.K, ldy);                         // bf16 matrix
        big.bf16_w(dX, (const uint16_t*) s.key, Y1, s.T, s.N, s.K, ldy);     // copy, one slice
        small.bf16_w(dX, (const uint16_t*) s.key, Y2, s.T, s.N, s.K, ldy);   // copy, row slices
        cudaStreamSynchronize(st);
        const std::vector<float> y0 = get(Y0, (size_t) s.T * ldy), y1 = get(Y1, (size_t) s.T * ldy),
                                 y2 = get(Y2, (size_t) s.T * ldy);
        Err eb, ex;
        for (int64_t t = 0; t < s.T; ++t)
            for (int64_t n = 0; n < s.N; ++n) eb.add(y1[t * ldy + n], y0[t * ldy + n]);
        std::uniform_int_distribution<int64_t> pt(0, s.T - 1), pn(0, s.N - 1);
        for (int k = 0; k < 256; ++k) {   // the copy's exact values in double
            const int64_t t = pt(rng), n = pn(rng);
            double ref = 0;
            for (int64_t i = 0; i < s.K; ++i) {
                const double x = (double) f32(X[t * s.K + i]);
                ref += x * (double) hq[n * s.K + i] * (double) f16f(hs[n * v.srow + i / 32]);
            }
            ex.add(y1[t * ldy + n], (float) ref);
        }
        Err es;   // row slices: cuBLAS may pick another kernel for the narrower product (another summation order)
        for (int64_t t = 0; t < s.T; ++t)
            for (int64_t n = 0; n < s.N; ++n) es.add(y2[t * ldy + n], y1[t * ldy + n]);
        const bool sliced = es.rel() < 1e-5;
        std::printf("%s: vs bf16 max abs %.3e (rel %.3e) | vs the copy in double max abs %.3e (rel %.3e) | sliced vs whole "
                    "rel %.1e\n", s.what, eb.abs, eb.rel(), ex.abs, ex.rel(), es.rel());
        expect(ex.rel() < 5e-3, "the product against the copy in double");
        expect(eb.rel() < 5e-2, "the product against bf16");
        expect(sliced, "row slices");
        cudaFree(dX); cudaFree(Y0); cudaFree(Y1); cudaFree(Y2);
    }

    // ---------------------------------------------------------------- 3. times
    if (bench) {
        void* evict = nullptr;
        const size_t eb = (size_t) 256 << 20;
        cudaMalloc(&evict, eb);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        auto time_us = [&](auto&& fn, int iters, bool cold) {
            double total = 0;
            for (int i = 0; i < iters + 3; ++i) {
                if (cold) cudaMemsetAsync(evict, i & 0xff, eb, st);
                cudaEventRecord(e0, st);
                fn();
                cudaEventRecord(e1, st);
                cudaEventSynchronize(e1);
                float ms = 0;
                cudaEventElapsedTime(&ms, e0, e1);
                if (i >= 3) total += ms;
            }
            return total * 1e3 / iters;
        };
        const double b8 = (double) (vd.bytes + vu.bytes + vi.bytes), b16 = (double) (wd.size() + wu.size() + wi.size()) * 2;
        for (int T : {1, 3, 8}) {
            K::FusedGrArgs a[2][K::kFusedGrMaxT];
            for (int w = 0; w < 2; ++w)
                for (int t = 0; t < T; ++t) {
                    K::FusedGrArgs& x = a[w][t];
                    x.R = o[w].R + (size_t) t * D; x.R_out = o[w].R + (size_t) t * D; x.apply = true;
                    x.bo_prev = dbo + (size_t) t * N; x.inj_prev = dinj + (size_t) t * HC; x.w_norm = dwn;
                    x.w_down = (const uint16_t*) (w ? (const void*) dwd : kd);
                    x.w_up = (const uint16_t*) (w ? (const void*) dwu : ku);
                    x.w_inject = (const uint16_t*) (w ? (const void*) dwi : ki);
                    x.eps = EPS; x.lo = o[w].lo + (size_t) t * LR; x.rs = o[w].rs + (size_t) t * HC;
                    x.inject_out = o[w].io + (size_t) t * HC; x.mixed = o[w].mix + (size_t) t * N;
                }
            const double u8 = time_us([&] { K::fused_gr_read_multi(a[0], T, o[0].xn, st); }, 50, true);
            const double u16 = time_us([&] { K::fused_gr_read_multi(a[1], T, o[1].xn, st); }, 50, true);
            std::printf("read T %d (cold L2): 8-bit %.1f us (%.0f GB/s of %.1f MB) | bf16 fused_gr %.1f us (%.0f GB/s of %.1f MB)\n",
                        T, u8, b8 / u8 / 1e3, b8 / 1e6, u16, b16 / u16 / 1e3, b16 / 1e6);
        }
        for (int64_t T : {512, 2048, 8192}) {
            uint16_t* dX;
            float* Y;
            cudaMalloc((void**) &dX, (size_t) T * D * 2);
            cudaMalloc((void**) &Y, (size_t) T * D * 4);
            cudaMemset(dX, 0, (size_t) T * D * 2);
            const double dn16 = time_us([&] { big.bf16(dX, dwd, Y, T, LR, D); }, 20, false);
            const double dn8 = time_us([&] { big.bf16_w(dX, (const uint16_t*) kd, Y, T, LR, D); }, 20, false);
            const double up16 = time_us([&] { big.bf16(dX, dwu, Y, T, D, LR); }, 20, false);
            const double up8 = time_us([&] { big.bf16_w(dX, (const uint16_t*) ku, Y, T, D, LR); }, 20, false);
            const double dq = time_us([&] { K::hc8_dequant_bf16(vd, 0, LR, big.scratch(), st); }, 20, false);
            std::printf("product T %lld: down bf16 %.1f us, copy %.1f us | up bf16 %.1f us, copy %.1f us | dequantize one "
                        "matrix %.1f us (%.0f GB/s of %.1f MB)\n",
                        (long long) T, dn16, dn8, up16, up8, dq, (vd.bytes + wd.size() * 2.0) / dq / 1e3,
                        (vd.bytes + wd.size() * 2.0) / 1e6);
            cudaFree(dX); cudaFree(Y);
        }
    }
    std::printf("%s\n", g_fail ? "hc8_parity: FAILED" : "hc8_parity: ok");
    return g_fail ? 1 : 0;
}
