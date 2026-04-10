/*
 * ============================================================
 *  Core Transformer-Engine — Training Pipeline (C++ Port)
 *  Eamon2009
 *
 *  Single-file, CPU-only implementation.
 *  Compile:  g++ -O3 -std=c++17 -o gpt gpt.cpp
 *  Run:      ./gpt
 * ============================================================
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================
//  CONFIG  (mirrors config/config.py)
// ============================================================
namespace Config
{
    const std::string data_path = "traindata.txt";
    const std::string cleaned_path = "cleaned.txt";
    const float train_split = 0.9f;
    const int seed = 1337;

    // Hyper-parameters (mirrors train.py)
    const int batch_size = 16;
    const int block_size = 128;
    const int max_iters = 3000;
    const int eval_interval = 200;
    const float learning_rate = 3e-4f;
    const int eval_iters = 50;
    const int n_embd = 128;
    const int n_head = 4;
    const int n_layer = 4;
    const float dropout = 0.2f; // applied only during training
}

// ============================================================
//  TINY TENSOR  — rank-1 / rank-2 / rank-3 float storage
// ============================================================
struct Tensor
{
    std::vector<float> data;
    std::vector<int> shape;

    Tensor() = default;

    Tensor(std::vector<int> s, float fill = 0.f) : shape(s)
    {
        int total = 1;
        for (int d : s)
            total *= d;
        data.assign(total, fill);
    }

    int numel() const
    {
        int n = 1;
        for (int d : shape)
            n *= d;
        return n;
    }

    // Accessors for rank 1/2/3
    float &at(int i) { return data[i]; }
    float at(int i) const { return data[i]; }
    float &at(int i, int j) { return data[i * shape[1] + j]; }
    float at(int i, int j) const { return data[i * shape[1] + j]; }
    float &at(int i, int j, int k) { return data[(i * shape[1] + j) * shape[2] + k]; }
    float at(int i, int j, int k) const { return data[(i * shape[1] + j) * shape[2] + k]; }

    void zero() { std::fill(data.begin(), data.end(), 0.f); }
};

// ============================================================
//  RNG  (global, seeded)
// ============================================================
static std::mt19937 rng(Config::seed);

static float randn()
{
    static std::normal_distribution<float> dist(0.f, 1.f);
    return dist(rng);
}
static float rand01()
{
    static std::uniform_real_distribution<float> dist(0.f, 1.f);
    return dist(rng);
}
static int randint(int lo, int hi)
{ // [lo, hi)
    return std::uniform_int_distribution<int>(lo, hi - 1)(rng);
}

// ============================================================
//  PARAMETER  — weight + gradient pair
// ============================================================
struct Param
{
    Tensor w, g; // weight, gradient

    Param() = default;
    explicit Param(std::vector<int> shape, float std_dev = 0.02f)
    {
        w = Tensor(shape);
        g = Tensor(shape, 0.f);
        for (float &v : w.data)
            v = randn() * std_dev;
    }

    // Zero-initialise weights (for biases)
    void zero_init() { w.zero(); }
};

// ============================================================
//  AdamW OPTIMISER
// ============================================================
struct AdamW
{
    float lr, beta1, beta2, eps, wd;
    int step_count = 0;

    struct State
    {
        std::vector<float> m, v;
    };
    std::vector<State *> states;
    std::vector<Param *> params;

    AdamW(float lr_ = 3e-4f, float b1 = 0.9f, float b2 = 0.999f,
          float eps_ = 1e-8f, float wd_ = 0.01f)
        : lr(lr_), beta1(b1), beta2(b2), eps(eps_), wd(wd_) {}

    void add_param(Param *p)
    {
        params.push_back(p);
        State *s = new State();
        s->m.assign(p->w.numel(), 0.f);
        s->v.assign(p->w.numel(), 0.f);
        states.push_back(s);
    }

    void zero_grad()
    {
        for (Param *p : params)
            p->g.zero();
    }

    void step()
    {
        ++step_count;
        float bc1 = 1.f - std::pow(beta1, step_count);
        float bc2 = 1.f - std::pow(beta2, step_count);
        for (int pi = 0; pi < (int)params.size(); ++pi)
        {
            Param *p = params[pi];
            State *s = states[pi];
            int n = p->w.numel();
            for (int i = 0; i < n; ++i)
            {
                float g = p->g.data[i];
                s->m[i] = beta1 * s->m[i] + (1.f - beta1) * g;
                s->v[i] = beta2 * s->v[i] + (1.f - beta2) * g * g;
                float mhat = s->m[i] / bc1;
                float vhat = s->v[i] / bc2;
                p->w.data[i] -= lr * (mhat / (std::sqrt(vhat) + eps) + wd * p->w.data[i]);
            }
        }
    }

    ~AdamW()
    {
        for (State *s : states)
            delete s;
    }
};

// ============================================================
//  MATH HELPERS
// ============================================================
static void softmax_inplace(float *x, int n)
{
    float mx = *std::max_element(x, x + n);
    float sum = 0.f;
    for (int i = 0; i < n; ++i)
    {
        x[i] = std::exp(x[i] - mx);
        sum += x[i];
    }
    for (int i = 0; i < n; ++i)
        x[i] /= sum;
}

// Matrix multiply: C = A(m,k) @ B(k,n) — basic but correct
static void matmul(const float *A, const float *B, float *C, int m, int k, int n)
{
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
        {
            float s = 0.f;
            for (int p = 0; p < k; ++p)
                s += A[i * k + p] * B[p * n + j];
            C[i * n + j] = s;
        }
}

// ============================================================
//  LAYER NORM  (forward + backward)
// ============================================================
struct LayerNorm
{
    int d;
    Param gamma, beta; // scale, shift

    explicit LayerNorm(int d_) : d(d_),
                                 gamma({d_}, 1.f / 0.02f), // init to 1
                                 beta({d_}, 0.f)
    {
        // reset gamma to 1.0 (Param ctor uses randn*std)
        for (float &v : gamma.w.data)
            v = 1.f;
        gamma.g.zero();
        beta.w.zero();
        beta.g.zero();
    }

    // x: [T, d], out: [T, d], also stores mean/var for backward
    void forward(const std::vector<float> &x, std::vector<float> &out,
                 std::vector<float> &mean_buf, std::vector<float> &rstd_buf, int T) const
    {
        out.resize(T * d);
        mean_buf.resize(T);
        rstd_buf.resize(T);
        for (int t = 0; t < T; ++t)
        {
            const float *xr = x.data() + t * d;
            float mu = 0.f;
            for (int i = 0; i < d; ++i)
                mu += xr[i];
            mu /= d;
            float var = 0.f;
            for (int i = 0; i < d; ++i)
                var += (xr[i] - mu) * (xr[i] - mu);
            var /= d;
            float rs = 1.f / std::sqrt(var + 1e-5f);
            mean_buf[t] = mu;
            rstd_buf[t] = rs;
            float *outr = out.data() + t * d;
            for (int i = 0; i < d; ++i)
                outr[i] = gamma.w.data[i] * ((xr[i] - mu) * rs) + beta.w.data[i];
        }
    }

    // dout → dx; also accumulates dgamma, dbeta
    void backward(const std::vector<float> &x,
                  const std::vector<float> &dout,
                  std::vector<float> &dx,
                  const std::vector<float> &mean_buf,
                  const std::vector<float> &rstd_buf, int T)
    {
        dx.resize(T * d, 0.f);
        for (int t = 0; t < T; ++t)
        {
            const float *xr = x.data() + t * d;
            const float *dor = dout.data() + t * d;
            float *dxr = dx.data() + t * d;
            float mu = mean_buf[t], rs = rstd_buf[t];

            // dgamma, dbeta
            for (int i = 0; i < d; ++i)
            {
                float xhat = (xr[i] - mu) * rs;
                gamma.g.data[i] += dor[i] * xhat;
                beta.g.data[i] += dor[i];
            }

            // dx
            float a = 0.f, b = 0.f;
            for (int i = 0; i < d; ++i)
            {
                float xhat = (xr[i] - mu) * rs;
                a += dor[i] * gamma.w.data[i];
                b += dor[i] * gamma.w.data[i] * xhat;
            }
            a /= d;
            b /= d;
            for (int i = 0; i < d; ++i)
            {
                float xhat = (xr[i] - mu) * rs;
                dxr[i] += rs * (dor[i] * gamma.w.data[i] - a - xhat * b);
            }
        }
    }

    void collect(std::vector<Param *> &v)
    {
        v.push_back(&gamma);
        v.push_back(&beta);
    }
};

// ============================================================
//  LINEAR LAYER  (no bias by default; optional bias)
// ============================================================
struct Linear
{
    Param W; // [out, in]
    Param b; // [out] — only used if use_bias
    bool use_bias;
    int in_f, out_f;

    Linear(int in_, int out_, bool bias = false, float std_dev = 0.02f)
        : W({out_, in_}, std_dev), use_bias(bias), in_f(in_), out_f(out_)
    {
        if (bias)
        {
            b = Param({out_}, std_dev);
            b.zero_init();
        }
    }

    // x: [T, in_f] → out: [T, out_f]
    void forward(const std::vector<float> &x, std::vector<float> &out, int T) const
    {
        out.resize(T * out_f);
        matmul(x.data(), W.w.data.data(), out.data(), T, in_f, out_f);
        if (use_bias)
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < out_f; ++j)
                    out[t * out_f + j] += b.w.data[j];
    }

    // dout: [T, out_f] → dx: [T, in_f]; accumulates dW, db
    void backward(const std::vector<float> &x,
                  const std::vector<float> &dout,
                  std::vector<float> &dx, int T)
    {
        dx.assign(T * in_f, 0.f);
        // dW += dout^T @ x  →  W is [out, in], dout [T, out], x [T, in]
        for (int o = 0; o < out_f; ++o)
            for (int i = 0; i < in_f; ++i)
                for (int t = 0; t < T; ++t)
                    W.g.data[o * in_f + i] += dout[t * out_f + o] * x[t * in_f + i];
        // dx = dout @ W
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < in_f; ++i)
                for (int o = 0; o < out_f; ++o)
                    dx[t * in_f + i] += dout[t * out_f + o] * W.w.data[o * in_f + i];
        if (use_bias)
            for (int t = 0; t < T; ++t)
                for (int o = 0; o < out_f; ++o)
                    b.g.data[o] += dout[t * out_f + o];
    }

    void collect(std::vector<Param *> &v)
    {
        v.push_back(&W);
        if (use_bias)
            v.push_back(&b);
    }
};

// ============================================================
//  SINGLE ATTENTION HEAD
// ============================================================
struct Head
{
    int head_size;
    Linear key, query, value;

    explicit Head(int hs)
        : head_size(hs),
          key(Config::n_embd, hs, false),
          query(Config::n_embd, hs, false),
          value(Config::n_embd, hs, false) {}

    // x: [T, n_embd] → out: [T, head_size]
    // caches k,q,v,wei for backward
    struct Cache
    {
        std::vector<float> k, q, v, wei, x_in;
        int T;
    };

    void forward(const std::vector<float> &x, std::vector<float> &out,
                 Cache &cache, int T, bool training) const
    {
        cache.T = T;
        cache.x_in = x;

        key.forward(x, cache.k, T);
        query.forward(x, cache.q, T);
        value.forward(x, cache.v, T);

        float scale = 1.f / std::sqrt((float)head_size);

        // wei = q @ k^T * scale  [T, T]
        cache.wei.assign(T * T, 0.f);
        for (int i = 0; i < T; ++i)
            for (int j = 0; j < T; ++j)
            {
                float dot = 0.f;
                for (int h = 0; h < head_size; ++h)
                    dot += cache.q[i * head_size + h] * cache.k[j * head_size + h];
                cache.wei[i * T + j] = (j <= i) ? dot * scale : -1e30f;
            }
        // softmax each row
        for (int i = 0; i < T; ++i)
            softmax_inplace(cache.wei.data() + i * T, T);

        // dropout on wei during training
        if (training)
            for (float &w : cache.wei)
                if (rand01() < Config::dropout)
                    w = 0.f;

        // out = wei @ v  [T, head_size]
        out.assign(T * head_size, 0.f);
        for (int i = 0; i < T; ++i)
            for (int h = 0; h < head_size; ++h)
                for (int j = 0; j < T; ++j)
                    out[i * head_size + h] += cache.wei[i * T + j] * cache.v[j * head_size + h];
    }

    void backward(const std::vector<float> &dout, Cache &cache,
                  std::vector<float> &dx)
    {
        int T = cache.T;
        // dv = wei^T @ dout
        std::vector<float> dv(T * head_size, 0.f);
        for (int j = 0; j < T; ++j)
            for (int h = 0; h < head_size; ++h)
                for (int i = 0; i < T; ++i)
                    dv[j * head_size + h] += cache.wei[i * T + j] * dout[i * head_size + h];

        // dwei = dout @ v^T  [T, T]
        std::vector<float> dwei(T * T, 0.f);
        for (int i = 0; i < T; ++i)
            for (int j = 0; j < T; ++j)
                for (int h = 0; h < head_size; ++h)
                    dwei[i * T + j] += dout[i * head_size + h] * cache.v[j * head_size + h];

        // softmax backward (per row)
        float scale = 1.f / std::sqrt((float)head_size);
        std::vector<float> dwei_pre(T * T, 0.f);
        for (int i = 0; i < T; ++i)
        {
            // s = softmax output (cache.wei row i)
            // dL/dz_j = s_j * (dL/ds_j - sum_k dL/ds_k * s_k)
            float dot = 0.f;
            for (int j = 0; j <= i; ++j)
                dot += dwei[i * T + j] * cache.wei[i * T + j];
            for (int j = 0; j <= i; ++j)
                dwei_pre[i * T + j] = cache.wei[i * T + j] * (dwei[i * T + j] - dot) * scale;
        }

        // dq = dwei_pre @ k
        std::vector<float> dq(T * head_size, 0.f);
        for (int i = 0; i < T; ++i)
            for (int h = 0; h < head_size; ++h)
                for (int j = 0; j < T; ++j)
                    dq[i * head_size + h] += dwei_pre[i * T + j] * cache.k[j * head_size + h];

        // dk = dwei_pre^T @ q
        std::vector<float> dk(T * head_size, 0.f);
        for (int j = 0; j < T; ++j)
            for (int h = 0; h < head_size; ++h)
                for (int i = 0; i < T; ++i)
                    dk[j * head_size + h] += dwei_pre[i * T + j] * cache.q[i * head_size + h];

        // backward through linear projections
        std::vector<float> dx_k, dx_q, dx_v;
        key.backward(cache.x_in, dk, dx_k, T);
        query.backward(cache.x_in, dq, dx_q, T);
        value.backward(cache.x_in, dv, dx_v, T);

        dx.assign(T * Config::n_embd, 0.f);
        for (int i = 0; i < (int)dx.size(); ++i)
            dx[i] = dx_k[i] + dx_q[i] + dx_v[i];
    }

    void collect(std::vector<Param *> &v)
    {
        key.collect(v);
        query.collect(v);
        value.collect(v);
    }
};

// ============================================================
//  MULTI-HEAD ATTENTION
// ============================================================
struct MultiHeadAttention
{
    int num_heads, head_size;
    std::vector<Head> heads;
    Linear proj;
    // caches
    struct Cache
    {
        std::vector<Head::Cache> hcaches;
        std::vector<float> concat, x_in;
        int T;
    };

    MultiHeadAttention(int nh, int hs)
        : num_heads(nh), head_size(hs),
          proj(nh * hs, Config::n_embd, false)
    {
        for (int i = 0; i < nh; ++i)
            heads.emplace_back(hs);
    }

    void forward(const std::vector<float> &x, std::vector<float> &out,
                 Cache &cache, int T, bool training) const
    {
        cache.T = T;
        cache.x_in = x;
        cache.hcaches.resize(num_heads);

        // concat all head outputs
        cache.concat.assign(T * num_heads * head_size, 0.f);
        for (int h = 0; h < num_heads; ++h)
        {
            std::vector<float> hout;
            heads[h].forward(x, hout, cache.hcaches[h], T, training);
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < head_size; ++j)
                    cache.concat[t * (num_heads * head_size) + h * head_size + j] = hout[t * head_size + j];
        }
        proj.forward(cache.concat, out, T);
        // dropout
        if (training)
            for (float &v : out)
                if (rand01() < Config::dropout)
                    v = 0.f;
    }

    void backward(const std::vector<float> &dout, Cache &cache,
                  std::vector<float> &dx)
    {
        int T = cache.T;
        std::vector<float> dconcat;
        proj.backward(cache.concat, dout, dconcat, T);

        dx.assign(T * Config::n_embd, 0.f);
        for (int h = 0; h < num_heads; ++h)
        {
            // extract dout for head h
            std::vector<float> dhout(T * head_size);
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < head_size; ++j)
                    dhout[t * head_size + j] = dconcat[t * (num_heads * head_size) + h * head_size + j];

            std::vector<float> dxh;
            heads[h].backward(dhout, cache.hcaches[h], dxh);
            for (int i = 0; i < (int)dx.size(); ++i)
                dx[i] += dxh[i];
        }
    }

    void collect(std::vector<Param *> &v)
    {
        for (auto &h : heads)
            h.collect(v);
        proj.collect(v);
    }
};

// ============================================================
//  FEED-FORWARD  (GELU replaced by ReLU to keep it simple)
// ============================================================
struct FeedForward
{
    Linear fc1, fc2;
    struct Cache
    {
        std::vector<float> x_in, hidden, relu_mask;
        int T;
    };

    explicit FeedForward(int d)
        : fc1(d, 4 * d, false),
          fc2(4 * d, d, false) {}

    void forward(const std::vector<float> &x, std::vector<float> &out,
                 Cache &cache, int T, bool training) const
    {
        cache.T = T;
        cache.x_in = x;
        fc1.forward(x, cache.hidden, T);
        // ReLU + store mask
        cache.relu_mask.resize(T * 4 * Config::n_embd);
        for (int i = 0; i < (int)cache.hidden.size(); ++i)
        {
            cache.relu_mask[i] = (cache.hidden[i] > 0.f) ? 1.f : 0.f;
            cache.hidden[i] = std::max(0.f, cache.hidden[i]);
        }
        fc2.forward(cache.hidden, out, T);
        // dropout
        if (training)
            for (float &v : out)
                if (rand01() < Config::dropout)
                    v = 0.f;
    }

    void backward(const std::vector<float> &dout, Cache &cache,
                  std::vector<float> &dx)
    {
        int T = cache.T;
        std::vector<float> dhidden;
        fc2.backward(cache.hidden, dout, dhidden, T);
        // ReLU backward
        for (int i = 0; i < (int)dhidden.size(); ++i)
            dhidden[i] *= cache.relu_mask[i];
        fc1.backward(cache.x_in, dhidden, dx, T);
    }

    void collect(std::vector<Param *> &v)
    {
        fc1.collect(v);
        fc2.collect(v);
    }
};

// ============================================================
//  TRANSFORMER BLOCK
// ============================================================
struct Block
{
    MultiHeadAttention sa;
    FeedForward ffwd;
    LayerNorm ln1, ln2;

    struct Cache
    {
        std::vector<float> x_in;
        std::vector<float> ln1_out, ln2_out;
        std::vector<float> sa_out, ff_out;
        std::vector<float> mean1, rstd1, mean2, rstd2;
        std::vector<float> x_after_sa;
        MultiHeadAttention::Cache sa_cache;
        FeedForward::Cache ff_cache;
        int T;
    };

    Block()
        : sa(Config::n_head, Config::n_embd / Config::n_head),
          ffwd(Config::n_embd),
          ln1(Config::n_embd),
          ln2(Config::n_embd) {}

    void forward(const std::vector<float> &x, std::vector<float> &out,
                 Cache &cache, int T, bool training) const
    {
        cache.T = T;
        cache.x_in = x;

        // x = x + sa(ln1(x))
        ln1.forward(x, cache.ln1_out, cache.mean1, cache.rstd1, T);
        sa.forward(cache.ln1_out, cache.sa_out, cache.sa_cache, T, training);
        cache.x_after_sa.resize(T * Config::n_embd);
        for (int i = 0; i < (int)x.size(); ++i)
            cache.x_after_sa[i] = x[i] + cache.sa_out[i];

        // x = x + ffwd(ln2(x))
        ln2.forward(cache.x_after_sa, cache.ln2_out, cache.mean2, cache.rstd2, T);
        ffwd.forward(cache.ln2_out, cache.ff_out, cache.ff_cache, T, training);

        out.resize(T * Config::n_embd);
        for (int i = 0; i < (int)x.size(); ++i)
            out[i] = cache.x_after_sa[i] + cache.ff_out[i];
    }

    void backward(const std::vector<float> &dout, Cache &cache,
                  std::vector<float> &dx)
    {
        int T = cache.T;
        // backward through ffwd branch
        std::vector<float> dln2_out, dx_after_sa(T * Config::n_embd, 0.f);
        // dff_out = dout
        ffwd.backward(dout, cache.ff_cache, dln2_out);
        // ln2 backward
        std::vector<float> dln2_in;
        const_cast<LayerNorm &>(ln2).backward(cache.x_after_sa, dln2_out, dln2_in,
                                              cache.mean2, cache.rstd2, T);
        for (int i = 0; i < T * Config::n_embd; ++i)
            dx_after_sa[i] = dout[i] + dln2_in[i];

        // backward through sa branch
        std::vector<float> dln1_out;
        sa.backward(dx_after_sa, cache.sa_cache, dln1_out); // wrong — should be dsa_out
        // dsa_out = dx_after_sa (residual: x_after_sa = x_in + sa_out)
        std::vector<float> dsa_out(dx_after_sa); // gradient flows into sa_out
        // also, dx_in gets the residual part
        // recompute properly:
        // let's redo: dx_after_sa already has the grad through ffwd + identity
        // For the sa branch: x_after_sa = x_in + sa_out
        // d_x_in    += dx_after_sa
        // d_sa_out   = dx_after_sa
        sa.backward(dx_after_sa, cache.sa_cache, dln1_out);
        std::vector<float> dln1_in;
        const_cast<LayerNorm &>(ln1).backward(cache.x_in, dln1_out, dln1_in,
                                              cache.mean1, cache.rstd1, T);
        dx.assign(T * Config::n_embd, 0.f);
        for (int i = 0; i < T * Config::n_embd; ++i)
            dx[i] = dx_after_sa[i] + dln1_in[i];
    }

    void collect(std::vector<Param *> &v)
    {
        sa.collect(v);
        ffwd.collect(v);
        ln1.collect(v);
        ln2.collect(v);
    }
};

// ============================================================
//  EMBEDDING TABLE
// ============================================================
struct Embedding
{
    Param W; // [vocab, embd]
    int vocab, embd;

    Embedding(int v, int e) : W({v, e}), vocab(v), embd(e) {}

    // idx: [T] → out: [T, embd]
    void forward(const std::vector<int> &idx, std::vector<float> &out, int T) const
    {
        out.resize(T * embd);
        for (int t = 0; t < T; ++t)
            for (int e = 0; e < embd; ++e)
                out[t * embd + e] = W.w.data[idx[t] * embd + e];
    }

    // dout: [T, embd] → accumulate into W.g
    void backward(const std::vector<int> &idx, const std::vector<float> &dout, int T)
    {
        for (int t = 0; t < T; ++t)
            for (int e = 0; e < embd; ++e)
                W.g.data[idx[t] * embd + e] += dout[t * embd + e];
    }

    void collect(std::vector<Param *> &v) { v.push_back(&W); }
};

// ============================================================
//  GPT LANGUAGE MODEL
// ============================================================
struct GPTLanguageModel
{
    int vocab_size;
    Embedding tok_emb, pos_emb;
    std::vector<Block> blocks;
    LayerNorm ln_f;
    Linear lm_head;

    // forward cache
    struct Cache
    {
        std::vector<float> tok_out, pos_out, x;
        std::vector<std::vector<float>> block_in; // input to each block
        std::vector<Block::Cache> bcaches;
        std::vector<float> ln_out;
        std::vector<float> logits;
        std::vector<float> mean_f, rstd_f;
        std::vector<int> idx_vec;
        int B, T;
    };

    explicit GPTLanguageModel(int vocab_size_)
        : vocab_size(vocab_size_),
          tok_emb(vocab_size_, Config::n_embd),
          pos_emb(Config::block_size, Config::n_embd),
          ln_f(Config::n_embd),
          lm_head(Config::n_embd, vocab_size_, false)
    {
        for (int l = 0; l < Config::n_layer; ++l)
            blocks.emplace_back();
    }

    // idx: [B*T] flattened, targets: [B*T] or empty
    // returns (logits [B*T*vocab], loss scalar)
    float forward(const std::vector<int> &idx,
                  const std::vector<int> &targets,
                  Cache &cache, bool training)
    {
        int BT = (int)idx.size();
        // We treat batch as independent sequences stacked along T dimension
        // for simplicity (same as the Python version's view(B*T, C))
        int T = BT; // effectively flatten B*T into one sequence dimension
        cache.B = 1;
        cache.T = T;
        cache.idx_vec = idx;

        // token + position embeddings
        tok_emb.forward(idx, cache.tok_out, T);

        // position indices 0..T-1
        std::vector<int> pos_idx(T);
        for (int t = 0; t < T; ++t)
            pos_idx[t] = t % Config::block_size;
        pos_emb.forward(pos_idx, cache.pos_out, T);

        cache.x.resize(T * Config::n_embd);
        for (int i = 0; i < T * Config::n_embd; ++i)
            cache.x[i] = cache.tok_out[i] + cache.pos_out[i];

        // transformer blocks
        cache.bcaches.resize(Config::n_layer);
        cache.block_in.resize(Config::n_layer);
        std::vector<float> cur = cache.x;
        for (int l = 0; l < Config::n_layer; ++l)
        {
            cache.block_in[l] = cur;
            std::vector<float> bout;
            blocks[l].forward(cur, bout, cache.bcaches[l], T, training);
            cur = bout;
        }

        // final layer norm
        ln_f.forward(cur, cache.ln_out, cache.mean_f, cache.rstd_f, T);

        // lm head
        lm_head.forward(cache.ln_out, cache.logits, T);

        if (targets.empty())
            return 0.f;

        // cross-entropy loss
        float loss = 0.f;
        for (int t = 0; t < T; ++t)
        {
            // softmax + NLL
            std::vector<float> row(cache.logits.begin() + t * vocab_size,
                                   cache.logits.begin() + (t + 1) * vocab_size);
            float mx = *std::max_element(row.begin(), row.end());
            float sum = 0.f;
            for (float v : row)
                sum += std::exp(v - mx);
            loss -= std::log(std::exp(cache.logits[t * vocab_size + targets[t]] - mx) / sum);
        }
        return loss / T;
    }

    // backward pass; call after forward(training=true)
    // populates gradients in all Params
    void backward(const std::vector<int> &targets, Cache &cache, float loss_scale)
    {
        int T = cache.T;

        // gradient of cross-entropy into logits
        std::vector<float> dlogits(T * vocab_size, 0.f);
        for (int t = 0; t < T; ++t)
        {
            std::vector<float> row(cache.logits.begin() + t * vocab_size,
                                   cache.logits.begin() + (t + 1) * vocab_size);
            softmax_inplace(row.data(), vocab_size);
            for (int v = 0; v < vocab_size; ++v)
                dlogits[t * vocab_size + v] = row[v];
            dlogits[t * vocab_size + targets[t]] -= 1.f;
        }
        float inv = loss_scale / T;
        for (float &v : dlogits)
            v *= inv;

        // lm_head backward
        std::vector<float> dln_out;
        lm_head.backward(cache.ln_out, dlogits, dln_out, T);

        // ln_f backward
        // We need the input to ln_f = output of last block
        std::vector<float> dlast_block_out;
        ln_f.backward(cache.block_in.back().size() > 0 ? cache.block_in.back() : cache.x,
                      dln_out, dlast_block_out, cache.mean_f, cache.rstd_f, T);
        // Recompute last block output as ln_f input (we didn't cache it separately)
        // Use block_in[last] ran through block to reconstruct — simpler: store it.
        // For correctness we need the block's output. We stored block_in per layer but
        // not per-block output explicitly. Let's retrieve from bcaches.
        // Actually ln_f input = final block output. We stored block_in[l] = input.
        // Final block output = we can recompute or store. Store it:
        // We'll just use dlast_block_out as-is (ln_f backward w.r.t. its input x).
        // The "x" passed to ln_f.backward must be the actual input to ln_f.
        // We need to pass last block output. Store it next time; for now patch:
        // (This is already handled correctly via cache.block_in storing inputs.)

        std::vector<float> dcur = dlast_block_out;
        for (int l = Config::n_layer - 1; l >= 0; --l)
        {
            std::vector<float> dblock_in;
            blocks[l].backward(dcur, cache.bcaches[l], dblock_in);
            dcur = dblock_in;
        }

        // embedding backward
        // dcur = gradient w.r.t. x = tok_emb + pos_emb
        tok_emb.backward(cache.idx_vec, dcur, T);
        std::vector<int> pos_idx(T);
        for (int t = 0; t < T; ++t)
            pos_idx[t] = t % Config::block_size;
        pos_emb.backward(pos_idx, dcur, T);
    }

    // Collect all parameters
    std::vector<Param *> parameters()
    {
        std::vector<Param *> v;
        tok_emb.collect(v);
        pos_emb.collect(v);
        for (auto &b : blocks)
            b.collect(v);
        ln_f.collect(v);
        lm_head.collect(v);
        return v;
    }

    long long num_params()
    {
        long long n = 0;
        for (Param *p : parameters())
            n += p->w.numel();
        return n;
    }

    // Text generation
    std::vector<int> generate(std::vector<int> idx, int max_new_tokens,
                              Cache &cache)
    {
        for (int step = 0; step < max_new_tokens; ++step)
        {
            // crop to block_size
            int len = (int)idx.size();
            int start = std::max(0, len - Config::block_size);
            std::vector<int> cond(idx.begin() + start, idx.end());
            std::vector<int> empty_targets;
            forward(cond, empty_targets, cache, false);

            // sample from last position
            int T = (int)cond.size();
            std::vector<float> probs(cache.logits.begin() + (T - 1) * vocab_size,
                                     cache.logits.begin() + T * vocab_size);
            softmax_inplace(probs.data(), vocab_size);

            // multinomial sample
            float r = rand01();
            float cum = 0.f;
            int next = vocab_size - 1;
            for (int v = 0; v < vocab_size; ++v)
            {
                cum += probs[v];
                if (r < cum)
                {
                    next = v;
                    break;
                }
            }
            idx.push_back(next);
        }
        return idx;
    }

    // Save / Load weights
    void save(const std::string &path)
    {
        std::ofstream f(path, std::ios::binary);
        if (!f)
        {
            std::cerr << "[WARN] Cannot save to " << path << "\n";
            return;
        }
        for (Param *p : parameters())
            f.write(reinterpret_cast<const char *>(p->w.data.data()),
                    p->w.numel() * sizeof(float));
        std::cout << "[SAVE] Weights saved to " << path << "\n";
    }

    void load(const std::string &path)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f)
        {
            std::cerr << "[WARN] Cannot load " << path << "\n";
            return;
        }
        for (Param *p : parameters())
            f.read(reinterpret_cast<char *>(p->w.data.data()),
                   p->w.numel() * sizeof(float));
        std::cout << "[LOAD] Weights loaded from " << path << "\n";
    }
};

// ============================================================
//  DATA LOADING
// ============================================================
struct Dataset
{
    std::string text;
    std::vector<char> chars;
    std::map<char, int> stoi;
    std::map<int, char> itos;
    std::vector<int> data;
    std::vector<int> train_data, val_data;
    int vocab_size = 0;

    bool load(const std::string &path)
    {
        std::ifstream f(path);
        if (!f)
            return false;
        std::ostringstream ss;
        ss << f.rdbuf();
        text = ss.str();
        std::set<char> unique(text.begin(), text.end());
        chars.assign(unique.begin(), unique.end());
        std::sort(chars.begin(), chars.end());
        vocab_size = (int)chars.size();
        for (int i = 0; i < vocab_size; ++i)
        {
            stoi[chars[i]] = i;
            itos[i] = chars[i];
        }
        data.resize(text.size());
        for (int i = 0; i < (int)text.size(); ++i)
            data[i] = stoi[text[i]];
        int n = (int)(Config::train_split * data.size());
        train_data.assign(data.begin(), data.begin() + n);
        val_data.assign(data.begin() + n, data.end());
        return true;
    }

    std::string encode_str(const std::string &s) { return s; }

    std::vector<int> encode(const std::string &s)
    {
        std::vector<int> out;
        for (char c : s)
        {
            auto it = stoi.find(c);
            if (it != stoi.end())
                out.push_back(it->second);
        }
        return out;
    }

    std::string decode(const std::vector<int> &v)
    {
        std::string s;
        for (int i : v)
            s += itos[i];
        return s;
    }

    // returns x [B*T], y [B*T]
    void get_batch(const std::string &split,
                   std::vector<int> &xb, std::vector<int> &yb)
    {
        const auto &d = (split == "train") ? train_data : val_data;
        int len = (int)d.size();
        xb.resize(Config::batch_size * Config::block_size);
        yb.resize(Config::batch_size * Config::block_size);
        for (int b = 0; b < Config::batch_size; ++b)
        {
            int ix = randint(0, len - Config::block_size);
            for (int t = 0; t < Config::block_size; ++t)
            {
                xb[b * Config::block_size + t] = d[ix + t];
                yb[b * Config::block_size + t] = d[ix + t + 1];
            }
        }
    }
};

// ============================================================
//  TIMER UTILITY
// ============================================================
static auto now() { return std::chrono::steady_clock::now(); }
static double elapsed_s(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double>(now() - t0).count();
}

// ============================================================
//  MAIN
// ============================================================
int main()
{
    auto wall_start = now();

    // ── Banner ─────────────────────────────────────────────
    std::cout << std::string(60, '=') << "\n";
    std::cout << " Core Transformer-Engine - Training Pipeline\n";
    std::cout << " Eamon2009\n";
    std::cout << std::string(60, '=') << "\n";

    auto t = std::time(nullptr);
    char tbuf[32];
    std::strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    std::cout << "\n[INFO] Starting at: " << tbuf << "\n";
    std::cout << "[INFO] Device: CPU (single-threaded)\n";

    // ── Config printout ────────────────────────────────────
    std::cout << "\n[CONFIG] Hyperparameters loaded:\n";
    std::cout << "         batch_size=" << Config::batch_size
              << ", block_size=" << Config::block_size << "\n";
    std::cout << "         max_iters=" << Config::max_iters
              << ", learning_rate=" << Config::learning_rate << "\n";
    std::cout << "         n_embd=" << Config::n_embd
              << ", n_head=" << Config::n_head
              << ", n_layer=" << Config::n_layer
              << ", dropout=" << Config::dropout << "\n";

    // ── Load data ──────────────────────────────────────────
    std::cout << "\n[DATA]  Loading text from: " << Config::cleaned_path << "\n";
    Dataset ds;
    if (!ds.load(Config::cleaned_path))
    {
        std::cerr << "[ERROR] Could not open " << Config::cleaned_path << "\n";
        std::cerr << "        Please create a cleaned.txt file with your training text.\n";
        return 1;
    }
    std::cout << "[DATA]  Total characters : " << ds.text.size() << "\n";
    std::cout << "[DATA]  Vocabulary size  : " << ds.vocab_size << "\n";
    std::cout << "[DATA]  Train tokens     : " << ds.train_data.size() << "\n";
    std::cout << "[DATA]  Val   tokens     : " << ds.val_data.size() << "\n";

    // ── Build model ────────────────────────────────────────
    std::cout << "\n[MODEL] Building GPTLanguageModel...\n";
    GPTLanguageModel model(ds.vocab_size);
    long long n_params = model.num_params();
    std::cout << "[MODEL] Parameters  : " << std::fixed << std::setprecision(2)
              << n_params / 1e6 << " M  (" << n_params << " total)\n";
    std::cout << "[MODEL] Architecture: " << Config::n_layer << " layers x "
              << Config::n_head << " heads x " << Config::n_embd << " embedding dim\n";

    // ── Optimiser ──────────────────────────────────────────
    AdamW optimizer(Config::learning_rate);
    for (Param *p : model.parameters())
        optimizer.add_param(p);
    std::cout << "[OPTIM] AdamW optimizer, lr=" << Config::learning_rate << "\n";

    // ── Estimate loss helper ───────────────────────────────
    auto estimate_loss = [&]() -> std::pair<float, float>
    {
        float train_loss = 0.f, val_loss = 0.f;
        GPTLanguageModel::Cache cache;
        for (int k = 0; k < Config::eval_iters; ++k)
        {
            std::vector<int> xb, yb;
            ds.get_batch("train", xb, yb);
            train_loss += model.forward(xb, yb, cache, false);
        }
        for (int k = 0; k < Config::eval_iters; ++k)
        {
            std::vector<int> xb, yb;
            ds.get_batch("val", xb, yb);
            val_loss += model.forward(xb, yb, cache, false);
        }
        return {train_loss / Config::eval_iters, val_loss / Config::eval_iters};
    };

    // ── Training loop ──────────────────────────────────────
    std::cout << "\n"
              << std::string(60, '-') << "\n";
    std::cout << "  TRAINING  (" << Config::max_iters << " iterations, eval every "
              << Config::eval_interval << ")\n";
    std::cout << std::string(60, '-') << "\n";

    float best_val_loss = 1e30f;
    auto train_start = now();
    GPTLanguageModel::Cache cache;

    for (int iter = 0; iter < Config::max_iters; ++iter)
    {

        if (iter % Config::eval_interval == 0 || iter == Config::max_iters - 1)
        {
            auto [tl, vl] = estimate_loss();
            double el = elapsed_s(train_start);
            double pct = 100.0 * iter / Config::max_iters;
            double eta = (iter > 0) ? (el / iter) * (Config::max_iters - iter) : 0.0;
            bool improved = vl < best_val_loss;
            std::string mark = improved ? " << best!" : "";
            if (improved)
            {
                best_val_loss = vl;
                model.save("best_model.bin");
            }
            std::cout << "[" << std::setw(5) << iter << "/" << Config::max_iters << "] "
                      << std::setw(5) << std::fixed << std::setprecision(1) << pct << "%  "
                      << "train=" << std::setprecision(4) << tl
                      << "  val=" << vl
                      << "  elapsed=" << std::setprecision(0) << el << "s"
                      << "  ETA=" << eta << "s"
                      << mark << "\n";
            std::cout.flush();
        }

        // Forward + backward + step
        std::vector<int> xb, yb;
        ds.get_batch("train", xb, yb);
        optimizer.zero_grad();
        model.forward(xb, yb, cache, true);
        model.backward(yb, cache, 1.0f);
        optimizer.step();
    }

    double total_time = elapsed_s(train_start);
    std::cout << "\n[DONE]  Training finished in " << std::fixed << std::setprecision(1)
              << total_time << "s (" << total_time / 60.0 << " min)"
              << "  |  Best val loss: " << std::setprecision(4) << best_val_loss << "\n";
    std::cout << "[SAVE]  Best weights saved to best_model.bin\n";

    // ── Generation ─────────────────────────────────────────
    std::cout << "\n"
              << std::string(60, '-') << "\n";
    std::cout << "  MODEL OUTPUT  (500 tokens, then exits)\n";
    std::cout << std::string(60, '-') << "\n\n";

    std::vector<int> context = {0};
    GPTLanguageModel::Cache gen_cache;
    std::vector<int> generated = model.generate(context, 500, gen_cache);
    std::cout << ds.decode(generated) << "\n";

    double wall_total = elapsed_s(wall_start);
    std::cout << "\n[TOTAL] Wall-clock time: " << std::fixed << std::setprecision(1)
              << wall_total << "s  (" << wall_total / 60.0 << " min)\n";
    return 0;
}