#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <algorithm>

__global__ void ddt_batch(const uint32_t* vectors, const uint32_t* outputs,
                          const uint32_t* add, const uint32_t* sub,
                          const uint32_t* places, const uint32_t* a_indices,
                          uint32_t* hist, uint32_t q, uint32_t s,
                          uint32_t n, uint32_t batch_size) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t row = blockIdx.y;
    if (x >= n || row >= batch_size) return;
    const size_t xoff = size_t(x) * s;
    const size_t aoff = size_t(a_indices[row]) * s;
    uint32_t xa = 0;
    for (uint32_t i = 0; i < s; ++i) {
        const uint32_t xi = vectors[xoff + i], ai = vectors[aoff + i];
        xa += add[size_t(ai) * q + xi] * places[i];
    }
    const size_t xaoff = size_t(xa) * s;
    uint32_t b = 0;
    for (uint32_t j = 0; j < s; ++j) {
        const uint32_t fx = outputs[xoff + j], fxa = outputs[xaoff + j];
        b += sub[size_t(fxa) * q + fx] * places[j];
    }
    atomicAdd(&hist[size_t(row) * n + b], 1u);
}

struct Partial { uint32_t maximum; uint32_t index; unsigned long long sum; };
struct RowResult { uint32_t maximum; uint32_t index; unsigned long long sum; };

// Stage 1: 256 threads reduce 1024 consecutive histogram bins.
__global__ void histogram_partials(const uint32_t* hist, Partial* partials,
                                    uint32_t n, uint32_t parts_per_row) {
    const uint32_t row = blockIdx.y;
    const uint32_t part = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t first = part * 1024u;
    uint32_t best = 0, best_index = UINT32_MAX;
    unsigned long long total = 0;
    for (uint32_t k = tid; k < 1024u; k += 256u) {
        const uint32_t idx = first + k;
        if (idx < n) {
            const uint32_t value = hist[size_t(row) * n + idx];
            total += value;
            if (value > best || (value == best && idx < best_index)) {
                best = value;
                best_index = idx;
            }
        }
    }
    __shared__ uint32_t max_buf[256], index_buf[256];
    __shared__ unsigned long long sum_buf[256];
    max_buf[tid] = best;
    index_buf[tid] = best_index;
    sum_buf[tid] = total;
    __syncthreads();
    for (uint32_t stride = 128; stride; stride >>= 1) {
        if (tid < stride) {
            const uint32_t other = tid + stride;
            sum_buf[tid] += sum_buf[other];
            if (max_buf[other] > max_buf[tid] ||
                (max_buf[other] == max_buf[tid] && index_buf[other] < index_buf[tid])) {
                max_buf[tid] = max_buf[other];
                index_buf[tid] = index_buf[other];
            }
        }
        __syncthreads();
    }
    if (tid == 0) partials[size_t(row) * parts_per_row + part] =
        {max_buf[0], index_buf[0], sum_buf[0]};
}

// Stage 2: one block per row, reducing all stage-1 results.
__global__ void histogram_row_results(const Partial* partials, RowResult* results,
                                       uint32_t parts_per_row) {
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    uint32_t best = 0, best_index = UINT32_MAX;
    unsigned long long total = 0;
    for (uint32_t i = tid; i < parts_per_row; i += 256u) {
        const Partial p = partials[size_t(row) * parts_per_row + i];
        total += p.sum;
        if (p.maximum > best || (p.maximum == best && p.index < best_index)) {
            best = p.maximum;
            best_index = p.index;
        }
    }
    __shared__ uint32_t max_buf[256], index_buf[256];
    __shared__ unsigned long long sum_buf[256];
    max_buf[tid] = best;
    index_buf[tid] = best_index;
    sum_buf[tid] = total;
    __syncthreads();
    for (uint32_t stride = 128; stride; stride >>= 1) {
        if (tid < stride) {
            const uint32_t other = tid + stride;
            sum_buf[tid] += sum_buf[other];
            if (max_buf[other] > max_buf[tid] ||
                (max_buf[other] == max_buf[tid] && index_buf[other] < index_buf[tid])) {
                max_buf[tid] = max_buf[other];
                index_buf[tid] = index_buf[other];
            }
        }
        __syncthreads();
    }
    if (tid == 0) results[row] = {max_buf[0], index_buf[0], sum_buf[0]};
}

struct Context {
    uint32_t *vectors{}, *outputs{}, *add{}, *sub{}, *places{}, *a_indices{}, *hist{};
    Partial* partials{};
    RowResult* results{};
    uint32_t q{}, s{}, n{}, max_batch{}, parts_per_row{};
};

static thread_local char last_error[512] = "";
static int fail(cudaError_t e, const char* step) {
    if (e == cudaSuccess) return 0;
    std::snprintf(last_error, sizeof(last_error), "%s: %s", step, cudaGetErrorString(e));
    return 1;
}
extern "C" const char* cuda_ddt_last_error() { return last_error; }
extern "C" void cuda_ddt_destroy(Context* c) {
    if (!c) return;
    cudaFree(c->vectors); cudaFree(c->outputs); cudaFree(c->add);
    cudaFree(c->sub); cudaFree(c->places); cudaFree(c->a_indices);
    cudaFree(c->hist); cudaFree(c->partials); cudaFree(c->results);
    delete c;
}
extern "C" Context* cuda_ddt_create(uint32_t q, uint32_t s, uint32_t n,
    uint32_t max_batch, const uint32_t* vectors, const uint32_t* outputs,
    const uint32_t* add, const uint32_t* sub, const uint32_t* places) {
    last_error[0] = 0;
    if (!q || !s || !n || !max_batch) {
        std::snprintf(last_error, sizeof(last_error), "Invalid CUDA dimensions"); return nullptr;
    }
    Context* c = new Context();
    c->q=q; c->s=s; c->n=n; c->max_batch=max_batch;
    c->parts_per_row = uint32_t((uint64_t(n) + 1023u) / 1024u);
    auto upload = [&](uint32_t** dst, const uint32_t* src, size_t count) -> bool {
        if (fail(cudaMalloc(reinterpret_cast<void**>(dst), count*sizeof(uint32_t)), "cudaMalloc")) return false;
        return !fail(cudaMemcpy(*dst, src, count*sizeof(uint32_t), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    };
    bool ok = upload(&c->vectors, vectors, size_t(n)*s)
           && upload(&c->outputs, outputs, size_t(n)*s)
           && upload(&c->add, add, size_t(q)*q)
           && upload(&c->sub, sub, size_t(q)*q)
           && upload(&c->places, places, s);
    if (ok) ok = !fail(cudaMalloc(reinterpret_cast<void**>(&c->a_indices), size_t(max_batch)*sizeof(uint32_t)), "allocate a indices");
    if (ok) ok = !fail(cudaMalloc(reinterpret_cast<void**>(&c->hist), size_t(max_batch)*n*sizeof(uint32_t)), "allocate histograms");
    if (ok) ok = !fail(cudaMalloc(reinterpret_cast<void**>(&c->partials), size_t(max_batch)*c->parts_per_row*sizeof(Partial)), "allocate partials");
    if (ok) ok = !fail(cudaMalloc(reinterpret_cast<void**>(&c->results), size_t(max_batch)*sizeof(RowResult)), "allocate row results");
    if (!ok) { cuda_ddt_destroy(c); return nullptr; }
    return c;
}

extern "C" int cuda_ddt_run_reduced(Context* c, const uint32_t* a_indices,
                                      uint32_t batch_size, RowResult* output) {
    if (!c || !a_indices || !output || !batch_size || batch_size > c->max_batch) {
        std::snprintf(last_error, sizeof(last_error), "Invalid batch arguments"); return 1;
    }
    const size_t a_bytes = size_t(batch_size)*sizeof(uint32_t);
    const size_t hist_bytes = size_t(batch_size)*c->n*sizeof(uint32_t);
    if (fail(cudaMemcpy(c->a_indices, a_indices, a_bytes, cudaMemcpyHostToDevice), "copy a indices")) return 1;
    if (fail(cudaMemset(c->hist, 0, hist_bytes), "clear histogram")) return 1;
    const dim3 blocks((uint64_t(c->n) + 255u)/256u, batch_size);
    ddt_batch<<<blocks,256>>>(c->vectors,c->outputs,c->add,c->sub,c->places,
                              c->a_indices,c->hist,c->q,c->s,c->n,batch_size);
    if (fail(cudaGetLastError(), "DDT kernel launch")) return 1;
    histogram_partials<<<dim3(c->parts_per_row, batch_size),256>>>(c->hist,c->partials,c->n,c->parts_per_row);
    if (fail(cudaGetLastError(), "partial reduction launch")) return 1;
    histogram_row_results<<<batch_size,256>>>(c->partials,c->results,c->parts_per_row);
    if (fail(cudaGetLastError(), "row reduction launch")) return 1;
    if (fail(cudaMemcpy(output, c->results, size_t(batch_size)*sizeof(RowResult), cudaMemcpyDeviceToHost), "copy reduced results D2H")) return 1;
    return 0;
}

