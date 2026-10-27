// Receiver-driven ("pull") sparse row collectives on NVSHMEM: GPU-initiated gets by row-index table, and a sparse
// reduce whose summation rides in the transfer.  Intra-node PEs are served through NVSHMEM's P2P mapping, remote PEs
// through the libfabric (Slingshot) transport, from the same kernels.  C ABI for ctypes (gt3d/dist/pullcomm/__init__.py).
//
//   pull_gather : out[k, :] = SRC_pe[k][row[k], :]  for k < n        (SRC = the symmetric chunk buffer of every PE)
//   pull_reduce : out[i, :] = sum_{e in row_ptr[i]..row_ptr[i+1]} PART_pe[e][idx[e], :]   (PART = symmetric partials)
// One warp per row; rows of the compact buffer are grouped by source PE, so the warp -> row map is round-robin over G
// slices of the row range to keep every link busy (RESULTS 2026-10-27: one PE at a time = one NVLink pair).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cstdio>
#include <cstring>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "[pullcomm] CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); return -1; } } while (0)

extern "C" {

int pull_uid_size() { return (int)sizeof(nvshmemx_uniqueid_t); }

int pull_get_uid(void* buf) {
    nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;
    if (nvshmemx_get_uniqueid(&id) != 0) return -1;
    memcpy(buf, &id, sizeof(id));
    return 0;
}

int pull_init(const void* uid, int rank, int nranks) {
    nvshmemx_uniqueid_t id;
    memcpy(&id, uid, sizeof(id));
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    nvshmemx_set_attr_uniqueid_args(rank, nranks, &id, &attr);
    if (nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr) != 0) return -1;
    return 0;
}

int pull_my_pe() { return nvshmem_my_pe(); }
int pull_n_pes() { return nvshmem_n_pes(); }
void* pull_malloc(size_t bytes) { return nvshmem_malloc(bytes); }
void pull_free(void* p) { nvshmem_free(p); }
void pull_barrier(cudaStream_t s) { nvshmemx_barrier_all_on_stream(s); }
void pull_finalize() { nvshmem_finalize(); }

}  // extern "C"

// ---------------------------------------------------------------------------------------------------------------
// gather: warp w handles compact row k(w); round-robin over G slices of the row-block range
// ---------------------------------------------------------------------------------------------------------------
__global__ void pull_gather_kernel(char* out, const char* sym_src, const int* src_pe, const int* src_row, long n,
                                   int row_bytes, long src_stride, long out_stride, int G, int nbi) {
    int warps_per_block = blockDim.x >> 5;
    long gw = (long)blockIdx.x * warps_per_block + (threadIdx.x >> 5);   // global warp id
    long n_warps = (long)gridDim.x * warps_per_block;
    long per = (n_warps + G - 1) / G;
    long k = (gw % G) * per + gw / G;                                    // interleave the G source slices
    if (k >= n) return;
    int pe = src_pe[k];
    const char* src = sym_src + (long)src_row[k] * src_stride;
    char* dst = out + k * out_stride;
    if (nbi) nvshmemx_getmem_nbi_warp(dst, src, row_bytes, pe);
    else     nvshmemx_getmem_warp(dst, src, row_bytes, pe);
}

__global__ void quiet_kernel() { if (threadIdx.x == 0) nvshmem_quiet(); }

extern "C" int pull_gather(void* out, const void* sym_src, const int* src_pe, const int* src_row, long n, int row_bytes,
                           long src_stride, long out_stride, int G, int nbi, cudaStream_t s) {
    const int WARPS = 8;
    long blocks = (n + WARPS - 1) / WARPS;
    pull_gather_kernel<<<(unsigned)blocks, WARPS * 32, 0, s>>>((char*)out, (const char*)sym_src, src_pe, src_row, n, row_bytes,
                                                              src_stride, out_stride, G, nbi);
    CHECK(cudaGetLastError());
    if (nbi) { quiet_kernel<<<1, 32, 0, s>>>(); CHECK(cudaGetLastError()); }
    return 0;
}

// ---------------------------------------------------------------------------------------------------------------
// reduce: warp per owned row i; entries e in [row_ptr[i], row_ptr[i+1]) give (pe, idx) of the partials to sum.
// Each entry: warp-level blocking get of one row into a per-warp staging buffer (shared memory), then the lanes add it
// to their fp32 accumulators.  out is bf16.
// ---------------------------------------------------------------------------------------------------------------
__global__ void pull_reduce_bf16_kernel(__nv_bfloat16* out, const __nv_bfloat16* sym_part, const int* row_ptr, const int* ent_pe,
                                        const int* ent_idx, long n_rows, int F, long part_stride, long out_stride) {
    extern __shared__ __nv_bfloat16 stage[];                     // [warps][F]
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;
    long i = (long)blockIdx.x * warps_per_block + warp;
    if (i >= n_rows) return;
    __nv_bfloat16* st = stage + (long)warp * F;
    // lanes own columns lane, lane+32, ... ; up to 16 per lane (F <= 512)
    float acc[16];
    #pragma unroll
    for (int q = 0; q < 16; ++q) acc[q] = 0.f;
    int e0 = row_ptr[i], e1 = row_ptr[i + 1];
    for (int e = e0; e < e1; ++e) {
        int pe = ent_pe[e];
        const __nv_bfloat16* src = sym_part + (long)ent_idx[e] * part_stride;
        nvshmemx_getmem_warp(st, src, (size_t)F * 2, pe);
        __syncwarp();
        #pragma unroll
        for (int q = 0; q < 16; ++q) { int c = lane + 32 * q; if (c < F) acc[q] += __bfloat162float(st[c]); }
        __syncwarp();
    }
    __nv_bfloat16* o = out + i * out_stride;
    #pragma unroll
    for (int q = 0; q < 16; ++q) { int c = lane + 32 * q; if (c < F) o[c] = __float2bfloat16(acc[q]); }
}

extern "C" int pull_reduce_bf16(void* out, const void* sym_part, const int* row_ptr, const int* ent_pe, const int* ent_idx,
                                long n_rows, int F, long part_stride, long out_stride, cudaStream_t s) {
    if (F > 512) return -2;
    const int WARPS = 8;
    long blocks = (n_rows + WARPS - 1) / WARPS;
    size_t shm = (size_t)WARPS * F * 2;
    pull_reduce_bf16_kernel<<<(unsigned)blocks, WARPS * 32, shm, s>>>((__nv_bfloat16*)out, (const __nv_bfloat16*)sym_part, row_ptr,
                                                                      ent_pe, ent_idx, n_rows, F, part_stride, out_stride);
    CHECK(cudaGetLastError());
    return 0;
}

// ---------------------------------------------------------------------------------------------------------------
// diagnostics: one contiguous block from one PE, host-initiated (on stream) and device-initiated (one thread)
// ---------------------------------------------------------------------------------------------------------------
__global__ void get_block_kernel(char* dst, const char* src, size_t bytes, int pe) {
    if (threadIdx.x == 0 && blockIdx.x == 0) nvshmem_getmem(dst, src, bytes, pe);
}
extern "C" int pull_get_block_host(void* dst, const void* src, size_t bytes, int pe, cudaStream_t s) {
    nvshmemx_getmem_on_stream(dst, src, bytes, pe, s);
    return 0;
}
extern "C" int pull_get_block_device(void* dst, const void* src, size_t bytes, int pe, cudaStream_t s) {
    get_block_kernel<<<1, 32, 0, s>>>((char*)dst, (const char*)src, bytes, pe);
    CHECK(cudaGetLastError());
    return 0;
}
