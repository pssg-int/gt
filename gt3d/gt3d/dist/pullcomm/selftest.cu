// Standalone NVSHMEM init check (no torch): uid bootstrap with nranks = 1, a symmetric malloc, a barrier.
#include <cstdio>
#include <nvshmem.h>
#include <nvshmemx.h>
int main() {
    cudaSetDevice(0);
    nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;
    nvshmemx_get_uniqueid(&id);
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    nvshmemx_set_attr_uniqueid_args(0, 1, &id, &attr);
    printf("init...\n"); fflush(stdout);
    int r = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
    printf("init rc=%d pe=%d/%d\n", r, nvshmem_my_pe(), nvshmem_n_pes()); fflush(stdout);
    void* p = nvshmem_malloc(1 << 20);
    printf("malloc %p\n", p); fflush(stdout);
    nvshmem_barrier_all();
    nvshmem_free(p);
    nvshmem_finalize();
    printf("ok\n");
    return 0;
}
