"""ctypes binding of libpullcomm.so (NVSHMEM receiver-driven sparse row collectives; build with build.sh).

    pullcomm.init(group)                       # once per process; every rank of the group becomes an NVSHMEM PE
    x = pullcomm.sym_empty((rows, F), dtype)   # symmetric buffer (collective: same size on every PE)
    pullcomm.gather(out, x, src_pe, src_row)   # out[k] = x_on_PE[src_pe[k]][src_row[k]]
    pullcomm.reduce(out, part, row_ptr, ent_pe, ent_idx)   # out[i] = sum over the row's entries of part_on_PE[pe][idx]
    pullcomm.barrier()                         # all PEs, on the current stream
"""
import ctypes
import os

import torch
import torch.distributed as dist

_lib = None
_initialized = False


def lib():
    global _lib
    if _lib is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libpullcomm.so")
        L = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        c = ctypes
        L.pull_uid_size.restype = c.c_int
        L.pull_get_uid.argtypes = [c.c_void_p]; L.pull_get_uid.restype = c.c_int
        L.pull_init.argtypes = [c.c_void_p, c.c_int, c.c_int]; L.pull_init.restype = c.c_int
        L.pull_my_pe.restype = c.c_int; L.pull_n_pes.restype = c.c_int
        L.pull_malloc.argtypes = [c.c_size_t]; L.pull_malloc.restype = c.c_void_p
        L.pull_free.argtypes = [c.c_void_p]
        L.pull_barrier.argtypes = [c.c_void_p]
        L.pull_gather.argtypes = [c.c_void_p, c.c_void_p, c.c_void_p, c.c_void_p, c.c_long, c.c_int, c.c_long, c.c_long, c.c_int, c.c_int, c.c_void_p]
        L.pull_gather.restype = c.c_int
        L.pull_reduce_bf16.argtypes = [c.c_void_p, c.c_void_p, c.c_void_p, c.c_void_p, c.c_void_p, c.c_long, c.c_int, c.c_long, c.c_long, c.c_void_p]
        L.pull_reduce_bf16.restype = c.c_int
        _lib = L
    return _lib


def init(group=None):
    """Bootstrap NVSHMEM over the ranks of `group` (default: the world) with a unique id broadcast by torch.distributed."""
    global _initialized
    if _initialized:
        return
    L = lib()
    n = L.pull_uid_size()
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    obj = [None]
    if rank == 0:
        buf = ctypes.create_string_buffer(n)
        assert L.pull_get_uid(buf) == 0, "nvshmemx_get_uniqueid failed"
        obj = [bytes(buf.raw)]
    src = dist.get_global_rank(group, 0) if group is not None else 0
    dist.broadcast_object_list(obj, src=src, group=group)
    uid = ctypes.create_string_buffer(obj[0], n)
    assert L.pull_init(uid, rank, world) == 0, "nvshmemx_init_attr failed"
    assert L.pull_my_pe() == rank and L.pull_n_pes() == world, (L.pull_my_pe(), L.pull_n_pes(), rank, world)
    _initialized = True


class _CAI:
    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 3, "strides": None}


def sym_empty(shape, dtype):
    """A torch tensor over an NVSHMEM symmetric allocation (collective call; same shape on every PE).  The tensor keeps
    the allocation alive through its base object; free() releases it."""
    L = lib()
    numel = 1
    for d in shape:
        numel *= int(d)
    nbytes = numel * torch.empty(0, dtype=dtype).element_size()
    ptr = L.pull_malloc(max(nbytes, 1))
    assert ptr, "nvshmem_malloc failed (raise NVSHMEM_SYMMETRIC_SIZE?)"
    base = _CAI(ptr, max(nbytes, 1))
    t = torch.as_tensor(base, device="cuda")                    # uint8 view sharing the symmetric memory
    t = t.view(dtype).view(*shape)
    t._pull_ptr = ptr
    return t


def free(t):
    lib().pull_free(ctypes.c_void_p(t._pull_ptr))


def barrier(stream=None):
    s = torch.cuda.current_stream() if stream is None else stream
    lib().pull_barrier(ctypes.c_void_p(s.cuda_stream))


def gather(out, sym_src, src_pe, src_row, nbi=False, stream=None):
    """out [n, F] (local) <- rows src_row[k] of PE src_pe[k]'s copy of sym_src [rows, F] (symmetric)."""
    n, F = out.shape
    es = out.element_size()
    s = torch.cuda.current_stream() if stream is None else stream
    G = dist.get_world_size()
    r = lib().pull_gather(ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(sym_src.data_ptr()), ctypes.c_void_p(src_pe.data_ptr()),
                          ctypes.c_void_p(src_row.data_ptr()), n, F * es, sym_src.stride(0) * es, out.stride(0) * es, G, int(nbi),
                          ctypes.c_void_p(s.cuda_stream))
    assert r == 0, r
    return out


def reduce(out, sym_part, row_ptr, ent_pe, ent_idx, stream=None):
    """out [n_rows, F] bf16 <- for every row i, the sum over entries e in [row_ptr[i], row_ptr[i+1]) of
    PE ent_pe[e]'s sym_part[ent_idx[e]] (bf16, symmetric)."""
    n_rows, F = out.shape
    assert out.dtype == torch.bfloat16 and sym_part.dtype == torch.bfloat16
    s = torch.cuda.current_stream() if stream is None else stream
    r = lib().pull_reduce_bf16(ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(sym_part.data_ptr()), ctypes.c_void_p(row_ptr.data_ptr()),
                               ctypes.c_void_p(ent_pe.data_ptr()), ctypes.c_void_p(ent_idx.data_ptr()), n_rows, F,
                               sym_part.stride(0), out.stride(0), ctypes.c_void_p(s.cuda_stream))
    assert r == 0, r
    return out
