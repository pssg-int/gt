"""Guard against upstream drift: gt3d uses Plexus and AxoNN unmodified, so the symbols it imports and their
signatures are checked here (CPU, no process group).  Run: python tests/test_plexus_api.py"""
import inspect
import os
import subprocess
import sys

PLEXUS_DIR = "/pscratch/sd/c/cunyang/gnn/plexus"
PLEXUS_PINNED = "c3f2660"          # git commit gt3d was validated against (2026-07-20 main)
AXONN_PINNED = "0.2.0"
sys.path.insert(0, PLEXUS_DIR)


def params(fn):
    return list(inspect.signature(fn).parameters)


def main():
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("ok   " if cond else "FAIL ") + msg)
        ok &= bool(cond)

    import axonn
    from axonn import axonn as ax
    check(getattr(axonn, "__version__", AXONN_PINNED) == AXONN_PINNED, f"axonn version {getattr(axonn, '__version__', '?')} == {AXONN_PINNED}")
    check(params(ax.init)[:5] == ["G_inter", "G_data", "G_intra_r", "G_intra_c", "G_intra_d"], "axonn.init(G_inter, G_data, G_intra_r, G_intra_c, G_intra_d, ...)")
    check("enable_internal_timers" in params(ax.init), "axonn.init(enable_internal_timers=)")
    from axonn.communication import communication_handle
    check(hasattr(communication_handle, "get_intra_layer_groups"), "communication_handle.get_intra_layer_groups")
    src = inspect.getsource(communication_handle.get_intra_layer_groups)
    check("reshape(G_intra_d, G_intra_r, G_intra_c)" in src and "[i, j, :]" in src and "[i, :, j]" in src,
          "AxoNN rank layout [G_d, G_r, G_c]; inner = [i, j, :] (column rank varies), outer = [i, :, j]")
    from axonn.timers import Timers
    check(all(hasattr(Timers, m) for m in ("start", "stop", "get_times")), "axonn.timers.Timers.start/stop/get_times")

    from plexus.utils.general import get_process_groups_info, pad_dimension, print_axonn_timer_data
    check(params(get_process_groups_info) == ["groups"], "plexus.utils.general.get_process_groups_info(groups)")
    check(pad_dimension(10, 4) == 12 and pad_dimension(8, 4) == 8, "pad_dimension semantics")
    check(params(print_axonn_timer_data) == ["times"], "print_axonn_timer_data(times)")
    from plexus import gcn_conv_rs as rs
    check(params(rs._all_gather_rows) == ["x", "group", "wire_dtype", "async_op"], "gcn_conv_rs._all_gather_rows(x, group, wire_dtype, async_op)")
    check(params(rs._reduce_scatter_rows) == ["x", "group", "wire_dtype"], "gcn_conv_rs._reduce_scatter_rows(x, group, wire_dtype)")
    check(params(rs._ag_rows_uneven) == ["x", "group", "recv_counts", "wire_dtype"], "gcn_conv_rs._ag_rows_uneven(x, group, recv_counts, wire_dtype)")
    check(params(rs._rs_rows_uneven) == ["x", "group", "chunk_counts", "wire_dtype"], "gcn_conv_rs._rs_rows_uneven(x, group, chunk_counts, wire_dtype)")
    check(rs._splits(10, 4) == [3, 3, 2, 2], "gcn_conv_rs._splits")
    from plexus.scattered import ScatteredRMSNorm
    check(params(ScatteredRMSNorm.__init__) == ["self", "size", "eps"], "scattered.ScatteredRMSNorm(size, eps)")
    try:
        head = subprocess.check_output(["git", "-C", PLEXUS_DIR, "rev-parse", "--short", "HEAD"], text=True).strip()
        check(head.startswith(PLEXUS_PINNED), f"plexus HEAD {head} == pinned {PLEXUS_PINNED} (update the pin after re-validating)")
    except Exception as e:  # noqa: BLE001
        print("warn: could not read plexus git HEAD:", e)
    print("[plexus-api] ALL PASSED" if ok else "[plexus-api] FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
