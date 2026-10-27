"""Numerical test: gt3d.models.ExphormerRef vs the official graphgps Exphormer model.

Builds the official `MultiModel` (GCN+Exphormer, arxiv.yaml hyper-parameters) on a
small random graph, copies its parameters into ExphormerRef and compares
(a) eval-mode forward outputs in fp64, (b) train-mode outputs, gradients and
BatchNorm running statistics with dropout disabled.

Run:  python tests/test_ref_vs_official.py   (also works under pytest)
Note: importing graphgps is slow on NERSC login nodes (several minutes).
"""
import os
import sys
import time

import numpy as np
import torch

EXPHORMER_DIR = os.environ.get("EXPHORMER_DIR", "/pscratch/sd/c/cunyang/gnn/baseline/Exphormer")
GT3D_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (EXPHORMER_DIR, GT3D_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

NUM_NODES, DIM_IN, DIM_OUT, NUM_DIRECTED_EDGES = 257, 128, 40, 2000


def _import_official():
    t0 = time.time()
    import graphgps  # noqa: F401  (registers custom modules; slow)
    from torch_geometric.graphgym.config import cfg, set_cfg
    from graphgps.network.multi_model import MultiModel
    print(f"[test] graphgps imported in {time.time() - t0:.1f}s", flush=True)
    return cfg, set_cfg, MultiModel


def _build_cfg(cfg, set_cfg, dropout):
    set_cfg(cfg)
    cfg.merge_from_file(os.path.join(EXPHORMER_DIR, "configs/Exphormer/arxiv.yaml"))
    cfg.gt.dropout = dropout
    cfg.gnn.dropout = dropout
    cfg.gt.attn_dropout = 0.0
    cfg.share.dim_in = DIM_IN
    cfg.share.dim_out = DIM_OUT
    cfg.accelerator = "cpu"
    cfg.device = "cpu"
    cfg.wandb.use = False
    return cfg


def _random_graph(seed=0):
    """Random graph preprocessed the way preformat_ogbn does for arxiv:
    add_reverse_edges -> add_self_loops (arxiv.yaml) -> random expander (degree 3)."""
    from torch_geometric.data import Data, Batch
    from graphgps.transform.dist_transforms import add_reverse_edges, add_self_loops
    from graphgps.transform.expander_edges import generate_random_expander

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(NUM_NODES, DIM_IN, generator=g, dtype=torch.float64)
    y = torch.randint(0, DIM_OUT, (NUM_NODES, 1), generator=g)
    ei = torch.randint(0, NUM_NODES, (2, NUM_DIRECTED_EDGES), generator=g)
    data = Data(x=x, y=y, edge_index=ei, num_nodes=NUM_NODES)
    data = add_reverse_edges(data)
    data = add_self_loops(data)
    data = generate_random_expander(data, degree=3, algorithm="Random-d",
                                    rng=np.random.default_rng(seed), max_num_iters=3)
    batch = Batch.from_data_list([data])
    return data, batch


def _build_models(cfg, MultiModel, dropout, seed=0):
    from gt3d.models.exphormer_ref import ExphormerRef, load_official_state_dict
    torch.manual_seed(seed)
    official = MultiModel(DIM_IN, DIM_OUT).double()
    ref = ExphormerRef(DIM_IN, cfg.gt.dim_hidden, DIM_OUT, cfg.gt.layers, cfg.gt.n_heads,
                       dropout=dropout, gnn_dropout=dropout).double()
    load_official_state_dict(ref, official.state_dict(), strict=True)
    return official, ref


def _run_official(official, batch):
    b = batch.clone()  # ExpanderEdgeFixer deletes batch.expander_edges in place
    pred, _ = official(b)
    return pred


def _run_ref(ref, data):
    return ref(data.x, data.edge_index, data.expander_edges.t())


def _max_rel_err(a, b, floor=1e-30):
    """max|a-b| / max(max|b|, floor).  For gradients use floor=1e-8: biases that feed a
    BatchNorm in training mode have an exactly-zero gradient, so their fp64 round-off
    (~1e-18) would otherwise dominate the relative error."""
    return ((a - b).abs().max() / b.abs().max().clamp_min(floor)).item()


def test_eval_forward():
    cfg, set_cfg, MultiModel = _import_official()
    cfg = _build_cfg(cfg, set_cfg, dropout=0.3)
    data, batch = _random_graph()
    official, ref = _build_models(cfg, MultiModel, dropout=0.3)
    official.eval(); ref.eval()
    with torch.no_grad():
        p_off = _run_official(official, batch)
        p_ref = _run_ref(ref, data)
    err = _max_rel_err(p_ref, p_off)
    print(f"[test] eval forward: max rel err = {err:.3e}", flush=True)
    assert p_ref.shape == p_off.shape == (NUM_NODES, DIM_OUT)
    assert err < 1e-10, err


def test_train_forward_backward():
    cfg, set_cfg, MultiModel = _import_official()
    cfg = _build_cfg(cfg, set_cfg, dropout=0.0)
    data, batch = _random_graph(seed=1)
    official, ref = _build_models(cfg, MultiModel, dropout=0.0, seed=1)
    official.train(); ref.train()

    p_off = _run_official(official, batch)
    p_ref = _run_ref(ref, data)
    err = _max_rel_err(p_ref, p_off)
    print(f"[test] train forward: max rel err = {err:.3e}", flush=True)
    assert err < 1e-10, err

    y = data.y.view(-1)
    torch.nn.functional.cross_entropy(p_off, y).backward()
    torch.nn.functional.cross_entropy(p_ref, y).backward()

    from gt3d.models.exphormer_ref import official_to_ref_key
    ref_params = dict(ref.named_parameters())
    worst = 0.0
    n_checked = 0
    for name, p in official.named_parameters():
        tgt = official_to_ref_key(name)
        assert tgt is not None, name
        if "[" in tgt:  # edge-type embedding rows
            base, row = tgt[:-1].split("[")
            g_ref = ref_params[base].grad[int(row)].reshape(-1)
        else:
            g_ref = ref_params[tgt].grad
        assert p.grad is not None and g_ref is not None, name
        e = _max_rel_err(g_ref.reshape(-1), p.grad.reshape(-1), floor=1e-8)
        worst = max(worst, e)
        n_checked += 1
    print(f"[test] gradients: {n_checked} params, worst rel err = {worst:.3e}", flush=True)
    assert worst < 1e-8, worst

    # BatchNorm running statistics after one training forward
    off_sd, ref_sd = official.state_dict(), ref.state_dict()
    worst_bn = 0.0
    for name, val in off_sd.items():
        if name.endswith("running_mean") or name.endswith("running_var"):
            tgt = official_to_ref_key(name)
            worst_bn = max(worst_bn, _max_rel_err(ref_sd[tgt], val))
    print(f"[test] BN running stats: worst rel err = {worst_bn:.3e}", flush=True)
    assert worst_bn < 1e-10, worst_bn


if __name__ == "__main__":
    test_eval_forward()
    test_train_forward_backward()
    print("[test] ALL PASSED")
