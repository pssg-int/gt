import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_ref_vs_official import _import_official, _build_cfg, _random_graph, _build_models, _run_official, _run_ref, _max_rel_err
from gt3d.models.exphormer_ref import official_to_ref_key
cfg, set_cfg, MultiModel = _import_official()
cfg = _build_cfg(cfg, set_cfg, dropout=0.0)
data, batch = _random_graph(seed=1)
official, ref = _build_models(cfg, MultiModel, dropout=0.0, seed=1)
official.train(); ref.train()
p_off = _run_official(official, batch); p_ref = _run_ref(ref, data)
y = data.y.view(-1)
torch.nn.functional.cross_entropy(p_off, y).backward(); torch.nn.functional.cross_entropy(p_ref, y).backward()
ref_params = dict(ref.named_parameters())
rows = []
for name, p in official.named_parameters():
    tgt = official_to_ref_key(name)
    if "[" in tgt:
        base, row = tgt[:-1].split("["); g_ref = ref_params[base].grad[int(row)].reshape(-1)
    else:
        g_ref = ref_params[tgt].grad
    g_off = p.grad
    rows.append((_max_rel_err(g_ref.reshape(-1), g_off.reshape(-1)), name, g_off.abs().max().item(), g_ref.abs().max().item()))
for r in sorted(rows, reverse=True)[:12]:
    print(f"relerr {r[0]:.3e}  |g_off| {r[2]:.3e}  |g_ref| {r[3]:.3e}  {r[1]}")
