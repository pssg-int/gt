"""Stage-by-stage comparison of ExphormerRef against the official model (eval mode, fp64)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_ref_vs_official import (_import_official, _build_cfg, _random_graph, _build_models,  # noqa: E402
                                  _max_rel_err, NUM_NODES)


def main():
    cfg, set_cfg, MultiModel = _import_official()
    cfg = _build_cfg(cfg, set_cfg, dropout=0.3)
    data, batch = _random_graph()
    official, ref = _build_models(cfg, MultiModel, dropout=0.3)
    official.eval(); ref.eval()
    print(official)
    b = batch.clone()
    with torch.no_grad():
        # --- official, stage by stage ---
        b = official.encoder(b)
        print("official expander_edge_index", tuple(b.expander_edge_index.shape),
              "edge_attr", tuple(b.edge_attr.shape), "expander_edge_attr", tuple(b.expander_edge_attr.shape))
        b = official.pre_mp(b)
        x_off = b.x.clone()
        # --- ref up to pre_mp ---
        h = ref.pre_mp(data.x)
        print("after pre_mp: rel err", _max_rel_err(h, x_off))
        attn_ei, etype = ref.build_attention_edges(data.edge_index, data.expander_edges.t())
        print("ref attention edges", tuple(attn_ei.shape), "official", tuple(b.expander_edge_index.shape),
              "equal:", torch.equal(attn_ei, b.expander_edge_index))
        for i, layer in enumerate(official.layers):
            # official sub-blocks
            h_local_off = layer.models[0](b)
            h_attn_off = layer.models[1](b)
            # ref sub-blocks on the SAME input as the official layer (b.x)
            xin = b.x.clone()
            rl = ref.layers[i]
            gl = rl.gcn(xin, data.edge_index)
            gl = rl.bn_local(xin + gl)
            at = rl.attention(xin, attn_ei, etype, ref.edge_type_emb.weight)
            at = rl.bn_attn(xin + at)
            print(f"layer {i}: local rel err {_max_rel_err(gl, h_local_off):.2e}  attn rel err {_max_rel_err(at, h_attn_off):.2e}")
            # also raw GCNConv output vs PyG
            raw_off = layer.models[0].local_model(xin, b.edge_index)
            raw_ref = rl.gcn(xin, data.edge_index)
            print(f"          raw GCNConv rel err {_max_rel_err(raw_ref, raw_off):.2e}")
            raw_attn_off = layer.models[1].self_attn(b)
            raw_attn_ref = rl.attention(xin, attn_ei, etype, ref.edge_type_emb.weight)
            print(f"          raw attention rel err {_max_rel_err(raw_attn_ref, raw_attn_off):.2e}")
            b = layer(b)
            hr = rl(xin, data.edge_index, attn_ei, etype, ref.edge_type_emb.weight)
            print(f"          full layer rel err {_max_rel_err(hr, b.x):.2e}")
        h_final = b.x.clone()
        pred, _ = official.post_mp(b)
        print("head rel err", _max_rel_err(ref.head(h_final), pred))
        # full forward of both models (what the unit test compares)
        p_ref = ref(data.x, data.edge_index, data.expander_edges.t())
        b2 = batch.clone()
        p_off, _ = official(b2)
        print("full forward rel err", _max_rel_err(p_ref, p_off), "| ref vs staged official pred", _max_rel_err(p_ref, pred),
              "| official full vs staged", _max_rel_err(p_off, pred))
        print("ref.training", ref.training, "official.training", official.training)


if __name__ == "__main__":
    main()
