import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_ref_vs_official import _import_official, _build_cfg, _random_graph, _build_models, _max_rel_err
cfg, set_cfg, MultiModel = _import_official()
cfg = _build_cfg(cfg, set_cfg, dropout=0.3)
data, batch = _random_graph()
official, ref = _build_models(cfg, MultiModel, dropout=0.3)
official.eval(); ref.eval()
with torch.no_grad():
    b = batch.clone()
    print("batch.x vs data.x:", _max_rel_err(b.x, data.x), b.x.dtype, data.x.dtype)
    b = official.encoder(b)
    print("after encoder x vs data.x:", _max_rel_err(b.x, data.x))
    W_off = official.pre_mp.Layer_0.layer.model.weight
    print("W shapes", tuple(W_off.shape), tuple(ref.pre_lin.weight.shape), "W diff", _max_rel_err(ref.pre_lin.weight, W_off))
    lin_off = official.pre_mp.Layer_0.layer(b.x)
    print("official Linear module output type:", type(lin_off))
    lin_off_t = official.pre_mp.Layer_0.layer.model(b.x)
    print("linear rel err:", _max_rel_err(ref.pre_lin(data.x), lin_off_t))
    bn_off = official.pre_mp.Layer_0.post_layer[0]
    print("BN params diff:", _max_rel_err(ref.pre_bn.weight, bn_off.weight), _max_rel_err(ref.pre_bn.bias, bn_off.bias),
          "rm", (ref.pre_bn.running_mean - bn_off.running_mean).abs().max().item(),
          "rv", (ref.pre_bn.running_var - bn_off.running_var).abs().max().item(),
          "eps", ref.pre_bn.eps, bn_off.eps, "training", ref.pre_bn.training, bn_off.training)
    y_off = official.pre_mp.Layer_0.post_layer(lin_off_t)
    y_ref = torch.relu(ref.pre_bn(ref.pre_lin(data.x)))
    print("post_layer rel err:", _max_rel_err(y_ref, y_off))
    full_off = official.pre_mp(b).x
    print("pre_mp full rel err:", _max_rel_err(y_ref, full_off))
    print(official.pre_mp.Layer_0)
    print("GeneralLayer forward src:")
    import inspect; print(inspect.getsource(type(official.pre_mp.Layer_0).forward))
