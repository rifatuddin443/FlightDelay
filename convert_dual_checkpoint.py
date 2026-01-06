import torch, glob, os
paths = glob.glob('kan_gat_dp_three_stage_sigma*.pth') or glob.glob('kan_gat_dp_three_stage_*.pth')
if not paths:
    raise SystemExit('No model files found')
path = max(paths, key=os.path.getmtime)
ckpt = torch.load(path, map_location='cpu', weights_only=False)
if not all(k in ckpt for k in ('encoder','classifier','regressor')):
    raise SystemExit('Unexpected checkpoint format')
ckpt.setdefault('regressor_delayed', ckpt['regressor'])
ckpt.setdefault('regressor_nondelayed', ckpt['regressor'])
out = os.path.splitext(path)[0] + '_DUALDUMMY.pth'
torch.save(ckpt, out)
print('Loaded:', path)
print('Wrote:', out)
