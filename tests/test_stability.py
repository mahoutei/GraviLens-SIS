import torch
import torch.nn as nn

from gravilens_sis.stability.adaptive_clip import adaptive_gradient_clip
from gravilens_sis.stability.early_stopping import EarlyStopping
from gravilens_sis.stability.ema import ModelEMA
from gravilens_sis.stability.grad_accum import GradientAccumulator
from gravilens_sis.stability.label_smoothing import LabelSmoothingCrossEntropy
from gravilens_sis.stability.lookahead import Lookahead
from gravilens_sis.stability.warmup_cosine import WarmupCosineScheduler


def _opt(lr=1.0):
    return torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=lr)


def test_warmup_cosine_schedule():
    opt = _opt()
    sched = WarmupCosineScheduler(opt, warmup_epochs=2, total_epochs=10,
                                  min_lr=0.0, max_lr=1.0)
    lrs = []
    for _ in range(10):
        sched.step()
        lrs.append(opt.param_groups[0]['lr'])
    assert lrs[0] > 0 and lrs[1] > lrs[0]      # warmup rises
    assert lrs[9] < lrs[2]                     # cosine anneals after warmup
    assert all(l >= 0 for l in lrs)


def test_early_stopping_triggers():
    es = EarlyStopping(patience=2, min_delta=0.0, mode='min')
    stop = False
    for v in [1.0, 0.5, 0.51, 0.52, 0.53]:     # improve, then 3 non-improvements
        stop = es(v)
    assert stop


def test_lookahead_state_roundtrip():
    m = nn.Linear(3, 3)
    la = Lookahead(torch.optim.SGD(m.parameters(), lr=0.1), k=2, alpha=0.5)
    for _ in range(3):
        la.zero_grad()
        m(torch.randn(2, 3)).sum().backward()
        la.step()
    la2 = Lookahead(torch.optim.SGD(m.parameters(), lr=0.1), k=2)
    la2.load_state_dict(la.state_dict())       # must not raise


def test_label_smoothing_loss():
    crit = LabelSmoothingCrossEntropy(smoothing=0.1)
    loss = crit(torch.randn(8, 3), torch.randint(0, 3, (8,)))
    assert torch.isfinite(loss) and loss.ndim == 0


def test_grad_accum_and_ema_run():
    acc = GradientAccumulator(accumulation_steps=2)
    assert isinstance(bool(acc.should_step()), bool)
    acc.reset()
    m = nn.Linear(2, 2)
    ema = ModelEMA(m)
    ema.update(m)                              # must not raise


def test_adaptive_gradient_clip_reduces_norm():
    m = nn.Linear(4, 4)
    m(torch.randn(2, 4)).sum().backward()
    with torch.no_grad():
        for p in m.parameters():
            p.grad *= 100
    before = torch.cat([p.grad.flatten() for p in m.parameters()]).norm()
    adaptive_gradient_clip(m, clip_factor=0.01)
    after = torch.cat([p.grad.flatten() for p in m.parameters()]).norm()
    assert torch.isfinite(after) and after <= before
