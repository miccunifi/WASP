import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _t(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=DEVICE)


def _affine(W, mu_s, mu_t):
    return lambda x: ((_t(x) - mu_s) @ W + mu_t).cpu().numpy()


def _cov(x, eps):
    xc = x - x.mean(0)
    return xc.T @ xc / max(len(x) - 1, 1) + eps * torch.eye(x.shape[1], device=DEVICE)


def _power(m, p, eps):
    w, v = torch.linalg.eigh(m)
    return (v * w.clamp_min(eps).pow(p)) @ v.T


def coral(xs, xt, eps=1e-5):
    xs, xt = _t(xs), _t(xt)
    W = _power(_cov(xs, eps), -0.5, eps) @ _power(_cov(xt, eps), 0.5, eps)
    return _affine(W, xs.mean(0), xt.mean(0))


def cca(xs, xt, eps=1e-6):
    xs, xt = _t(xs), _t(xt)
    cross = (xs - xs.mean(0)).T @ (xt - xt.mean(0)) / max(len(xs) - 1, 1)
    ss, tt = _power(_cov(xs, eps), -0.5, eps), _cov(xt, eps)
    u, _, vh = torch.linalg.svd(ss @ cross @ _power(tt, -0.5, eps), full_matrices=False)
    return _affine(ss @ u @ vh @ _power(tt, 0.5, eps), xs.mean(0), xt.mean(0))


def ridge(xs, xt, lam):
    xs, xt = _t(xs), _t(xt)
    u, s, vh = torch.linalg.svd(xs - xs.mean(0), full_matrices=False)
    W = (vh.T * (s / (s**2 + lam))) @ (u.T @ (xt - xt.mean(0)))
    return _affine(W, xs.mean(0), xt.mean(0))


class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, d_out)
        self.residual = d_in == d_out

    def forward(self, x):
        y = self.fc2(F.gelu(self.fc1(x)))
        return y + x if self.residual else y


def mlp(xs, xt, weight_decay, hidden, lr=1e-3, epochs=600, patience=80):
    xs, xt = _t(xs), _t(xt)
    model = MLP(xs.shape[1], xt.shape[1], hidden).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, state, wait = float("inf"), None, 0
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = F.mse_loss(model(xs), xt)
        loss.backward()
        optimizer.step()
        if loss.item() + 1e-8 < best:
            best, state, wait = loss.item(), {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(state)
    model.eval()

    def transform(x):
        with torch.no_grad():
            return model(_t(x)).cpu().numpy()

    return transform
