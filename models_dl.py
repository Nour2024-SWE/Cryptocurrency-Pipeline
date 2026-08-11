"""
models_dl.py
=============
Framework points 26-27: LSTM and TCN sequence models over
[RV_{t-L+1}, ..., RV_t] (we use LogRV internally for numerical stability, and
exponentiate the forecast back per framework point 16).

Trained with PyTorch. Retraining cadence follows point 28's practical rule:
retrain monthly (config.DL_RETRAIN_FREQ), forecast daily between retrains.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_sequences(logrv: np.ndarray, L: int):
    """Build (X, y) with X_i = logrv[i:i+L], y_i = logrv[i+L] (i.e. target = t+1)."""
    n = len(logrv)
    if n <= L:
        return np.empty((0, L)), np.empty((0,))
    X = np.stack([logrv[i:i + L] for i in range(n - L)])
    y = logrv[L:n]
    return X, y


class LSTMForecaster(nn.Module):
    def __init__(self, hidden=32, dropout=0.1, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=layers,
                             batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):  # x: (batch, L, 1)
        out, (h_n, _) = self.lstm(x)
        h_last = self.drop(h_n[-1])
        return self.fc(h_last).squeeze(-1)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    """Causal dilated conv block (core building unit of a TCN)."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation))
        self.chomp1 = Chomp1d(pad)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation))
        self.chomp2 = Chomp1d(pad)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.relu2(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNForecaster(nn.Module):
    def __init__(self, channels=(16, 16, 16), kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        in_ch = 1
        for i, out_ch in enumerate(channels):
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation=2 ** i, dropout=dropout))
            in_ch = out_ch
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1], 1)

    def forward(self, x):  # x: (batch, L, 1) -> conv expects (batch, ch, L)
        x = x.transpose(1, 2)
        out = self.net(x)
        last = out[:, :, -1]
        return self.fc(last).squeeze(-1)


def _train_model(model, X, y, epochs=60, lr=1e-3, batch_size=64, weight_decay=1e-5):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(DEVICE)  # (n, L, 1)
    yt = torch.tensor(y, dtype=torch.float32).to(DEVICE)
    n = len(Xt)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred = model(Xt[idx])
            loss = loss_fn(pred, yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    return model


def fit_lstm(train: pd.DataFrame, L=22, hidden=32, dropout=0.1, epochs=60):
    logrv = train["LogRV"].to_numpy()
    X, y = _make_sequences(logrv, L)
    model = LSTMForecaster(hidden=hidden, dropout=dropout)
    if len(X) == 0:
        return model, L
    return _train_model(model, X, y, epochs=epochs), L


def fit_tcn(train: pd.DataFrame, L=22, channels=(16, 16, 16), dropout=0.1, epochs=60):
    logrv = train["LogRV"].to_numpy()
    X, y = _make_sequences(logrv, L)
    model = TCNForecaster(channels=channels, dropout=dropout)
    if len(X) == 0:
        return model, L
    return _train_model(model, X, y, epochs=epochs), L


def forecast_dl(model_and_L, recent_logrv: np.ndarray) -> float:
    model, L = model_and_L
    if len(recent_logrv) < L:
        return config.RV_FLOOR
    x = recent_logrv[-L:]
    xt = torch.tensor(x, dtype=torch.float32).reshape(1, L, 1).to(DEVICE)
    model.eval()
    with torch.no_grad():
        pred_logrv = model(xt).cpu().item()
    return max(float(np.exp(pred_logrv)), config.RV_FLOOR)
