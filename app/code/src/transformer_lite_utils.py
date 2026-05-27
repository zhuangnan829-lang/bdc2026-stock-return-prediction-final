from pathlib import Path

import numpy as np
import torch
from torch import nn

from lstm_utils import build_dataloader, get_device, run_epoch, set_torch_seed


class LitePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerLiteRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 64,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = LitePositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model * 2)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(x)
        hidden = self.pos_encoder(hidden)
        encoded = self.encoder(hidden)
        last_state = encoded[:, -1, :]
        mean_state = encoded.mean(dim=1)
        pooled = self.norm(torch.cat([last_state, mean_state], dim=1))
        return self.head(pooled).squeeze(-1)


def predict_transformer_lite(
    model: nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = build_dataloader(x, None, None, batch_size=batch_size, shuffle=False)
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            preds.append(model(batch_x).detach().cpu().numpy())
    if not preds:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(preds).astype(np.float32)


def train_transformer_lite_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_weight: np.ndarray | None,
    valid_x: np.ndarray | None,
    valid_y: np.ndarray | None,
    valid_weight: np.ndarray | None,
    input_size: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    patience: int,
    seed: int,
    grad_clip_norm: float = 1.0,
) -> tuple[TransformerLiteRegressor, dict]:
    set_torch_seed(seed)
    device = get_device()
    model = TransformerLiteRegressor(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    train_loader = build_dataloader(train_x, train_y, train_weight, batch_size=batch_size, shuffle=True)
    valid_loader = None
    if valid_x is not None and valid_y is not None and len(valid_x) > 0:
        valid_loader = build_dataloader(valid_x, valid_y, valid_weight, batch_size=batch_size, shuffle=False)

    best_state = None
    best_valid_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        for batch in train_loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            sample_weight = batch[2].to(device) if len(batch) > 2 else torch.ones_like(y)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = ((pred - y) ** 2 * sample_weight).sum() / sample_weight.sum().clamp_min(1e-12)
            loss.backward()
            if grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            batch_size_actual = int(x.size(0))
            running_loss += float(loss.item()) * batch_size_actual
            total += batch_size_actual

        train_loss = running_loss / max(total, 1)
        valid_loss = train_loss
        if valid_loader is not None:
            valid_loss = run_epoch(model, valid_loader, optimizer=None, device=device)

        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
        if valid_loss < best_valid_loss - 1e-8:
            best_valid_loss = valid_loss
            best_epoch = epoch
            no_improve = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            no_improve += 1
            if valid_loader is not None and no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "device": str(device),
        "best_valid_loss": float(best_valid_loss),
        "best_epoch": int(best_epoch),
        "epochs_ran": len(history),
        "history": history,
    }


def save_transformer_lite_checkpoint(
    path: Path,
    model: TransformerLiteRegressor,
    feature_columns: list[str],
    sequence_length: int,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_columns": feature_columns,
            "sequence_length": int(sequence_length),
            "input_size": int(len(feature_columns)),
            "d_model": int(d_model),
            "nhead": int(nhead),
            "num_layers": int(num_layers),
            "dim_feedforward": int(dim_feedforward),
            "dropout": float(dropout),
            "scaler_mean": scaler_mean.astype(np.float32),
            "scaler_std": scaler_std.astype(np.float32),
        },
        path,
    )


def load_transformer_lite_checkpoint(path: Path) -> tuple[TransformerLiteRegressor, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = TransformerLiteRegressor(
        input_size=int(checkpoint["input_size"]),
        d_model=int(checkpoint["d_model"]),
        nhead=int(checkpoint["nhead"]),
        num_layers=int(checkpoint["num_layers"]),
        dim_feedforward=int(checkpoint["dim_feedforward"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
