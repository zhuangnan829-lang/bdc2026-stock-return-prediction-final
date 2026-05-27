from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils_seed import set_seed


@dataclass
class SequenceDatasetBundle:
    x: np.ndarray
    y: np.ndarray | None
    sample_weight: np.ndarray | None
    meta: pd.DataFrame


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.head(last_hidden).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / max(d_model, 1)))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(x)
        hidden = self.pos_encoder(hidden)
        encoded = self.encoder(hidden)
        pooled = self.norm(encoded[:, -1, :])
        return self.head(pooled).squeeze(-1)


def set_torch_seed(seed: int) -> None:
    set_seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_sequence_dataset(
    df: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
    target_dates: set[pd.Timestamp] | None = None,
    label_column: str | None = None,
    raw_label_column: str | None = None,
    sample_weight_column: str | None = None,
) -> SequenceDatasetBundle:
    working = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
    sequences: list[np.ndarray] = []
    labels: list[float] = []
    sample_weights: list[float] = []
    meta_rows: list[dict] = []

    for stock_id, stock_df in working.groupby("stock_id", sort=False):
        stock_df = stock_df.reset_index(drop=True)
        feature_values = stock_df[feature_columns].to_numpy(dtype=np.float32, copy=False)
        for idx in range(sequence_length - 1, len(stock_df)):
            current_date = stock_df.at[idx, "date"]
            if target_dates is not None and current_date not in target_dates:
                continue
            if label_column is not None and pd.isna(stock_df.at[idx, label_column]):
                continue

            sequences.append(feature_values[idx - sequence_length + 1 : idx + 1])
            meta = {
                "stock_id": str(stock_id).zfill(6),
                "date": pd.Timestamp(current_date),
            }
            if raw_label_column and raw_label_column in stock_df.columns:
                meta[raw_label_column] = stock_df.at[idx, raw_label_column]
            if label_column is not None:
                labels.append(float(stock_df.at[idx, label_column]))
                meta[label_column] = float(stock_df.at[idx, label_column])
            if sample_weight_column is not None and sample_weight_column in stock_df.columns:
                sample_weights.append(float(stock_df.at[idx, sample_weight_column]))
                meta[sample_weight_column] = float(stock_df.at[idx, sample_weight_column])
            meta_rows.append(meta)

    x = np.stack(sequences).astype(np.float32) if sequences else np.empty((0, sequence_length, len(feature_columns)), dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32) if label_column is not None else None
    weight_array = np.asarray(sample_weights, dtype=np.float32) if sample_weights else None
    meta_df = pd.DataFrame(meta_rows)
    return SequenceDatasetBundle(x=x, y=y, sample_weight=weight_array, meta=meta_df)


def fit_sequence_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        raise ValueError("Cannot fit scaler on empty sequence array")
    flat = x.reshape(-1, x.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def transform_sequences(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32)
    return ((x - mean[None, None, :]) / std[None, None, :]).astype(np.float32)


def build_dataloader(
    x: np.ndarray,
    y: np.ndarray | None,
    sample_weight: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    x_tensor = torch.from_numpy(x)
    if y is None:
        dataset = TensorDataset(x_tensor)
    else:
        if sample_weight is None:
            sample_weight = np.ones(len(y), dtype=np.float32)
        dataset = TensorDataset(x_tensor, torch.from_numpy(y), torch.from_numpy(sample_weight))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    running_loss = 0.0
    total = 0
    train_mode = optimizer is not None
    model.train(mode=train_mode)

    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].to(device) if len(batch) > 1 else None
        sample_weight = batch[2].to(device) if len(batch) > 2 else None

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        pred = model(x)
        if y is None:
            raise ValueError("Training/evaluation loader must provide labels")
        if sample_weight is None:
            sample_weight = torch.ones_like(y)
        loss = ((pred - y) ** 2 * sample_weight).sum() / sample_weight.sum().clamp_min(1e-12)
        if optimizer is not None:
            loss.backward()
            optimizer.step()

        batch_size = int(x.size(0))
        running_loss += float(loss.item()) * batch_size
        total += batch_size

    return running_loss / max(total, 1)


def predict_sequences(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    loader = build_dataloader(x, None, None, batch_size=batch_size, shuffle=False)
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            batch_pred = model(batch_x).detach().cpu().numpy()
            preds.append(batch_pred)
    if not preds:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(preds).astype(np.float32)


def train_lstm_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_weight: np.ndarray | None,
    valid_x: np.ndarray | None,
    valid_y: np.ndarray | None,
    valid_weight: np.ndarray | None,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    patience: int,
    seed: int,
    snapshot_top_k: int = 0,
) -> tuple[LSTMRegressor, dict]:
    set_torch_seed(seed)
    device = get_device()
    model = LSTMRegressor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = build_dataloader(train_x, train_y, train_weight, batch_size=batch_size, shuffle=True)
    valid_loader = None
    if valid_x is not None and valid_y is not None and len(valid_x) > 0:
        valid_loader = build_dataloader(valid_x, valid_y, valid_weight, batch_size=batch_size, shuffle=False)

    best_state = None
    best_valid_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    history: list[dict] = []
    snapshot_states: list[dict] = []
    last_state = None
    capture_snapshots = int(snapshot_top_k) > 0

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer=optimizer, device=device)
        valid_loss = train_loss
        if valid_loader is not None:
            valid_loss = run_epoch(model, valid_loader, optimizer=None, device=device)

        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
        if capture_snapshots:
            last_state = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "valid_loss": float(valid_loss),
                "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            }
            snapshot_states.append(last_state)
            snapshot_states = sorted(snapshot_states, key=lambda item: item["valid_loss"])[: int(snapshot_top_k)]

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
        "best_valid_loss": best_valid_loss,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "history": history,
        "snapshot_states": snapshot_states,
        "last_state": last_state,
    }


def train_transformer_model(
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
) -> tuple[TransformerRegressor, dict]:
    set_torch_seed(seed)
    device = get_device()
    model = TransformerRegressor(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

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
        train_loss = run_epoch(model, train_loader, optimizer=optimizer, device=device)
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
        "best_valid_loss": best_valid_loss,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "history": history,
    }


def save_lstm_checkpoint(
    path: Path,
    model: LSTMRegressor,
    feature_columns: list[str],
    sequence_length: int,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_columns": feature_columns,
            "sequence_length": int(sequence_length),
            "input_size": int(len(feature_columns)),
            "hidden_size": int(hidden_size),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "scaler_mean": scaler_mean.astype(np.float32),
            "scaler_std": scaler_std.astype(np.float32),
        },
        path,
    )


def load_lstm_checkpoint(path: Path) -> tuple[LSTMRegressor, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = LSTMRegressor(
        input_size=int(checkpoint["input_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def save_transformer_checkpoint(
    path: Path,
    model: TransformerRegressor,
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


def load_transformer_checkpoint(path: Path) -> tuple[TransformerRegressor, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = TransformerRegressor(
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
