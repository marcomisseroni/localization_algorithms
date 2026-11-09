import numpy as np
import argparse
import tomllib
import torch
import json
import csv
import os

from pathlib import Path
from torch.utils.data import DataLoader
from localization.cnn_attention import SELDNet, INPUT_NORM
from training.loader import FootstepDataset, output_frames
from training.plots import plot_history, LOG_DIR

ROOT = Path(__file__).parent.parent
TRAIN_CONF_PATH = ROOT / "configs" / "train.toml"
DEFAULT_CONF_PATH = ROOT / "configs" / "default.toml"

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the neural network of the CnnAttention localizer for one sensor configuration")
    parser.add_argument("configuration", help="name of the configuration, the same subfolder of train_dir and val_dir, e.g. y_7m_10db")
    parser.add_argument("--train_conf", default=TRAIN_CONF_PATH, help="Path of the configuration file train.toml")
    parser.add_argument("--def_conf", default=DEFAULT_CONF_PATH,
                        help="Path of default.toml: filter, hop and amp_thr used to read and label the data, the same of evaluate.py")
    return parser.parse_args()

def load_conf(conf_path: Path) -> dict:
    with open(conf_path, "rb") as f:
        data = tomllib.load(f)
    return data

def dataset_info(data_dir: Path) -> tuple[int, int, list]:
    """
    Number of sensors, sampling frequency and sensor positions of the dataset in data_dir.
    """
    with open(data_dir / "dataset.json") as f:
        data = json.load(f)
    return len(data["sensor_pos"]), data["fs"], data["sensor_pos"]

def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    MSE loss, y_pred and y_true are (B, T, 3).
    The error on (dx, dy) is computed on all the frames, so that the network learns also the detection.
    The error on d is computed only on the frames with a step in the target (norm of the target (dx, dy) is 1,
    otherwise it is 0), since the distance of a frame without a step has no meaning.
    """
    loss = torch.mean((y_pred[:, :, 0:2] - y_true[:, :, 0:2])**2)
    mask = torch.linalg.norm(y_true[:, :, 0:2], dim=-1) > 0.5       # (B, T) frames with a step
    if(mask.any()):
        loss = loss + torch.mean((y_pred[:, :, 2][mask] - y_true[:, :, 2][mask])**2)
    return loss

def train_one_epoch(model: SELDNet, training_loader: DataLoader, 
                    optimizer: torch.optim.Optimizer, device: str) -> float:
    """
    One pass over the training set, the weights are updated after every batch.
    Returns the mean loss of the batches.
    """
    model.train()
    running_loss = 0
    for inputs, labels in training_loader:
        inputs = inputs.to(device)              # (B, N, input_size)
        labels = labels.to(device)              # (B, T, 3)
        optimizer.zero_grad()
        outputs = model(inputs)                 # (B, T, 3)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(training_loader)

def val_one_epoch(model: SELDNet, validation_loader: DataLoader, device: str) -> tuple[float, np.ndarray, np.ndarray]:
    """
    One pass over the validation set, without updating the weights.
    Returns the mean loss of the batches and the outputs and targets of every frame, (n_windows, T, 3),
    used to compute the metrics.
    """
    model.eval()
    running_loss = 0
    preds = []
    targets = []
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs = inputs.to(device)          # (B, N, input_size)
            labels = labels.to(device)          # (B, T, 3)
            outputs = model(inputs)             # (B, T, 3)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            preds.append(outputs.cpu().numpy())
            targets.append(labels.cpu().numpy())
    return running_loss / len(validation_loader), np.concatenate(preds), np.concatenate(targets)

def frame_metrics(y_pred: np.ndarray, y_true: np.ndarray, det_thr: float, dist_scale: float) -> dict[str, float]:
    """
    Metrics of the frames, computed as the network is used by CNNAttention:
    - detection: a frame is detected if the norm of (dx, dy) is above det_thr, it has a step if its target is not 0;
      precision, recall and F1 of the detected frames;
    - localization of the frames with a step that are detected: median distance [m] between the estimated and the
      true position (relative to the array centre) and median error of the direction [deg].
    """
    # y_pred, y_true: (n_windows, T, 3)
    pred = y_pred.reshape(-1, 3)
    true = y_true.reshape(-1, 3)
    norm_pred = np.linalg.norm(pred[:, 0:2], axis=1)
    detected = norm_pred > det_thr
    active = np.linalg.norm(true[:, 0:2], axis=1) > 0.5
    tp = np.sum(detected & active)
    precision = tp / max(np.sum(detected), 1)
    recall = tp / max(np.sum(active), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    both = detected & active
    if(both.any()):
        u_pred = pred[both, 0:2] / norm_pred[both, None]            # (n, 2) unit vectors
        pos_pred = u_pred * pred[both, 2:3] * dist_scale            # (n, 2) [m] from the array centre
        pos_true = true[both, 0:2] * true[both, 2:3] * dist_scale   # (n, 2) [m] from the array centre
        error = np.median(np.linalg.norm(pos_pred - pos_true, axis=1))
        cos = np.clip(np.sum(u_pred * true[both, 0:2], axis=1), -1, 1)
        angle = np.median(np.degrees(np.arccos(cos)))
    else:
        error = np.nan
        angle = np.nan
    return {"val_precision": precision, "val_recall": recall, "val_f1": f1,
            "val_median_error_m": error, "val_median_angle_deg": angle}

def main() -> None:
    """
    Train the network of one sensor configuration. The best weights (lowest validation loss) are saved in
    <checkpoint_dir>/<configuration>.pt, the loss and the validation metrics of every epoch and their plots in
    training/log/<configuration>/.
    """
    args = parse_arguments()
    train_conf = load_conf(args.train_conf)
    def_conf = load_conf(args.def_conf)
    train_dir = ROOT / train_conf["data"]["train_dir"] / args.configuration
    val_dir = ROOT / train_conf["data"]["val_dir"] / args.configuration
    N, fs, sensor_pos = dataset_info(train_dir)
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    T = output_frames(N, train_conf["window"]["input_size"])
    train_dataset = FootstepDataset(train_dir, train_conf["window"]["input_size"], 
                                                  train_conf["window"]["window_stride"], train_conf["window"]["frame_thr"],
                                                  train_conf["window"]["dist_scale"], T, def_conf)
    val_dataset = FootstepDataset(val_dir, train_conf["window"]["input_size"], 
                                                      train_conf["window"]["window_stride"], train_conf["window"]["frame_thr"],
                                                      train_conf["window"]["dist_scale"], T, def_conf)
    train_dataloader = DataLoader(train_dataset, train_conf["training"]["batch_size"], True)
    val_dataloader = DataLoader(val_dataset, train_conf["training"]["batch_size"], False)
    torch.manual_seed(train_conf["training"]["seed"])
    seld_net = SELDNet(N).to(device)
    optimizer = torch.optim.Adam(seld_net.parameters(), lr=train_conf["training"]["lr"])
    epoch_number = 0
    epochs = train_conf["training"]["max_epochs"]
    checkpoint_path = ROOT / train_conf["output"]["checkpoint_dir"] / f"{args.configuration}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(".tmp.pt")
    log_dir = LOG_DIR / args.configuration
    det_thr = def_conf["localization"]["cnn_attention"]["det_thr"]
    history = []

    best_loss = np.inf
    no_improv_count = 0
    for epoch in range(epochs):
        train_loss = train_one_epoch(seld_net, train_dataloader, optimizer, device)
        val_loss, y_pred, y_true = val_one_epoch(seld_net, val_dataloader, device)
        metrics = frame_metrics(y_pred, y_true, det_thr, train_conf["window"]["dist_scale"])
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, **metrics})
        print(f"EPOCH {epoch_number + 1}:")
        print(f"training loss: {train_loss}")
        print(f"validation loss: {val_loss}")
        print(f"validation F1: {100 * metrics['val_f1']:.1f} %, median error: {metrics['val_median_error_m']:.2f} m")
        epoch_number += 1
        if(val_loss < best_loss):
            best_loss = val_loss
            no_improv_count = 0
            torch.save({
                "model_state": seld_net.state_dict(),
                "N": N,
                "fs": fs,
                "sensor_pos": sensor_pos,
                "input_size": train_conf["window"]["input_size"],
                "T": T,
                "frame_thr": train_conf["window"]["frame_thr"],
                "dist_scale": train_conf["window"]["dist_scale"],
                "input_norm": INPUT_NORM,
                "data_params": {
                    "band": def_conf["band"],
                    "hop": def_conf["stream"]["hop"],
                    "pre_margin": def_conf["event"]["pre_margin"],
                    "amp_thr": def_conf["detection"]["amp_thr"],
                },
                "configuration": args.configuration,
                "epoch": epoch + 1,
                "val_loss": val_loss,
            }, tmp_path)

        else:
            no_improv_count += 1
            if(no_improv_count >= train_conf["training"]["patience"]): break

    tmp_path.replace(checkpoint_path)

    # Loss and validation metrics of every epoch, and their plots
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    plot_history(log_dir / "history.csv")

if __name__ == "__main__":
    main()