import numpy as np
import argparse
import tomllib
import torch
import json
import csv

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
    parser.add_argument("--conf", default=TRAIN_CONF_PATH, help="Path of the configuration file train.toml")
    parser.add_argument("--config", default=DEFAULT_CONF_PATH,
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

def loss_function(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    MSE on (dx, dy) of all the frames, so that the network learns also the detection, plus MSE on the
    distance only of the active frames, since the distance of an inactive frame has no meaning.
    """
    # y_pred, y_true: (B, T, 3)
    dir_loss = torch.mean((y_pred[:, :, 0:2] - y_true[:, :, 0:2])**2)
    mask = torch.linalg.norm(y_true[:, :, 0:2], dim=-1) > 0                 # (B, T) active frames
    if(mask.any()):
        dist_loss = torch.mean((y_pred[:, :, 2][mask] - y_true[:, :, 2][mask])**2)
    else:
        dist_loss = torch.zeros((), device=y_pred.device)
    return dir_loss + dist_loss

def run_epoch(model: SELDNet, loader: DataLoader, device: str,
              optimizer: torch.optim.Optimizer | None) -> tuple[float, np.ndarray, np.ndarray]:
    """
    One pass over the dataset, the weights are updated only if an optimizer is given (training).
    Returns the mean loss of the batches and, without optimizer (validation), the outputs and the targets
    of every frame, (n_windows, T, 3), used for the metrics; empty arrays with the optimizer.
    """
    if(optimizer is not None):
        model.train()
    else:
        model.eval()
    total_loss = 0
    preds = []
    targets = []
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x = x.to(device)                    # (B, N, input_size)
            y = y.to(device)                    # (B, T, 3)
            y_pred = model(x)                   # (B, T, 3)
            loss = loss_function(y_pred, y)
            if(optimizer is not None):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                preds.append(y_pred.cpu().numpy())
                targets.append(y.cpu().numpy())
            total_loss += loss.item()
    if(optimizer is not None):
        return total_loss / len(loader), np.empty(0), np.empty(0)
    return total_loss / len(loader), np.concatenate(preds), np.concatenate(targets)

def frame_metrics(y_pred: np.ndarray, y_true: np.ndarray, det_thr: float, dist_scale: float) -> dict[str, float]:
    """
    Metrics of the frames, as the network is used by CNNAttention:
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
    active = np.linalg.norm(true[:, 0:2], axis=1) > 0
    tp = np.sum(detected & active)
    precision = tp / max(np.sum(detected), 1)
    recall = tp / max(np.sum(active), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    both = detected & active
    if(both.any()):
        u_pred = pred[both, 0:2] / norm_pred[both, None]                        # unit vectors
        pos_pred = u_pred * pred[both, 2:3] * dist_scale                        # (n, 2) [m] from the array centre
        pos_true = true[both, 0:2] * true[both, 2:3] * dist_scale
        error = np.median(np.linalg.norm(pos_pred - pos_true, axis=1))
        cos = np.clip(np.sum(u_pred * true[both, 0:2], axis=1), -1, 1)
        angle = np.median(np.degrees(np.arccos(cos)))
    else:
        error = np.nan
        angle = np.nan
    return {"val_precision": precision, "val_recall": recall, "val_f1": f1,
            "val_median_error_m": error, "val_median_angle_deg": angle}

def main() -> None:
    args = parse_arguments()
    conf = load_conf(Path(args.conf))
    def_toml = load_conf(Path(args.config))
    win = conf["window"]
    tr = conf["training"]
    train_dir = ROOT / conf["data"]["train_dir"] / args.configuration
    val_dir = ROOT / conf["data"]["val_dir"] / args.configuration
    checkpoint_path = ROOT / conf["output"]["checkpoint_dir"] / f"{args.configuration}.pt"
    # The best weights are written in a temporary file renamed at the end, so that an interrupted training
    # does not leave a checkpoint that looks complete
    tmp_path = checkpoint_path.with_suffix(".tmp.pt")
    log_dir = LOG_DIR / args.configuration
    det_thr = def_toml["localization"]["cnn_attention"]["det_thr"]

    torch.manual_seed(tr["seed"])
    np.random.seed(tr["seed"])
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    N, fs, sensor_pos = dataset_info(train_dir)
    _, _, val_sensor_pos = dataset_info(val_dir)
    if(val_sensor_pos != sensor_pos):
        raise ValueError(f"{train_dir} and {val_dir} have different sensor positions")
    # Parameters used to read and label the data, the network must be used with the same ones
    data_params = {
        "band": def_toml["band"],
        "hop": def_toml["stream"]["hop"],
        "pre_margin": def_toml["event"]["pre_margin"],
        "amp_thr": def_toml["detection"]["amp_thr"],
    }
    T = output_frames(N, win["input_size"])
    train_set = FootstepDataset(train_dir, win["input_size"], win["window_stride"], win["frame_thr"], win["dist_scale"], T, def_toml)
    val_set = FootstepDataset(val_dir, win["input_size"], win["window_stride"], win["frame_thr"], win["dist_scale"], T, def_toml)
    print(f"N = {N}, fs = {fs}, T = {T}, training windows: {len(train_set)}, validation windows: {len(val_set)}")
    train_loader = DataLoader(train_set, batch_size=tr["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=tr["batch_size"], shuffle=False)

    model = SELDNet(N).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tr["lr"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_loss = np.inf
    epochs_no_improve = 0
    history = []
    for epoch in range(tr["max_epochs"]):
        train_loss, _, _ = run_epoch(model, train_loader, device, optimizer)
        val_loss, y_pred, y_true = run_epoch(model, val_loader, device, None)
        metrics = frame_metrics(y_pred, y_true, det_thr, win["dist_scale"])
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, **metrics})
        print(f"Epoch {epoch + 1}/{tr['max_epochs']}: training loss {train_loss:.5f}, validation loss {val_loss:.5f}, "
              f"F1 {100 * metrics['val_f1']:.1f} %, median error {metrics['val_median_error_m']:.2f} m")

        if(val_loss < best_loss):
            best_loss = val_loss
            epochs_no_improve = 0
            # Weights and everything needed to use them in the same conditions of the training
            torch.save({
                "model_state": model.state_dict(),
                "N": N,
                "fs": fs,
                "input_size": win["input_size"],
                "T": T,
                "frame_thr": win["frame_thr"],
                "dist_scale": win["dist_scale"],
                "input_norm": INPUT_NORM,
                "data_params": data_params,
                "configuration": args.configuration,
                "sensor_pos": sensor_pos,
                "train_dir": str(train_dir.relative_to(ROOT)),
                "val_dir": str(val_dir.relative_to(ROOT)),
                "epoch": epoch + 1,
                "val_loss": val_loss,
            }, tmp_path)
            print(f"Saved best weights of epoch {epoch + 1}")
        else:
            epochs_no_improve += 1
            if(epochs_no_improve >= tr["patience"]):
                print(f"No improvement for {tr['patience']} epochs, stop")
                break

    tmp_path.replace(checkpoint_path)
    print(f"Best validation loss {best_loss:.5f}, weights in {checkpoint_path}")
    # Loss and validation metrics of every epoch, and their plots
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    plot_history(log_dir / "history.csv")

if __name__ == "__main__":
    main()
