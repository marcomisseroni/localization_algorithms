import numpy as np
import argparse
import tomllib
import torch
import json
import csv

from pathlib import Path
from torch.utils.data import DataLoader
from localization.cnn_attention import SELDNet
from training.loader import FootstepDataset, output_frames

ROOT = Path(__file__).parent.parent
TRAIN_CONF_PATH = ROOT / "configs" / "train.toml"

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the neural network of the CnnAttention localizer")
    parser.add_argument("--conf", default=TRAIN_CONF_PATH, help="Path of the configuration file train.toml")
    return parser.parse_args()

def load_conf(conf_path: Path) -> dict:
    with open(conf_path, "rb") as f:
        data = tomllib.load(f)
    return data

def recording_info(data_dir: Path) -> tuple[int, int, dict]:
    """
    Number of sensors, sampling frequency and parameters of params.json of the first dataset in data_dir.
    """
    d = [d for d in sorted(data_dir.iterdir()) if d.is_dir() and (d / "index.csv").is_file()][0]
    with open(d / "index.csv", newline="") as f:
        first_file = next(csv.DictReader(f))["file"]
    data = np.load(d / first_file)
    with open(d / "params.json") as f:
        params = json.load(f)
    return data["signal"].shape[0], int(data["fs"]), params

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

def run_epoch(model: SELDNet, loader: DataLoader, device: str, optimizer: torch.optim.Optimizer | None) -> float:
    """
    One pass over the dataset, the weights are updated only if an optimizer is given (training).
    Returns the mean loss of the batches.
    """
    if(optimizer is not None):
        model.train()
    else:
        model.eval()
    total_loss = 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x = x.to(device)                    # (B, N, input_size)
            y = y.to(device)                    # (B, T, 3)
            loss = loss_function(model(x), y)
            if(optimizer is not None):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
    return total_loss / len(loader)

def main() -> None:
    args = parse_arguments()
    conf = load_conf(Path(args.conf))
    win = conf["window"]
    tr = conf["training"]
    train_dir = ROOT / conf["data"]["train_dir"]
    val_dir = ROOT / conf["data"]["val_dir"]
    checkpoint_path = ROOT / conf["output"]["checkpoint"]

    torch.manual_seed(tr["seed"])
    np.random.seed(tr["seed"])
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    N, fs, data_params = recording_info(train_dir)
    T = output_frames(N, win["input_size"])
    train_set = FootstepDataset(train_dir, win["input_size"], win["window_stride"], win["frame_thr"], win["dist_scale"], T)
    val_set = FootstepDataset(val_dir, win["input_size"], win["window_stride"], win["frame_thr"], win["dist_scale"], T)
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
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device, None)
        history.append([epoch + 1, train_loss, val_loss])
        print(f"Epoch {epoch + 1}/{tr['max_epochs']}: training loss {train_loss:.5f}, validation loss {val_loss:.5f}")

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
                "data_params": data_params,
                "train_dir": conf["data"]["train_dir"],
                "val_dir": conf["data"]["val_dir"],
                "epoch": epoch + 1,
                "val_loss": val_loss,
            }, checkpoint_path)
            print(f"Saved {conf['output']['checkpoint']}")
        else:
            epochs_no_improve += 1
            if(epochs_no_improve >= tr["patience"]):
                print(f"No improvement for {tr['patience']} epochs, stop")
                break

    # Loss of every epoch, for the learning curve
    with open(checkpoint_path.with_suffix(".csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        writer.writerows(history)
    print(f"Best validation loss {best_loss:.5f}")

if __name__ == "__main__":
    main()
