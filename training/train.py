import numpy as np
import argparse
import tomllib
import torch
import json
import csv
import os
import time

from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from localization.cnn_attention import SELDNet, INPUT_NORM
from training.loader import FootstepDataset, output_frames
from training.plots import plot_history, LOG_DIR

ROOT = Path(__file__).parent.parent
TRAIN_CONF_PATH = ROOT / "configs" / "train.toml"
DEFAULT_CONF_PATH = ROOT / "configs" / "default.toml"

NUM_WORKERS_TRAIN = 12
NUM_WORKERS_VAL = 2

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

def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor, bce_weight: float, mse_weight: float) -> torch.Tensor:
    """
    Weighted sum of a detection loss and a position loss, y_pred and y_true are (B, T, 4) = (logit, dx, dy, d).
    - detection: binary cross entropy between the logit and the activity of the frame (1 step, 0 no step), mean
      over all the frames of the batch;
    - position: Euclidean error between the estimated position d * (dx, dy) / |(dx, dy)| (the direction is
      normalized as in decode_pos) and the true one, both relative to the array centre and divided by dist_scale,
      mean over the frames with a step only, since the position of a frame without a step has no meaning.
    """
    bce = torch.nn.functional.binary_cross_entropy_with_logits(y_pred[:, :, 0], y_true[:, :, 0])
    mask = y_true[:, :, 0] > 0.5
    u = y_pred[:, :, 1:3] / (torch.linalg.norm(y_pred[:, :, 1:3], dim=-1, keepdim=True) + 1e-8)
    est_pos = (u * y_pred[:, :, 3:4])[mask]
    true_pos = (y_true[:, :, 1:3] * y_true[:, :, 3:4])[mask]
    mse = torch.mean(torch.linalg.norm(est_pos - true_pos, dim=-1)) if mask.any() else 0
    return bce_weight * bce + mse_weight * mse

def train_one_epoch(model: SELDNet, training_loader: DataLoader, optimizer: torch.optim.Optimizer,
                    device: str, bce_weight: float, mse_weight: float) -> float:
    """
    One pass over the training set, the weights are updated after every batch.
    Returns the mean loss of the batches.
    """
    model.train()
    running_loss = 0
    bar = tqdm(training_loader, desc="Training", leave=False)
    for inputs, labels in bar:
        inputs = inputs.to(device)              # (B, N, input_size)
        labels = labels.to(device)              # (B, T, 4)
        optimizer.zero_grad()
        outputs = model(inputs)                 # (B, T, 4)
        loss = loss_fn(outputs, labels, bce_weight, mse_weight)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / len(training_loader)

def val_one_epoch(model: SELDNet, validation_loader: DataLoader, device: str,
                  bce_weight: float, mse_weight: float) -> tuple[float, np.ndarray, np.ndarray]:
    """
    One pass over the validation set, without updating the weights.
    Returns the mean loss of the batches and the outputs and targets of every frame, (n_windows, T, 4),
    used to compute the metrics.
    """
    model.eval()
    running_loss = 0
    preds = []
    targets = []
    with torch.no_grad():
        for inputs, labels in tqdm(validation_loader, desc="Validation", leave=False):
            inputs = inputs.to(device)          # (B, N, input_size)
            labels = labels.to(device)          # (B, T, 4)
            outputs = model(inputs)             # (B, T, 4)
            loss = loss_fn(outputs, labels, bce_weight, mse_weight)
            running_loss += loss.item()
            preds.append(outputs.cpu().numpy())
            targets.append(labels.cpu().numpy())
    return running_loss / len(validation_loader), np.concatenate(preds), np.concatenate(targets)

def frame_metrics(y_pred: np.ndarray, y_true: np.ndarray, det_thr: float, dist_scale: float) -> dict[str, float]:
    """
    Metrics of the frames, computed as the network is used by CNNAttention:
    - detection: a frame is detected if sigmoid(logit) is above det_thr, it has a step if its target activity is 1;
      precision, recall and F1 of the detected frames;
    - localization of the frames with a step that are detected: median distance [m] between the estimated and the
      true position (relative to the array centre) and median error of the direction [deg].
    """
    # y_pred, y_true: (n_windows, T, 4)
    pred = y_pred.reshape(-1, 4)
    true = y_true.reshape(-1, 4)
    norm_pred = np.linalg.norm(pred[:, 1:3], axis=1)
    detected = 1 / (1 + np.exp(-pred[:, 0])) > det_thr
    active = true[:, 0] > 0.5
    tp = np.sum(detected & active)
    precision = tp / max(np.sum(detected), 1)
    recall = tp / max(np.sum(active), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    both = detected & active
    if(both.any()):
        u_pred = pred[both, 1:3] / norm_pred[both, None]            # (n, 2) unit vectors
        pos_pred = u_pred * pred[both, 3:4] * dist_scale            # (n, 2) [m] from the array centre
        pos_true = true[both, 1:3] * true[both, 3:4] * dist_scale   # (n, 2) [m] from the array centre
        error = np.median(np.linalg.norm(pos_pred - pos_true, axis=1))
        cos = np.clip(np.sum(u_pred * true[both, 1:3], axis=1), -1, 1)
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
                                                  train_conf["window"]["dist_scale"], T, def_conf, random_offset=True, seed=train_conf["training"]["seed"])
    val_dataset = FootstepDataset(val_dir, train_conf["window"]["input_size"], 
                                                      train_conf["window"]["val_stride"], train_conf["window"]["frame_thr"],
                                                      train_conf["window"]["dist_scale"], T, def_conf, random_offset=False, seed=train_conf["training"]["seed"])
    # Workers not persistent in training: they are created again at every epoch with a copy of the dataset, so they
    # see the new windows of make_windows()
    train_dataloader = DataLoader(train_dataset, train_conf["training"]["batch_size"], True,
                                  num_workers=NUM_WORKERS_TRAIN, persistent_workers=False)
    val_dataloader = DataLoader(val_dataset, train_conf["training"]["batch_size"], False, 
                                num_workers=NUM_WORKERS_VAL, persistent_workers=False)
    torch.manual_seed(train_conf["training"]["seed"])
    seld_net = SELDNet(N).to(device)
    optimizer = torch.optim.Adam(seld_net.parameters(), lr=train_conf["training"]["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=train_conf["training"]["factor"],
                                                           patience=train_conf["training"]["patience_lr"], 
                                                           min_lr=train_conf["training"]["min_lr"])
    epochs = train_conf["training"]["max_epochs"]
    patience = train_conf["training"]["patience"]
    print(f"Configuration {args.configuration}: {N} sensors, fs {fs} Hz, {T} frames per window")
    print(f"Windows: {len(train_dataset)} training, {len(val_dataset)} validation")
    print(f"Network: {seld_net.model_size():,} parameters, learning rate {train_conf['training']['lr']}, "
          f"batch size {train_conf['training']['batch_size']}, at most {epochs} epochs, patience {patience}")
    print(f"Scheduler ReduceLROnPlateau: learning rate x {train_conf['training']['factor']} after "
          f"{train_conf['training']['patience_lr']} epochs without improvement, min {train_conf['training']['min_lr']}")
    checkpoint_path = ROOT / train_conf["output"]["checkpoint_dir"] / f"{args.configuration}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(".tmp.pt")
    log_dir = LOG_DIR / args.configuration
    det_thr = def_conf["localization"]["cnn_attention"]["det_thr"]
    history = []

    best_loss = np.inf
    best_epoch = 0
    no_improv_count = 0
    start_time = time.perf_counter()
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        if(epoch > 0):
            train_dataset.make_windows()        # new random offsets of the training windows
        train_loss = train_one_epoch(seld_net, train_dataloader, optimizer, device, train_conf["training"]["bce_weight"], train_conf["training"]["mse_weight"])
        val_loss, y_pred, y_true = val_one_epoch(seld_net, val_dataloader, device, train_conf["training"]["bce_weight"], train_conf["training"]["mse_weight"])
        metrics = frame_metrics(y_pred, y_true, det_thr, train_conf["window"]["dist_scale"])
        lr = optimizer.param_groups[0]["lr"]        # learning rate used in this epoch, read before the scheduler changes it
        history.append({"epoch": epoch + 1, "lr": lr, "train_loss": train_loss, "val_loss": val_loss, **metrics})
        scheduler.step(val_loss)
        improved = val_loss < best_loss
        no_improv_count = 0 if improved else no_improv_count + 1
        print(f"Epoch {epoch + 1:3d}/{epochs} | lr {lr:.1e} | train loss {train_loss:.5f} | val loss {val_loss:.5f} {'*' if improved else ' '} | "
              f"F1 {100 * metrics['val_f1']:5.1f} % | median error {metrics['val_median_error_m']:5.2f} m | "
              f"angle {metrics['val_median_angle_deg']:5.2f} deg | {time.perf_counter() - epoch_start:4.1f} s | "
              f"no improvement {no_improv_count}/{patience}")
        if(improved):
            best_loss = val_loss
            best_epoch = epoch + 1
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

        elif(no_improv_count >= patience):
            print(f"No improvement for {patience} epochs, stop")
            break

    tmp_path.replace(checkpoint_path)
    best = history[best_epoch - 1]
    print(f"Best epoch {best_epoch}: val loss {best['val_loss']:.5f}, F1 {100 * best['val_f1']:.1f} %, "
          f"median error {best['val_median_error_m']:.2f} m, angle {best['val_median_angle_deg']:.2f} deg")
    print(f"Training time {(time.perf_counter() - start_time) / 60:.1f} min, weights in {checkpoint_path}")

    # Loss and validation metrics of every epoch, and their plots
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    plot_history(log_dir / "history.csv")

if __name__ == "__main__":
    main()