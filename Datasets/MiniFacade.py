import os
import png
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

DATA_PATH = "Datasets/Data"
SAMPLE_PATH = "Datasets/Samples/MiniFacade.png"

N_SIZE = {
    "train": 905 + 1,
    "test_dev": 113 + 1,
    "test": 0,  # [!] To be filled
}
N_CLASS = 5
COLOR_MAP = {0: "others", 1: "facade", 2: "pillar", 3: "window", 4: "balcony"}

# [!] To use the real test under "test" folder, set this flag to False,
# or it will be using "test_dev" folder for test set.
FLAG_DEV = True


def load_raw(set_name="train"):
    X = []
    y = []
    for i in tqdm(range(N_SIZE[set_name]), desc=f"Loading {set_name} data"):
        img = Image.open(os.path.join(DATA_PATH, set_name, "ee616_%04d.jpg" % i))
        pngreader = png.Reader(
            filename=os.path.join(DATA_PATH, set_name, "ee616_%04d.png" % i)
        )
        w, h, row, info = pngreader.read()
        label = np.array(list(row)).astype("uint8")

        # Normalize input image uint8(0-255) -> float(-1, 1)
        # (H, W, C) -> (C, H, W)
        img = np.asarray(img).astype("f").transpose(2, 0, 1) / 128.0 - 1.0

        # Convert to n_class-dimensional onehot matrix
        label_ = np.asarray(label)
        label = np.zeros((N_CLASS, img.shape[1], img.shape[2])).astype("i")
        for j in range(N_CLASS):
            label[j, :] = label_ == j

        X.append(img)
        y.append(label)

    np.savez_compressed(f"Datasets/Data/{set_name}.npz", X=X, y=y)
    return np.array(X), np.array(y)  # (N, C, H, W), (N, n_class, H, W)


def load_npz(set_name="train"):
    try:
        print(f"Loading preprocessed {set_name} data from pre-saved .npz file...")
        data = np.load(f"Datasets/Data/{set_name}.npz")
    except FileNotFoundError:
        print(
            f".npz file not found. Loading raw {set_name} data and saving to .npz file..."
        )
        load_raw(set_name)
        data = np.load(f"Datasets/Data/{set_name}.npz")
    X, y = data["X"], data["y"]
    return X, y


def load_dataset():
    X_train, y_train = load_npz("train")
    X_test, y_test = load_npz("test_dev" if FLAG_DEV else "test")
    X_train, y_train = (
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float(),
    )
    X_test, y_test = (
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).float(),
    )
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    return train_ds, test_ds


def load_loader(batch_size=32):
    train_ds, test_ds = load_dataset()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, test_loader


def stats():
    train_ds, test_ds = load_dataset()
    print("=== MiniFacade Dataset Statistics ===")
    for split_name, ds in zip(["train", "test"], [train_ds, test_ds]):
        total_pixels = 0
        class_counts = np.zeros(N_CLASS, dtype=np.int64)
        for _, y in tqdm(
            ds, desc=f"Calculating {split_name} set statistics", leave=False
        ):
            y_np = y.numpy()  # (n_class, H, W)
            total_pixels += y_np.shape[1] * y_np.shape[2]
            for c in range(N_CLASS):
                class_counts[c] += np.sum(y_np[c, :, :])
        print(f"--- {split_name} Set ---")
        for c in range(N_CLASS):
            class_ratio = class_counts[c] / total_pixels
            print(
                f"Class {c} ({COLOR_MAP[c]}): "
                f"Pixels = {class_counts[c]}, "
                f"Ratio = {class_ratio:.4f}"
            )
        print(f"Total Pixels: {total_pixels}")


def plot():
    sample_loader, _ = load_loader(batch_size=4)
    X, y = next(iter(sample_loader))
    X, y = X.numpy(), y.numpy()

    # Get png palette
    pngreader = png.Reader(filename=os.path.join(DATA_PATH, "train", "ee616_0000.png"))
    w, h, row, info = pngreader.read()
    palette = np.array(info["palette"], dtype=np.uint8)

    fig, axes = plt.subplots(4, 2, figsize=(5, 10))
    for i in range(4):
        axes[i, 0].imshow(((X[i].transpose(1, 2, 0) + 1.0) * 128).astype("uint8"))
        axes[i, 0].axis("off")
        axes[i, 1].imshow(palette[np.argmax(y[i], axis=0)].astype("uint8"))
        axes[i, 1].axis("off")
    fig.suptitle(f"X: {X.shape}, y: {y.shape}", y=0.98, fontsize=16)
    plt.tight_layout()
    plt.savefig(SAMPLE_PATH)


if __name__ == "__main__":
    stats()
    plot()
