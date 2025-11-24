import time

import torch
from torchmetrics import AveragePrecision
from sklearn.metrics import average_precision_score

from settings import DEVICE, N_CLASS


def ap_torch(outs, ys):
    ap = AveragePrecision(task="multilabel", num_labels=N_CLASS, average="none")
    aps = ap(outs, ys)
    del outs, ys
    return aps.cpu().numpy().tolist()


def ap_np(outs, ys):
    outs = outs.cpu().numpy()
    ys = ys.cpu().numpy()
    aps = []
    for c in range(N_CLASS):
        preds_c = outs[:, c].reshape(-1)
        ys_c = ys[:, c].reshape(-1)

        if ys_c.max() == 0:
            ap = float("nan")
        else:
            ap = average_precision_score(ys_c, preds_c)
            aps.append(ap)
    return aps


if __name__ == "__main__":
    out_torch = torch.rand((100, 5, 256, 256)).float().to(DEVICE)
    y_torch = torch.randint(0, 2, (100, 5, 256, 256)).long().to(DEVICE)

    start = time.time()
    result = ap_torch(out_torch, y_torch)
    end = time.time()
    print(f"torchmetrics AP result: {result}, used time: {end - start:.4f} seconds")

    start = time.time()
    result = ap_np(out_torch, y_torch)
    end = time.time()
    print(f"NumPy AP result: {result}, used time: {end - start:.4f} seconds")
