import os
import shutil
from datetime import datetime
from uuid import uuid4

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from Networks.UNetMini import Network
from Train.lib.criterion import get_criterion
from Train.lib.evaluation import ap_torch as ap_evaluator
from Train.lib.train import run

from settings import DEVICE
from Datasets.MiniFacade import load_loader

BATCH_SIZE = 16
NUM_EPOCHS = 100
EVAL_AFTER = 0
NUM_RUNS = 1

INPUT_SHAPE = (3, 256, 256)
OUTPUT_SHAPE = (5, 256, 256)

_DIM_BEGIN = 32
_DIM_DEPTH = 5
DIM = [_DIM_BEGIN * (2**i) for i in range(_DIM_DEPTH)]
DROPOUT = 0.1

CRITERION = "ce"
_RATIO = np.array([0.3548, 0.4057, 0.0236, 0.1650, 0.0509])
# [0.72, 0.69, 2.41, 0.93, 1.74]
CLASS_WEIGHTS = torch.tensor(np.float32(1 / np.sqrt(_RATIO))).to(DEVICE)
DICE_WEIGHT = 0.3
LR = 5e-4
PCT_START = 0.09

RESULT_PATH = "Train/Results/UNetMini"

if __name__ == "__main__":
    os.makedirs(RESULT_PATH, exist_ok=True)
    train_loader, test_loader = load_loader(BATCH_SIZE)

    for _ in range(NUM_RUNS):
        run_id = str(uuid4())[:8]
        print(f"=== UNetMini Run {run_id} ===")
        model = Network(
            INPUT_SHAPE,
            OUTPUT_SHAPE[0],
            dim=DIM,
            dropout=DROPOUT,
        ).to(DEVICE)
        # model = torch.compile(model, mode="reduce-overhead")
        optimizer = optim.AdamW(model.parameters(), lr=LR)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=LR,
            total_steps=NUM_EPOCHS * len(train_loader),
            pct_start=PCT_START,
            anneal_strategy="linear",
        )
        criterion = get_criterion(
            name=CRITERION,
            weight=CLASS_WEIGHTS,
            dice_weight=DICE_WEIGHT,
        )
        best_model_path = os.path.join(RESULT_PATH, "Checkpoints", f"best_{run_id}.pt")
        log, best_meta = run(
            model,
            train_loader,
            test_loader,
            optimizer,
            scheduler,
            criterion,
            ap_evaluator,
            NUM_EPOCHS,
            eval_after=EVAL_AFTER,
            save_best_path=best_model_path,
        )

        if best_meta["best_path"] is not None:
            latest_best_path = os.path.join(
                RESULT_PATH, "Checkpoints", "best_latest.pt"
            )
            shutil.copy(best_meta["best_path"], latest_best_path)
            print(
                f"Best checkpoint saved at epoch {best_meta['best_epoch']} "
                f"with eval={best_meta['best_eval']:.4f}. "
                f"Paths: {best_meta['best_path']} (run), {latest_best_path} (latest)."
            )
        else:
            print("No evaluation was run; best checkpoint not saved.")

        # Save results to CSV
        result_file = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}_{run_id}.csv"
        with open(os.path.join(RESULT_PATH, result_file), "w") as f:
            f.write("epoch,train_loss,train_eval,test_loss,test_eval\n")
            for epoch_idx, tr_loss, tr_eval, te_loss, te_eval in log:
                f.write(
                    f"{epoch_idx},"
                    f"{tr_loss},{str(tr_eval).replace(',', ';')},"
                    f"{te_loss},{str(te_eval).replace(',', ';')}\n"
                )
