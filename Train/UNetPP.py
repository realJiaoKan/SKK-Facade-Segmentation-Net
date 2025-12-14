import os
from datetime import datetime
from uuid import uuid4

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from Networks.UNetPP import Network
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

DROPOUT = 0.18

_RATIO = np.array([0.3548, 0.4057, 0.0236, 0.1650, 0.0509])
# [0.72, 0.69, 2.41, 0.93, 1.74]
CLASS_WEIGHTS = torch.tensor(np.float32(1 / np.sqrt(_RATIO))).to(DEVICE)
LR = 1e-3

RESULT_PATH = "Train/Results/UNetMini"

if __name__ == "__main__":
    train_loader, test_loader = load_loader(BATCH_SIZE)

    for _ in range(NUM_RUNS):
        run_id = str(uuid4())[:8]
        print(f"=== UNetMini Run {run_id} ===")
        model = Network(
            INPUT_SHAPE,
            OUTPUT_SHAPE[0],
            dropout=DROPOUT,
        ).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LR)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=1e-3,
            total_steps=NUM_EPOCHS * len(train_loader),
            pct_start=0.2,
            anneal_strategy="linear",
        )
        criterion = get_criterion(
            name="ce_dice",
            weight=CLASS_WEIGHTS,
            dice_weight=1.0,
        )
        log = run(
            model,
            train_loader,
            test_loader,
            optimizer,
            scheduler,
            criterion,
            ap_evaluator,
            NUM_EPOCHS,
            eval_after=EVAL_AFTER,
        )
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
