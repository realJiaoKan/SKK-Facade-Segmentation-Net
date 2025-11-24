import os
from datetime import datetime
from uuid import uuid4

import torch
import torch.optim as optim

from Networks.UNetMini import Network
from Train.lib.evaluation import ap_torch as ap_evaluator
from Train.lib.train import run

from settings import DEVICE
from Datasets.MiniFacade import load_loader

BATCH_SIZE = 64
NUM_EPOCHS = 70
EVAL_AFTER = 0
NUM_RUNS = 3

INPUT_SHAPE = (3, 256, 256)
OUTPUT_SHAPE = (5, 256, 256)

DROPOUT = 0.05

CLASS_WEIGHTS = torch.tensor([0.72, 0.69, 2.41, 0.93, 1.74]).to(DEVICE)

RESULT_PATH = "Train/Results/UNetMini"

if __name__ == "__main__":
    train_loader, test_loader = load_loader(BATCH_SIZE)

    for _ in range(NUM_RUNS):
        run_id = str(uuid4())[:8]
        print(f"=== UNetMini Run {run_id} ===")
        model = Network(
            INPUT_SHAPE,
            OUTPUT_SHAPE,
            dropout=DROPOUT,
        ).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
        log = run(
            model,
            train_loader,
            test_loader,
            optimizer,
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
