import os
from tqdm import tqdm

import torch

from settings import *


def train_one_epoch(
    model, loader, optimizer, scheduler, criterion, evaluator, eval_flag=True
):
    model.train()
    running_loss = 0
    evaluation = 0
    outs = []
    ys = []
    total = 0

    for X, y in tqdm(loader, desc="Training", leave=False):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Forward pass and backward pass
        optimizer.zero_grad()
        out = model(X)

        # Convert one-hot to label index for CE loss
        y_label = y.argmax(dim=1)
        loss = criterion(out, y_label)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Loss and evaluation
        running_loss += loss.item() * y.size(0)
        outs.append(out.detach())
        ys.append(y.detach())
        total += y.size(0)

    # Train loss and train evaluation
    if eval_flag and False:  # For lack of GPU memory, disable train evaluation
        evaluation = evaluator(
            torch.cat(outs, dim=0).float(), torch.cat(ys, dim=0).long()
        )
    else:
        evaluation = [0.0 for _ in range(N_CLASS)]
    return running_loss / total, evaluation


def evaluate(model, loader, criterion, evaluator, eval_flag=True):
    model.eval()
    running_loss = 0
    evaluation = 0
    outs = []
    ys = []
    total = 0

    with torch.no_grad():
        for X, y in tqdm(loader, desc="Evaluating", leave=False):
            X, y = X.to(DEVICE), y.to(DEVICE)

            # Forward pass
            out = model(X)
            y_label = y.argmax(dim=1)
            loss = criterion(out, y_label)

            # Loss and evaluation
            running_loss += loss.item() * y.size(0)
            outs.append(out)
            ys.append(y)
            total += y.size(0)

    # Test loss and test evaluation
    if eval_flag:
        evaluation = evaluator(
            torch.cat(outs, dim=0).float(), torch.cat(ys, dim=0).long()
        )
    else:
        evaluation = [0.0 for _ in range(N_CLASS)]
    return running_loss / total, evaluation


def run(
    model,
    train_loader,
    test_loader,
    optimizer,
    scheduler,
    criterion,
    evaluator,
    epochs,
    eval_after=0.8,
    save_best_path=None,
):
    log = []
    best_eval = -float("inf")
    best_epoch = None

    if save_best_path:
        os.makedirs(os.path.dirname(save_best_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        eval_ready = epoch / epochs >= eval_after

        # Training
        tr_loss, tr_eval = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            evaluator,
            eval_flag=eval_ready,
        )

        # Evaluation
        te_loss, te_eval = evaluate(
            model,
            test_loader,
            criterion,
            evaluator,
            eval_flag=eval_ready,
        )

        # Save the best-performing model checkpoint based on evaluation metric
        if eval_ready:
            current_eval = float(np.nanmean(te_eval))
            if save_best_path and current_eval > best_eval:
                best_eval = current_eval
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "eval": current_eval,
                    },
                    save_best_path,
                )
                print(
                    f"Saved new best model to {save_best_path} "
                    f"(epoch={epoch}, eval={current_eval:.4f})"
                )

        # Logging
        print(
            f"Epoch {epoch}: train_loss={tr_loss:.4f}, train_eval={np.average(tr_eval):.4f}, "
            f"test_loss={te_loss:.4f}, test_eval={np.average(te_eval):.4f}"
        )
        log.append((epoch, tr_loss, tr_eval, te_loss, te_eval))

    best_meta = {
        "best_eval": best_eval if best_epoch is not None else None,
        "best_epoch": best_epoch,
        "best_path": save_best_path if best_epoch is not None else None,
    }
    return log, best_meta
