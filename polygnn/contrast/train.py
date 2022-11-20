import torch
from torch import optim
from torch.cuda import amp
from polygnn_trainer.utils import analyze_gradients
from polygnn_trainer.train import initialize_training
from polygnn_trainer import constants
from torch import save as torch_save
import numpy as np
from torch_geometric.loader import DataLoader
from collections import deque
import time


def amp_train(model, view1, view2, optimizer, tc, apply_grad):
    """
    This function handles the parts of the per-epoch loop that torch's
    autocast methods can speed up. See https://pytorch.org/docs/1.9.1/notes/amp_examples.html
    """
    if tc.amp:
        scaler = torch.cuda.amp.GradScaler()
        with amp.autocast(enabled=True):
            output = model(view1, view2)
            loss = tc.loss_obj(output)

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not r
        # ecommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        if apply_grad:
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()
    else:
        output = model(view1, view2)
        loss = tc.loss_obj(output)
        loss.backward()
        # nn.utils.clip_grad_value_(model.temperature_param, 0.1)
        optimizer.step()

    return output, loss.item()


def train(
    model,
    train_pts,
    val_pts,
    cfg,
    transforms,
    max_time=np.inf,
    batches_per_step=1,
):
    """
    Train a model and save it to cfg.model_save_path. Models are
    checkpointed each time the epoch-wise ***validation*** loss is
    the lowest yet.

    Keyword arguments:
        model (polygnn_trainer.std_module.StandardModule): The model
            architecture.
        train_pts (List[pyg.data.Data]): The training data.
        val_pts (List[pyg.data.Data]): The validation data.
        cfg (polygnn_trainer.train.trainConfig)
        transforms (List[callable]): The contrastive augmentations.
        max_time (float): The training loop will break after the
            first epoch that exceeds `max_time` seconds.
    """
    # error handle inputs
    if cfg.model_save_path:
        if not cfg.model_save_path.endswith(".pt"):
            raise ValueError(f"The model_save_path you passed in does not end in .pt")
    # create the epoch suffix for this submodel
    epoch_suffix = f"{cfg.epoch_suffix}, fold {cfg.fold_index}"
    model.to(cfg.device, non_blocking=True)
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.hps.r_learn.value
    )  # Adam optimization

    val_loader = DataLoader(
        val_pts,
        batch_size=cfg.hps.batch_size.value * 2,
        shuffle=True,
        pin_memory=True,
        num_workers=10,
    )

    # intialize a few variables that get reset during the training loop
    min_tr_loss = np.inf  # epoch-wise loss
    min_val_loss = np.inf  # epoch-wise loss
    best_val_epoch = 0
    vanishing_grads = False
    exploding_grads = False
    grad_hist_per_epoch = deque(
        maxlen=constants.GRADIENT_HISTORY_LENGTH
    )  # gradients for last maxlen epochs

    # if we do not need to make a new dataloader inside each epoch,
    # let us make the dataloader now.
    if not cfg.get_train_dataloader:
        train_loader = DataLoader(
            train_pts,
            batch_size=cfg.hps.batch_size.value,
            shuffle=True,
            pin_memory=True,
            num_workers=10,
        )
    # Enable benchmarking for optimal function execution
    torch.backends.cudnn.benchmark = True
    start = time.time()
    for epoch in range(cfg.epochs):
        # Let's stop training and not waste time if we have vanishing
        # gradients early in training. We won't
        # be able to learn anything anyway.
        if vanishing_grads:
            print("Vanishing gradients detected")
        if exploding_grads:
            print("Exploding gradients detected")
        if (
            (vanishing_grads or exploding_grads)
            and (epoch < 50)
            and (cfg.break_on_bad_grads)
        ):
            break
        # if the errors or gradients are messed up later in training,
        # let us just re-initialize
        # the model. Perhaps this new initial point on the loss surface
        # will lead to a better local minima
        elif vanishing_grads or exploding_grads:
            model, optimizer = initialize_training(
                model, cfg.hps.r_learn.value, cfg.device
            )
        # augment data, if necessary
        if cfg.get_train_dataloader:
            train_pts = cfg.get_train_dataloader()
            train_loader = DataLoader(
                train_pts,
                batch_size=cfg.hps.batch_size.value,
                shuffle=True,
            )
        # ################################################################
        # Loop through training batches and compute the training loss
        # ################################################################
        model.train()
        epoch_tr_loss = 0
        for ind, data in enumerate(train_loader):
            data = data.to(cfg.device, non_blocking=True)
            with torch.no_grad():
                view1, view2 = data.clone(), data.clone()
                del data  # save space
                for fn in transforms:
                    view1, view2 = fn(view1), fn(view2)
            if (ind + 1) % batches_per_step == 0 or (ind + 1) == len(train_loader):
                apply_grad = True
            else:
                apply_grad = False
            if apply_grad == False:
                optimizer.zero_grad(set_to_none=True)
            _, loss_item = amp_train(model, view1, view2, optimizer, cfg, apply_grad)
            epoch_tr_loss += loss_item

        with torch.no_grad():
            epoch_tr_loss = epoch_tr_loss / len(train_pts)
            # ################################################################
            # Loop through validation batches and compute the validation loss
            # ################################################################
            model.eval()
            epoch_val_loss = 0
            for ind, data in enumerate(val_loader):
                data = data.to(cfg.device, non_blocking=True)
                view1, view2 = data.clone(), data.clone()
                del data  # save space
                for fn in transforms:
                    view1, view2 = fn(view1), fn(view2)
                output = model(view1, view2)
                loss_item = cfg.loss_obj(output).item()
                epoch_val_loss += loss_item
            epoch_val_loss = epoch_val_loss / len(val_pts)
            # ################################################################
            # Compute and print the gradient statistics
            # ################################################################
            _, ave_grads, _ = analyze_gradients(
                model.named_parameters(), allow_errors=False
            )
            grad_hist_per_epoch.append(ave_grads)
            if np.sum(grad_hist_per_epoch) == 0:
                vanishing_grads = True
            else:
                vanishing_grads = False
            if int(np.sum(np.isnan(grad_hist_per_epoch))) == len(grad_hist_per_epoch):
                exploding_grads = True
            else:
                exploding_grads = False
            # ################################################################
            # Print the epoch summary
            # ################################################################
            print(f"\nEpoch {epoch}{epoch_suffix}", flush=True)
            print(
                f"[avg. train loss] {epoch_tr_loss} [avg. val loss] {epoch_val_loss}",
                flush=True,
            )
            # Checkpoint model, if necessary.
            model_saved = False
            if getattr(cfg, "save_each_epoch", False):
                save_path = f"epoch{epoch}__{cfg.model_save_path}"
                torch_save(model.state_dict(), save_path)
                print(f"Model saved to {save_path}.", flush=True)
                model_saved = True
            if epoch_val_loss < min_val_loss:
                min_tr_loss = epoch_tr_loss
                min_val_loss = epoch_val_loss
                best_val_epoch = epoch
                if not model_saved and cfg.model_save_path:
                    torch_save(model.state_dict(), cfg.model_save_path)
                    print("Best model saved according to validation loss.", flush=True)
            print(
                f"[best val epoch] {best_val_epoch} [best avg. train loss] {min_tr_loss} [best avg. val loss] {min_val_loss}",
                flush=True,
            )
            if max_time < np.inf:
                end = time.time()
                so_far = end - start
                time_remaining = max_time - so_far
                print(
                    f"Time so far / time remaining: {str(round(so_far, 3))}s / {str(round(time_remaining, 3))}s",
                    flush=True,
                )
                if time_remaining < 0:
                    print(
                        f"Breaking training loop after {epoch}/{cfg.epochs} epochs.",
                        flush=True,
                    )
                    break
            # ################################################################
    return min_tr_loss
