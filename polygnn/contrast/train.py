from torch import optim
from torch.cuda import amp
from polygnn_trainer.utils import analyze_gradients
from polygnn_trainer.train import initialize_training, minibatch
from polygnn_trainer import constants
from torch import save as torch_save
import numpy as np
from torch_geometric.loader import DataLoader
from collections import deque


def amp_train(model, view1, view2, optimizer, tc, selector_dim):
    """
    This function handles the parts of the per-epoch loop that torch's
    autocast methods can speed up. See https://pytorch.org/docs/1.9.1/notes/amp_examples.html
    """
    if tc.amp:
        with amp.autocast(device_type=tc.device, enabled=True):
            output = model(view1, view2)
            loss = tc.loss_obj(output)
            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            amp.scaler.scale(loss).backward()

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            amp.scaler.step(optimizer)

            # Updates the scale for next iteration.
            amp.scaler.update()
    else:
        output = model(view1, view2)
        loss = tc.loss_obj(output)
        loss.backward()
        optimizer.step()

    return output, loss.item()


def train(
    model,
    train_pts,
    # val_pts,
    cfg,
    transforms,
):
    """
    Train a model and save it to cfg.model_save_path. Models are
    checkpointed each time the epoch-wise ***training*** loss is
    the lowest yet.

    Keyword arguments:
        model (polygnn_trainer.std_module.StandardModule): The model architecture.
        train_pts (List[pyg.data.Data]): The training data.
        val_pts (List[pyg.data.Data]): The validation data.
        cfg (polygnn_trainer.train.trainConfig)
        transforms (List[callable]): The contrastive augmentations.
    """
    # error handle inputs
    if cfg.model_save_path:
        if not cfg.model_save_path.endswith(".pt"):
            raise ValueError(f"The model_save_path you passed in does not end in .pt")
    # create the epoch suffix for this submodel
    epoch_suffix = f"{cfg.epoch_suffix}, fold {cfg.fold_index}"
    model.to(cfg.device)
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.hps.r_learn.value
    )  # Adam optimization

    # val_loader = DataLoader(
    #     val_pts, batch_size=cfg.hps.batch_size.value * 2, shuffle=True
    # )

    # intialize a few variables that get reset during the training loop
    min_tr_loss = np.inf  # epoch-wise loss
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
            train_pts, batch_size=cfg.hps.batch_size.value, shuffle=True
        )
    for epoch in range(cfg.epochs):
        # Let's stop training and not waste time if we have vanishing
        # gradients early in training. We won't
        # be able to learn anything anyway.
        if vanishing_grads:
            print("Vanishing gradients detected")
        if exploding_grads:
            print("Exploding gradients detected")
        if (vanishing_grads or exploding_grads) and (epoch < 50):
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
                train_pts, batch_size=cfg.hps.batch_size.value, shuffle=True
            )
        # ################################################################
        # train loop
        # ################################################################
        model.train()
        epoch_loss = 0
        for ind, data in enumerate(train_loader):  # loop through training batches
            data = data.to(cfg.device)
            view1, view2 = transforms[0](data), transforms[0](data)
            del data  # save space
            for fn in transforms[1:]:
                view1, view2 = fn(view1), fn(view2)
            # TODO: In an ideal world, we should not have to send view1/2
            # to the GPU here.
            view1, view2 = view1.to(cfg.device), view2.to(cfg.device)
            optimizer.zero_grad()
            _, loss_item = amp_train(
                model, view1, view2, optimizer, cfg, selector_dim=None
            )
            epoch_loss += loss_item
        epoch_loss = epoch_loss / (ind + 1)
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
        print(f"\nEpoch {epoch}{epoch_suffix}", flush=True)
        print(f"[avg. train loss] {epoch_loss}", flush=True)

        if epoch_loss < min_tr_loss:
            min_tr_loss = epoch_loss
            best_val_epoch = epoch
            if cfg.model_save_path:
                torch_save(model.state_dict(), cfg.model_save_path)
                print("Best model saved according to training loss.", flush=True)

        print(
            f"[best val epoch] {best_val_epoch} [best avg. train loss] {min_tr_loss}",
            flush=True,
        )

    return min_tr_loss
