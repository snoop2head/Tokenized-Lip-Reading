import os
import glob
import wandb
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from utils.metric import AverageMeter, Summary, ProgressMeter


def validate(CONFIG, val_loader, model, criterion, device):
    # Reference: https://github.com/pytorch/examples/blob/00ea159a99f5cb3f3301a9bf0baa1a5089c7e217/imagenet/main.py#L313-L353
    losses = AverageMeter("Loss", ":.4f", Summary.AVERAGE)
    top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader), [losses, top1, top5], prefix="Validation: "
    )
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():

        for i, (
            images,
            flipped_images,
            videos,
            flipped_videos,
            attention_mask,
            input_ids,
        ) in enumerate(tqdm(val_loader)):
            images = images.to(device)
            flipped_images = flipped_images.to(device)
            videos = videos.to(device)
            flipped_videos = flipped_videos.to(device)
            # attention mask: https://github.com/huggingface/transformers/blob/799cea64ac1029d66e9e58f18bc6f47892270723/src/transformers/models/roberta/modeling_roberta.py#L234-L262
            attention_mask = attention_mask.to(device)
            input_ids = input_ids.to(device)

            # get cross entropy loss
            ce_loss_org = model(images, videos, attention_mask, input_ids)
            ce_loss_flip = model(
                flipped_images, flipped_videos, attention_mask, input_ids
            )
            ce_loss = (ce_loss_org + ce_loss_flip) / 2
            losses.update(ce_loss.mean().item(), images.size(0))

        progress.display_summary()
    return (
        losses.avg,
        top1.avg,
        top5.avg,
    )


class EarlyStopPatience(nn.Module):
    def __init__(self, patience=10):
        self.patience = patience
        self.count = 0
        self.best_score = None
        self.bool_early_stop = False

    def __call__(self, score: float):
        if self.best_score is None:
            self.best_score = score
        # accumulate counting if the score is not better than the best score
        elif score < self.best_score:
            self.count += 1
            if self.count >= self.patience:
                self.bool_early_stop = True
                print("Early stopping")
        # renew count and best score if the maximum score is achieved
        elif score >= self.best_score:
            self.best_score = score
            self.count = 0
        return self.bool_early_stop


def train(
    CONFIG,
    model,
    train_loader,
    valid_loader,
    optimizer,
    save_path=None,
    scheduler=None,
    criterion=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    CONFIG.logging_steps = len(train_loader)
    early_stop_acc1 = EarlyStopPatience(patience=CONFIG.early_stop_patience)
    best_valid_acc1 = 0.0
    best_valid_acc5 = 0.0
    best_valid_loss = 1e10

    for epoch in range(CONFIG.num_epochs):
        # Train Code Reference: https://github.com/pytorch/examples/blob/00ea159a99f5cb3f3301a9bf0baa1a5089c7e217/imagenet/main.py#L266-L310
        losses = AverageMeter("Loss", ":.4f")
        top1 = AverageMeter("Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")
        progress = ProgressMeter(
            len(train_loader),
            [losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch),
        )

        # switch to train mode
        model.train()

        for iter, (
            images,
            flipped_images,
            videos,
            flipped_videos,
            attention_mask,
            input_ids,
        ) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            flipped_images = flipped_images.to(device)
            videos = videos.to(device)
            flipped_videos = flipped_videos.to(device)
            attention_mask = attention_mask.to(device)
            input_ids = input_ids.to(device)

            # get cross entropy loss
            ce_loss_org = model(images, videos, attention_mask, input_ids)
            ce_loss_flip = model(
                flipped_images, flipped_videos, attention_mask, input_ids
            )
            ce_loss = (ce_loss_org + ce_loss_flip) / 2

            losses.update(ce_loss.mean().item(), images.size(0))

            # compute gradient and do step
            optimizer.zero_grad()
            ce_loss.backward(torch.ones(ce_loss.shape).to(device))  # for multi-gpu
            optimizer.step()

        progress.display(iter)  # display train status
        wandb.log(
            {
                "epoch": epoch + 1,
                "train/loss": losses.avg,
                "learning_rate": optimizer.param_groups[0]["lr"],
                # "train/top5_accuracy": top5.avg,
                # "train/top1_accuracy": top1.avg,
            }
        )

        # Validate on each epoch
        print(f"Epoch {epoch+1} Finished... Validating")
        (
            valid_loss,
            valid_acc1,
            valid_acc5,
        ) = validate(CONFIG, valid_loader, model, criterion, device)

        wandb.log(
            {
                "valid/loss": valid_loss,
                # "valid/top5_accuracy": valid_acc5,
                # "valid/top1_accuracy": valid_acc1,
            }
        )

        if best_valid_loss > valid_loss:
            # find previous checkpoint that contains
            # if found, delete it

            previous_checkpoints = glob.glob(os.path.join(save_path, f"loss_*.ckpt"))
            for checkpoint in previous_checkpoints:
                os.remove(checkpoint)

            print("New valid model for val loss! saving the model...")
            torch.save(
                model.state_dict(),
                os.path.join(
                    save_path,
                    f"loss_{valid_loss:4.2}.ckpt",
                ),
            )

            best_valid_acc1 = valid_acc1
            best_valid_acc5 = valid_acc5
            best_valid_loss = valid_loss

        if scheduler is not None:
            scheduler.step()  # update learning rate

        # early stop based on validation top1 accuracy patience
        # if early_stop_acc1(valid_acc1):
        #     break

    # load best model
    print("Loading best model...")
    print(f"Best valid loss: {best_valid_loss:4.2}")
    print(f"Best valid top5 accuracy: {best_valid_acc5:4.2}")
    print(f"Best valid top1 accuracy: {best_valid_acc1:4.2}")
    wandb.log({"best/valid_top5_acc": best_valid_acc5})
    wandb.log({"best/valid_top1_acc": best_valid_acc1})

    model.load_state_dict(
        torch.load(
            os.path.join(
                save_path,
                f"loss_{best_valid_loss:4.2}.ckpt",
            )
        )
    )
    return model
