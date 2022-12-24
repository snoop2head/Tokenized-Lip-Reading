import os
import argparse
import wandb
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torch.nn import DataParallel
from x_transformers import TransformerWrapper, Encoder, Decoder
import warnings
from dataset.dataset import RDropDataset
from dataset.dataset_finetune import RDropDataset_finetune
from pretrain import train, validate
from finetune import train_finetune, validate_finetune
from utils import *
from models import *


def main():
    CONSTANTS = load_config("CONSTANTS")
    parser = argparse.ArgumentParser(
        description="Yonsei Final Project : Tokenized Lip Reading"
    )
    parser.add_argument("--run", type=str, choices=["pretrain", "finetune"])
    running_args = parser.parse_args()

    # path definition
    ROOT_DIR = CONSTANTS["ROOT_DIR"]
    LABEL_DIR = CONSTANTS["LABEL_DIR"]
    LOAD_PATH = CONSTANTS["LOAD_PATH"]
    SAVE_GPT_PATH = CONSTANTS["SAVE_GPT_PATH"]
    SAVE_PATH = CONSTANTS["SAVE_PATH"]
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # constant definition
    SEED = CONSTANTS["SEED"]
    NUM_CLASS = CONSTANTS["NUM_CLASS"]
    CHANNEL = CONSTANTS["CHANNEL"]
    IMG_RES = CONSTANTS["IMG_RES"]

    # hyperparameter definition
    seed_everything(SEED)
    HYPERPARAMS = load_config("MLP_HYPERPARAMS")  # load hyperparams

    wandb.init(project=HYPERPARAMS.wandb_project_name, entity=CONSTANTS.WANDB_USER)
    wandb.config.update(HYPERPARAMS)  # add hyperparams to wandb
    print("current parameters: ", HYPERPARAMS)

    # image transformation
    mean = [0.5037278, 0.503253, -7.063131e-05]
    std = [0.11415035, 0.13153046, 0.07024033]

    vanilla_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    if running_args == "pretrain":
        train_dataset = RDropDataset(
            label_dir=LABEL_DIR, split="train", transforms=True
        )
        train_dataset.set_transform(vanilla_transform)
        valid_dataset = RDropDataset(label_dir=LABEL_DIR, split="val", transforms=True)
        valid_dataset.set_transform(vanilla_transform)
        test_dataset = RDropDataset(label_dir=LABEL_DIR, split="test", transforms=True)
        test_dataset.set_transform(vanilla_transform)
    elif running_args == "finetune":
        train_dataset = RDropDataset_finetune(
            root_dir=ROOT_DIR, label_dir=LABEL_DIR, split="train", transforms=True
        )
        train_dataset.set_transform(vanilla_transform)
        valid_dataset = RDropDataset_finetune(
            root_dir=ROOT_DIR, label_dir=LABEL_DIR, split="val", transforms=True
        )
        valid_dataset.set_transform(vanilla_transform)
        test_dataset = RDropDataset_finetune(
            root_dir=ROOT_DIR, label_dir=LABEL_DIR, split="test", transforms=True
        )
        test_dataset.set_transform(vanilla_transform)
    else:
        raise ValueError("Invalid run argument: run with --run [pretrain|finetune]")

    device, NUM_WORKERS = check_device()

    train_loader = DataLoader(
        train_dataset,
        batch_size=HYPERPARAMS.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=HYPERPARAMS.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=HYPERPARAMS.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )

    model = VideoImageModel(
        num_tokens=NUM_CLASS,
        max_seq_len=58,
        attn_layers=Encoder(
            dim=HYPERPARAMS.working_dim,
            depth=HYPERPARAMS.num_layers,
            heads=HYPERPARAMS.heads,
            layer_dropout=HYPERPARAMS.layer_dropout,
            ff_dropout=HYPERPARAMS.ff_dropout,  # Let's set this as 0.3 or higher
            use_rmsnorm=True,
            ff_glu=True,
            rotary_pos_emb=True,
        ),
        channels=CHANNEL,
        image_size=IMG_RES,
        patch_size=HYPERPARAMS.patch_size,
        dim=HYPERPARAMS.working_dim,
        post_emb_norm=False,
        emb_dropout=HYPERPARAMS.emp_dropout,  # Let's set this as 0.15 (lower than 0.25)
    )

    model = DataParallel(model)
    model.to(device)

    if running_args == "pretrain":
        pass
    elif running_args == "finetune":
        model.load_state_dict(torch.load(LOAD_PATH))

        del model.module.decoder
        del model.module.gpt

        clf_model = VideoImageModelForClassification(
            num_tokens=500,
            max_seq_len=58,
            attn_layers=model.module.attn_layers,
            channels=CHANNEL,
            image_size=IMG_RES,
            patch_size=HYPERPARAMS.patch_size,
            dim=HYPERPARAMS.working_dim,
            post_emb_norm=False,
            emb_dropout=HYPERPARAMS.emp_dropout,  # Let's set this as 0.15 (lower than 0.25)
            num_classes=NUM_CLASS,
        )
        clf_model = DataParallel(clf_model)
        clf_model.to(device)

    else:
        raise ValueError("Invalid run argument: run with --run [pretrain|finetune]")

    if running_args == "pretrain":
        criterion = nn.CrossEntropyLoss()
    elif running_args == "finetune":
        # labelsmoothing loss
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = AdamW(
        model.parameters(),
        lr=HYPERPARAMS.learning_rate,
        weight_decay=HYPERPARAMS.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=HYPERPARAMS.num_epochs, verbose=True)

    if running_args == "pretrain":
        best_model = train(
            CONFIG=HYPERPARAMS,
            save_path=SAVE_GPT_PATH,
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
        )

        # run test
        test_loss, _, _ = validate(
            HYPERPARAMS, test_loader, best_model, criterion, device
        )

        print(f"Test loss: {test_loss:4.2}")
        wandb.log(
            {
                "test/loss": test_loss,
            }
        )

    elif running_args == "finetune":
        best_clf_model = train_finetune(
            CONFIG=HYPERPARAMS,
            save_path=SAVE_PATH,
            model=clf_model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
        )
        # run test
        test_loss, test_acc1, test_acc5 = validate_finetune(
            HYPERPARAMS, test_loader, best_clf_model, criterion, device
        )

        print(
            f"Test loss: {test_loss:4.2}, test acc1: {test_acc1:4.2}, test acc5: {test_acc5:4.2}"
        )
        wandb.log(
            {
                "test/loss": test_loss,
                "test/top5_accuracy": test_acc5,
                "test/top1_accuracy": test_acc1,
            }
        )

    wandb.finish()


if __name__ == "__main__":
    main()
