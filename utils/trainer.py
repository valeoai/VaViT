# Copyright 2026 - Valeo Comfort and Driving Assistance - valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import warnings

import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.metrics import fast_hist, overall_accuracy, per_class_accuracy, per_class_iu


class TrainingManager:
    def __init__(
        self,
        net,
        loss,
        loader_train,
        loader_val,
        train_sampler,  # If provided, we assume distributed training
        optim,
        scheduler,
        max_epoch,
        path,
        rank,
        world_size,
        fp16=True,
        class_names=None,
        tensorboard=True,
    ):
        # Optim. methods
        self.optim = optim
        self.fp16 = fp16
        self.scaler = GradScaler() if fp16 else None
        self.dtype = torch.float16 if fp16 else torch.bfloat16
        self.scheduler = scheduler

        # Dataloaders
        self.max_epoch = max_epoch
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.train_sampler = train_sampler
        self.class_names = class_names

        # Network
        self.net = net
        self.rank = rank
        self.world_size = world_size
        print(f"Trainer on gpu: {self.rank}. World size:{self.world_size}.")

        # Loss
        self.loss = loss

        # Checkpoints
        self.best_miou = 0
        self.current_epoch = 0
        self.path_to_ckpt = path

        # Monitoring
        if tensorboard and (self.rank == 0 or self.rank is None):
            self.writer_train = SummaryWriter(
                path + "/tensorboard/train/",
                purge_step=self.current_epoch * len(self.loader_train),
                flush_secs=30,
            )
            self.writer_val = SummaryWriter(
                path + "/tensorboard/val/",
                purge_step=self.current_epoch,
                flush_secs=30,
            )
        else:
            self.writer_val = None
            self.writer_train = None

    def print_log(self, running_loss, oAcc, mAcc, mIoU, ious, class_names):
        if self.rank == 0 or self.rank is None:
            # Global score
            log = (
                f"\nEpoch: {self.current_epoch:d} :\n"
                + f" Loss = {running_loss:.3f}"
                + f" - oAcc = {oAcc:.1f}"
                + f" - mAcc = {mAcc:.1f}"
                + f" - mIoU = {mIoU:.1f}"
            )
            print(log)
            # Per class score
            log = ""
            for i, s in enumerate(ious):
                if class_names is None:
                    log += f"Class {i}: {100 * s:.1f} - "
                else:
                    log += f"{class_names[i]}: {100 * s:.1f} - "
            print(log[:-3])
            # Recall best mIoU
            print(f"Best mIoU was {self.best_miou:.1f}.")
            sys.stdout.flush()

    def gather_scores(self, list_tensors):
        if self.rank == 0:
            tensor_reduced = [
                [torch.empty_like(t) for _ in range(self.world_size)]
                for t in list_tensors
            ]
            for t, t_reduced in zip(list_tensors, tensor_reduced):
                torch.distributed.gather(t, t_reduced)
            tensor_reduced = [sum(t).cpu() for t in tensor_reduced]
            return tensor_reduced
        else:
            for t in list_tensors:
                torch.distributed.gather(t)

    def one_epoch(self, training=True):

        # Train or eval mode
        self.optim.zero_grad(set_to_none=True)
        if training:
            net = self.net.train()
            loader = self.loader_train
            if self.rank == 0 or self.rank is None:
                print("\nTraining: %d/%d epochs" % (self.current_epoch, self.max_epoch))
            writer = self.writer_train
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(self.current_epoch)
        else:
            net = self.net.eval()
            loader = self.loader_val
            if self.rank == 0 or self.rank is None:
                print(
                    "\nValidation: %d/%d epochs" % (self.current_epoch, self.max_epoch)
                )
            writer = self.writer_val
        print_freq = np.max((len(loader) // 10, 1))

        # Stat.
        running_loss = 0.0
        confusion_matrix = 0.0

        # Loop over mini-batches
        if self.rank == 0 or self.rank is None:
            bar_format = "{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}"
            loader = tqdm(loader, bar_format=bar_format)
        for it, batch in enumerate(loader):
            # Network inputs
            feat = batch["feat"].cuda(self.rank, non_blocking=True)
            neighbors_emb = batch["neighbors"].cuda(self.rank, non_blocking=True)
            idx_to_bev = batch["idx_bev"].cuda(self.rank, non_blocking=True)
            xy_bev = batch["xy_bev"].cuda(self.rank, non_blocking=True)
            idx_occ = batch["idx_occ"].cuda(self.rank, non_blocking=True)
            if batch["idx_pad"] is not None:
                idx_pad = batch["idx_pad"].cuda(self.rank, non_blocking=True)
            else:
                idx_pad = None
            bev_splits = batch["splits_bev"].cuda(self.rank, non_blocking=True)
            labels = batch["labels"].cuda(self.rank, non_blocking=True)
            upsample = batch["upsample"].cuda(self.rank, non_blocking=True)

            net_inputs = (
                feat,
                neighbors_emb,
                xy_bev,
                idx_to_bev,
                idx_occ,
                idx_pad,
                bev_splits,
            )

            # Get prediction and loss
            with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                # Logits
                if training:
                    out = net(*net_inputs)
                    out = out[upsample]
                else:
                    with torch.no_grad():
                        out = net(*net_inputs)
                        out = out[upsample]
                loss = self.loss(out, labels)
            running_loss += loss.item()

            # Confusion matrix
            with torch.no_grad():
                nb_class = out.shape[1]
                pred_label = out.max(1)[1]
                where = labels != 255
                confusion_matrix += (
                    fast_hist(pred_label[where], labels[where], nb_class).detach().cpu()
                )

            # Logs
            if it % print_freq == print_freq - 1 or it == len(loader) - 1:
                # Gather scores
                temp_cm = confusion_matrix.cuda(self.rank)
                temp_loss = torch.tensor(running_loss).cuda(self.rank)
                if self.train_sampler is not None:
                    out = self.gather_scores([temp_loss, temp_cm])
                else:
                    out = [temp_loss.cpu(), temp_cm.cpu()]

                if self.rank == 0 or self.rank is None:
                    running_loss_reduced = out[0].item() / self.world_size / (it + 1)
                    oAcc = 100 * overall_accuracy(out[1])
                    mAcc = 100 * np.nanmean(per_class_accuracy(out[1]))
                    ious = per_class_iu(out[1])
                    mIoU = 100 * np.nanmean(ious)

                    # Print score
                    self.print_log(
                        running_loss_reduced,
                        oAcc,
                        mAcc,
                        mIoU,
                        ious,
                        self.class_names[0],
                    )

                    # Save in tensorboard
                    if (writer is not None) and (training or it == len(loader) - 1):
                        header = "Train" if training else "Test"
                        step = (
                            self.current_epoch * len(loader) + it
                            if training
                            else self.current_epoch
                        )
                        writer.add_scalar(header + "/loss", running_loss_reduced, step)
                        writer.add_scalar(header + "/oAcc", oAcc, step)
                        writer.add_scalar(header + "/mAcc", mAcc, step)
                        writer.add_scalar(header + "/mIoU", mIoU, step)
                        for i in range(len(self.optim.param_groups)):
                            writer.add_scalar(
                                header + f"/lr_{i}",
                                self.optim.param_groups[i]["lr"],
                                step,
                            )
                        for i, s in enumerate(ious):
                            cname = (
                                f"Class {i}"
                                if self.class_names is None
                                else self.class_names[0][i]
                            )
                            writer.add_scalar(header + f"/ious/{cname}", 100 * s, step)

            # Gradient step
            if training:
                if self.fp16:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optim.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optim.zero_grad(set_to_none=True)

        # Return score
        if self.rank == 0 or self.rank is None:
            return mIoU
        else:
            return None

    def load_state(self, best=False):
        filename = self.path_to_ckpt
        filename += "/ckpt_best.pth" if best else "/ckpt_last.pth"
        rank = 0 if self.rank is None else self.rank
        ckpt = torch.load(
            filename,
            map_location=f"cuda:{rank}",
        )
        self.net.load_state_dict(ckpt["net"])
        if ckpt.get("optim") is None:
            warnings.warn("Optimizer state not available")
        else:
            self.optim.load_state_dict(ckpt["optim"])
        if self.scheduler is not None:
            if ckpt.get("scheduler") is None:
                warnings.warn("Scheduler state not available")
            else:
                self.scheduler.load_state_dict(ckpt["scheduler"])
        if self.fp16:
            if ckpt.get("scaler") is None:
                warnings.warn("Scaler state not available")
            else:
                self.scaler.load_state_dict(ckpt["scaler"])
        if ckpt.get("best_miou") is not None:
            self.best_miou = ckpt["best_miou"]
        if ckpt.get("epoch") is not None:
            self.current_epoch = ckpt["epoch"] + 1
        print(
            f"Checkpoint loaded on {torch.device(rank)} (cuda:{rank}): {self.path_to_ckpt}"
        )

    def save_state(self, best=False):
        if self.rank == 0 or self.rank is None:
            dict_to_save = {
                "epoch": self.current_epoch,
                "net": self.net.state_dict(),
                "optim": self.optim.state_dict(),
                "scheduler": self.scheduler.state_dict()
                if self.scheduler is not None
                else None,
                "scaler": self.scaler.state_dict() if self.fp16 else None,
                "best_miou": self.best_miou,
            }
            filename = self.path_to_ckpt
            filename += "/ckpt_best.pth" if best else "/ckpt_last.pth"
            torch.save(dict_to_save, filename)

    def train(self):
        for _ in range(self.current_epoch, self.max_epoch):
            # Train
            self.one_epoch(training=True)
            # Val
            miou = self.one_epoch(training=False)
            # Save best checkpoint
            if miou is not None and miou > self.best_miou:
                self.best_miou = miou
                self.save_state(best=True)
                print(f"\n\n*** New best mIoU: {self.best_miou:.1f}.\n")
            # Save last checkpoint
            self.save_state()
            # Increase epoch number
            self.current_epoch += 1
        if self.rank == 0 or self.rank is None:
            print("Finished Training")
