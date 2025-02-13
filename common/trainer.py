import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import common.models as models
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import Subset
from common.logger import TensorboardLogger
import common.utils as utils

from torch.utils.data import Dataset
#from ignite.contrib import metrics

from tqdm import tqdm
import common.visualize_utilities as plt_util
import datetime
import os
import sys

import argparse
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cv2
from PIL import Image
#from imports.SelfMedMAE.lib.trainers.mae3d_trainer import MAE3DTrainer

from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

def add_training_args(parser):
    group = parser.add_argument_group("Training Argmunets", "Arguments for the training")
    group.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    group.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    group.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    group.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    group.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    group.add_argument('--device', type=str, default=None, 
                        help='device to use (default: None)')
    group.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    group.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    group.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    group.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    group.add_argument('--scheduler', type=str, default="CosineAnnealingWarmRestarts", help='scheduler to use (default: CosineAnnealingWarmRestarts)')
    group.add_argument('--loss', type=str, default="mse", help='loss function to use (default: "mse")')
    group.add_argument('--optimizer', type=str, default="sgd", help='optimizer to use (default: "sgd")')

    group.add_argument('--loss_reduction', type=str, default="mean", help='loss reduction (default: mean)')
    group.add_argument('--sgd_momentum', type=float, default=0.9, help='sgd momentum (default: 0.9)')
    group.add_argument('--wd', type=float, default=5e-4, help='sgd weight decay (default: 5e-4)')
    group.add_argument('--adam_beta1', type=float, default=0.9, help='adam beta1 (default: 0.9)')
    group.add_argument('--adam_beta2', type=float, default=0.999, help='adam beta2 (default: 0.999)')
    group.add_argument('--steplr_stepsize', type=int, default=2, help='stepLR step size (default: 1)')
    group.add_argument('--steplr_gamma', type=float, default=0.92, help='stepLR gamma (default: 0.92)')
    group.add_argument('--cosine-eta-min', type=float, default=1e-5, help='cosine annealing eta_min (default: 1e-5)')

    #MAE implementation detils
    group.add_argument('--layer-decay', type=float, default=0.0, help='layer decay (default: 0.0)')
    group.add_argument('--smoothing', type=float, default=0.0, help='label smoothing (default: 0.1), applies only to cross entropy loss and when mixup is not used')
    group.add_argument('--mixup', type=float, default=0.0, help='mixup alpha (default: 0.0)') # https://arxiv.org/pdf/1710.09412 (MAE finetune 0.8, 1.0)
    group.add_argument('--cutmix', type=float, default=0.0, help='cutmix alpha (default: 0.0)') # https://arxiv.org/abs/1905.04899, https://github.com/clovaai/CutMix-PyTorch
    group.add_argument('--cutmix-minmax', type=float, nargs='+', default=None, help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    group.add_argument('--mixup-prob', type=float, default=1.0, help='Probability of performing mixup or cutmix when either/both is enabled')
    group.add_argument('--mixup-switch-prob', type=float, default=0.5, help='Probability of switching to cutmix when both mixup and cutmix enabled')
    group.add_argument('--mixup-mode', type=str, default='batch', help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')


def parse_scheduler(scheduler_name, args):
    if scheduler_name is None:
        return None, None
    elif scheduler_name == "StepLR":
        return StepLR, {"step_size": args.steplr_stepsize, "gamma": args.steplr_gamma}  # 1, 0.92
    elif scheduler_name == "CosineAnnealingWarmRestarts":
        return CosineAnnealingWarmRestarts, {'T_0': 10, 'T_mult':2, 'eta_min': args.cosine_eta_min}  # 10, 2, 1e
    elif scheduler_name == "CosineAnnealingLR":
        return CosineAnnealingLR, {'T_max': 100}
    elif scheduler_name == "ReduceLROnPlateau":
        return ReduceLROnPlateau, {'factor': 0.1, 'patience': 2}
    elif scheduler_name == "OneCycleLR":
        return optim.lr_scheduler.OneCycleLR, {'max_lr': args.lr, 'steps_per_epoch': 100, 'epochs': args.epochs}
    else:
        raise ValueError(f"Unknown scheduler {scheduler_name}")

def parse_loss(loss_name, args):
    if args.smoothing > 0:
        return LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    elif args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None:
        return SoftTargetCrossEntropy()
    elif loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(reduction=args.loss_reduction)
    elif loss_name == "mse":
        return nn.MSELoss(reduction=args.loss_reduction)
    elif loss_name == "bce":
        return nn.BCELoss(reduction=args.loss_reduction)
    else:
        raise ValueError(f"Unknown loss {loss_name}")

def parse_optimizer(optimizer_name, args):
    if optimizer_name == "sgd":
        return optim.SGD, {"lr": args.lr, "weight_decay": args.wd, "momentum": args.sgd_momentum}
    elif optimizer_name == "adam":
        return optim.Adam, {"lr": args.lr, "weight_decay": args.wd, "betas": (args.adam_beta1, args.adam_beta2)}
    elif optimizer_name == "adamw":
        return optim.AdamW, {"lr": args.lr, "weight_decay": args.wd, "betas": (args.adam_beta1, args.adam_beta2)}
    elif optimizer_name == "rmsprop":
        return optim.RMSprop, {"lr": args.lr, "weight_decay": args.wd}
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")

def parse_mixup_args(args):
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)

    return mixup_fn

def parse_train_args(args):    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    torch.manual_seed(args.seed)

    if args.device is not None:
        device = torch.device(args.device)
    elif use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size, 'num_workers': 8, 'prefetch_factor': min(64,args.batch_size//2), 'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size, 'num_workers': 8, 'prefetch_factor': min(64,args.test_batch_size//2)}
    if use_cuda:
        cuda_kwargs = {'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    args.lr = args.lr * args.batch_size / 256
    loss = parse_loss(args.loss, args)
    optimizer, optimizer_args = parse_optimizer(args.optimizer, args)
    scheduler, scheduler_args = parse_scheduler(args.scheduler, args)

    return device, train_kwargs, test_kwargs, loss, optimizer, optimizer_args, scheduler, scheduler_args

class Trainer():

    def __init__(self, train_args, test_args, optimizer, criterion, scheduler, epochs, device, logger:TensorboardLogger) -> None:
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.epochs = epochs
        self.device = device
        self.logger = logger
        self.train_args = train_args
        self.test_args = test_args
        self.current_epoch = 1
    
    def train(self, model, train_dataset: Dataset, test_dataset: Dataset, pred_fn, log_interval=1, dry_run=False, save_callback=None, checkpoint_callback=None, mixup_fn=None):
        self.train_accs, self.test_accs, self.train_losses, self.test_losses = [], [], [], []
        self.logger.log("Started training with args:")
        self.logger.log_dict(self.train_args)

        batch_scheduler = None
        scheduler = self.scheduler
        if self.scheduler.__class__.__name__ == "OneCycleLR":
            batch_scheduler = self.scheduler
            batch_scheduler.total_steps = len(train_dataset) * self.epochs
            scheduler = None
            self.logger.log("Using batch scheduler OneCycleLR, setting scheduler to None.")

        for epoch in range(self.current_epoch, self.epochs + 1):
            self.logger.log("Starting epoch: " + str(epoch))
            train_dataloader = torch.utils.data.DataLoader(train_dataset, **self.train_args)
            train_acc, train_loss = self.train_step(model, train_dataloader, pred_fn, epoch, log_interval, dry_run, batch_scheduler, mixup_fn=mixup_fn)
            self.logger.log("Train Accuracy: " + str(train_acc) + ", Average Loss: " + str(train_loss))

            test_dataloader = torch.utils.data.DataLoader(test_dataset, **self.test_args)
            test_acc, test_loss = self.test_step(model, test_dataloader, pred_fn, epoch, dry_run=dry_run)
            self.logger.log("Test Accuracy: " + str(test_acc) + ", Average Loss: " + str(test_loss))

            self.train_accs.append(train_acc)
            self.test_accs.append(test_acc)
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)

            if scheduler:
                if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                    scheduler.step(test_acc)
                else:
                    scheduler.step()
                    self.logger.log("Setting learning rate to: " + str(self.scheduler.get_last_lr()[0]))

            self.current_epoch += 1
            if save_callback:
                save_callback(epoch)
            if checkpoint_callback:
                checkpoint_callback(epoch, {'acc': test_acc, 'loss': test_loss})

            fig = get_training_fig(self.train_accs, self.test_accs)
            self.logger.log_figure(fig, "training")
            self.logger.log("_______________________________________________________")
            if dry_run:
                break
            
    
    def test(self, model, test_dataset: Dataset, pred_fn, dry_run=False):
        test_dataloader = torch.utils.data.DataLoader(test_dataset, **self.test_args)
        test_acc, test_loss = self.test_step(model, test_dataloader, pred_fn, 0, dry_run=dry_run)
        self.logger.log("Test Accuracy: " + str(test_acc) + ", Average Loss: " + str(test_loss))
        return test_acc, test_loss

    def gradcam(self, model, dataset_test: Dataset, pred_fn, target_layers, num_images=20, mask=None, size=None):
        model.eval()
        features = {}

        def get_features(name):
            def hook(model, input, output):
                features[name] = output.detach()
            return hook

        try:
            model.attention.register_forward_hook(get_features('attention'))
            self.logger.log("Registered attention hook.")
        except:
            self.logger.log("Failed to register attention, using gradcam instead.")

        cam = utils.GradCAM(model=model, target_layers=target_layers)
        transform, target_transform = dataset_test.transform, dataset_test.target_transform
        dataset_test.transform, dataset_test.target_transform = None, None
        for i in range(num_images):
            img, y = dataset_test.__getitem__(i)
            if mask is not None:
                m = cv2.resize(mask, [img.shape[1], img.shape[0]], interpolation=cv2.INTER_NEAREST)
                img = np.where(m == np.max(m), img, np.mean(img))
            visualization, grayscale_cam = utils.createGradCamExplanation(model, target_layers=target_layers, img=img, transform=transform, cam=cam)
            data = transform(img)[np.newaxis,:,:,:].to(self.device)
            output = model(data)
            pred = pred_fn(output).item()

            f, axis = plt_util.plt.subplots(1, 3, figsize=(24, 12))
            plt_util.plt.suptitle(f"Class: {dataset_test.CLASSES[y]}, pred: {pred}R")
            axis[0].imshow(img.squeeze(), cmap='gray')

            axis[0].axis('off')
            axis[1].imshow(visualization)
            axis[1].axis('off')

            image_dir = f"visualization/{i}"
            self.logger.log_image(img.squeeze(), f"{image_dir}/img", cmap='gray')
            self.logger.log_image(visualization.squeeze(), f"{image_dir}/gradcam")

            if 'attention' in features:
                att = torch.sigmoid(features['attention'][0]).squeeze().cpu().numpy()
                att = cv2.resize(att, (img.shape[1], img.shape[0]))
                att = np.where(att > 0.5, 1.0, 0.0)
                attimg = att * img
                axis[2].imshow(att, cmap='gray')
                self.logger.log_image(att, f"{image_dir}/attention", cmap='gray')
            else:
                axis[2].imshow(grayscale_cam, cmap='gray')
                self.logger.log_image(grayscale_cam, f"{image_dir}/attention", cmap='gray')
            axis[2].axis('off')
            self.logger.log_figure(f, f"{image_dir}/all")
            plt.close(f)

        dataset_test.transform, dataset_test.target_transform = transform, target_transform

    def train_step(self, model, train_dataloader, pred_fn, epoch, log_interval, dry_run=False, scheduler=None, mixup_fn=None):
        model.train()
        correct = 0
        
        train_loss = 0
        targets, preds = [], []

        for batch_idx, (data, target) in (pbar := tqdm(enumerate(train_dataloader), total=len(train_dataloader), ncols=100)):
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                data, target = data.to(self.device), target.to(self.device)
                if mixup_fn is not None:
                    data, target = mixup_fn(data, target)
                self.optimizer.zero_grad()
                output = model(data)

                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                # if self.scheduler is not None:
                #     self.scheduler.step()

                train_loss += loss.item()

                pred = pred_fn(output)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
            pbar.set_description(f'Loss: {(train_loss / (batch_idx+1)):.4f}, Accuracy: {(correct)}')

            targets.extend(target.cpu().numpy().reshape((-1,)))
            preds.extend(pred.cpu().detach().numpy().reshape((-1,)))

            if batch_idx % log_interval == 0:
                self.logger.log_scalar('Train/loss', loss.item(), (epoch-1) * len(train_dataloader) + batch_idx)
                if dry_run:
                    break
            if scheduler:
                scheduler.step()

        clr = utils.createClassificationReport(targets, preds, output_dict=True)
        f1 = clr['macro avg']['f1-score']
        
        f = self.logger.log_cm(targets, preds, epoch)
        plt.close(f)
        
        return f1, train_loss / len(train_dataloader.dataset)

    def test_step(self, model, test_loader, pred_fn, epoch, name="Valid", dry_run=False):
        model.eval()
        test_loss = 0
        correct = 0

        targets, preds = [], []

        with torch.no_grad():
            for batch_idx, (data, target) in (pbar := tqdm(enumerate(test_loader), total=len(test_loader), ncols=100)):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += self.criterion(output, target).item()  # sum up batch loss

                pred = pred_fn(output)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                pbar.set_description(f'Loss {name}: {(test_loss / (batch_idx+1)):.4f}, Accuracy: {(correct)}')

                targets.extend(target.cpu().numpy().reshape((-1,)))
                preds.extend(pred.cpu().numpy().reshape((-1,)))

        test_loss /= len(test_loader.dataset)

        clr = utils.createClassificationReport(targets, preds, output_dict=True)
        f1 = clr['macro avg']['f1-score']
        self.logger.log('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), f1))
        f = self.logger.log_cm(targets, preds, epoch, name=f"{name}_confusion_matrix")
        plt.close(f)
        self.logger.log_classification_report(targets, preds)

        return f1, test_loss / len(test_loader.dataset)

class MAE3DTrainerWrapper(Trainer):
    def __init__(self, train_args, test_args, optimizer, criterion, scheduler, epochs, device, logger: TensorboardLogger) -> None:
        super().__init__(train_args, test_args, optimizer, criterion, scheduler, epochs, device, logger)

class VitMAETrainer(Trainer):
    def __init__(self, train_args, test_args, optimizer, criterion, scheduler, epochs, device, logger: TensorboardLogger) -> None:
        super().__init__(train_args, test_args, optimizer, criterion, scheduler, epochs, device, logger)

    def train_step(self, model, train_loader, pred_fn, epoch, log_interval, dry_run=False, batch_scheduler=None, mixup_fn=None):
        model.train()
        running_loss = 0.0
        self.log_interval = log_interval
        for batch_idx, (data, _) in (pbar := tqdm(enumerate(train_loader), total=len(train_loader), ncols=100)):
            x = data.to(self.device)
            self.optimizer.zero_grad()
            output = model(x)
            loss = output.loss
            pbar.set_description(f'Loss: {loss:.4f}')

            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

            if batch_scheduler:
                batch_scheduler.step()
                lr = batch_scheduler.get_last_lr()[0]
                self.logger.log_scalar('Train/lr', lr, (epoch-1) * len(train_loader) + batch_idx)
            if batch_idx % log_interval == 0:
                self.logger.log_scalar('Train/loss', loss.item(), (epoch-1) * len(train_loader) + batch_idx)
                if dry_run:
                    break

        return running_loss / len(train_loader.dataset), None
    
    def test_step(self, model, test_loader, pred_fn, epoch, name="Valid", dry_run=False):
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch_idx, (x, _) in (pbar := tqdm(enumerate(test_loader), total=len(test_loader), ncols=100)):
                x = x.to(self.device)
                output = model(x)
                loss = output.loss

                pbar.set_description(f'Loss: {loss:.4f}')
                running_loss += loss.item()
                if dry_run:
                    break

            log_interval = self.log_interval if hasattr(self, 'log_interval') else 1
            if epoch % log_interval == 0:
                img = utils.visualize_vitmae(x, model)
                self.logger.log_image(img, f"visualization/reconstruction_{epoch}")

        return running_loss / len(test_loader.dataset), None
    
class VitTrainer(Trainer):
    def __init__(self, train_args, test_args, optimizer, criterion, scheduler, epochs, device, logger: TensorboardLogger) -> None:
        super().__init__(train_args, test_args, optimizer, criterion, scheduler, epochs, device, logger)
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train_step(self, model, train_loader, pred_fn, epoch, log_interval, dry_run=False, batch_scheduler=None, mixup_fn=None):
        model.train()
        running_loss = 0.0
        correct = 0
        self.log_interval = log_interval
        targets, preds = [], []
        for batch_idx, (data, target) in (pbar := tqdm(enumerate(train_loader), total=len(train_loader), ncols=100)):
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                x = data.to(self.device)
                data, target = data.to(self.device), target.to(self.device)
                if mixup_fn is not None:
                    data, target = mixup_fn(data, target)
                
                
                output = model(x) #model(x, labels=target)
                if hasattr(output, 'loss'):
                    loss = output.loss
                else:
                    loss = self.criterion(output, target)
                
                if hasattr(output, 'logits'):
                    pred = pred_fn(output.logits)  # get the index of the max log-probability
                else:
                    pred = pred_fn(output)

                
                #if target.shape[-1] > 1 and len(target.shape) > 1:
                if mixup_fn is not None:
                    target = target.argmax(dim=1)
                #print(target.shape, pred.shape)

            correct += pred.eq(target.view_as(pred)).sum().item()
            
            targets.extend(target.cpu().numpy().reshape((-1,)))
            preds.extend(pred.cpu().detach().numpy().reshape((-1,)))

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            running_loss += loss.item()

            pbar.set_description(f'Loss: {(running_loss / (batch_idx+1)):.4f}, Accuracy: {(correct)}')

            if batch_scheduler:
                batch_scheduler.step()
                lr = batch_scheduler.get_last_lr()[0]
                self.logger.log_scalar('Train/lr', lr, (epoch-1) * len(train_loader) + batch_idx)
            if batch_idx % log_interval == 0:
                self.logger.log_scalar('Train/loss', loss.item(), (epoch-1) * len(train_loader) + batch_idx)
                if dry_run:
                    break

        clr = utils.createClassificationReport(targets, preds, output_dict=True)
        f1 = clr['macro avg']['f1-score']
        
        f = self.logger.log_cm(targets, preds, epoch)
        plt.close(f)
        
        return f1, running_loss / len(train_loader.dataset)
    
    def test_step(self, model, test_loader, pred_fn, epoch, name="Valid", dry_run=False):
        model.eval()
        running_loss = 0.0
        correct = 0
        targets, preds = [], []
        with torch.no_grad():
            for batch_idx, (x, target) in (pbar := tqdm(enumerate(test_loader), total=len(test_loader), ncols=100)):
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    data, target = x.to(self.device), target.to(self.device)
                    output = model(data) #model(data, labels=target)
                    if hasattr(output, 'loss'):
                        loss = output.loss
                    else:
                        if isinstance(self.criterion, SoftTargetCrossEntropy):
                            target_oh = torch.nn.functional.one_hot(target, num_classes=output.shape[-1])
                            loss = self.criterion(output, target_oh)
                        else:
                            loss = self.criterion(output, target)

                    running_loss += loss.item()

                if hasattr(output, 'logits'):
                    pred = pred_fn(output.logits)  # get the index of the max log-probability
                else:
                    pred = pred_fn(output)
                correct += pred.eq(target.view_as(pred)).sum().item()

                targets.extend(target.cpu().numpy().reshape((-1,)))
                preds.extend(pred.cpu().detach().numpy().reshape((-1,)))

                if dry_run:
                    break

            log_interval = self.log_interval if hasattr(self, 'log_interval') else 1
            clr = utils.createClassificationReport(targets, preds, output_dict=True)
            f1 = clr['macro avg']['f1-score']
            self.logger.log('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
                running_loss, correct, len(test_loader.dataset), f1))
            if epoch % log_interval == 0:
                f = self.logger.log_cm(targets, preds, epoch, name=f"{name}_confusion_matrix")
                plt.close(f)
                self.logger.log_classification_report(targets, preds)

        return f1, running_loss / len(test_loader.dataset)

class VAETrainer(Trainer):
    def __init__(self, train_args, test_args, optimizer, criterion, scheduler, epochs, device, logger: TensorboardLogger) -> None:
        super().__init__(train_args, test_args, optimizer, criterion, scheduler, epochs, device, logger)

    def train_step(self, model, train_loader, pred_fn, epoch, log_interval, dry_run=False):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, _) in (pbar := tqdm(enumerate(train_loader), total=len(train_loader), ncols=100)):
            x = data.to(self.device)
            self.optimizer.zero_grad()
            recon_x, mu, logvar = model(x)

            # Compute reconstruction loss and KL divergence
            BCE = self.criterion(recon_x, x)
            
            if mu is None or logvar is None:
                KLD = 0
            else:
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            pbar.set_description(f'Loss: {BCE:.4f}, KLD: {(KLD)}')

            loss = BCE + KLD
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

            if batch_idx % log_interval == 0:
                self.logger.log_scalar('Train/loss', loss.item(), (epoch-1) * len(train_loader) + batch_idx)
                if dry_run:
                    break

        return running_loss / len(train_loader.dataset), None
    
    def test_step(self, model, test_loader, pred_fn, epoch, name="Valid", dry_run=False):
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch_idx, (x, _) in (pbar := tqdm(enumerate(test_loader), total=len(test_loader), ncols=100)):
                x = x.to(self.device)
                recon_x, mu, logvar = model(x)
                BCE = self.criterion(recon_x, x)
                if mu is None or logvar is None:
                    KLD = 0
                else:
                    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                pbar.set_description(f'Loss: {BCE:.4f}, KLD: {(KLD)}')
                loss = BCE + KLD
                running_loss += loss.item()
        return running_loss / len(test_loader.dataset), None

    def anomaly_score(self, model, dataset_test: Dataset, indices=range(20)):
        scores = []
        for e, i in enumerate(indices):
            img, y = dataset_test.__getitem__(i)
            x = img[np.newaxis, :, :, :].to(self.device)

            recon_x, mu, logvar = model(x)
            BCE = self.criterion(recon_x, x)
            if mu is None or logvar is None:
                KLD = 0
            else:
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            score = BCE + KLD
            scores.append((y, score.item()))

            f, axis = plt_util.plt.subplots(1, 3, figsize=(24, 12))
            plt_util.plt.suptitle(f"Class: {dataset_test.CLASSES[y]}, anomaly score: {score}")
            axis[0].imshow(img.squeeze().cpu().detach().numpy(), cmap='gray')
            axis[0].axis('off')
            axis[1].imshow(recon_x.squeeze().cpu().detach().numpy(), cmap='gray')
            axis[1].axis('off')
            rec = img * (img - recon_x.to(img.device))
            axis[2].imshow(torch.where(rec > 0.65 * rec.max(), torch.ones_like(rec), torch.zeros_like(rec)).squeeze().cpu().detach().numpy(), cmap='gray')
            axis[2].axis('off')
            self.logger.log_figure(f, f"anomaly_{i}")
        
        self.logger.log("Anomaly scores:")
        self.logger.log_list(scores)
        
        return scores

class FastFlowTrainer(Trainer):

    def train_step(self, model, train_dataloader, pred_fn, epoch, log_interval, dry_run=False):
        model.train()
        loss_meter = AverageMeter()
        for batch_idx, (data, _) in (pbar := tqdm(enumerate(train_dataloader), total=len(train_dataloader), ncols=100)):
            # forward
            data = data.to(self.device)
            ret = model(data)
            loss = ret["loss"]
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # log
            loss_meter.update(loss.item())
            if (batch_idx) % log_interval == 0 or (batch_idx) == len(train_dataloader):
                self.logger.log_scalar('Train/loss', loss_meter.val, (epoch-1) * len(train_dataloader) + batch_idx)
            pbar.set_description(f'Loss: {loss_meter.val:.4f}, AVG: {(loss_meter.avg)}')
            if dry_run:
                break
        return loss_meter.avg, 0

    def test_step(self, model, test_loader, pred_fn, epoch, name="Valid"):
        model.eval()
        model = model.to(self.device)
        loss_meter = AverageMeter()
        logged = True
        for batch_idx, (x, _) in (pbar := tqdm(enumerate(test_loader), total=len(test_loader), ncols=100)):
            data = x.to(self.device)
            with torch.no_grad():
                ret = model(data)
                loss = ret["loss"]
                loss_meter.update(loss.item())
            outputs = ret["anomaly_map"].cpu().detach()
            if batch_idx > 10 and not logged:
                outputs = outputs[0]
                fig = plt.figure()
                plt.imshow(outputs, cmap='gray')
                self.logger.log_figure(fig, "anomaly_map", epoch)
                plt.close(fig)
                logged = True

            pbar.set_description(f'Loss {name}: {loss_meter.val:.4f}, AVG: {(loss_meter.avg)}')
        return loss_meter.avg, 0
    
    def anomaly_score(self, model, dataset_test: Dataset, indices=range(20)):
        scores = []
        model = model.to(self.device)
        model.eval()
        for e, i in enumerate(indices):
            img, y = dataset_test.__getitem__(i)
            x = img[np.newaxis, :, :, :].to(self.device)

            with torch.no_grad():
                ret = model(x)
                loss = ret["loss"]

            outputs = ret["anomaly_map"]
            scores.append((y, loss.item()))

            f, axis = plt_util.plt.subplots(1, 2, figsize=(24, 12))
            plt_util.plt.suptitle(f"Class: {dataset_test.CLASSES[y]}, anomaly score: {loss}")
            axis[0].imshow(img.squeeze().cpu().detach().numpy(), cmap='gray')
            axis[0].axis('off')
            axis[1].imshow(outputs.squeeze().cpu().detach().numpy(), cmap='viridis')
            axis[1].axis('off')
            self.logger.log_figure(f, f"anomaly_{i}")
        
        self.logger.log("Anomaly scores:")
        self.logger.log_list(scores)
        
        return scores
            
class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_training_fig(accs, test_accs) -> matplotlib.figure.Figure:
    fig = plt.figure()
    plt.plot(accs, label='Train Acc', marker='o')
    plt.plot(test_accs, label='Test Acc', marker='s')
    plt.legend()
    return fig
