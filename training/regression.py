import sys,os
sys.path.append(os.getcwd())

import common.logger as logger
import common.dataset as dataset
import common.record as record
import common.trainer as trainer
import common.models as models

import torch
import argparse
from datetime import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
import sys


def main():
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--log', type=str, default='tmp/test', help='log directory (default: tmp/test)')
    parser.add_argument('--record_path', type=str, default=None, help='train record path (default: None)')
    parser.add_argument('--regression', action='store_true', default=False, help='Toggles regression')
    parser.add_argument('--backbone-name', type=str, default='resnet18', help='backbone name (default: resnet18)')

    parser.add_argument("--test", action='store_true', default=False, help="test mode, default: False")
    parser.add_argument("--gradcam", action='store_true', default=False, help="Use GradCAM, default: False")
    parser.add_argument("--checkpoint-interval", type=int, default=-1, help="checkpoint interval, default: -1 (disabled)")
    parser.add_argument("--load-checkpoint", type=str, default=None, help="load checkpoint, default: None")
    
    dataset.add_dataset_args(parser)
    models.add_model_args(parser)
    trainer.add_training_args(parser)
    
    args = parser.parse_args()
    log = logger.TensorboardLogger(args.log)

    log.log_list(sys.argv)

    if args.regression:
        pred_fn = models.regression_pred_fn
        target_transform = dataset.regression_target_transform
    else:
        pred_fn = models.classification_pred_fn
        target_transform = None

    if args.record_path is not None:
        log.log("Loading train record from " + args.record_path)
        recorder = record.Record.load(args.record_path)
        recorder.path = args.log
        model = recorder.model
        model = model.to(torch.device(args.device))
        train_dataset = recorder.train_dataset
        valid_dataset = recorder.valid_dataset
        test_dataset = recorder.test_dataset
        model_trainer = recorder.trainer
        model_trainer.logger = log
        model_trainer.epochs = args.epochs
        model_trainer.device = torch.device(args.device)
        if args.load_checkpoint is not None:
            model.load_state_dict(recorder.checkpoints[args.load_checkpoint])
        log.log("Loaded train record from " + args.record_path)
        if args.test:
            try:
                log.log("Loading test dataset from args.")
                device, train_kwargs, test_kwargs, loss, optimizer, optimizer_args, scheduler, scheduler_args = trainer.parse_train_args(args)
                train_dataset, valid_dataset, test_dataset, classes = dataset.parse_dataset_args(args)
            except:
                log.log("Failed to load dataset from args, using the one loaded from record.")
    else:
        device, train_kwargs, test_kwargs, loss, optimizer, optimizer_args, scheduler, scheduler_args = trainer.parse_train_args(args)
        train_dataset, valid_dataset, test_dataset, classes = dataset.parse_dataset_args(args)

        train_dataset.target_transform = target_transform
        valid_dataset.target_transform = target_transform
        test_dataset.target_transform = target_transform

        args.num_classes = len(classes)

        model = models.parse_model_args(getattr(models, args.model), args)
        model = model.to(device)

        optimizer = optimizer(model.parameters(), **optimizer_args)
        if scheduler is not None:
            scheduler = scheduler(optimizer, **scheduler_args)

        model_trainer = trainer.Trainer(train_kwargs, test_kwargs, optimizer, loss, scheduler, args.epochs, device, log)
        recorder = record.Record(args.log, train_dataset, valid_dataset, test_dataset, model, model_trainer, record_name='test')

    def save(epoch):
        writer = log.writer

        #Remove the writer because of the thread lock error
        log.writer = None
        recorder.save()
        
        log.log(f"Saved record at epoch {epoch}.")
        log.writer = writer
    
    checkpoint = record.Checkpoint(recorder, save, args.checkpoint_interval)
    if args.gradcam:
        if args.path.find("biopsies_small") == -1:
            test_dataset.transform.transforms.insert(0, dataset.RemoveImagingArtifacts())
        model_trainer.gradcam(model, test_dataset, pred_fn, [model.tanh3])#, mask=dataset.load_image_greyscale('data/mask_cropped.tif'))

    if args.test:
        model_trainer.test(model, test_dataset, pred_fn)
    else:
        try:
            model_trainer.train(model, train_dataset, valid_dataset, pred_fn, save_callback=save, checkpoint_callback=checkpoint.checkpoint)
            model_trainer.test(model, test_dataset, pred_fn)
        except Exception as e:
            log.log(e)
            raise e

if __name__ == '__main__':
    main()
