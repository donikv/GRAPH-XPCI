from torchvision.models import *
import torch
import torch.nn as nn
import torch.nn.functional as F
#from imports.FastFlow.fastflow import FastFlow
from common.classification_models import *
from common.vae import *
from common.itae import *
from common.vit_mae import *
import os
import sys

def add_model_args(parser):
    parser.add_argument('--model', type=str, default="ClassificationModelSmall", metavar='M', help='model to use (default: "ClassificationModelSmall")')
    parser.add_argument("--model-path", type=str, default=None, help="path to save model, default: None")
    parser.add_argument("--weights", type=str, default=None, help="path to model weights, default: None")
    parser.add_argument("--bn", action='store_true', default=False, help="use batch normalization, default: False")
    parser.add_argument("--scat-upsample", action='store_true', default=False, help="use scat upsample, default: False")


def parse_model_args(model, args):
    if args.model.find('VAE') != -1:
        model = model(args.in_channels, args.latent_dim, args.size)
    elif args.model.find('ITAE') != -1:
        model = model(args.in_channels)
    elif args.model.find('ViTMAE') != -1:
        model = model(args.in_channels, args.latent_dim, args.size, args.num_classes)
    elif args.model.find('FastFlow') != -1:
        model = model(args.backbone_name, args.flow_steps, args.size, args.conv3x3_only, args.hidden_ratio)
    elif args.model.find('VariableEncoder') != -1:
        model = model(num_classes=args.num_classes, regression=args.regression, encoder_type=args.backbone_name, rgb=args.rgb)
    elif args.model.find('Scat') != -1:
        model = model(num_classes=args.num_classes, regression=args.regression, bn=args.bn, upsample=args.scat_upsample)
    else:
        model = model(num_classes=args.num_classes, regression=args.regression, bn=args.bn)
    return model


def save_model(model, folder, model_name="model.pt"):
    os.makedirs(folder, exist_ok=True)
    torch.save(model.state_dict(), f"{folder}/{model_name}")

    s = ' '.join(sys.argv)
    with open(f"{folder}/args.txt", "w") as f:
        f.write(s)
    print("Model saved to", folder)

def build_fastflow_model(backbone_name="wide_resnet50_2", flow_steps=8, input_size=256, conv3x3_only=False, hidden_ratio=1.0):
    model = FastFlow(
        backbone_name=backbone_name,
        flow_steps=flow_steps,
        input_size=input_size,
        conv3x3_only=conv3x3_only,
        hidden_ratio=hidden_ratio,
    )
    return model

def regression_pred_fn(x):
    x = torch.clamp(x, 0, 3)
    return x.round()

def classification_pred_fn(x):
    return x.argmax(dim=1, keepdim=True)