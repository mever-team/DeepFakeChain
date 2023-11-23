from torch import nn
from torchvision import models
from pathlib import Path

import random
import numpy as np
import torch
import timm

from dirs import BASE_DIR


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def get_optimizer(model, args):

    if args[0] == "adam":
        return torch.optim.Adam(model.parameters(), lr=float(args[1]), weight_decay=float(args[2]))
    else:
        raise Exception(f"{args[0]} is not an optimizer that I know of")


def get_model_target_size(model_type):
    if model_type in ("resnet",
                      "vgg",
                      "squeezenet",
                      "densenet",
                      "efficientnetv1b0"): 
        return (224, 224)
    
    elif model_type in ("efficientnetv1b1"):
        return (240, 240)
    
    elif model_type in ("inception", "xception"):
        return (299, 299)
    
    elif model_type in ("efficientnetv2"):
        return (384, 384)
    
    else:
        raise Exception(f"unknown target size of model '{model_type}'")


def get_scheduler(opt, args):

    if args[0] == "steplr":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=int(args[1]), gamma=float(args[2]))
    else:
        raise Exception(f"{args[0]} is not a scheduler that I know of")
 

def initialise_model(model_type, num_labels, feature_extract, use_pretrained=True):

    if model_type == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_labels)

    elif model_type == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_labels)

    elif model_type == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_labels)

    elif model_type == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_labels, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_labels

    elif model_type == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_labels)

    elif model_type == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_labels)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_labels)

    elif model_type == "efficientnetv1b0":
        """ EfficientNet v1 b0
        """
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if use_pretrained else None
        model_ft = models.efficientnet_b0(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_labels)

    elif model_type == "efficientnetv1b1":
        """ EfficientNet v1 b1
        """
        weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1 if use_pretrained else None
        model_ft = models.efficientnet_b1(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_labels)

    elif model_type == "efficientnetv2":
        """ EfficientNet v2
        """
        model_ft = timm.create_model('efficientnetv2_rw_s', pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier = nn.Linear(model_ft.num_features, num_labels)

    elif model_type == "xception":
        """ Xception
        """
        model_ft = timm.create_model('xception', pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_labels)
        
    else:
        raise NotImplementedError(f"{model_type} is not currently supported")

    return model_ft


def register_model(model_name, args):

    args = args if isinstance(args, dict) else vars(args)

    path = Path(__file__).parent / "model_register.txt"
    with open(path, "a", encoding="utf8") as f:
        f.write(f"MODEL {model_name}\n")
        for k, v in args.items():
            f.write(f"\t{str(k):<10}: {v}\n")
        f.write("\n")


def set_reproducible_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



