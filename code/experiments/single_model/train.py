from inputs import *
from .model import Model
from utils import random_filename, write_csv, read_csv
from ..utils import register_model, set_reproducible_seed, get_optimizer, get_scheduler, get_model_target_size

import argparse


parser = argparse.ArgumentParser()
## dataset arguments
parser.add_argument("-d",   "--dataset")
parser.add_argument("-g",   "--margin", type=float, default=1.3)
parser.add_argument("-bs",  "--batch-size", type=int, default=64)
parser.add_argument("-a",   "--augmentation", type=str, nargs="+", default=['selimsef'])
parser.add_argument("-p",   "--preprocessing", type=str, nargs="+", default=['none'])
parser.add_argument("-ot",  "--output-type", type=str, nargs="+", default=["attribute"])
parser.add_argument("-c",   "--categories", type=str, nargs="+", default=None)
parser.add_argument("-q",   "--qualities", type=str, nargs="+", default=None)
parser.add_argument("-s",   "--split", type=float, nargs="+", default=None)
parser.add_argument("-max", "--max-num-samples", type=int, default=None)
## model arguments
parser.add_argument("-mn",  "--model-name", type=str, default=None)
parser.add_argument("-mt",  "--model-type", type=str, default="efficientnetv1b0")
## train arguments
parser.add_argument("-e",   "--epochs", type=int, default=40)
parser.add_argument("-opt", "--optimizer", type=str, nargs="+", default=["adam", 0.001, 5e-4])
parser.add_argument("-sch", "--scheduler", type=str, nargs="+", default=["steplr", 20, 0.1])
parser.add_argument("-ebt", "--eval-before-train", action="store_true")
parser.add_argument("-sd",  "--seed", type=int, default=0)
parser.add_argument("-D",   "--device", type=str, default="cuda:0")

args = parser.parse_args()

set_reproducible_seed(args.seed)

target_size = get_model_target_size(args.model_type)

# input pipeline
aug = get_augmentation(args.augmentation[0], args.augmentation[1:], target_size)
pre = get_preprocessing(args.preprocessing[0], args.preprocessing[1:])
output_type = {"task": args.output_type[0],
               "category": int(args.output_type[1]) if args.output_type[0] == "detect_category" else None}

train_dset = get_dataset(args.dataset,
                    target_size=target_size,
                    augmentation=aug,
                    margin=args.margin,
                    output_type=output_type,
                    mode="train",
                    qualities=args.qualities,
                    categories=args.categories,
                    split=args.split,
                    preprocessing=pre,
                    max_num_samples=args.max_num_samples)

val_dset = get_dataset(args.dataset,
                    target_size=target_size,
                    augmentation=None, # no augmentations for evaluation
                    margin=args.margin,
                    output_type=output_type,
                    mode="val",
                    qualities=args.qualities,
                    categories=args.categories,
                    split=args.split,
                    preprocessing=pre,
                    max_num_samples=args.max_num_samples)
                    
train_dl = get_dataloader(train_dset, args.batch_size, balanced=True)
val_dl = get_dataloader(val_dset, args.batch_size, balanced=True)

# model
model = Model(args.model_type, train_dset.num_labels, args.device)

model_name = args.model_name or random_filename()
optimizer = get_optimizer(model, args.optimizer) 
scheduler = get_scheduler(optimizer, args.scheduler)

model.train(train_dl,
            args.epochs,
            model_save_name=model_name,
            val_dl=val_dl,
            optimizer=optimizer,
            scheduler=scheduler,
            eval_before_train=args.eval_before_train)


path = model.get_model_path(model_name) / "hyperparams"

hyperparams = list(vars(args).items())
hyperparams.append(("num_labels", train_dset.num_labels))
write_csv(path, hyperparams)
register_model(model_name, args)




