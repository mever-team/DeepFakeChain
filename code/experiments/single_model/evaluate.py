from inputs import *
from experiments.single_model.model import Model
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
from matplotlib import pyplot as plt
from experiments.utils import get_model_target_size

import argparse
import torchmetrics as tm

parser = argparse.ArgumentParser()
## dataset arguments
parser.add_argument("-d", "--dataset")
parser.add_argument("-m", "--mode", type=str, default=None)
parser.add_argument("-g", "--margin", type=float, default=1.3)
parser.add_argument("-bs", "--batch-size", type=int, default=20)
parser.add_argument("-a", "--augmentation", type=str, nargs="+", default=['none'])
parser.add_argument("-p", "--preprocessing", type=str, nargs="+", default=['none'])
parser.add_argument("-ot", "--output-type", type=str, choices=("attribute", "detect_fake"))
parser.add_argument("-c", "--categories", type=str, nargs="+", default=None)
parser.add_argument("-q", "--qualities", type=str, nargs="+", default=None)
parser.add_argument("-s", "--split", type=float, nargs="+", default=None)
parser.add_argument("-max", "--max-num-samples", type=int, default=None)
## model arguments
parser.add_argument("-mn", "--model-name", type=str, default=None)
## other arguments
parser.add_argument("-D", "--device", type=str, default="cuda:0")


args = parser.parse_args()

if args.output_type=="attribute" and not args.dataset.startswith("ff++"):
    print("warning, evaluation of attribution should only work with ff++ in this study")

model = Model.easy_load(model_name=args.model_name, device=args.device)

# augmentation on the test set can be used only for experimentation
aug = get_augmentation(args.augmentation[0], args.augmentation[1:], model.target_size)
pre = get_preprocessing(args.preprocessing[0], args.preprocessing[1:])
dset = get_dataset(args.dataset,
                    target_size=model.target_size,
                    augmentation=aug,
                    margin=args.margin,
                    output_type={"task": args.output_type},
                    mode=args.mode,
                    qualities=args.qualities,
                    categories=args.categories,
                    split=args.split,
                    preprocessing=pre,
                    max_num_samples=args.max_num_samples)
dl = get_dataloader(dset, args.batch_size, balanced=False) # no need for balanced labels due to balanced accuracy

metric_fns = [tm.ConfusionMatrix(task="multiclass", num_classes=model.num_classes, normalize='true')]

loss, acc, metrics = model.evaluate(dl, extra_metrics=metric_fns, binarize_model=(args.output_type=="detect_fake")) 

print(f"\tloss = {loss}")
print(f"\tacc  = {acc}")
for name, metric in metrics.items():
    print(f"{name} = {metric}")

