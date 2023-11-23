from inputs import *
from experiments.single_model.model import Model
from experiments.ensemble_avgpool.model import Ensemble
from experiments.utils import get_balanced_accuracy, store_results

import argparse
import torchmetrics as tm


parser = argparse.ArgumentParser()

## dataset arguments
parser.add_argument("-d",   "--dataset")
parser.add_argument("-m",   "--mode", type=str, default=None)
parser.add_argument("-g",   "--margin", type=float, default=1.3)
parser.add_argument("-bs",  "--batch-size", type=int, default=20)
parser.add_argument("-a",   "--augmentation", type=str, nargs="+", default=['none'])
parser.add_argument("-p",   "--preprocessing", type=str, nargs="+", default=['none'])
parser.add_argument("-ot",  "--output-type", type=str, choices=("attribute", "detect_fake"))
parser.add_argument("-c",   "--categories", type=str, nargs="+", default=None)
parser.add_argument("-q",   "--qualities", type=str, nargs="+", default=None)
parser.add_argument("-s",   "--split", type=float, nargs="+", default=None)
parser.add_argument("-max", "--max-num-samples", type=int, default=None)
## model arguments
parser.add_argument("-mn",  "--model-names", type=str, nargs="+")
## other arguments
parser.add_argument("-D",   "--device", type=str, default="cuda:0")
parser.add_argument("-O",   "--output-filename", type=str, default=None)

args = parser.parse_args()

models = [Model.easy_load(model_name, device=args.device) for model_name in args.model_names]
ensemble = Ensemble(models, args.device)
ensemble_name =  f"avgpool_ensemble({', '.join(args.model_names)})" 

target_size = models[0].target_size
num_classes = models[0].num_classes

# augmentations should be used only for experimentation
aug = get_augmentation(args.augmentation[0], args.augmentation[1:], target_size)
pre = get_preprocessing(args.preprocessing[0], args.preprocessing[1:])
dset = get_dataset(args.dataset,
                    target_size=target_size,
                    augmentation=aug,
                    margin=args.margin,
                    output_type={"task": args.output_type},
                    mode=args.mode,
                    qualities=args.qualities,
                    categories=args.categories,
                    split=args.split,
                    preprocessing=pre,
                    max_num_samples=args.max_num_samples)
dl = get_dataloader(dset, args.batch_size, balanced=False)


print(f"evaluating task {args.output_type} of {ensemble_name} on dataset {args.dataset}{('/'+args.mode) if args.mode else ''}")

metric_fns = [tm.ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize='true')]
loss, _, metrics = ensemble.evaluate(dl, extra_metrics=metric_fns, binarize_model=(args.output_type=="detect_fake"))

acc = get_balanced_accuracy(metrics[0])
print(f"\tloss = {loss}")
print(f"\tacc  = {acc}")
print()

if args.output_filename:
    
    store_results(args.output_filename,
                  args.output_type+"_task",
                  args.dataset,
                  args.mode,
                  ensemble_name,
                  acc)
