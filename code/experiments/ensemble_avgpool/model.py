import torch
import torchmetrics as tm

from collections import defaultdict
from pathlib import Path


class Ensemble:
    
    def __init__(self, models, device):
        self.models = models
        self.device = device
        for model in self.models:
            model.set_device(device)

    def __call__(self, x):
        y_est = [model(x, softmax=True) for model in self.models]
        return torch.stack(y_est).mean(dim=0)
    
    def reset_metric(self, metric: tm.Metric):
        metric = metric.to(self.device)
        metric.reset()

    def evaluate(self,
                dl,
                loss_fn=None,
                acc=None,
                extra_metrics=None,
                binarize_model=False,
                debug_mode=False,
                logger=None):

        loss_fn = loss_fn or torch.nn.CrossEntropyLoss()
        acc = acc or tm.Accuracy(task="multiclass", num_classes=self.models[0].num_classes)
        self.reset_metric(acc)

        extra_metrics = extra_metrics or []
        for metric in extra_metrics:
            self.reset_metric(metric)

        for model in self.models:
            model.model.eval()

        log_interval = max(len(dl) // 100, 1)

        with torch.no_grad():
            epoch_loss = 0.0
            for i, (x, y) in enumerate(dl):

                if i % log_interval == 0:
                    if logger is None:
                        print(f"evaluating {100 * i / len(dl):.1f}% complete")
                    else:
                        logger.log(
                            f"evaluating {100 * i / len(dl):.1f}% complete")

                x, y = x.to(self.device), y.to(self.device)
                y_est = self(x)
                loss = loss_fn(y_est, y)
                preds = y_est.argmax(dim=1)

                if binarize_model:
                    preds = (preds > 0).int()

                epoch_loss += loss.item() / len(dl)
                acc(preds, y.data)
                for metric in extra_metrics:
                    metric(preds, y.data)

                if debug_mode:
                    if i == 3:
                        break

            epoch_acc = acc.compute().cpu().item()
            epoch_metrics = [metric.compute().cpu()
                            for metric in extra_metrics]

            if logger is None:
                print(f"evaluating 100% complete")
            else:
                logger.log(f"evaluating 100% complete")

            if len(epoch_metrics) == 0:
                return epoch_loss, epoch_acc
            else:
                return epoch_loss, epoch_acc, epoch_metrics