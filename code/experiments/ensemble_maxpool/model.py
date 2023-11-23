import torch
import torchmetrics as tm


class Ensemble:

    def __init__(self, models, device, threshold=0.5):
        self.models = models
        self.device = device
        for model in self.models:
            model.set_device(device)
        self.threshold = threshold

    def __call__(self, x, threshold=None):
        threshold = threshold or self.threshold
        y_est = [model(x, softmax=True)[:, 1] for model in self.models]
        confs, preds = torch.stack(y_est).max(dim=0)
        preds += 1 # the labels of manipulations start from 1
        preds[confs < threshold] = 0
        confs[confs < threshold] = 1 - confs[confs < threshold]
        return preds, confs

    def reset_metric(self, metric: tm.Metric):
        metric = metric.to(self.device)
        metric.reset()

    def evaluate(self,
                 dl,
                 metrics=None,
                 binarize_model=False,
                 debug_mode=False,
                 logger=None):

        metrics = metrics or []
        for metric in metrics:
            self.reset_metric(metric)

        for model in self.models:
            model.model.eval()

        log_interval = max(len(dl) // 100, 1)

        with torch.no_grad():
            for i, (x, y) in enumerate(dl):

                if i % log_interval == 0:
                    if logger is None:
                        print(f"evaluating {100 * i / len(dl):.1f}% complete")
                    else:
                        logger.log(
                            f"evaluating {100 * i / len(dl):.1f}% complete")

                x, y = x.to(self.device), y.to(self.device)
                preds, confs = self(x)

                if binarize_model:
                    preds = (preds > 0).int()

                for metric in metrics:
                    metric(preds, y.data)

                if debug_mode:
                    if i == 3:
                        break

            epoch_metrics = [metric.compute().cpu()
                             for metric in metrics]

            if logger is None:
                print(f"evaluating 100% complete")
            else:
                logger.log(f"evaluating 100% complete")

            return epoch_metrics
