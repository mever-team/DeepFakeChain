import torch
import logging
import torchmetrics as tm

from torch.utils.tensorboard import SummaryWriter
from ..utils import initialise_model, get_model_target_size
from torchmetrics import Accuracy
from copy import deepcopy
from utils import read_csv
from dirs import MODELS_DIR


class Model(torch.nn.Module):

    @classmethod
    def easy_load(cls, model_name, device=None):
        
        hyperparams = read_csv(cls.get_model_path(model_name) / "hyperparams")
        hyperparams = {k:v for k, v in hyperparams}
        model = Model( hyperparams["model_type"], int(hyperparams["num_labels"]), device)
        model.load(model_name)
        return model

    def __init__(self, model_type, num_classes, device=None):
        super().__init__()

        self.model = initialise_model(model_type, num_labels=num_classes,
                                                        feature_extract=False, use_pretrained=True)
        self.target_size = get_model_target_size(model_type)
        self.is_inception = (model_type == "inception")
        self.device = device or (
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.num_classes = num_classes

    def set_device(self, device):
        self.device = device
        self.model = self.model.to(device)

    def reset_metric(self, metric: tm.Metric):
        metric = metric.to(self.device)
        metric.reset()

    def train(self,
              train_dl,
              epochs,
              model_save_name,
              val_dl=None,
              loss_fn=None,
              optimizer=None,
              scheduler=None,
              eval_before_train=True):

        # initialize metrics
        loss_fn = loss_fn or torch.nn.CrossEntropyLoss()
        acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.reset_metric(acc)
        best_acc = 0.0
        best_model_wts = self.model.state_dict()

        writer = SummaryWriter(Model.get_model_path(model_save_name) / "runs")

        logger = self.get_logger(model_save_name, "train")

        if eval_before_train:

            logger.info("Pre-training evaluation")

            epoch_loss, epoch_acc = self.evaluate(
                train_dl, loss_fn, logger=logger)
            writer.add_scalar("Loss/train", epoch_loss, 0)
            writer.add_scalar("Accuracy/train", epoch_acc, 0)
            logger.info(f"epoch 0, train loss {epoch_loss} acc{epoch_acc}")

            epoch_loss, epoch_acc = self.evaluate(
                val_dl, loss_fn, logger=logger)
            writer.add_scalar("Loss/val", epoch_loss, 0)
            writer.add_scalar("Accuracy/val", epoch_acc, 0)
            logger.info(f"epoch 0, val loss {epoch_loss} acc{epoch_acc}")

        early_stop_counter = 0

        for epoch in range(1, epochs+1):

            logger.info(f'Epoch {epoch}/{epochs}')

            epoch_loss, epoch_acc = self.train_epoch(
                train_dl, loss_fn, acc, optimizer, scheduler, logger)
            writer.add_scalar("Loss/train", epoch_loss, epoch)
            writer.add_scalar("Accuracy/train", epoch_acc, epoch)
            logger.info(
                f"epoch {epoch}: train loss {epoch_loss} acc {epoch_acc}")

            if val_dl is not None:

                early_stop_counter += 1

                epoch_loss, epoch_acc = self.evaluate(val_dl, loss_fn, acc)
                writer.add_scalar("Loss/val", epoch_loss, epoch)
                writer.add_scalar("Accuracy/val", epoch_acc, epoch)
                logger.info(
                    f"epoch {epoch}: val loss {epoch_loss} acc {epoch_acc}")

                logger.info(f"early stopping counter = {early_stop_counter}")

                if epoch_acc > best_acc:

                    early_stop_counter = 0

                    logger.info(
                        f"reseting early stopping counter = {early_stop_counter}")

                    best_acc = epoch_acc
                    best_model_wts = deepcopy(self.model.state_dict())

                    if model_save_name is not None:
                        self.save(model_save_name)
                        logger.info("***checkpoint*** saving best model")

                if early_stop_counter > 5:
                    logger.info("early stopping")
                    break

        if val_dl is not None:
            self.model.load_state_dict(best_model_wts)

    def train_epoch(self,
                    dl,
                    loss_fn,
                    acc,
                    optimizer,
                    scheduler,
                    logger):

        self.model.train()

        epoch_loss = 0.0
        self.reset_metric(acc)

        log_interval = max(len(dl) // 100, 1)

        for i, (x, y) in enumerate(dl):

            if i % log_interval == 0:
                logger.info(f"training {100 * i / len(dl):.1f}% complete")

            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            if self.is_inception:
                # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                y_est, y_aux = self.model(x)
                loss1 = loss_fn(y_est, y)
                loss2 = loss_fn(y_aux, y)
                loss = loss1 + 0.4*loss2
            else:
                y_est = self.model(x)
                loss = loss_fn(y_est, y)
            preds = y_est.argmax(dim=1)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() / len(dl)
            acc(preds, y)

        scheduler.step()

        epoch_acc = acc.compute().cpu().item()
        logger.info(f"training 100% complete")
        return epoch_loss, epoch_acc

    def evaluate(self,
                 dl,
                 loss_fn=None,
                 acc=None,
                 extra_metrics=None,
                 binarize_model=False,
                 debug_mode=False,
                 logger=None):

        loss_fn = loss_fn or torch.nn.CrossEntropyLoss()
        acc = acc or Accuracy(task="multiclass", num_classes=self.num_classes)
        self.reset_metric(acc)

        extra_metrics = extra_metrics or []
        for metric in extra_metrics:
            self.reset_metric(metric)

        self.model.eval()

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

                if self.is_inception:
                    # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                    y_est, y_aux = self.model(x)
                    loss1 = loss_fn(y_est, y)
                    loss2 = loss_fn(y_aux, y)
                    loss = loss1 + 0.4*loss2
                else:
                    y_est = self.model(x)
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
            epoch_metrics = [metric.compute().cpu() for metric in extra_metrics]

            if logger is None:
                print(f"evaluating 100% complete")
            else:
                logger.log(f"evaluating 100% complete")

            if len(epoch_metrics) == 0:
                return epoch_loss, epoch_acc
            else:
                return epoch_loss, epoch_acc, epoch_metrics

    def forward(self, x, softmax=False):
        y = self.model(x)
        if softmax:
            y = torch.nn.functional.softmax(y, dim=1)
        return y
    
    @classmethod
    def get_model_path(cls, model_name):
        path = MODELS_DIR / model_name
        path.mkdir(exist_ok=True, parents=True)
        return path

    def save(self, model_name):
        path = Model.get_model_path(model_name) / "model.pt"
        torch.save(self.model.state_dict(), path)

    def load(self, model_name):
        path = Model.get_model_path(model_name) / "model.pt"
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def get_logger(self, model_name, mode):

        log_path = Model.get_model_path(model_name) / "training_logs"
        # clear logs
        open(log_path, "w").close()

        console_logging_format = "%(levelname)s %(message)s"
        file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"

        logging.basicConfig(level=logging.INFO, format=console_logging_format)
        logger = logging.getLogger()
        handler = logging.FileHandler(str(log_path))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(file_logging_format)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger
