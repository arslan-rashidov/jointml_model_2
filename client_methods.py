from __future__ import annotations

from typing import Union

import torch

from joint_ml._metric import Metric
from torch.utils.data import Dataset, DataLoader

from tsb_data import DatasetController
from tsb_model import TimeSeriesBERTModel, TimeSeriesBERTModelForTraining, TimeSeriesBERTLoss


def load_model(time_series_size, hidden_size, encoder_layers_count, heads_count, dropout_prob) -> torch.nn.Module:
    tsb_model = TimeSeriesBERTModel(time_series_size=time_series_size,
                                    hidden_size=hidden_size,
                                    encoder_layers_count=encoder_layers_count,
                                    heads_count=heads_count,
                                    dropout_prob=dropout_prob)
    model = TimeSeriesBERTModelForTraining(tsb_model)
    return model


def get_dataset(dataset_path: str, with_split: bool, lm, mask_prob, train_size, valid_size, test_size, train_dir, valid_dir, test_dir, rewrite) -> (Dataset, Dataset, Dataset):
    train_dataset, val_dataset, test_dataset = None, None, None
    if with_split:
        dataset_controller = DatasetController(file_path=dataset_path,
                                               lm=lm,
                                               mask_prob=mask_prob,
                                               train_size=train_size,
                                               valid_size=valid_size,
                                               test_size=test_size,
                                               train_dir=train_dir,
                                               valid_dir=valid_dir,
                                               test_dir=test_dir,
                                               rewrite=rewrite)
        train_dataset, val_dataset, test_dataset = dataset_controller.get_sets()

    return train_dataset, val_dataset, test_dataset


def train(model: torch.nn.Module, train_set: torch.utils.data.Dataset, batch_size, epochs_num, k,
          lr, valid_set: torch.utils.data.Dataset = None) -> tuple[list[Metric], torch.nn.Module]:
    model.train()

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    epochs_num = epochs_num

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = TimeSeriesBERTLoss(k=k)

    train_loss_metric = Metric("train_loss")

    for epoch in range(epochs_num):
        train_loss = 0.0

        model.train()

        for batch_id, data in enumerate(train_dataloader):
            if torch.cuda.is_available():
                for key in data.keys():
                    data[key] = data[key].to('cuda:0')

            batch_size = data["input_series"].shape[0]
            time_series_size = data["input_series"].shape[1]

            optimizer.zero_grad()

            pred_series = model(data["input_series"])

            loss, masked_pred, masked_true = lossf(
                pred_series,
                data["target_series"].reshape(batch_size, time_series_size, 1),
                data["mask"].reshape(batch_size, time_series_size, 1),
                epoch + 1,
            )

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        train_loss_metric.log_value(train_loss)

    return [train_loss_metric], model


def test(model: torch.nn.Module, test_set: torch.utils.data.Dataset) -> Union[
    list[Metric], tuple[list[Metric], list]]:
    model.eval()  # turn on evaluation mode

    test_loss = 0.0
    lossf = TimeSeriesBERTLoss()
    test_loss_metric = Metric('test_loss')

    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)

    with torch.no_grad():
        for batch_id, data in enumerate(test_dataloader):

            if torch.cuda.is_available():
                for key in data.keys():
                    data[key] = data[key].to('cuda:0')

            batch_size = data["input_series"].shape[0]
            time_series_size = data["input_series"].shape[1]

            pred_series = model(data["input_series"])

            loss, masked_pred, masked_true = lossf(
                pred_series,
                data["target_series"].reshape(batch_size, time_series_size, 1),
                data["mask"].reshape(batch_size, time_series_size, 1),
            )

            test_loss += loss.item()

        test_loss /= len(test_dataloader)
        test_loss_metric.log_value(test_loss)

        return [test_loss]


def get_prediction(model: torch.nn.Module, dataset_path: str) -> list:
    model.eval()

    predictions = []

    test_set = get_dataset(dataset_path, False, 3, 0.15, 1, 0, 0, '', '', '', False)

    test_dataloader = DataLoader(test_set, batch_size=1)

    with torch.no_grad():
        for batch_id, data in enumerate(test_dataloader):
            if torch.cuda.is_available():
                for key in data.keys():
                    data[key] = data[key].to('cuda:0')
            batch_size = data["input_series"].shape[0]
            time_series_size = data["input_series"].shape[1]

            pred_series = model(data["input_series"])
            predictions.append(list(pred_series))

    return predictions