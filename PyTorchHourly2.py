import math

import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot
from pytorch_forecasting import GroupNormalizer, Baseline
from pytorch_forecasting.data import TimeSeriesDataSet
from sklearn.metrics import mean_absolute_error, mean_squared_error

column_prefixes = [
    "arrivals",
    "occupancy_daily_peak",
    "weekday",
    "month",
    "holiday_name",
    "holiday_t-3",
    "holiday_t-2",
    "holiday_t-1",
    "holiday_t+0",
    "holiday_t+1",
    "holiday_t+2",
    "holiday_t+3",
    "is_working_day",
    "cloud_count",
    "air_pressure",
    "rel_hum",
    "rain_intensity",
    "snow_depth",
    "air_temp_min",
    "air_temp_max",
    "air_temp",
    "dew_point_temp",
    "visibility",
    "slip",
    #"heat",
    "ekströms_visits",
    "ekströms_ratio",
    "website_visits_tays",
    "website_visits_acuta",
    "public_events_num_of_daily_all"
]

time_varying_unknown_reals = [
    "arrivals",
    "occupancy_daily_peak",
    "cloud_count",
    "air_pressure",
    "rel_hum",
    "rain_intensity",
    "snow_depth",
    "air_temp_min",
    "air_temp_max",
    "air_temp",
    "dew_point_temp",
    "visibility",
    "ekströms_visits",
    "ekströms_ratio",
    "website_visits_tays",
    "website_visits_acuta",
    "public_events_num_of_daily_all"
]

time_varying_known_categoricals = [
    "weekday",
    "month",
    "holiday_t-3",
    "holiday_t-2",
    "holiday_t-1",
    "holiday_t+0",
    "holiday_t+1",
    "holiday_t+2",
    "holiday_t+3",
    "is_working_day",
    "holiday_name"
]

time_varying_unknown_categoricals = [
    "slip",
    #"heat",
]

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

if __name__ == '__main__':
    # Load data
    data = pd.read_pickle('data.pkl')['2015-05':'2019-9']
    # Include only predefined columns
    data = data[column_prefixes]

    data = data.replace([np.inf, -np.inf], np.nan).fillna(0)

    data[time_varying_known_categoricals] = data[time_varying_known_categoricals].astype("string")
    data[time_varying_known_categoricals] = data[time_varying_known_categoricals].astype("category")
    data[time_varying_unknown_categoricals] = data[time_varying_unknown_categoricals].astype("string")
    data[time_varying_unknown_categoricals] = data[time_varying_unknown_categoricals].astype("category")
    idx = pd.date_range(start='2015-05', end='2019-9', freq='H')
    data['time_idx'] = pd.Series(range(len(idx)), index=idx)
    data['series'] = "0"
    data['series'] = data['series'].astype("category")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(data.sample(10))
    print(data.dtypes)

    max_prediction_length = 24
    max_encoder_length = 168
    val_size = 9792

    training_cutoff = data["time_idx"].max() - val_size
    traning_data = data[lambda x: x.time_idx <= training_cutoff]

    training = TimeSeriesDataSet(
        traning_data,
        time_idx="time_idx",
        target="arrivals",
        group_ids=["series"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["series"],
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_unknown_categoricals=time_varying_unknown_categoricals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, data, predict=False, stop_randomization=True
    )

    # create dataloaders for model
    batch_size = 128
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0
    )

    # actuals = torch.cat([y for x, y in iter(val_dataloader)])
    # baseline_predictions = Baseline().predict(val_dataloader)
    # print((actuals - baseline_predictions).abs().mean().item())


    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        EarlyStopping
    )
    # from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_forecasting.models.temporal_fusion_transformer.tuning import \
        optimize_hyperparameters
    from pytorch_forecasting.metrics import QuantileLoss, MAPE, SMAPE, MASE, MAE
    from pytorch_forecasting.models import TemporalFusionTransformer
    import tensorboard as tb
    import tensorflow as tf

    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

    # create study. This is for optimizing for best hyperparameters.
    # study = optimize_hyperparameters(
    #     train_dataloader,
    #     val_dataloader,
    #     model_path="optuna_test",
    #     n_trials=200,
    #     max_epochs=50,
    #     gradient_clip_val_range=(0.01, 1.0),
    #     hidden_size_range=(8, 64),
    #     hidden_continuous_size_range=(8, 64),
    #     attention_head_size_range=(1, 4),
    #     learning_rate_range=(0.001, 0.1),
    #     dropout_range=(0.1, 0.3),
    #     trainer_kwargs=dict(limit_train_batches=30),
    #     reduce_on_plateau_patience=4,
    #     use_learning_rate_finder=True,
    #     # use Optuna to find ideal learning rate or use in-built learning rate finder
    # )
    # # show best hyperparameters
    # print(study.best_trial.params)

    # stop training, when loss metric does not improve on validation set
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=10,
        verbose=True,
        mode="min"
    )

    #lr_logger = LearningRateLogger()  # log the learning rate
    #logger = TensorBoardLogger("lightning_logs")  # log to tensorboard

    trainer = pl.Trainer(
        max_epochs=70,
        gpus=[0],
        weights_summary="top",
        gradient_clip_val=0.07491732457954926,
        limit_train_batches=30, # comment in for training, running valiation every 30 batches
        # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
        callbacks=[early_stop_callback]
    )

    # initialise model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=20,
        hidden_continuous_size=9,
        attention_head_size=4,
        learning_rate=0.00417,
        dropout=0.12522982138188382,
        reduce_on_plateau_patience=4,
        log_interval=100,  # log example every 10 batches
        loss=QuantileLoss(),
        output_size=7,  # QuantileLoss has 7 quantiles by default
    )
    print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")

    # # find optimal learning rate
    # res = trainer.lr_find(
    #     tft,
    #     train_dataloader=train_dataloader,
    #     val_dataloaders=val_dataloader,
    #     max_lr=10.0,
    #     min_lr=1e-6,
    # )
    #
    # print(f"suggested learning rate: {res.suggestion()}")

    # fit network
    trainer.fit(
        tft,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader
    )

    # load the best model according to the validation loss
    # (given that we use early stopping, this is not necessarily the last epoch)
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)


    actuals = torch.cat([y for x, y in iter(val_dataloader)])
    predictions = best_tft.predict(val_dataloader)

    train_actuals = actuals[0:-val_size].numpy()
    test_actuals = actuals[-val_size:-1].numpy()
    train_pred = predictions[0:-val_size].numpy()
    test_pred = predictions[-val_size:-1].numpy()

    test_actuals_daily_list = []
    for i in range(0, len(test_actuals), 24):
        daily = np.sum(test_actuals[i])
        test_actuals_daily_list.append(daily)

    test_pred_daily_list = []
    for i in range(0, len(test_pred), 24):
        daily = np.sum(test_pred[i])
        test_pred_daily_list.append(daily)

    mape = mean_absolute_percentage_error(test_actuals_daily_list, test_pred_daily_list)
    print(f"MAPE: {mape}")
    mae = mean_absolute_error(test_actuals_daily_list, test_pred_daily_list)
    print('MAE: %.2f ' % (mae))
    rmse = math.sqrt(mean_squared_error(test_actuals_daily_list, test_pred_daily_list))
    print('RMSE: %.2f ' % (rmse))

    pyplot.plot(test_actuals_daily_list, label='test_real')
    pyplot.plot(test_pred_daily_list, label='test_predicted')
    pyplot.legend()
    pyplot.show()

    # calculate mean absolute error on validation set
    print(f"MAE: {(actuals - predictions).abs().mean()}")

    smape = SMAPE(reduction="none")(predictions, actuals).mean(1)
    smape = smape.data[0]
    print(f"sMAPE: {smape}")


    raw_predictions, x = best_tft.predict(val_dataloader, mode="raw",
                                          return_x=True)
    for idx in range(0, max_prediction_length * 10, max_prediction_length):  # plot 10 examples
        best_tft.plot_prediction(x, raw_predictions, idx=idx,
                                 add_loss_to_title=True)
    pyplot.show()


    train_actuals = actuals[0:-val_size]
    test_actuals = actuals[-val_size:-1]
    train_pred = predictions[0:-val_size]
    test_pred = predictions[-val_size:-1]

    val_mse = (test_actuals - test_pred).abs().mean()
    print(f"validation MSE: {val_mse}")

    # # calcualte metric by which to display
    # indices = smape.argsort(descending=True)  # sort losses
    # for idx in range(10):  # plot 10 examples
    #     best_tft.plot_prediction(x, raw_predictions, idx=indices[idx],
    #                              add_loss_to_title=SMAPE())
    # pyplot.show()

    interpretation = best_tft.interpret_output(raw_predictions, reduction="sum")
    best_tft.plot_interpretation(interpretation)
    pyplot.show()

    mae = MAE(reduction="none")(predictions, actuals).mean(1)
    mae = mae.data[0]
    print(f"Mean average absolute error.: {mae}")

    predictions, x = best_tft.predict(val_dataloader, return_x=True)
    predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(x, predictions)
    best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals)
    pyplot.show()





