
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
import warnings

from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from datetime import datetime, timedelta
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting.models import BaseModel, RecurrentNetwork
from pytorch_forecasting import SMAPE, Baseline, TimeSeriesDataSet, QuantileLoss




import optuna
import joblib
warnings.filterwarnings('ignore')
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-start', default = '2016-01-04')
parser.add_argument('-end', default = '2019-11-30')
parser.add_argument('-pred_start', default = '2020-02-01')
parser.add_argument('-pred_end', default = '2020-02-20')
data_path = '../data_full.xlsx'



def preprocess_data(start, end, pred_start, pred_end):

    data=pd.read_excel(data_path)
    res_data = data[data["Date"].isin(pd.date_range(start, end))]
    # pd.date_range('2016-01-04', '2020-01-31')
    # pd.date_range('2016-01-04', '2020-03-31')
    res_data = data[data["Date"].isin(pd.date_range('2016-01-04', '2020-01-31'))]
    res_data.reset_index(drop=True, inplace=True)
    scale_cols = ['REV OBD']
    res_data['time_index'] = np.arange(len(res_data))
    res_data['time_index'] = res_data['time_index'].astype(int)
    
    # Scaling
    scaler = MinMaxScaler()
    scale_col = ['REV OBD', 'OBD NET+FSC_KRW', 'OBD A/R_KRW', 'REV CPN',
                'CPN NET+FSC_KRW', 'CPN A/R_KRW', 'REV TKT', 'TKT NET+FSC_KRW',
                'TKT A/R_KRW', 'WTI', 'exchanges', 'kospi', 'rates',
                'stock_a', 'stock_k', 'stock_kkj']
    scaled = scaler.fit_transform(res_data[scale_col])
    
    tmp_df_1 = res_data[['time_index', 'Date', 'Account DOW']]
    columns = ['REV OBD', 'OBD NET+FSC_KRW', 'OBD A/R_KRW', 'REV CPN',
            'CPN NET+FSC_KRW', 'CPN A/R_KRW', 'REV TKT', 'TKT NET+FSC_KRW',
            'TKT A/R_KRW', 'WTI', 'exchanges', 'kospi', 'rates',
            'stock_a', 'stock_k', 'stock_kkj']
    tmp_df_2 = pd.DataFrame(scaled, columns=columns)
    res_data = pd.concat([tmp_df_1, tmp_df_2], axis=1)
    
    res_data['time_index'] = res_data['time_index'].astype(int)
    res_data['market'] = 'OBD'
    
    # Loockback_period & forecasting_period
    max_prediction_length = 20
    lookback_length = 100
    training_data_max = len(res_data) - max_prediction_length

    # 학습용 데이터
    data_p = res_data.iloc[:training_data_max, :]
    training_data = scaler.fit_transform(data_p[scale_cols])

    
    # max_prediction_length 만큼의 데이터는 예측 데이터와 비교를 위해 분리
    # Training set에 없는 데이터로 구성
    # Input과 output의 pair로 정의
    pred_y_start = pred_start
    pred_y_end = pred_end
    
    pred_x_start = datetime.strftime(datetime.strptime(pred_start, '%Y-%m-%d') - timedelta( days = 60), '%Y-%m-%d')
    pred_x_end =datetime.strftime(datetime.strptime(pred_start, '%Y-%m-%d') - timedelta( days = 1), '%Y-%m-%d')
    print('pred_x_start = ', pred_x_start)
    print('pred_x_end = ', pred_x_end)
    
    
    x_for_metric = scaler.fit_transform(data[data["Date"].isin(pd.date_range(pred_x_start, pred_x_end))][scale_cols])
    y_for_metric = scaler.fit_transform(data[data["Date"].isin(pd.date_range(pred_y_start, pred_y_end))][scale_cols])
    
    # x_for_metric['time_index'] = np.arange(len(x_for_metric))
    # x_for_metric['market'] = 'OBD'
    print('x_for_metric', x_for_metric[:3])
    print('y_for_metric', y_for_metric[:3])
    
    return res_data, x_for_metric, y_for_metric
    
    


class LSTM(BaseModel):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = True)
        
        self.fc = nn.Linear(hidden_size  * num_layers, num_classes)
        
    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size, device = x.device))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size, device = x.device))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size * self.num_layers)
        out = self.fc(h_out)
        return out
    
    
def train(log_interval, model, train_dl, val_dl, optimizer, criterion, epoch):

    best_loss = np.inf
    for epoch in range(epoch):
        train_loss = 0.0
        model.train()
        for data, target in train_dl:

            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
                model = model.cuda()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target) # mean-squared error for regression
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # validation
        valid_loss = 0.0
        model.eval()
        for data, target in val_dl:

            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            output = model(data)         
            loss = criterion(output, target)
            valid_loss += loss.item()

        if ( epoch % log_interval == 0 ):
            print(f'\n Epoch {epoch} \t Training Loss: {train_loss / len(train_dl)} \t Validation Loss: {valid_loss / len(val_dl)} \n')

        if best_loss > (valid_loss / len(val_dl)):
            print(f'Validation Loss Decreased({best_loss:.6f}--->{(valid_loss / len(val_dl)):.6f}) \t Saving The Model')
            best_loss = (valid_loss / len(val_dl))
            torch.save(model.state_dict(), 'lstm_saved_model.pth')

    return best_loss



def objective(trial):

    cfg = { 
            'learning_rate' : trial.suggest_loguniform('learning_rate', 1e-3, 1e-1), #trial.suggest_loguniform('learning_rate', 1e-2, 1e-1), # learning rate을 0.01-0.1까지 로그 uniform 분포로 사용
            'hidden_size': trial.suggest_categorical('hidden_size',[16, 32, 64, 128, 256, 512, 1024]),
            'num_layers': trial.suggest_int('num_layers', 1, 5, 1),       
        }


    torch.manual_seed(42) 

    # create dataset and dataloader
    max_encoder_length = 100
    max_prediction_length = 20

    training_cutoff = res_data['time_index'].max() - max_prediction_length

    training = TimeSeriesDataSet(
            res_data[lambda x: x.time_index <= training_cutoff],
            time_idx="time_index",
            target="REV OBD",
            group_ids=["market"], # ["agency", "sku"],
            min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            static_categoricals=["market"],
            static_reals=[],
            time_varying_known_categoricals=[], #'Account DOW', 'kr_holiday','us_holiday'], #["special_days", "month"],
            # variable_groups= {"special_days": special_days},  # group of categorical variables can be treated as one variable
            time_varying_known_reals=["time_index"], # ["time_index", "price_regular", "discount_in_percent"],
            # time_varying_unknown_categoricals=["keyword"],     
            time_varying_unknown_reals=['REV OBD',], # 'OBD NET+FSC_KRW' 'OBD A/R_KRW', 'REV CPN', 'CPN NET+FSC_KRW', 'CPN A/R_KRW', 'REV TKT' 'TKT NET+FSC_KRW', 'TKT A/R_KRW', 'WTI', 'exchanges', 'kospi', 'rates' 'stock_a', 'stock_k', 'stock_kkj'
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

    validation = TimeSeriesDataSet.from_dataset(
            training,
            res_data,
            min_prediction_idx = training_cutoff + 1,
            predict=True,
            stop_randomization=True
        )

    batch_size = 128 
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

    # calculate baseline mean absolute error, i.e. predict next value as the last available value from the history
    actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
    baseline_predictions = Baseline().predict(val_dataloader)
    mae_with_baseline = (actuals - baseline_predictions).abs().mean().item()
    smape_with_baseline = SMAPE()(baseline_predictions, actuals)
    # (2 * (baseline_predictions - actuals).abs() / (baseline_predictions.abs() + actuals.abs() + 1e-8)).mean()

    print(f'mae_with_baseline : {mae_with_baseline}')
    print(f'smape_with_baseline : {smape_with_baseline}')

    # configure network and trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger(log_path)  # logging results to a tensorboard

    max_epochs = 10
    gradient_clip = .1
    limit_train_batches = max_prediction_length
    trainer = pl.Trainer(
        max_epochs = max_epochs,
        gpus = 0,
        weights_summary = 'top',
        gradient_clip_val = gradient_clip,
        callbacks = [lr_logger, early_stop_callback],
        limit_train_batches =  limit_train_batches, # coment in for training, running valiation every 30 batches
        # fast_dev_run=True,  # comment in to check that networkor datas et has no serious bugs
        enable_checkpointing = True,
        auto_lr_find = True,
        logger=logger
    )
        
    dropout = .1
    LSTM = RecurrentNetwork.from_dataset(
        training,
        cell_type = 'LSTM',
        # architecture hyperparameters
        learning_rate=cfg['learning_rate'],
        hidden_size = cfg['hidden_size'],
        log_gradient_flow=True,
        dropout = dropout,
        log_interval=1,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        reduce_on_plateau_patience= 4,
    )
    print(f"Number of parameters in network: {LSTM.size()/1e3:.1f}k")

    # fit network
    trainer.fit(
        LSTM,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


    #### Evaluate performance

    # load the best model according to the validation loss
    # (given that we use early stopping, this is not necessarily the last epoch)
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = RecurrentNetwork.load_from_checkpoint(best_model_path)

    actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
    predictions = best_model.predict(val_dataloader)

    # 여기서 x는 (sample, lookback_length, 1)의 크기를 지님. 따라서, 제일 앞의 시점을 제거하려면, x[:, -1, :]이 되어야 함
    x_pred = np.expand_dims(x_for_metric, 0)  # Inference에 사용할 lookback data를 x_pred로 지정. 앞으로 x_pred를 하나씩 옮겨 가면서 inference를 할 예정

    for j, i in enumerate(range(max_prediction_length)):

        # feed the last forecast back to the model as an input
        x_pred = np.append( x_pred[:, 1:, :], np.expand_dims(y_for_metric[j, :], (0,2)), axis=1)
        xt_pred = torch.Tensor(x_pred)

        if torch.cuda.is_available():
            xt_pred = xt_pred.cuda()
        # generate the next forecast
        yt_pred = best_model(xt_pred)
        # tensor to array
        # x_pred = xt_pred.cpu().detach().numpy()
        y_pred = yt_pred.cpu().detach().numpy()

        # save the forecast
        predict_data.append(y_pred)

    # transform the forecasts back to the original scale
    predict_data = np.array(predict_data).reshape(-1, 1)
    
    print('predicted = ', predict_data[-10:])
    print('actual = ', y_for_metric[-10:])

    print('SMAPE = ', SMAPE)

    smape = SMAPE()(actuals, predictions)

    print(f' \nSMAPE : {smape}')
    return smape




if __name__ == '__main__':
    args = parser.parse_args()
    print(f'start = {args.start}')
    print(f'end = {args.end}')
    res_data, x_for_metric, y_for_metric = preprocess_data(args.start, args.end, args.pred_start, args.pred_end)
    log_path = f"../lightning_logs/LSTM_{args.start}_{args.end}"
    sampler = optuna.samplers.TPESampler()
    print('start studying')
    study = optuna.create_study(sampler=sampler, direction='minimize')
    study.optimize(objective, n_trials= 5)
    
    joblib.dump(study, '../PyTorchForecasting-LSTM_optuna.pkl')