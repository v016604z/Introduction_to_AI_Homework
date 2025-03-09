import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Conv1D, Flatten, MultiHeadAttention, Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os

# 設置 TensorFlow 運行設備的最大性能
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 隱藏 TensorFlow 訊息

# 讀取真實數據
data = pd.read_csv('C:\\Users\\v0166\\Github\\Introduction_to_AI_Homework\src\\final\\中鋼.csv')

# 預處理數據
data['日期'] = pd.to_datetime(data['日期'])
data.set_index('日期', inplace=True)

# 直接選擇所需特徵
features = data[['股票開盤價', '股票最高價', '股票最低價', '股票收盤價', '股票交易量',
                 '期貨開盤', '期貨最高', '期貨最低', '期貨收盤']]

# 特徵縮放
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features.values)
scaled_df = pd.DataFrame(scaled_features, columns=features.columns, index=features.index)

# 目標值是明日的「股票收盤價」
scaled_df['target'] = scaled_df['股票收盤價'].shift(-1)

# 去除最後一行的空值
scaled_df = scaled_df.dropna()

# 基礎模型定義
class BaseModels:
    @staticmethod
    def lstm_model(input_shape):
        model = Sequential([
            LSTM(50, activation='relu', input_shape=input_shape),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    @staticmethod
    def gru_model(input_shape):
        model = Sequential([
            GRU(50, activation='relu', input_shape=input_shape),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    @staticmethod
    def tcn_model(input_shape):
        model = Sequential([
            Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
            Flatten(),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    @staticmethod
    def transformer_model(input_shape):
        input_layer = Input(shape=input_shape)
        attention = MultiHeadAttention(num_heads=4, key_dim=2)(input_layer, input_layer)
        flatten = Flatten()(attention)
        output = Dense(1)(flatten)
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer='adam', loss='mse')
        return model

# 集成學習模型
class EnsembleModel:
    def __init__(self):
        self.rf_model = RandomForestRegressor()
        self.xgb_model = XGBRegressor()

    def train(self, X, y):
        self.rf_model.fit(X, y)
        self.xgb_model.fit(X, y)

    def predict(self, X):
        rf_pred = self.rf_model.predict(X)
        xgb_pred = self.xgb_model.predict(X)
        return (rf_pred + xgb_pred) / 2

# 主流程
train_size = int(len(scaled_df) * 0.8)
train, test = scaled_df.iloc[:train_size], scaled_df.iloc[train_size:]

X_train = train.drop(columns=['target']).values
y_train = train['target'].values
X_test = test.drop(columns=['target']).values
y_test = test['target'].values

# 訓練基礎模型
base_predictions = []
model_scores = {}

# 在主流程中，對 X_test 進行相同的重塑
X_test_reshaped = X_test.reshape(-1, X_test.shape[1], 1)

# 訓練基礎模型後，對基礎模型預測結果進行集成
base_predictions = []
for model_idx in tqdm(range(4), desc="Training Models"):
    input_shape = (X_train.shape[1], 1)
    X_train_reshaped = X_train.reshape(-1, input_shape[0], 1)

    if model_idx == 0:
        model = BaseModels.lstm_model(input_shape)
    elif model_idx == 1:
        model = BaseModels.gru_model(input_shape)
    elif model_idx == 2:
        model = BaseModels.tcn_model(input_shape)
    elif model_idx == 3:
        model = BaseModels.transformer_model(input_shape)

    model.fit(X_train_reshaped, y_train, epochs=20, verbose=0)
    prediction = model.predict(X_train_reshaped).flatten()
    base_predictions.append(prediction)

# 集成學習層
ensemble = EnsembleModel()

# 轉置基礎模型的預測結果，確保其形狀為 (樣本數, 基礎模型數量)
X_base = np.array(base_predictions).T  # base_predictions的形狀應為 (4, n_samples)，這樣轉置後是 (n_samples, 4)

# 訓練集成模型
ensemble.train(X_base, y_train)

# 在進行預測前，將 X_test 變回二維資料
X_test_reshaped = X_test.reshape(-1, X_test.shape[1])  # 這樣就變成 (n_samples, n_features)

# 預測
ensemble_predictions = ensemble.predict(X_test_reshaped)

# 計算集成模型的MSE
ensemble_mse = mean_squared_error(y_test, ensemble_predictions)
model_scores['Ensemble'] = ensemble_mse

# 預測
ensemble_predictions = ensemble.predict(X_test_reshaped)  # 使用相同形狀的 X_test_reshaped
ensemble_mse = mean_squared_error(y_test, ensemble_predictions)
model_scores['Ensemble'] = ensemble_mse

