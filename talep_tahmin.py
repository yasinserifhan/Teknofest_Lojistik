import pandas as pd
import numpy as np
import xgboost as xgb
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel('Desi_talep.xlsx')
coordinates = pd.read_excel('Koordinatlar_v2.xlsx')
df['Tarih'] = pd.to_datetime(df['Tarih'])

kampanya_gunleri = ['2026-02-14', '2026-03-08', '2026-05-10', '2026-03-17', '2026-03-27', '2026-04-17']
kapali_gunler = ['2026-01-01', '2026-03-19', '2026-03-20', '2026-03-21', '2026-03-22',
                  '2026-04-23', '2026-05-01', '2026-04-15']

tatil_array = np.array(kampanya_gunleri, dtype='datetime64[D]')
kapali_array = np.array(kapali_gunler, dtype='datetime64[D]')
pazarlar = np.arange('2026-01-04', '2026-05-18', 7, dtype='datetime64[D]')
cumartesiler = np.arange('2026-01-03', '2026-05-18', 7, dtype='datetime64[D]')
haftasonu = np.append(pazarlar, cumartesiler)

df['Holiday'] = df['Tarih'].isin(tatil_array)
df['Closed'] = df['Tarih'].isin(kapali_array)
df['Haftasonu'] = df['Tarih'].isin(haftasonu)
df['dayID'] = df['Tarih'].dt.dayofweek
df['DayInMOnth'] = df['Tarih'].dt.day
df['Is_Payday_Window'] = ((df['DayInMOnth'] >= 28) | (df['DayInMOnth'] <= 3)).astype(int)

df = df.merge(coordinates, left_on='Çıkış Transfer Merkezi', right_on='Transfer Merkezi', how='left')
df = df.rename(columns={'Enlem': 'Cikis_Enlem', 'Boylam': 'Cikis_Boylam'}).drop('Transfer Merkezi', axis=1)
df = df.merge(coordinates, left_on='Varış Transfer Merkezi', right_on='Transfer Merkezi', how='left')
df = df.rename(columns={'Enlem': 'Varis_Enlem', 'Boylam': 'Varis_Boylam'}).drop('Transfer Merkezi', axis=1)


def haversine_vektorize(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


df['Mesafe_km'] = haversine_vektorize(df['Cikis_Enlem'], df['Cikis_Boylam'], df['Varis_Enlem'], df['Varis_Boylam'])
df['route'] = df['Çıkış Transfer Merkezi'] + ' → ' + df['Varış Transfer Merkezi']

df = df.sort_values(['Çıkış Transfer Merkezi', 'Varış Transfer Merkezi', 'Tarih']).reset_index(drop=True)
rg = df.groupby(['Çıkış Transfer Merkezi', 'Varış Transfer Merkezi'])
df['Lag_1_Desi'] = rg['Toplam Desi'].shift(1)
df['Lag_7_Desi'] = rg['Toplam Desi'].shift(7)
df['Son_3_Gun_Ort_Desi'] = rg['Toplam Desi'].transform(lambda x: x.shift(1).rolling(3).mean())
df['Son_7_Gun_Ort_Desi'] = rg['Toplam Desi'].transform(lambda x: x.shift(1).rolling(7).mean())
df['Trend_Ivmesi'] = df['Son_3_Gun_Ort_Desi'] / df['Son_7_Gun_Ort_Desi']
df = df.dropna(subset=['Lag_7_Desi']).reset_index(drop=True)

split_date = '2026-04-01'
train = df[df['Tarih'] < split_date].copy()
test = df[df['Tarih'] >= split_date].copy()

cikis_mean_encoded = train.groupby('Çıkış Transfer Merkezi')['Toplam Desi'].mean().to_dict()
varis_mean_encoded = train.groupby('Varış Transfer Merkezi')['Toplam Desi'].mean().to_dict()
global_mean = train['Toplam Desi'].mean()

train['Cikis_Merkezi_Encoded'] = train['Çıkış Transfer Merkezi'].map(cikis_mean_encoded)
train['Varis_Merkezi_Encoded'] = train['Varış Transfer Merkezi'].map(varis_mean_encoded)
test['Cikis_Merkezi_Encoded'] = test['Çıkış Transfer Merkezi'].map(cikis_mean_encoded).fillna(global_mean)
test['Varis_Merkezi_Encoded'] = test['Varış Transfer Merkezi'].map(varis_mean_encoded).fillna(global_mean)

drop_cols_xgb = [
    'Tarih', 'Çıkış Transfer Merkezi', 'Varış Transfer Merkezi', 'Toplam Desi', 'route',
    'Cikis_Enlem', 'Cikis_Boylam', 'Varis_Enlem', 'Varis_Boylam',
    'Holiday', 'Mesafe_km', 'Varis_Merkezi_Encoded'
]

X_train_xgb = train.drop(columns=[c for c in drop_cols_xgb if c in train.columns])
y_train_xgb = train['Toplam Desi']
X_test_xgb = test.drop(columns=[c for c in drop_cols_xgb if c in test.columns])
y_test_xgb = test['Toplam Desi']

model_xgb = xgb.XGBRegressor(
    n_estimators=1500, learning_rate=0.03, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    objective='reg:tweedie', tweedie_variance_power=1.5,
    eval_metric='mae', random_state=42, early_stopping_rounds=50
)
model_xgb.fit(X_train_xgb, y_train_xgb,
              eval_set=[(X_train_xgb, y_train_xgb), (X_test_xgb, y_test_xgb)], verbose=False)

y_pred_xgb = np.maximum(0, model_xgb.predict(X_test_xgb))

test_xgb = test.copy()
test_xgb['y_pred'] = y_pred_xgb
test_xgb['y_true'] = y_test_xgb.values

xgb_route_mae = {}
for (cikis, varis), grp in test_xgb.groupby(['Çıkış Transfer Merkezi', 'Varış Transfer Merkezi']):
    route = f'{cikis} → {varis}'
    xgb_route_mae[route] = mean_absolute_error(grp['y_true'], grp['y_pred'])


def add_time_features(d):
    d = d.copy()
    d['Ayin_Gunu'] = d['Tarih'].dt.day
    d['Haftanin_Gunu'] = d['Tarih'].dt.dayofweek
    yilin_haftasi = d['Tarih'].dt.isocalendar().week.astype('int32')
    d['Hafta_Sin'] = np.sin(2 * np.pi * yilin_haftasi / 52.0)
    d['Hafta_Cos'] = np.cos(2 * np.pi * yilin_haftasi / 52.0)
    return d


train = add_time_features(train)
test = add_time_features(test)

feature_cols = ['Toplam Desi', 'Mesafe_km', 'Hafta_Sin', 'Hafta_Cos', 'Ayin_Gunu', 'Haftanin_Gunu']
target_col = 'Toplam Desi'
LOOKBACK = 28
PREDICT_HORIZON = 7

feature_scaler = RobustScaler()
target_scaler = RobustScaler()
train_scaled = train.copy()
train_scaled[feature_cols] = feature_scaler.fit_transform(train[feature_cols])
target_scaler.fit(train[[target_col]])
test_scaled = test.copy()
test_scaled[feature_cols] = feature_scaler.transform(test[feature_cols])


def create_sequences(d, lookback, horizon):
    X, y, routes = [], [], []
    for (cikis, varis), grp in d.groupby(['Çıkış Transfer Merkezi', 'Varış Transfer Merkezi']):
        grp = grp.sort_values('Tarih')
        data = grp[feature_cols].values
        tgts = grp[target_col].values
        if len(data) <= lookback + horizon:
            continue
        for i in range(len(data) - lookback - horizon + 1):
            X.append(data[i:i + lookback])
            y.append(tgts[i + lookback:i + lookback + horizon])
            routes.append((cikis, varis))
    return np.array(X), np.array(y), routes


X_train_seq, y_train_seq, _ = create_sequences(train_scaled, LOOKBACK, PREDICT_HORIZON)
X_test_seq, y_test_seq, test_routes = create_sequences(test_scaled, LOOKBACK, PREDICT_HORIZON)

tf.random.set_seed(42)
model_lstm = Sequential([
    Input(shape=(LOOKBACK, len(feature_cols))),
    LSTM(32, return_sequences=False, activation='tanh'),
    Dropout(0.4),
    Dense(PREDICT_HORIZON, activation='linear')
])
model_lstm.compile(optimizer=Adam(0.0005), loss=tf.keras.losses.Huber(delta=1.0), metrics=['mae'])
es_lstm = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
rl_lstm = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=0.00005, verbose=0)
model_lstm.fit(X_train_seq, y_train_seq, epochs=100, batch_size=64,
               validation_split=0.15, callbacks=[es_lstm, rl_lstm], verbose=0)

y_pred_lstm_sc = model_lstm.predict(X_test_seq, verbose=0)
y_pred_lstm = np.maximum(0, target_scaler.inverse_transform(y_pred_lstm_sc))
y_test_lstm = target_scaler.inverse_transform(y_test_seq)

tf.random.set_seed(42)
model_gru = Sequential([
    Input(shape=(LOOKBACK, len(feature_cols))),
    GRU(32, return_sequences=False, activation='tanh'),
    Dropout(0.4),
    Dense(PREDICT_HORIZON, activation='linear')
])
model_gru.compile(optimizer=Adam(0.0005), loss=tf.keras.losses.Huber(delta=1.0), metrics=['mae'])
es_gru = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
rl_gru = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=0.00005, verbose=0)
model_gru.fit(X_train_seq, y_train_seq, epochs=100, batch_size=64,
              validation_split=0.15, callbacks=[es_gru, rl_gru], verbose=0)

y_pred_gru_sc = model_gru.predict(X_test_seq, verbose=0)
y_pred_gru = np.maximum(0, target_scaler.inverse_transform(y_pred_gru_sc))
y_test_gru = target_scaler.inverse_transform(y_test_seq)

lstm_route_dict, gru_route_dict = {}, {}
for i, (cikis, varis) in enumerate(test_routes):
    route = f'{cikis} → {varis}'
    true_flat = y_test_lstm[i].flatten()
    pred_lstm_flat = y_pred_lstm[i].flatten()
    pred_gru_flat = y_pred_gru[i].flatten()
    if route not in lstm_route_dict:
        lstm_route_dict[route] = {'true': [], 'pred': []}
        gru_route_dict[route] = {'true': [], 'pred': []}
    lstm_route_dict[route]['true'].extend(true_flat)
    lstm_route_dict[route]['pred'].extend(pred_lstm_flat)
    gru_route_dict[route]['true'].extend(true_flat)
    gru_route_dict[route]['pred'].extend(pred_gru_flat)

lstm_route_mae = {r: mean_absolute_error(d['true'], d['pred']) for r, d in lstm_route_dict.items()}
gru_route_mae = {r: mean_absolute_error(d['true'], d['pred']) for r, d in gru_route_dict.items()}

prophet_route_mae = {}
prophet_forecasts = {}
TEST_DAYS = 14
FORECAST_DATES = pd.date_range('2026-05-11', '2026-05-17')

for route in df['route'].unique():
    rdf = df[df['route'] == route].sort_values('Tarih')
    pdf = rdf[['Tarih', 'Toplam Desi']].rename(columns={'Tarih': 'ds', 'Toplam Desi': 'y'})
    if len(pdf) < TEST_DAYS + 20:
        continue
    train_p = pdf.iloc[:-TEST_DAYS]
    test_p = pdf.iloc[-TEST_DAYS:]
    m = Prophet(weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=False,
                seasonality_mode='multiplicative', changepoint_prior_scale=0.1,
                seasonality_prior_scale=5, interval_width=0.80)
    m.fit(train_p)
    future = m.make_future_dataframe(periods=TEST_DAYS + 7, freq='D')
    forecast = m.predict(future)
    test_pred = forecast[forecast['ds'].isin(test_p['ds'])]['yhat'].values.clip(0)
    if len(test_pred) != len(test_p):
        continue
    prophet_route_mae[route] = mean_absolute_error(test_p['y'].values, test_pred)
    fc_window = forecast[forecast['ds'].isin(FORECAST_DATES)]
    prophet_forecasts[route] = fc_window[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()

xgb_forecast_rows = []
for _, row in test.groupby(['Çıkış Transfer Merkezi', 'Varış Transfer Merkezi']).tail(1).iterrows():
    route = f"{row['Çıkış Transfer Merkezi']} → {row['Varış Transfer Merkezi']}"
    rdf_full = df[df['route'] == route].sort_values('Tarih')
    history = rdf_full.copy()
    for fd in FORECAST_DATES:
        lag1 = history['Toplam Desi'].iloc[-1]
        lag7 = history['Toplam Desi'].iloc[-7] if len(history) >= 7 else history['Toplam Desi'].mean()
        roll3 = history['Toplam Desi'].tail(3).mean()
        roll7 = history['Toplam Desi'].tail(7).mean()
        trend = roll3 / roll7 if roll7 != 0 else 1
        feat_row = pd.DataFrame([{
            'Closed': fd in kapali_array, 'Haftasonu': np.datetime64(fd.date()) in haftasonu,
            'dayID': fd.dayofweek, 'DayInMOnth': fd.day,
            'Is_Payday_Window': int((fd.day >= 28) or (fd.day <= 3)),
            'Lag_1_Desi': lag1, 'Lag_7_Desi': lag7,
            'Son_3_Gun_Ort_Desi': roll3, 'Son_7_Gun_Ort_Desi': roll7, 'Trend_Ivmesi': trend,
            'Cikis_Merkezi_Encoded': cikis_mean_encoded.get(row['Çıkış Transfer Merkezi'], global_mean),
        }])
        feat_row = feat_row[[c for c in X_train_xgb.columns]]
        pred_val = max(0, model_xgb.predict(feat_row)[0])
        xgb_forecast_rows.append({'route': route, 'Tarih': fd, 'Tahmini_Desi': pred_val})
        new_row = history.iloc[-1:].copy()
        new_row['Tarih'] = fd
        new_row['Toplam Desi'] = pred_val
        history = pd.concat([history, new_row], ignore_index=True)

xgb_forecast_df = pd.DataFrame(xgb_forecast_rows)

final_results = []
for route in df['route'].unique():
    maes = {
        'XGBoost': xgb_route_mae.get(route, np.inf),
        'Prophet': prophet_route_mae.get(route, np.inf),
        'LSTM': lstm_route_mae.get(route, np.inf),
        'GRU': gru_route_mae.get(route, np.inf),
    }
    best_model = min(maes, key=maes.get)
    hist_std = df[df['route'] == route]['Toplam Desi'].tail(14).std()

    if best_model == 'Prophet' and route in prophet_forecasts:
        fc = prophet_forecasts[route]
        for _, r in fc.iterrows():
            final_results.append({
                'route': route, 'Tarih': r['ds'], 'Model': 'Prophet',
                'Tahmini_Desi_Low': max(0, r['yhat_lower']),
                'Tahmini_Desi_Mean': max(0, r['yhat']),
                'Tahmini_Desi_High': r['yhat_upper'],
            })
    else:
        fc = xgb_forecast_df[xgb_forecast_df['route'] == route]
        for _, r in fc.iterrows():
            val = r['Tahmini_Desi']
            final_results.append({
                'route': route, 'Tarih': r['Tarih'], 'Model': 'XGBoost',
                'Tahmini_Desi_Low': max(0, val - 1.5 * hist_std),
                'Tahmini_Desi_Mean': val,
                'Tahmini_Desi_High': val + 1.5 * hist_std,
            })

final_df = pd.DataFrame(final_results)
final_df[['Çıkış Transfer Merkezi', 'Varış Transfer Merkezi']] = final_df['route'].str.split(' → ', expand=True)
final_df = final_df[['Çıkış Transfer Merkezi', 'Varış Transfer Merkezi', 'Tarih', 'Model',
                      'Tahmini_Desi_Low', 'Tahmini_Desi_Mean', 'Tahmini_Desi_High']]
final_df.to_csv('Talep_Tahmini_11_17_Mayis.csv', index=False)
