import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# 데이터 로드
train = pd.read_csv('../train_chungju_new.csv', encoding='utf-8')
train.columns
test = pd.read_csv('../test_2020.csv', encoding='utf-8')

# 결측치 확인
train.isna().sum()
test.isna().sum()

# train 전처리
train.columns = [col.split('(')[0].strip() for col in train.columns]
train

train.query('일시 == "2017-04-03 22:00"')
train.query('일시 == "2017-04-03 23:00"')
train.query('일시 == "2017-04-04 00:00"')

missing_hour = pd.DataFrame({
    "일시": ["2017-04-03 23:00"],
    "기온": [(13.2 + 11.4) / 2],
    "강수량": [0],
    "풍속": [(1.7 + 0.6) / 2],
    "습도": [(23 + 30) / 2],
    "일조": [0],
    "일사": [0]
})
train = pd.concat([train, missing_hour], ignore_index=True)

train['일시'] = pd.to_datetime(train['일시'])

# '일시' 열을 기준으로 정렬하고 인덱스 재설정
train = train.sort_values(by='일시').reset_index(drop=True)

train['year'] = train['일시'].dt.year
train['month'] = train['일시'].dt.month
train['hour'] = train['일시'].dt.hour

# 결측치 처리
train['일조'] = train['일조'].fillna(0.0)
train['일사'] = train['일사'].fillna(0.0)
train['강수량'] = train['강수량'].fillna(0.0)

train[train['풍속'].isna()] # 20971 번째 행이 문제.
# 날씨는 이어진다. 그 사이 데이터를 보자.
train.iloc[20961: 20981, :] # 그 전 후 1시간동안 2.5였음.

train['풍속'] = train['풍속'].fillna(2.5)

# test 전처리
test.columns = [col.split('(')[0].strip() for col in test.columns]

test['일시'] = pd.to_datetime(test['일시'])
test['year'] = test['일시'].dt.year
test['month'] = test['일시'].dt.month
test['hour'] = test['일시'].dt.hour

## 태양에너지 테스트 데이터
test.isna().sum()
test[test['습도'].isna()] # 673, 674 번째 행이 문제.
# 2020-06-29 1시와 2시에 습도 데이터가 없는 거구나!

# 날씨는 이어진다. 그 사이 데이터를 보자.
test.iloc[672: 676, :] # 2시간 사이에 64.0 에서 82.0으로 바뀌었다!

test.loc[673, '습도'] = 70
test.loc[674, '습도'] = 76

test['일조'] = test['일조'].fillna(0.0)
test['일사'] = test['일사'].fillna(0.0)
test['강수량'] = test['강수량'].fillna(0.0)

train.isna().sum()
test.isna().sum()

# '일시' 열을 인덱스로 설정하여 시간 순서에 따라 정렬
train.set_index('일시', inplace=True)
test.set_index('일시', inplace=True)

##############################
# Feature와 Target 분리
X_train = train.drop(['일사'], axis=1)
y_train = train['일사']
X_test = test.drop(['일사'], axis=1)
y_test = test['일사']

##############################
# 모델 학습 및 예측 함수
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, is_lstm=False):
    if is_lstm:
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
        test_predict = model.predict(X_test).flatten()
    else:
        model.fit(X_train, y_train)
        test_predict = model.predict(X_test)
    
    train_predict = model.predict(X_train) if not is_lstm else model.predict(X_train).flatten()
    
    mae_test = mean_absolute_error(y_test, test_predict)
    rmse_test = np.sqrt(mean_squared_error(y_test, test_predict))
    
    print(f"Test - MAE: {mae_test:.5f}, RMSE: {rmse_test:.5f}")
    return test_predict

# 데이터 준비 (train, test를 위 코드와 동일하게 전처리했다고 가정)
# Linear Regression
print("Linear Regression Results")
linear_model = LinearRegression()
linear_pred = train_and_evaluate_model(linear_model, X_train, y_train, X_test, y_test)

# Decision Tree Regressor
print("\nDecision Tree Regressor Results")
tree_model = DecisionTreeRegressor(random_state=42)
tree_pred = train_and_evaluate_model(tree_model, X_train, y_train, X_test, y_test)

# Random Forest Regressor
print("\nRandom Forest Regressor Results")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_pred = train_and_evaluate_model(rf_model, X_train, y_train, X_test, y_test)

# Gradient Boosting Regressor
print("\nGradient Boosting Regressor Results")
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_pred = train_and_evaluate_model(gb_model, X_train, y_train, X_test, y_test)

# LightGBM Regressor
print("\nLightGBM Regressor Results")
lgbm_model = LGBMRegressor(n_estimators=100, random_state=42)
lgbm_pred = train_and_evaluate_model(lgbm_model, X_train, y_train, X_test, y_test)

# CatBoost Regressor
print("\nCatBoost Regressor Results")
catboost_model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, random_seed=42,verbose=100)
catboost_pred = train_and_evaluate_model(catboost_model, X_train, y_train, X_test, y_test)

# LSTM Model
def create_sequences(X, Y, time_step=1):
    X_seq, Y_seq = [], []
    for i in range(len(X) - time_step):
        X_seq.append(X[i:(i + time_step)])
        Y_seq.append(Y[i + time_step])
    return np.array(X_seq), np.array(Y_seq)

X_seq_train, y_seq_train = create_sequences(X_train, y_train, time_step=24)
X_seq_test, y_seq_test = create_sequences(X_test, y_test, time_step=24)

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# LSTM 모델 학습을 위한 데이터 준비
# 이미 (samples, time_steps, features) 형태이므로 reshape 불필요
lstm_model = create_lstm_model((X_seq_train.shape[1], X_seq_train.shape[2]))
print("\nLSTM Results")
lstm_pred = train_and_evaluate_model(lstm_model, X_seq_train, y_seq_train, X_seq_test, y_seq_test, is_lstm=True)


# 예측값과 실제값 비교 시각화
train.reset_index(inplace=True)
test.reset_index(inplace=True)

train
test

plt.figure(figsize=(14, 7))
plt.plot(test['일시'], y_test, label='Actual', color='blue', alpha=0.6)
plt.plot(test['일시'], lgbm_pred, label='LightGBM Predicted', color='orange', alpha=0.6)
plt.title('Actual vs LightGBM Predicted')
plt.xlabel('Date')
plt.ylabel('Solar Radiation (일사)')
plt.legend()
plt.show()

#############
# 모델 별 성능 비교
# Linear Regression Results
# Test - MAE: 0.41063, RMSE: 0.53095

# Decision Tree Regressor Results
# Test - MAE: 0.12689, RMSE: 0.23214

# Random Forest Regressor Results
# Test - MAE: 0.09018, RMSE: 0.16290

# Gradient Boosting Regressor Results
# Test - MAE: 0.14619, RMSE: 0.21486

# LightGBM Regressor Results
# Test - MAE: 0.09005, RMSE: 0.15402

# CatBoost Regressor Results
# Test - MAE: 0.09885, RMSE: 0.15726

# LSTM Results
# Test - MAE: 1.57464, RMSE: 1.91500


###########################################

df = pd.read_csv('../data_week2.csv',encoding='euc-kr')
df.columns
df.columns = [col.split('(')[0].strip() for col in df.columns]
df.rename(columns={'date_time': '일시'}, inplace=True)

df['일시'] = pd.to_datetime(df['일시'])
df['year'] = df['일시'].dt.year
df['month'] = df['일시'].dt.month
df['hour'] = df['일시'].dt.hour

df.set_index('일시', inplace=True)
X_df = df[['기온', '강수량', '풍속', '습도', '일조', 'year', 'month', 'hour']]
y_pred = np.round(lgbm_model.predict(X_df), 2)
y_pred
df['예측일사'] = y_pred
df.reset_index(inplace=True)

###############################

df.rename(columns={'예측일사': '예측일사_MJ', '전력사용량': '전력사용량_kWh'}, inplace=True)

df['일자'] = df['일시'].dt.date
solar_sum_day = df.groupby(['num','일자'],as_index=False).agg(일일_전력_사용량_kWh=('전력사용량_kWh','sum'),일일_일사량_MJ=('예측일사_MJ','sum'))
MJ_to_kWh = 0.277778
solar_sum_day['일일_일사량_kWh'] = solar_sum_day['일일_일사량_MJ']*MJ_to_kWh
solar_sum_day.info()

solar_sum_day['일자'] = pd.to_datetime(solar_sum_day['일자'])
solar_sum_day['month'] = solar_sum_day['일자'].dt.month
solar_sum_day

solar_sum_month = solar_sum_day.groupby(['num','month'],as_index=False).agg(월별_전력_사용량_kWh=('일일_전력_사용량_kWh','sum'),월별_일사량_kWh = ('일일_일사량_kWh','sum'))
solar_sum_month['월별_전력_사용량_kWh'] = np.round(solar_sum_month['월별_전력_사용량_kWh'],3)
solar_sum_month

solar_sum_month.query('month == 7').sort_values('월별_전력_사용량_kWh',ascending=False).head()
solar_sum_month.query('month == 7').sort_values('월별_전력_사용량_kWh',ascending=False).tail()

solar_sum_total = solar_sum_day.groupby(['num'],as_index=False).agg(총_전력_사용량_kWh = ('일일_전력_사용량_kWh','sum'),총_일사량_kWh = ('일일_일사량_kWh','sum'))
solar_sum_total['총_전력_사용량_kWh'] = np.round(solar_sum_total['총_전력_사용량_kWh'],3)
solar_sum_total

is_solar = df.groupby(['num'],as_index=False).agg(태양광보유=('태양광보유','sum'))
is_solar['태양광보유'] = is_solar['태양광보유'].apply(lambda x: 1 if x > 0 else 0)
solar_sum_total = solar_sum_total.merge(is_solar, on='num', how='left').sort_values('총_일사량_kWh',ascending=False).reset_index(drop=True)
solar_sum_total

solar_sum_total.max() - solar_sum_total.min()

sns.boxplot(data=solar_sum_total, y='총_일사량_kWh', hue='태양광보유')
sns.violinplot(data=solar_sum_total, y='총_일사량_kWh', hue='태양광보유')
sns.barplot(data=solar_sum_total, x='num', y='총_일사량_kWh', hue='태양광보유')
sns.histplot(data=solar_sum_total, x='총_일사량_kWh', bins=30, hue='태양광보유')
sns.kdeplot(data=solar_sum_total, x='총_일사량_kWh', hue='태양광보유')

#####################

# # 낮과 밤을 구분하여 새로운 열 추가
# train['time_of_day'] = train['hour'].apply(lambda x: 'Day' if 8 <= x <= 18 else 'Night')

# # 일사가 0인 경우와 0 초과인 경우를 구분하는 열 추가
# train['일사_status'] = train['일사'].apply(lambda x: '일사 > 0' if x > 0 else '일사 = 0')

# import seaborn as sns
# plt.rc('font', family='Malgun Gothic') 
# sns.countplot(data=train, x='time_of_day', hue='일사_status')

# train.query('일사>0')['hour'].unique()
# train.query('일사>0')['hour'].unique()

# result = train.query('일사 > 0').groupby('month', as_index=False).agg(
#     min_hour=('hour', 'min'),
#     max_hour=('hour', 'max')
# )

# result

# plt.figure(figsize=(12, 6))
# plt.bar(result['month'] - 0.2, result['min_hour'], width=0.4, label='Min Hour', color='skyblue')
# plt.bar(result['month'] + 0.2, result['max_hour'], width=0.4, label='Max Hour', color='salmon')
# plt.xlabel('Month')
# plt.ylabel('Hour')
# plt.title('Monthly Minimum and Maximum Hours of Solar Radiation')
# plt.xticks(result['month'])
# plt.legend()
# plt.show()

# plt.figure(figsize=(12, 6))
# plt.plot(result['month'], result['min_hour'], marker='o', label='Min Hour', color='blue')
# plt.plot(result['month'], result['max_hour'], marker='o', label='Max Hour', color='red')
# plt.xlabel('Month')
# plt.ylabel('Hour')
# plt.title('Monthly Minimum and Maximum Hours of Solar Radiation')
# plt.xticks(result['month'])
# plt.legend()
# plt.show()

# plt.figure(figsize=(12, 6))
# plt.plot(result['min_hour'], result['month'], marker='o', label='Min Hour', color='blue')
# plt.plot(result['max_hour'], result['month'], marker='o', label='Max Hour', color='red')
# plt.ylabel('Month')
# plt.xlabel('Hour')
# plt.title('Monthly Minimum and Maximum Hours of Solar Radiation')
# plt.yticks(result['month'])
# plt.legend()
# plt.show()

# train.groupby('year',as_index=False).agg(mean=('일사', 'mean'))
# train.groupby('year',as_index=False).agg(mean=('일사', 'sum'))

# df.query('num==1 & 일시< "2020-06-02"')['예측일사_MJ'].sum()
# train.query('일시< "2017-01-02"')['일사'].mean() * 0.2778

# train_new = pd.read_csv('../train_chungju_new.csv', encoding='utf-8')
# train_new
