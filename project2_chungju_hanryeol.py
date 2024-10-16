import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# 데이터 불러오기
df = pd.read_csv("data/data_week2.csv", encoding='euc-kr')
test = pd.read_csv("data/test_2020.csv")
train = pd.read_csv('data/train_chungju_new.csv')
# 전처리 베이스라인
train.columns = [col.split('(')[0].strip() for col in train.columns]
test.columns = [col.split('(')[0].strip() for col in test.columns]

train.isna().sum()/len(train)
train.info()
test.info()

# month/day를 month와 day로 분리하고 hour를 추가
train['일시'] = pd.to_datetime(train['일시'])
train['month'] = train['일시'].dt.month
train['day'] = train['일시'].dt.day
train['hour'] = train['일시'].dt.hour
train['day_of_year'] = train['일시'].dt.dayofyear

train['일조'] = train['일조'].fillna(0.0)
train['일사'] = train['일사'].fillna(0.0)
train['강수량'] = train['강수량'].fillna(0.0)
train['풍속'] = train['풍속'].fillna(train['풍속'].mean())

# month/day를 month와 day로 분리하고 hour를 추가
test['일시'] = pd.to_datetime(test['일시'])
test['month'] = test['일시'].dt.month
test['day'] = test['일시'].dt.day
test['hour'] = test['일시'].dt.hour
test['day_of_year'] = test['일시'].dt.dayofyear

test['일조'] = test['일조'].fillna(0.0)
test['일사'] = test['일사'].fillna(0.0)
test['강수량'] = test['강수량'].fillna(0.0)
# test 습도 변수 결측치 2개 있음 이슈 -> mean값으로 대체
test['습도'] = test['습도'].fillna(test['습도'].mean())

train.isna().sum()
test.isna().sum()

# 시계열 분해
from statsmodels.tsa.seasonal import seasonal_decompose

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic' 

# 마이너스 폰트 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# '일시' 열을 datetime 형식으로 변환하고 인덱스로 설정 (필수 단계)
train['일시'] = pd.to_datetime(train['일시'])
train.set_index('일시', inplace=True)

# 시계열 분해: 주기를 24로 설정해 일간 패턴 분석
result = seasonal_decompose(train['일사'], model='additive', period=24)

# 분해된 시계열 시각화
result.plot()
plt.show()

# 시계열 분해: 주기를 365 * 24로 설정해 연간 패턴 분석
result = seasonal_decompose(train['일사'], model='additive', period=8760)

# 분해된 시계열 시각화
result.plot()
plt.show()

#트렌드 0.05오른게 유의미한가
#시즈널 확대해보고 둘 다 보여주기. 

# Seasonal 컴포넌트에서 처음 100개만 추출
seasonal_data = result.seasonal[:100]

# 추출한 데이터 시각화
plt.figure(figsize=(12, 6))
plt.plot(seasonal_data, linestyle='-', linewidth=1.5, alpha=0.8)
plt.title('Expanded View of Seasonal Component (First 100 Values)')
plt.xlabel('Date')
plt.ylabel('Seasonal Effect')
plt.grid(True)
plt.show()

train['일사'].mean()
train['일사'].max()
train['일사'].min()

# 정상성 테스트 (사리마 하려면 필요)
from statsmodels.tsa.stattools import adfuller

# ADF 테스트 함수 정의
def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    
    if result[1] <= 0.05:
        print("시계열 데이터는 정상입니다.")
    else:
        print("시계열 데이터는 비정상입니다.")

# '일사' 열에 대해 ADF 테스트 수행
adf_test(train['일사'])

# 정상임 -> 그래서 차분을 나타내는 d를 1로 설정

from statsmodels.tsa.statespace.sarimax import SARIMAX

# SARIMA 모델 설정 (p, d, q) * (P, D, Q, S)
model = SARIMAX(train['일사'], order=(7, 1, 5), seasonal_order=(2, 1, 2, 24))
model_fit = model.fit()

# 예측 수행
forecast_steps = 120
forecast = model_fit.forecast(steps=forecast_steps)

# 예측 데이터를 데이터프레임으로 변환하여 날짜별로 그룹화
forecast_dates = pd.date_range(start=train.index[-1], periods=forecast_steps+1, freq='H')[1:]
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})

# 일별로 그룹화하여 통계 요약 정보 출력
daily_forecast_summary = forecast_df.groupby(forecast_df['Date'].dt.date)['Forecast'].agg(['mean', 'min', 'max', 'median'])

print(daily_forecast_summary)

#                mean       min       max    median
# Date                                              
# 2020-06-01  0.957417 -0.004680  2.833132  0.372907
# 2020-06-02  0.974490  0.073810  2.828687  0.377892
# 2020-06-03  0.979025  0.080678  2.830784  0.382768
# 2020-06-04  0.982973  0.084592  2.834056  0.387466
# 2020-06-05  0.986951  0.089586  2.838940  0.390592


# 예측 결과 시각화
plt.figure(figsize=(10, 6))
plt.plot(train.index[-100:], train['일사'][-100:], label='Observed', color='blue')
plt.plot(pd.date_range(start=train.index[-1], periods=forecast_steps+1, freq='H')[1:], forecast, label='Forecast', color='red')
plt.title('SARIMA Forecast with Seasonal Index')
plt.xlabel('Time')
plt.ylabel('일사')
plt.legend()
plt.grid(True)
plt.show()

#sarima 는 y만 써서 x들의 정보를 사용하지 못함.
# 일사량은 구름, 온도, ,,,, 영향을 받을텐데 활용 못하는 단점.
# 3일치 예측해보자. 
# 실제값, 사리마, 모델. 

#######################################
# 2017년 9월 ~ 2018년 5월 전국 데이터 
# 2018년 1월에 전국적 이슈가 발생했는지에 대해 
# 즉 청주 데이터의 trend 팍 튄 게 자연스러운 건지

df_2017 = pd.read_csv("data/2017.csv",encoding='euc-kr')
df_2018 = pd.read_csv("data/2018.csv",encoding='euc-kr')
df_2019 = pd.read_csv("data/2019.csv",encoding='euc-kr')
df_2020 = pd.read_csv("data/2020.csv",encoding='euc-kr')


df_jg = pd.concat([df_2017, df_2018, df_2019, df_2020], axis=0, ignore_index=True)

df_jg['일시'] = pd.to_datetime(df_jg['일시'])
df_jg.set_index('일시', inplace=False)

df_jg.columns = [col.split('(')[0].strip() for col in df_jg.columns]

df_jg.isna().sum()

# '일시'를 기준으로 그룹화하고, 같은 시간대의 '일사' 값의 평균을 계산하여 중복된 시간대를 제거
df_jg_grouped = df_jg.groupby('일시', as_index=False)['일사'].mean()

df_jg_grouped

# raw 개수가 다른데, 이거 왠진 모르겠는데 전국 데이터로 다운 받으니까 08~ 18시까지밖에 안됨
# 나머지는 0으로 채울거임 그래서 

# 기존 데이터프레임인 df_jg_grouped에서 일시의 최소값과 최대값을 가져옴
start_date = df_jg_grouped['일시'].min().date()
end_date = df_jg_grouped['일시'].max().date()

# 19시부터 다음날 07시까지의 시간대를 생성하여 새로운 데이터프레임 생성
new_times = pd.date_range(start=start_date, end=end_date, freq='D')
new_rows = []

for date in new_times:
    # 19시부터 23시까지의 일시 추가
    for hour in range(19, 24):
        new_rows.append([pd.Timestamp(date.year, date.month, date.day, hour), 0.0])
    
    # 다음날 00시부터 07시까지의 일시 추가
    next_day = date + pd.Timedelta(days=1)
    for hour in range(0, 8):
        new_rows.append([pd.Timestamp(next_day.year, next_day.month, next_day.day, hour), 0.0])

# 새로운 시간대 데이터프레임 생성
df_new_times = pd.DataFrame(new_rows, columns=['일시', '일사'])

# 기존 데이터와 새로운 시간대 데이터를 결합
df_complete = pd.concat([df_jg_grouped, df_new_times], ignore_index=True)

# '일시'를 기준으로 정렬하여 전체 데이터를 시간순으로 정리
df_complete = df_complete.sort_values(by='일시').reset_index(drop=True)

df_complete

# '일시'를 인덱스로 설정하여 시계열 분해가 제대로 이루어지도록 함
df_complete.set_index('일시', inplace=True)

# 시계열 분해: 주기를 24로 설정해 일간 패턴 분석
result = seasonal_decompose(df_complete['일사'], model='additive', period=24)

# 분해된 시계열 시각화
result.plot()
plt.show()

# 시계열 분해: 주기를 365 * 24로 설정해 연간 패턴 분석
result = seasonal_decompose(df_complete['일사'], model='additive', period=8760)

# 분해된 시계열 시각화
result.plot()
plt.show()

# Seasonal 컴포넌트에서 처음 100개만 추출
seasonal_data = result.seasonal[:100]

# 추출한 데이터 시각화
plt.figure(figsize=(12, 6))
plt.plot(seasonal_data, linestyle='-', linewidth=1.5, alpha=0.8)
plt.title('Expanded View of Seasonal Component (First 100 Values)')
plt.xlabel('Date')
plt.ylabel('Seasonal Effect')
plt.grid(True)
plt.show()

#####################################
# 청주 히스토그램
# (0포함 + 0제외) 

# 서브플롯 설정 (1행 2열로 구성)
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

# 0값을 포함한 일사량 히스토그램
axes[0].hist(df_complete['일사'], bins=50, color='blue', alpha=0.7)
axes[0].set_title('청주 일사량 분포 (0값 포함)', fontsize=14)
axes[0].set_xlabel('일사', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].grid(True)

# 최대 빈도수를 Y축 최대 범위로 설정
axes[0].set_ylim(0, np.max(np.histogram(df_complete['일사'], bins=50)[0]))

# 0값을 제외한 일사량 히스토그램
axes[1].hist(df_complete[df_complete['일사'] > 0]['일사'], bins=50, color='green', alpha=0.7)
axes[1].set_title('청주 일사량 분포 (0값 제외)', fontsize=14)
axes[1].set_xlabel('일사', fontsize=12)

# 최대 빈도수를 Y축 최대 범위로 설정
axes[1].set_ylim(0, np.max(np.histogram(df_complete[df_complete['일사'] > 0]['일사'], bins=50)[0]))

axes[1].grid(True)

# 그래프 간격 조정
plt.tight_layout()

# 그래프 출력
plt.show()


###########################################

# 시간별 일사량 변화

# 시간별 평균 일사량 계산
hourly_avg = train.groupby('hour')['일사'].mean()

# 시간별 일사량 그래프 그리기
plt.figure(figsize=(12, 6))
plt.plot(hourly_avg.index, hourly_avg.values, marker='o', linestyle='-', color='blue')
plt.title('시간별 평균 일사량 변화', fontsize=14)
plt.xlabel('시간', fontsize=12)
plt.ylabel('평균 일사량', fontsize=12)
plt.grid(True)
plt.xticks(range(0, 24))  # x축 눈금을 0부터 23까지 표시
plt.show()

#############################################

# 상관행렬 

# 관심 있는 열을 선택하여 상관행렬 계산
correlation_matrix = train[['기온', '강수량', '풍속', '습도', '일조', '일사']].corr()

# 상관행렬 히트맵 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True, cbar_kws={"shrink": .8})
plt.title('상관행렬', fontsize=16)
plt.show()

###########################################

# 서브플롯 설정 (3행 2열로 구성)
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# 기온 변화
axes[0, 0].plot(train.index, train['기온'], color='red', label='기온', linewidth=2)
axes[0, 0].set_title('청주 기온 변화', fontsize=12)
axes[0, 0].set_ylabel('기온 (°C)', fontsize=10)
axes[0, 0].grid(True)

# 습도 변화
axes[0, 1].plot(train.index, train['습도'], color='blue', label='습도', linewidth=2)
axes[0, 1].set_title('청주 습도 변화', fontsize=12)
axes[0, 1].set_ylabel('습도 (%)', fontsize=10)
axes[0, 1].grid(True)

# 일조 변화
axes[1, 0].plot(train.index, train['일조'], color='orange', label='일조', linewidth=2)
axes[1, 0].set_title('청주 일조 변화', fontsize=12)
axes[1, 0].set_ylabel('일조 (시간)', fontsize=10)
axes[1, 0].grid(True)

# 풍속 변화
axes[1, 1].plot(train.index, train['풍속'], color='green', label='풍속', linewidth=2)
axes[1, 1].set_title('청주 풍속 변화', fontsize=12)
axes[1, 1].set_ylabel('풍속 (m/s)', fontsize=10)
axes[1, 1].grid(True)

# 일사량 변화
axes[2, 0].plot(train.index, train['일사'], color='purple', label='일사량', linewidth=2)
axes[2, 0].set_title('청주 일사량 변화', fontsize=12)
axes[2, 0].set_ylabel('일사량 (MJ/m²)', fontsize=10)
axes[2, 0].grid(True)

# 빈 서브플롯 비워두기 (2행 2열에서 하나 비워두기)
axes[2, 1].axis('off')

# 그래프 간격 조정
plt.tight_layout()
plt.show()

###########################

# 특정 기간 설정 (예: 2018년 1월부터 3월까지)
start_date = '2018-01-01'
end_date = '2018-01-05'

# 특정 기간에 해당하는 데이터 필터링
subset_train = train.loc[start_date:end_date]

# 일조 데이터를 리스트로 변환
일조_리스트 = subset_train['일조'].tolist()

# 일시 데이터와 일조 데이터를 함께 시각화
plt.figure(figsize=(12, 6))
plt.plot(subset_train.index, 일조_리스트, color='orange', label='일조', linewidth=2)
plt.title('청주 일조 변화 (2018.01.01 ~ 2018.01.05)', fontsize=16)
plt.xlabel('일시', fontsize=12)
plt.ylabel('일조 (시간)', fontsize=12)
plt.xticks(rotation=45)  # x축 날짜 각도 조정
plt.grid(True)
plt.legend()
plt.show()

###############################

# 특정 기간 설정 (예: 2018년 1월부터 3월까지)
start_date = '2018-01-01'
end_date = '2018-01-05'

# 특정 기간에 해당하는 데이터 필터링
subset_train = train.loc[start_date:end_date]

# 일조 데이터를 리스트로 변환
일조_리스트 = subset_train['일사'].tolist()

# 일시 데이터와 일조 데이터를 함께 시각화
plt.figure(figsize=(12, 6))
plt.plot(subset_train.index, 일조_리스트, color='orange', label='일사', linewidth=2)
plt.title('청주 일사 변화 (2018.01.01 ~ 2018.01.05)', fontsize=16)
plt.xlabel('일시', fontsize=12)
plt.ylabel('일사 (시간)', fontsize=12)
plt.xticks(rotation=45)  # x축 날짜 각도 조정
plt.grid(True)
plt.legend()
plt.show()



