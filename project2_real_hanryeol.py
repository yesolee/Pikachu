import pandas as pd

df = pd.read_csv("data/data_week2.csv", encoding='euc-kr')
df


df.head()
df.describe()
df.info()

df.isnull().sum().sum()

df.dtypes

df.select_dtypes(include='object').nunique() # 2040

# 열 이름에서 단위 제거
df.columns = [col.split('(')[0].strip() for col in df.columns]
df.columns
df

df['비전기냉방설비운영'].value_counts()
df['태양광보유'].value_counts()
df['num'].value_counts()
df['date_time'].value_counts()
df['전력사용량'].value_counts()
df['기온'].value_counts()
df['풍속'].value_counts()
df['습도'].value_counts()
df['강수량'].value_counts() # 0이 개많음
df['일조'].value_counts() # 0.0이나 1.0이 개많음




# 분포 시각화

import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 환경에서 사용
plt.rcParams['axes.unicode_minus'] = False  # 음수 기호 깨짐 방지


# 수치형 변수만 선택
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

# 한 그래프에 모든 변수의 박스플롯과 히스토그램을 그리기 위해 서브플롯 생성
fig, axes = plt.subplots(nrows=2, ncols=len(numeric_columns), figsize=(15, 8))

# 박스플롯 그리기
for i, col in enumerate(numeric_columns):
    axes[0, i].boxplot(df[col].dropna(), vert=False)
    axes[0, i].set_title(f'Boxplot of {col}')

# 히스토그램 그리기
for i, col in enumerate(numeric_columns):
    axes[1, i].hist(df[col].dropna(), bins=20)
    axes[1, i].set_title(f'Histogram of {col}')

plt.tight_layout()
plt.show()

import seaborn as sns

# 숫자형 변수만 선택하여 상관행렬 계산
numeric_df = df.select_dtypes(include=['int64', 'float64'])
numeric_correlation_matrix = numeric_df.corr()

# 숫자형 변수의 상관행렬 히트맵 그리기
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('숫자형 변수 간 상관행렬')
plt.show()

# 피어슨 상관계수를 계산하기 위해 pandas를 사용
# 비전기냉방설비운영과 태양광보유가 이진 변수, 전력사용량이 연속형 숫자형 변수라고 가정

# 상관계수 계산
pearson_corr_비전기냉방설비운영 = df['비전기냉방설비운영'].corr(df['전력사용량'])
pearson_corr_태양광보유 = df['태양광보유'].corr(df['전력사용량'])

pearson_corr_비전기냉방설비운영, pearson_corr_태양광보유
