# 모듈 불러오기
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 불러오기
rawdata = pd.read_csv("../data/data_week2.csv", encoding='cp949')

rawdata.head()

### 건물끼리 완전 같은 데이터가 있는지 확인해보기.
## 1번 건물과 2040시간 동안 기온이 완전 같았던 건물이 있는지 알아보기.

# 1번 건물의 2040시간 동안 기온
rawdata[rawdata['num'] == 1]['기온(°C)']

# 2번 건물의 2040시간 동안 기온
rawdata[rawdata['num'] == 2]['기온(°C)']


####### 반복문
my_result_dict_temp = dict() # 빈 딕셔너리 생성
pass_list = [] # 이미 일치됐다고 메모한 건물은 패스 할거임.


for i in range(1,61):
    if i in pass_list:
        pass
    else:
        my_result_dict_temp[f'{i}번 건물과 기온이 같은 것'] = [] # 딕셔너리에 새로운 리스트 생성.
        for j in range(1,61):
            first_building_list = list(rawdata[rawdata['num'] == i]['기온(°C)'])
            second_building_list = list(rawdata[rawdata['num'] == j]['기온(°C)'])
            if i == j:
                pass
            elif first_building_list == second_building_list:
                my_result_dict_temp[f'{i}번 건물과 기온이 같은 것'].append(j)
                pass_list.append(j)
                print(f"{i} 건물과 {j} 건물의 기온이 2040시간동안 정확히 일치함!")
########## 반복문 끝

my_result_dict_temp
len(my_result_dict_temp) # 22
# 지역이 17개일건데 왜 22개가 나왔을까?


### 반복문 검증

# 60번 건물의 2040시간 동안 습도
rawdata[rawdata['num'] == 60]['습도(%)']

# 48번 건물의 2040시간 동안 습도
rawdata[rawdata['num'] == 48]['습도(%)']

# 두 건물의 2040시간동안 습도가 같냐?
list(rawdata[rawdata['num'] == 60]['습도(%)']) == list(rawdata[rawdata['num'] == 48]['습도(%)'])
# TRUE

# 두 건물의 2040시간동안 일조 같냐?
list(rawdata[rawdata['num'] == 60]['일조(hr)']) == list(rawdata[rawdata['num'] == 48]['일조(hr)'])
# TRUE

# 결국 날씨 데이터 5가지 중 하나라도 같으면 나머지도 같음.
# 5데이터 전부 하나의 기관에서 받아온 것임을 유추 가능.

####### 반복문 풍속!!
my_result_dict_WindSpeed = dict() # 빈 딕셔너리 생성
pass_list = [] # 이미 일치됐다고 메모한 건물은 패스 할거임.


for i in range(1,61):
    if i in pass_list:
        pass
    else:
        my_result_dict_WindSpeed[f'{i}번 건물과 풍속이 같은 것'] = [] # 딕셔너리에 새로운 리스트 생성.
        for j in range(1,61):
            first_building_list = list(rawdata[rawdata['num'] == i]['풍속(m/s)'])
            second_building_list = list(rawdata[rawdata['num'] == j]['풍속(m/s)'])
            if i == j:
                pass
            elif first_building_list == second_building_list:
                my_result_dict_WindSpeed[f'{i}번 건물과 풍속이 같은 것'].append(j)
                pass_list.append(j)
                print(f"{i} 건물과 {j} 건물의 풍속이 2040시간동안 정확히 일치함!")
########## 반복문 끝
my_result_dict_WindSpeed
len(my_result_dict_WindSpeed)