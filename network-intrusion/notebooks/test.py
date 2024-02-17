import pandas as pd
import numpy as np
import holidays
min_value = 0.2
max_value = 0.8
tw_holidays = holidays.country_holidays('TW')
def preprocessing(df):
    '''
    資料前處理
    1. MinMax Scaler: 將資料壓縮至0.2與0.8之間
    2. 日期資訊: 加入資料於禮拜幾與幾點產生的資訊(one hot encoding)
    --------------------------------------------------------
    Input: eps raw data
    Output: nparray[shape: (feature_count, feature_dim)]
    '''
    max_count = df['count'].max()
    min_count = df['count'].min()
    df['create_time'] = pd.to_datetime(df['create_time'])
    print(df)
    start_time = df.iloc[0]['create_time']
    end_time = df.iloc[-1]['create_time']
    freq = '1min'
    date_range = pd.date_range(start=start_time, end=end_time, freq=freq)
    df_all = pd.DataFrame({'create_time': date_range})
    print(df_all)

    df = pd.merge(df,df_all, on=['create_time'],how='outer')
    print(df[-10:])
    df['count'].fillna(0, inplace=True)

    df['date'] = df['create_time'].dt.date
    df['hour'] = df['create_time'].dt.hour
    df['minute'] = df['create_time'].dt.minute
    df['day'] = df['create_time'].dt.dayofweek
    first_index = df.index[df['minute'] == 0].min()
    df = df[first_index:]

    data = []
    for i in range(int(len(df)/60)):
        # expected_minutes = range(60)
        # min_list = df[60*i:(i+1)*60]['minute'].values.tolist()
        # missing_minutes = set(expected_minutes) - set(min_list)
        # if missing_minutes:
        #     print(df.iloc[60*i]['create_time'])
        #     print(missing_minutes)
        temp = df[60*i:(i+1)*60]['count'].values.tolist()
        temp = min_value+(max_value-min_value)*(temp-min_count)/(max_count-min_count)

        # 星期幾
        week = [0]*7
        week[df.iloc[60*i]['day']] = 1
        temp = np.concatenate((temp, week))

        # 幾點
        hour = [0]*24
        hour[df.iloc[60*i]['hour']] = 1
        temp = np.concatenate((temp, hour))

        # 國定假日
        date = df.iloc[60*i]['create_time']
        if date in tw_holidays:
            temp = np.concatenate((temp, [1]))
            #print(date)
        else:
            temp = np.concatenate((temp, [0]))

        data.append(temp)

    return np.array(data)
csv_file_path = 'eps_hour.csv'
df_total = pd.read_csv(csv_file_path)
df_total.columns = ['create_time','count']
df_scaled = preprocessing(df_total)