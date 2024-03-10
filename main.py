import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

if __name__ == '__main__':
    dataTrain = pd.read_csv('train.csv')
    dataTest = pd.read_csv('test.csv')

    # Разделение на признаки и целевую переменную
    X = dataTrain.drop('is_canceled', axis=1)
    y = dataTrain['is_canceled']

    # Подготовка датасета
    # Преобразования с помощью One-Hot encode
    X = pd.get_dummies(X, columns=['hotel', 'meal', 'distribution_channel', 'deposit_type', 'customer_type'])
    dataTest = pd.get_dummies(dataTest, columns=['hotel', 'meal', 'distribution_channel',
                                                 'deposit_type', 'customer_type'])

    # Преобразование с помощью Label Encoder
    label_encoder = LabelEncoder()
    label_for_encode = ['arrival_date_month', 'country', 'market_segment', 'reserved_room_type', 'assigned_room_type']
    for i in label_for_encode:
        X[i] = label_encoder.fit_transform(X[i])
        dataTest[i] = label_encoder.fit_transform(dataTest[i])

    # Преобразование даты
    date = X['reservation_status_date']
    X = X.drop('reservation_status_date', axis=1)
    date = date.str.split('-', expand=True)
    date.columns = ['year', 'month', 'day']
    date['year'] = date['year'].astype(int)
    date['month'] = date['month'].astype(int)
    date['day'] = date['day'].astype(int)
    X = pd.concat([X, date], axis=1)

    date = dataTest['reservation_status_date']
    dataTest = dataTest.drop('reservation_status_date', axis=1)
    date = date.str.split('-', expand=True)
    date.columns = ['year', 'month', 'day']
    date['year'] = date['year'].astype(int)
    date['month'] = date['month'].astype(int)
    date['day'] = date['day'].astype(int)
    dataTest = pd.concat([dataTest, date], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Масштабирование признаков
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
