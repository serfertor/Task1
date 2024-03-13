import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

if __name__ == '__main__':
    dataTrain = pd.read_csv('train.csv')
    dataTest = pd.read_csv('test.csv')

    # Разделение на признаки и целевую переменную
    X = dataTrain.drop('is_canceled', axis=1)
    y = dataTrain['is_canceled']

    labels = X.columns
    labels = labels.drop('reservation_status_date')
    label_encoder = LabelEncoder()

    # Подготовка датасета
    for i in labels:
        if len(X[i].value_counts()) <= 4:
            # Преобразования с помощью One-Hot encode
            X = pd.get_dummies(X, columns=[i])
            dataTest = pd.get_dummies(dataTest, columns=[i])
        else:
            # Преобразование с помощью Label Encoder
            label_for_encode = i
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


    print (X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Масштабирование признаков
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
