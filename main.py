import pandas as pd
import tensorflow as tf
from keras.src.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Подготовка датасета
def rebuild_dataset(dataset):
    labels = dataset.columns
    labels = labels.drop('reservation_status_date')
    label_encoder = LabelEncoder()

    # Преобразование с помощью Label Encoder
    for i in labels:
        dataset[i] = label_encoder.fit_transform(dataset[i])

    # Преобразование даты
    date = dataset['reservation_status_date']
    dataset = dataset.drop('reservation_status_date', axis=1)
    date = date.str.split('-', expand=True)
    date.columns = ['year', 'month', 'day']
    date['year'] = date['year'].astype(int)
    date['month'] = date['month'].astype(int)
    date['day'] = date['day'].astype(int)
    dataset = pd.concat([dataset, date], axis=1)

    return dataset


# Обучение и сохранение модели
def train_model():
    data_train = pd.read_csv('train.csv')
    # Разделение на признаки и целевую переменную
    x = data_train.drop('is_canceled', axis=1)
    y = data_train['is_canceled']
    x = rebuild_dataset(x)

    # Разбитие датасета на обучающий и тестовый
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Масштабирование признаков
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Постройка и обучение модели
    model = tf.keras.Sequential()
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(x_test_scaled, y_test))

    model.save('hotel_model.h5')


if __name__ == '__main__':
    # Загрузка обученной модели
    model = tf.keras.models.load_model('hotel_model.h5')

    # Работа с данными
    data_test = pd.read_csv('test.csv')
    data_test = rebuild_dataset(data_test)
    scaler = StandardScaler()
    data_test_scaled = scaler.fit_transform(data_test)

    # Получение предсказаний модели
    answer = model.predict(data_test_scaled)
    answer = (answer > 0.5).astype(int)

    # Формирование итогового файла с ответами
    answer_csv = pd.DataFrame(answer, columns=['is_canceled'])
    # answer_csv.reset_index(inplace=True)
    # answer_csv.rename(columns={'index': 'Index_Column'}, inplace=True)
    answer_csv.to_csv('answer.csv', index_label='index')
