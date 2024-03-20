Первое тестовое задание по направлению Искуственный интелект

Для решения задачи была выполнена предобработка датасета (использован Label Encoding и вручную размечен столбец с датой). Далее была построена, обучена модель на основе keras.Sequential затем сохранена. Точность модели на тренировочныхх данных 0.99. Полученные результаты по тестовым данным сохранены в файле answer.csv.

Модель представляет собой три слоя: первый и второй из 64 нейронов и с функцией активации relu, выходной слой содержит один нейрон и функцию активации sigmoid. 
Для запуска требуется: Python 3.12, установить библиотеки из файла requirements.txt и запустить файл main.py.

![image](https://github.com/serfertor/Task1/assets/37975885/2d721d41-222c-4aa5-a71c-efaa26282c60)

Скриншот 1 - Структура проекта

![image](https://github.com/serfertor/Task1/assets/37975885/89f3edc8-f1a0-44d8-b9ae-b6b994a63d18)

Скриншот 2 - Выходной файл

![image](https://github.com/serfertor/Task1/assets/37975885/e505e1c5-e382-4079-8a3b-9d44b0bd0833)

Скриншот 3 - Точность модели после обучения
