## Michael Retriever DistilBERT
###Введение

В этом репозитории находится веб-сервис (чат-бот), который по сути является ретривером – системой поиска среди реплик Майкла Скота из сериала The Office (US). Сервис построен на fine-tuned модели DistilBERT, обученной на парах вопрос-ответ, извлечённых из диалогов сериала. Сервис работает исключительно на английском языке.

[![Screenshot-2025-03-09-at-16-46-25.png](https://i.postimg.cc/DZKsykQr/Screenshot-2025-03-09-at-16-46-25.png)](https://postimg.cc/SXDjDPvR)
Скриншот интерфейса бота 

Перед запуском убедитесь, что:
- Файл модели (michael_scott_model.bin) и база ответов Майкла загружены и доступны.

### Структура проекта

#### Backend  
Содержит файлы для запуска модели, загрузки датасета, а также ноутбук test.ipynb для тестирования в формате Jupyter. Веб-сервер реализован с помощью Flask и использует Git LFS для хранения больших файлов (например, модели и базы эмбеддингов).

#### Frontend  
Простой HTML‑чат, предоставляющий интерфейс для общения с сервисом. История переписки сохраняется только в рамках сессии (не сохраняется на сервере).

#### Запуск локально 
Сервис запускается через Docker‑Compose. По умолчанию веб-сервер будет доступен по адресу http://localhost:5001 (порт можно изменить в настройках docker‑compose).

#### Шаги для запуска: 

Клонируйте репозиторий:  
```git clone https://github.com/andkhalov/Michael_retriever_distilbert.git```  
```cd Michael_retriever_distilbert```  

Запустите Docker‑Compose:  
```docker-compose up --build```  

Откройте браузер и перейдите по адресу:  
```http://localhost:5001```
(Если порт изменён, используйте соответствующий URL.)

### Отчет о ходе эксперимента

#### Датасет:
В качестве датасета использован [The Office (US) - Complete Dialogue/Transcript](https://www.kaggle.com/datasets/nasirkhalid24/the-office-us-complete-dialoguetranscript/data) с Kaggle. Из исходного датасета извлекались пары реплик, где всё, что предшествует реплике Майкла, рассматривалось как запрос, а сама реплика Майкла – как корректный ответ.

[![download.png](https://i.postimg.cc/9QKZKqrN/download.png)](https://postimg.cc/nXKXQrKv)
EDA датасета

#### Формирование данных:  
Пары были собраны по наивному принципу, после чего датасет был дополнен негативными примерами (неверными парами). Отрицательных примеров было сгенерировано в соотношении 10:1 (на один положительный пример 10 отрицательных), что дало итоговый датасет примерно из 100 записей.

#### Обучение модели:  
Модель DistilBERT fine-tuned обучалась на 10 эпохах. Первый проход не давал результатов, поэтому датасет был предварительно отфильтрован: оставлены только те примеры, где вопросы содержат не менее 5 слов, а ответы – не менее 8 слов. После этого кривая обучения стала показывать стабильное снижение ошибки, хоть результаты и остаются неидеальными.
Выходной слой модели состоит из одного нейрона, выдающего логит, на основе которого определяется класс (бинарная классификация). Для оптимизации использовалась функция потерь BCEWithLogitsLoss, а скорость обучения (LR) была установлена осторожно.

[![download-1.png](https://i.postimg.cc/6qtdzJyW/download-1.png)](https://postimg.cc/cgkg4zR2)
Кривая обучения модели

#### Эмбеддинги:  
После обучения модели были получены эмбеддинги как из базовой (pre-trained) версии модели DistilBERT, так и из fine-tuned версии. Были построены 2D отображения пар эмбеддингов с использованием TSNE и PCA. Визуальное сравнение показало, что хотя структура распределения пар эмбеддингов изменилась после обучения, четкого разделения между позитивными и негативными примерами не наблюдается.

[![download-3-PCA-base.png](https://i.postimg.cc/NfG6Hc3b/download-3-PCA-base.png)](https://postimg.cc/QVP92RxW)
Пары ответ-вопрос в двумерном векторном пространстве (PCA), базовая модель

[![download-3-PCA-ft.png](https://i.postimg.cc/FsPjb6mv/download-3-PCA-ft.png)](https://postimg.cc/gnZXmNnt)
Пары ответ-вопрос в двумерном векторном пространстве (PCA), модель после обучения

>> Вывод: Метод обучения на задаче бинарной классификации дает приближение для семантического сопоставления, однако для улучшения результатов целесообразно использовать контрастивное обучение с triplet loss.

#### Ретривер: 
Были получены эмбеддинги для базы ответов Майкла Скота, после чего реализована функция поиска по косинусной близости. Это позволяет системе быстро возвращать ответ, который наиболее семантически близок к входящему запросу. Преимущество такого подхода в том, что ответы всегда точны и не содержат галлюцинаций, поскольку они берутся из заранее подготовленной базы.

#### Пример использования
В интерактивном режиме можно ввести запрос, и сервис вернет соответствующий ответ. Например:

>> I’m in waiting room!   
Michaels Response: I burned my foot!!! Ok, twenty minutes, conference room, everybody's in there!   
Схожесть: 0.8755
>>

### Запуск веб-сервиса
Веб-сервис запускается командой python app.py (или через Docker‑Compose). API принимает POST‑запросы по адресу /api/chat, где JSON с ключом query передается в систему, а в ответ возвращается текст ответа и значение схожести.

### Дополнительные материалы
Подробный экспериментальный ноутбук доступен по [ссылке.](https://colab.research.google.com/drive/1MY-tDxPdmsh-iOYIFpwsxLCNOqdAzyGP?usp=sharing)

В рамках учебного эксперимента был разработан и протестирован retrieval‑based чат-бот, использующий fine-tuned DistilBERT. Несмотря на ограничения объема датасета и использованного подхода (бинарная классификация), полученные результаты демонстрируют потенциал метода. Для дальнейшего улучшения семантического сопоставления рекомендуется исследовать методы контрастивного обучения, такие как triplet loss.
