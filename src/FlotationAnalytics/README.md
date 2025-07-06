Данная работа представляет автоматизацию анализа флотации. Вы можете загрузить видеозапись, выбрать интересующий трекер и на выходе будет детальный анализ трекинга пузырей.
=====================

Выполните следующий код в терминале, чтобы создать и перейти в виртуальное окружение:

`python3 -m venv .venv`

`source ./.venv/bin/activate`

Установите все необходимые зависимости:

`pip install -r requirements.txt`

В моем коде используются вызовы следующих моделей: **DeepSORT** и **CounTR**. Их сначала необходимо склонировать себе в проект:

DeepSORT: `git clone https://github.com/nwojke/deep_sort.git`

CounTR: `git clone https://github.com/Verg-Avesta/CounTR.git`

Также в репозитории CounTR нужно внести некоторые изменения, чтобы корректно запустить проект:
- в файле **CounTR/util/misc.py**:
  
  закомментировать строчку`from torch._six import inf` и прописать `inf = float('inf')`
  
- в файле **CounTR/util/pos_embed.py**:

  `omega = np.arange(embed_dim // 2, dtype=np.float)` на `omega = np.arange(embed_dim // 2, dtype=float)`

В файле **CounTR/models_mae_cross.py** нужно написать правильный путь, а именно добавить название папки

`from CounTR.models_crossvit import CrossAttentionBlock`

`from CounTR.util.pos_embed import get_2d_sincos_pos_embed`


Скачайте веса и добавьте в папку **model**: 

https://drive.google.com/file/d/1CzYyiYqLshMdqJ9ZPFJyIzXBa7uFUIYZ/view?usp=sharing

В данной работе используются следующие существующие методы трекинга:
- SORT
- DeepSORT
- Bot-SORT
- ByteTrack

Для детекции объектов ограничивающими рамками:
- YOLOv11

Для детекции центров объектов:
- CounTR
- PseCO

### Запуск основного сервиса: 

`PYTHONPATH=абсолютный_путь_до_проекта HYDRA_FULL_ERROR=1 python app/main.py tracker=botsort output_dir="resuts_end"`,

где tracker - выбор трекера, output_dir - папка для сохранения результатов

### Запуск тестов: 

`PYTHONPATH=абсолютный_путь_до_проекта pytest tests/test_metrics.py -v`

Также представлены скрипты, которые не связаны с сервисом, но позволяют провести анализ.
-----------------------------------

**Конвертация из YOLO формата в CVAT аннотацию:** `fromYOLOtoCVAT.py`

**Подсчет контролируемых метрик: MOTP и MOTA.** А также визуализация сопоставлений истинных ограничивающих рамок с предсказанными лежит в `metrics.py`

В коде используются `cvat_annotations.xml` и `output_botsort`.

`cvat_annotations.xml` - разметка в формате CVAT

`output_botsort` - разметка в расширенном формате YOLO

Код, который по сегментации делает разметку, представлен в `fromMaskToYolo.py`.

Запустить код можно с помощью команды:

`!python fromMaskToYolo.py --image_folder '/path/to/images' --mask_folder '/path/to/masks' --output_folder '/path/to/output' --class_id 0`, где

- **image_folder** - путь к папке с изображениями,

- **mask_folder** - путь к папке с масками,

- **output_folder** - путь для сохранения YOLO разметки

### Проведенное исследование

Для оценки качества работы разных методов использовался Google Colab. Код из него был экспортирован в формате `.py` и находится в папке **research**.

Он включает в себя обучение детектора, запуск моделей, замер FPS, подсчет среднего перемещения и других метрик. Также в нем реализована визуализация трекинга.
