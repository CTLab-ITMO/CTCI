

CTCI - Однородные текстурные данные
==============================

Clumped texture composite images - изображения со скучкованными сложными данными, образующие текстуру. Такое изображение (или видео) подразумевает, что на одном изображении будут находиться множество объектов одного класса, находящихся в куче, т.е. перекрывая друг друга, в случайном порядке. Примерами таких данных могут являться пузыри, камни, любые одинаковые изделия на конвейере.

![](data/clumped_data.png)

Трудностью работой с такими данными являются отсутствие разметки в свободном доступе, трудоемкость разметки, требование к высокому качеству результатов.
Библиотека содержит методы, позволяющие автоматизировать процесс создания лейблов с помощью методов слабой разметки, методы самообучения и переноса между доменами.  Также представлены результаты обучения нескольких видов моделей на однотипных данных.

# Установка
1. Скачайте репозиторий по ссылке:
   ```git clone https://github.com/CTLab-ITMO/CTCI.git```
2. Перейдите в директорию проект
   ```cd CTCI```
3. Установите зависимости
```pip -r install requirements.txt```
4. Скачайте веса из [папки](https://disk.yandex.ru/client/disk/Веса%20CTCI) и поместите их рядом с выполняемым кодом.
# Слабая разметка

Слабая разметка однородных данных реализована с использованием моделей YOLOv8 и Segment Anything, а также с помощью алгоритма сегментации водоразделом. YOLOv8 используется для поисков маркеров, соответствующих крупным пузырям, SAM для поиска контуров крупных пузырей, водораздел для поиска контуров мелких пузырей. 

Функционал слабой разметки расположен в модуле `src/data/weakly_segmentation/annotation.py` в методе `annotation`. Разметка выполняется для указанной в аргументах папки с данными

Пример использования:

```python
import sys

from src.data.weakly_segmentation.annotation import annotation


data_dir = sys.argv[1]  # Директория, в которой лежат папки для разметки.
folder = sys.argv[2]    # Название папки, которую надо разметить
                        # ├──data_dir
                        #         ├── folder1
                        #         ├── folder2
                        #         ....

custom_yolo_checkpoint_path = sys.argv[3]  # Путь до весов модели YOLOv8
sam_checkpoint = sys.argv[4]  # Путь до весов модели SAM
sam_model_type = sys.argv[5]  # Тип модели SAM

narrowing = 0.20  # Значение сужения масок сегменатации. Предотвращает сливание масок однородных объектов
erode_iterations = 1  # Количество итераций эрозии 
processes_num = 3  # Количество параллельных процессов сегментации изображений
prompt_points = False  # Использование точек в промпте SAM

device = "cpu"  # Девайс, на котором выполняется разметка

annotation(
    data_dir, folder,
    custom_yolo_checkpoint_path, sam_checkpoint, sam_model_type, narrowing=narrowing,
    erode_iterations=erode_iterations, processes_num=processes_num, prompt_points=prompt_points,
    device=device
)
```
Результаты аннотации:
![Original - SAM - Watershed - Both](data/readme/Image.png)

Результаты аннотации на различных данных. Важно отметить, что алгоритм не работает в режиме реального времени, медиа показывает визуализацию.
![SAM + Watershed performance](data/readme/images_masks_output_video_masked.gif)

# Самообучение

## Алгоритмы самообучения 

Для реализации самообучения были выбраны алгоритмы barlow twins и MoCo

Barlow twins возможно можно реализовать при помощи соответствующего класс: 
```python
from barlow_twins import BarlowTwins

args = { 
    "batch_size" : 50, # Размер батчей при обучении
    "MLP" : "8192-8192-8912" # Структура  полносвязной сети 
}

model = BarlowTwins(args)

```
Энкодеры можно обучить в формате MoCo при помощи соответсвующих функций: 
```python
from train_moco import train

device = "cpu"

model_q = Net().to(device) #  2 энкодера, которые в связке обучаются 
model_k = copy.deepcopy(model_q) 

optimizer = optim.SGD(model_q.parameters(), lr=0.0001, weight_decay=0.0001) # Алгоритм оптимизации 
queue = initialize_queue(model_k, device, train_loader) # Очередь из объектов на которых обучается сеть 
epoch = 50 # Количество эпох 

train(model_q, model_k, device, train_loader, queue, optimizer, epoch)

```

## Обучение моделей 

### Скрипт обучения модели на основе Barlow Twins

```bash
python src/models/'barlow_twins'/barlow_twins.py images_path masks_path target_height target_width batch_size epochs
```

```images_path``` - путь к изображениям 

```masks_path``` - путь к маскам

```target_height``` - итоговая высота изображения

```target_width``` - итоговая ширина изображения 

```batch_size```  - размер батча при обучении 

```epochs`` - количество эпох обучения 

### Скрипт обучения для моделей на основе MoCo (Momentum contrast) 

```bash
python src/models/moco/train_moco.py images_path masks_path out_dir batch_size epochs
```

`images_path``` - путь к изображениям 

```masks_path``` - путь к маскам

```out_dir``` - путь сохранения результата 

```batch_size``` - размер батча при обучении 

```epochs``` - количесто эпох обучения 

## Результаты работы алгоритмов самообучения 

| Исходное изорбражение  | Barlow twins | MoCo | 
| ------------- | ------------- | -------------| 
| ![](data/orig/0.png)  | ![](data/bt/1.png)  | ![](data/moco/0.png) |
| ![](data/orig/15.png)  | ![](data/bt/15.png)  | ![](data/moco/15.png) |


## Запуск моделей 
Barlow twins 


```bash
python src/models/barlow_twins/unet/inference_bt_unet images tar_dir height width
```

**MoCo**


```bash
python src/models/moco/inference_moco images tar_dir height width
```

images - директория с фотографиями для обработки

tar_dir - директория результатов работы нейросети

height, width - размер изображений 


# Сегментация

## Конфигурационные файлы

Мы используем hydra для конфигурации проекта. Все файлы конфигураций находятся в папке `configs`. Ниже приведена структура этой папки:
```
└── configs/
    ├── arch/
    │   ├── deeplabv3.yaml
    │   └── ...
    ├── augmentations/
    │   ├── train.yaml
    │   └── valid.yaml
    ├── data/
    │   └── data.yaml
    ├── experiment/
    │   └── experiment.yaml
    ├── module/
    │   └── module.yaml
    ├── trainer/
    │   └── trainer.yaml
    ├── config.yaml
    └── preprocess.yaml
```

## Модели сегментации

Для сегментации изображений однородных данных реализованы такие модели, как: SegFormer, Swin+UNETR, DeepLabv3, HRNet. Модели инициализируются из файла конфигурации. Добавьте нужную вам модель в `configs/arch` и измените файл config.yaml. Проект поддерживает timm, pytorch-segmentation-models, transformers.
Если модели не нужно реализовать отдельную логику метода `forward`, достаточно добавить файл конфигурации в таком виде:
```yaml
_target_: segmentation_models_pytorch.DeepLabV3Plus
encoder_name: resnet34
in_channels: 3
classes: 1
```

Возможна инициализация модели с помощью соответствующего класса.

[//]: # (я не шарю за хтмл поэтому оставлю это здесь)


<details>
    <summary> Segformer </summary>

Результаты обучения модели:
![Segformer performance](data/readme/segformer_output_video_masked.gif)



</details>


<details>
    <summary> Swin-UNETR  </summary>

Результаты обучения модели:
![Swin performance](data/readme/swinv2_output_video_masked.gif)

</details>


<details>
    <summary>  HRNet  </summary>

Результаты обучения модели:
![HRNet performance](data/readme/hrnet_w18_small_v2_output_video_masked.gif)

</details>

<details>
    <summary>  DeepLabV3  </summary>

Результаты обучения модели:
![DeepLab performance](data/readme/resnet34-run2_output_video_masked.gif)

</details>


## Обучение моделей

Для того чтобы запустить обучение моделей, 

```bash
make run_training
```

В директории `src/infrastructure/models_tracking`  расположены скрипты, позволяющие обучить или дообучить модели “из коробки” с использованием конфигурационного файла. Пример использования:




### ADELE
Библиотека поддерживает [Adaptive Early-Learning Correction for Segmentation from Noisy Annotations](https://arxiv.org/abs/2110.03740). Для использования достаточно создать датасет **без аугментаций**, то есть без аффинных преобразований, но в том виде, в котором модель будет получать изображения на инференсе. В класс тренировщика необходимо будет передавать отдельно инициализированный датасет из тренировочных файлов. В файле конфигураций необходимо добавить шаг, через который будет применяться метод. 

```python
# В файле трекинга добавить следующие строки

from src.models.utils.config import read_yaml_config
from src.features.segmentation.dataset import get_train_dataset_by_config

config_handler = read_yaml_config(config_path)
adele_dataset = get_train_dataset_by_config(
        config_handler,
        transform=tr, # стандартные преобразования
        augmentation_transform=None
    )
adele_dataset.return_names = True
```


# Экспорт в ONNX

Описанные выше модели могут быть экспортированы в onnx формат для дальнейшего запуска в любом окружении, поддерживающем onnx-runtime. 

Экспорт каждой из моделей можно выполнить с использованием заготовленных скриптов, расположенных в директории `src/infrastructure/models_inference` . Например:

```bash
python src/infrastructure/models_inference/segformer_export.py <config_path>
```

Исходный код конвертации и квантизации находиться в модуле `src/models/inference.py` .


Организация проекта
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
