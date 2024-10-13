

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
```make install_all```
4. Скачайте веса из [папки](https://disk.yandex.ru/client/disk/Веса%20CTCI) и поместите их рядом с выполняемым кодом.


## Конфигурационные файлы

Мы используем hydra для конфигурации проекта. Все файлы конфигураций находятся в папке `configs`. Ниже приведена структура этой папки:
```
└── configs/
    ├── augmentations/           # конфигурации аугментаций
    │   ├── train.yaml
    │   └── valid.yaml
    ├── data/                    # конфигурации датамодуля и коррекции масок
    │   └── data.yaml
    ├── experiment/              # конфигурации эксперимента
    │   └── experiment.yaml
    ├── module/                  # конфигурации модуля модели
    │   ├── arch/                # конфигурации архитектуры модели
    │   │   ├── deeplabv3.yaml  
    │   │   └── ...
    │   └── module.yaml
    ├── preprocess/              # конфигурации водораздела
    │   └── preprocess.yaml
    ├── sam_yolo/                # конфигурации моделей SAM и YOLO для аннотации
    │   └── sam_yolo.yaml
    ├── trainer/                 # конфигурации класса обучения
    │   └── trainer.yaml
    ├── config.yaml
    └── annotation.yaml
```
В папках `sam_yolo` и `preprocess` находятся конфигурации для слабой разметки. Они объединяются файлом `annotation.yaml`.
Для задачи обучения сегментационных моделей используются все остальные папки, которые объединяются файлом `config.yaml`.

# Слабая разметка

Слабая разметка однородных данных реализована с использованием моделей YOLOv8 и Segment Anything, а также с помощью алгоритма сегментации водоразделом. YOLOv8 используется для поисков маркеров, соответствующих крупным пузырям, SAM для поиска контуров крупных пузырей, водораздел для поиска контуров мелких пузырей. 

Функционал слабой разметки расположен в модуле `src/annotation/data_annotation.py` в методе `run_annotation`. Разметка выполняется для указанной в файле конфигураций папки.

Файл конфигураций для моделей SAM и YOLOv8:
```yaml
yolo_checkpoint_path: 'models/annotation/best.pt'
sam:
  checkpoint_path: 'models/annotation/sam_vit_h_4b8939.pth'
  model_type: 'vit_h'

target_length: 1024
narrowing: 0.20
erode_iterations: 1
prompt_points: False
```
Файл конфигураций водораздела:
```yaml
single_scale_retinex:
  sigma: 80
bilateral_filtering:
  diameter: 5
  sigma_color: 75
  sigma_space: 75
morphological_transform:
  kernel_size: [3, 3]
gaussian_blur:
  kernel_size: [5, 5]
  sigma_x: 0
minmax:
  span: [0, 255]

```


Пример использования:

```bash
make run_annotation
```
Через переменную `ARGS` можно передавать и переопределять любые параметры в конфигурациях. Например, так можно передать другой путь к весам модели SAM:

```bash
make run_annotation ARGS="sam_yolo.sam.checkpoint_path={ваш путь}"
```

Результаты аннотации:
![Original - SAM - Watershed - Both](data/readme/Image.png)

Результаты аннотации на разнообразных данных. Важно отметить, что алгоритм не работает в режиме реального времени, медиа показывает визуализацию.
![SAM + Watershed performance](data/readme/images_masks_output_video_masked.gif)

## Обучение моделей 

# Сегментация

Обучение моделей запускаются следующей командой:

```bash
make run_training
```
Чтобы запустить обучение модели определенной архитектуры, убедитесь, что файл конфигурации для этой архитектуры находится в `configs/module/arch`.
Запустить обучение можно следующим образом:

```bash
make run_training ARGS="module/arch={ваша архитектура}.yaml"
```
Библиотека поддерживает логирование в ClearML. В файле конфигураций `configs/experiment/experiment.yaml` есть переменная `track_in_clearml`, с помощью которой можно включить поддержку ClearML. Прежде, чем запускать обучение, обязательно необходимо инициализировать ClearML [согласно документации](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps/).

## Модели сегментации

Для сегментации изображений однородных данных реализованы такие модели, как: SegFormer, Swin+UNETR, DeepLabv3, HRNet. Модели инициализируются из файла конфигурации. Добавьте нужную вам модель в `configs/arch` и измените файл config.yaml. Проект поддерживает timm, pytorch-segmentation-models, transformers.
Если модели не нужно реализовать отдельную логику метода `forward`, достаточно добавить файл конфигурации в таком виде:
```yaml
_target_: segmentation_models_pytorch.DeepLabV3Plus
encoder_name: resnet34
in_channels: ${module.num_channels}
classes: ${module.num_classes}
```
Если для вашей модели нужно реализовывать отдельную логику, достаточно передать путь для инициализации как в примере. Например, так выглядит файл конфигураций HRNet, в которой в качестве аргумента принимается сама модель из `timm` и дополнительные аргументы для инициализации класса-обертки:
```yaml
_target_: src.models.HRNetModel
image_size: ${data.img_size}
net:
  _target_: timm.create_model
  model_name: hrnet_w18_small_v2
  features_only: true
  pretrained: true
```


## Обучение моделей

Для того чтобы запустить обучение моделей, выполните следующую команду: 

```bash
make run_training
```

### ADELE
Библиотека поддерживает [Adaptive Early-Learning Correction for Segmentation from Noisy Annotations](https://arxiv.org/abs/2110.03740). В файле конфигураций `configs/dta/data.yaml` за это отвечает переменная `adele_correction`. 
Функционал реализован в качестве callback для тренировщика. На локальный диск в процессе обучения в папку рядом с тренировочными данными, определенную параметром `adele_dir`, будут сохраняться скорректированные маски.


## Результаты обучения

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
