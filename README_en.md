# CTCI - Clumped Texture Composite Images Projects 

**Clumped Texture Composite Images** - images with clumped complex data forming texture. Such images (or videos) imply that multiple objects of the same class will be present on one image, clustered together, i.e., overlapping each other, in a random order. Examples of such data could be bubbles, rocks, any identical items on a conveyor belt.

![](data/clumped_data.png)

Working with such data poses difficulties due to the lack of openly available annotations, the labor-intensive nature of annotation, and the requirement for high-quality results.

The library contains methods to automate the label creation process using weak annotation methods, self-learning methods, and domain transfer methods. It also presents the results of training several types of models on homogeneous data.

# Installation
1. Download the repository using the following link:
   
   ```git clone https://github.com/CTLab-ITMO/CTCI.git```
   
2. Navigate to the project directory
   
   ```cd CTCI```
   
3. Install dependencies
   
   ```pip -r install requirements.txt```


# Weak Annotation

Weak annotation of homogeneous data is implemented using YOLOv8 and Segment Anything models, as well as using a watershed segmentation algorithm. YOLOv8 is used to search for markers corresponding to large bubbles, SAM for contouring large bubbles, watershed for contouring small bubbles.

The functionality of weak annotation is located in the `src/data/weakly_segmentation/annotation.py` module in the `annotation` method. Annotation is performed for the specified data folder.

```python
import sys

from src.data.weakly_segmentation.annotation import annotation

data_dir = sys.argv[1]  # Directory with unannotated data
folder = sys.argv[2]    # Folder with unannotated data

custom_yolo_checkpoint_path = sys.argv[3]  # Path to YOLOv8 model weights
sam_checkpoint = sys.argv[4]  # Path to SAM model weights
sam_model_type = sys.argv[5]  # SAM model type

narrowing = 0.20  # Segmentation mask narrowing value. Prevents merging of masks of homogeneous objects
erode_iterations = 1  # Number of erosion iterations 
processes_num = 3  # Number of parallel image segmentation processes
prompt_points = False  # Use SAM prompt points

device = "cpu"  # Device on which annotation is performed

annotation(
    data_dir, folder,
    custom_yolo_checkpoint_path, sam_checkpoint, sam_model_type, narrowing=narrowing,
    erode_iterations=erode_iterations, processes_num=processes_num, prompt_points=prompt_points,
    device=device
)
```

Annotation results:
![Original - SAM - Watershed - Both](data/readme/Image.png)

Annotation results on different data. It is important to note that the algorithm does not work in real-time mode; the media shows only visualization.
![SAM + Watershed performance](data/readme/images_masks_output_video_masked.gif)

# Self-supervised learning

## Self-supervised learning Algorithms

For implementing self-supervised learning, the Barlow Twins and MoCo algorithms were chosen.

Barlow Twins can possibly be implemented using the corresponding class:

```python
from barlow_twins import BarlowTwins
args = { 
    "batch_size": 50,  # Batch size during training
    "MLP": "8192-8192-8912"  # Fully connected network structure 
}

model = BarlowTwins(args)

```

Encoders can be trained in the MoCo format using the respective functions:

```python
from train_moco import train

device = "cpu"

model_q = Net().to(device)  # Two encoders trained together
model_k = copy.deepcopy(model_q) 

optimizer = optim.SGD(model_q.parameters(), lr=0.0001, weight_decay=0.0001)  # Optimization algorithm
queue = initialize_queue(model_k, device, train_loader)  # Queue of objects on which the network is trained 
epoch = 50  # Number of epochs 

train(model_q, model_k, device, train_loader, queue, optimizer, epoch)
```

## Model Training
### Script for training the model based on Barlow Twins

```bash
python src/models/'barlow twins'/barlow_twins.py images_path masks_path target_height target_width batch_size epochs
```

```images_path``` - path to the images 

```masks_path``` - path to the masks

```target_height``` - target image height

```target_width``` - target image width 

```batch_size```  - batch size during training 

```epochs``` - number of training epochs 

### Script for training models based on MoCo (Momentum Contrast) 

```bash
python src/models/moco/train_moco.py images_path masks_path out_dir batch_size epochs
```

```images_path``` - path to the images

```masks_path``` - path to the masks

```out_dir``` - path to save the result

```batch_size``` - batch size during training

```epochs``` - number of training epochs

Results of Self-supervised learning Algorithms

| Source  | Barlow twins | MoCo | 
| ------------- | ------------- | -------------| 
| ![](data/orig/0.png)  | ![](data/bt/1.png)  | ![](data/moco/0.png) |
| ![](data/orig/15.png)  | ![](data/bt/15.png)  | ![](data/moco/15.png) |

# Segmentation

## Configuration Files

For the convenience of training and using models, we use configuration files, examples of which can be found in the directory `src/infrastructure/configs`. We recommend adhering to the structure specified in them.

## Segmentation Models

For segmenting homogeneous data images, models such as Yolov8, SegFormer, Swin+UNETR, DeepLabv3, HRNet are implemented. These models are located in the directories `src/models/<model_name>`.

Models can be initialized using the corresponding class, for example:

```python
net = transformers.SegformerForSemanticSegmentation.from_pretrained(
    f"nvidia/{model_name}-{model_type}-finetuned-ade-512-512",
    num_labels=1,
    image_size=image_size_height,
    ignore_mismatched_sizes=True
)

segformer = SegFormer(
    net=net, mask_head=final_layer, loss_fn=loss_fn,
    image_size=image_size, device=device
)

```
All segmentation models inherit from the `BaseModel` class.

Alternatively, using the `build_<model_name>` method, for example:

```python
config_handler = read_yaml_config(config_path) # Configuration file handler
model = build_segformer(config_handler)
```


<details>
    <summary> Segformer </summary>

Initialization of Segformer from a configuration file.

```python
from src.models.segformer.model import build_segformer
config_handler = read_yaml_config(config_path) # обработчик конфигурационных файлов
model = build_segformer(config_handler)
```

Training results of the model:
![Segformer performance](data/readme/segformer_output_video_masked.gif)

</details>


<details>
    <summary> Swin-UNETR  </summary>
  
Initialization of Swin-UNETR from a configuration file.

```python
from src.models.swin.model import build_swin
config_handler = read_yaml_config(config_path) # Configuration file handler
model = build_swin(config_handler)
```

Training results of the model:

![Swin performance](data/readme/swinv2_output_video_masked.gif)

</details>


<details>
    <summary>  HRNet  </summary>

Initialization of HRNet from a configuration file.
    
```python
from src.models.hrnet.model import build_hrnet
config_handler = read_yaml_config(config_path) # Configuration file handler
model = build_hrnet(config_handler)
```

Training results of the model:
![HRNet performance](data/readme/hrnet_w18_small_v2_output_video_masked.gif)

</details>

<details>
    <summary>  DeepLabV3  </summary>

Initialization of DeepLabV3 from a configuration file.

```python
from src.models.deeplab.model import build_deeplab
config_handler = read_yaml_config(config_path) # обработчик конфигурационных файлов
model = build_deeplab(config_handler)
```

Training results of the model:
![DeepLab performance](data/readme/resnet34-run2_output_video_masked.gif)

</details>

<details>
    <summary> YOLOv8 </summary>
For YOLOv8 inference, navigate to the directory CTCI/src/models/yolov8 and run the following command in the command line:

```bash
python3 CTCI/src/models/yolov8/<task_script.py> <path to input image> <path to output image> <path to model weights>
```
</details>

## Model Training

To train or fine-tune models, a trainer class `Trainer` has been added, located in the module `src/models/train.py`:

```python
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    metrics=metrics,
    main_metric_name=main_metric_name,
    save_dir=model_save_dir,
    device=device
)
```

In the directory ```src/infrastructure/models_tracking```, there are scripts allowing you to train or fine-tune models "out of the box" using a configuration file. Usage example:

```bash
python src/infrastructure/models_tracking/segformer_tracking.py <config_path>
```

### ADELE

The library supports [Adaptive Early-Learning Correction for Segmentation from Noisy Annotations](https://arxiv.org/abs/2110.03740). To use it, simply create a dataset **without augmentations**, i.e., without affine transformations, but in the same form that the model will receive images during inference. In the trainer class, you will need to pass a separately initialized dataset from the training files. In the configuration file, you need to add a step at which the method will be applied.

```python
# Add the following lines to the tracking file

from src.models.utils.config import read_yaml_config
from src.features.segmentation.dataset import get_train_dataset_by_config

config_handler = read_yaml_config(config_path)
adele_dataset = get_train_dataset_by_config(
        config_handler,
        transform=tr, # standard transformations
        augmentation_transform=None
    )
adele_dataset.return_names = True
```

# Export to ONNX

The models described above can be exported to the ONNX format for further execution in any environment supporting ONNX Runtime.

The export of each model can be done using predefined scripts located in the directory `src/infrastructure/models_inference`. For example:

```bash
python src/infrastructure/models_inference/segformer_export.py <config_path>
```

The source code for conversion and quantization is located in the module src/models/inference.py.

#Project Organization
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

