# Ninpy #

Ninnart's collection of reuse-able Python functions, classes and scripts. <br>
**Note that this repository is very very unstable. The update that may break this module occurs oftenly.**

## Install ##

### Recommend!: Local install ###

```bash
git clone https://github.com/ninfueng/ninpy
cd ninpy/
python setup.py develop
```

### Using pip install ###

```bash
pip install ninpy
```

## Unittest ##

```bash
pytest
```

## Directory ##

```bash
ninpy
├── Makefile
├── ninpy
│   ├── common.py
│   ├── crypto.py
│   ├── data.py
│   ├── datasets
│   │   ├── camvid.py
│   │   ├── cinic10.py
│   │   ├── datasets.py
│   │   ├── imagenet.py
│   │   ├── __init__.py
│   │   ├── kitti_road.py
│   │   ├── toy_datasets.py
│   │   └── voc2012.py
│   ├── debug.py
│   ├── experiment.py
│   ├── hw.py
│   ├── hyper.py
│   ├── __init__.py
│   ├── int8.py
│   ├── job.py
│   ├── layer_converter.py
│   ├── log.py
│   ├── losses.py
│   ├── metrics.py
│   ├── models
│   │   ├── cifar_resnet.py
│   │   ├── cifar_vgg.py
│   │   ├── __init__.py
│   │   └── small_models.py
│   ├── notify.py
│   ├── quant.py
│   ├── README.md
│   ├── resize.py
│   ├── torch2
│   │   ├── __init__.py
│   │   ├── module.py
│   │   └── torch2.py
│   ├── torch_utils.py
│   └── yaml2.py
├── README.md
├── requirements.txt
├── scripts
│   ├── files_to_npy.py
│   ├── prepare_camvid.sh
│   ├── prepare_cinic10.sh
│   ├── prepare_imagenet.sh
│   ├── prepare_imagenet_small.sh
│   ├── prepare_stl10.py
│   ├── prepare_voc2012.py
│   └── resize_imagenet.py
├── setup.py
├── templates
│   ├── camvid
│   │   ├── hyper.yaml
│   │   ├── main.py
│   │   ├── segnet.py
│   │   └── utils.py
│   └── imagenet
│       ├── hyper.yaml
│       ├── main_gpus_basic.py
│       ├── main_gpus.py
│       ├── main.py
│       └── utils.py
└── tests
    ├── check_camvid.ipynb
    ├── config.yaml
    ├── dog.jpg
    ├── imagenet
    │   ├── train
    │   │   └── n04552348
    │   │       └── n04552348_9.JPEG
    │   └── val
    │       └── n04552348
    │           └── n04552348_9.JPEG
    ├── imgs
    │   ├── airplane.png
    │   ├── arctichare.png
    │   └── baboon.png
    ├── test_crypto.py
    ├── test_data.py
    ├── test_datasets.py
    ├── test_kitti.ipynb
    ├── test_metrics.py
    ├── test_net.yaml
    ├── test_resize_imagenet.py
    └── test_utils.py
```

## TODO ##

* [ ] Adding base module with tensorboard tracking.
* [ ] Dataset with tensorboard add_images.
* [ ] Tensorboard with add_scalars with a dict input.
* [ ] Base dataset with mode "RGB" and "BGR".
* [ ] Base dataset with from_txt and folder.
* [ ] A script for adding license header to package using Makefile.
* [ ] Make this tensorboard covers all statistic in each layer of NN.
* [ ] Cover dry_run and eval_every_batch.
* [ ] Updating hyperparameter tunning visualization.

