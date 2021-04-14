# Ninpy #

My collection of reuse-able Python scripts and others.

## Install ##

```bash
python setup.py develop
```

## Directory ##

```bash
ninpy
├── common.py
├── crypto.py
├── data.py
├── dataset.py
├── datasets
│   ├── cinic10.py
│   ├── imagenet.py
│   ├── __init__.py
│   ├── kitti_road.py
│   └── voc2012.py
├── debug.py
├── experiment.py
├── hw.py
├── hyper.py
├── __init__.py
├── int8.py
├── job.py
├── layer_converter.py
├── log.py
├── losses.py
├── models
│   ├── cifar_resnet.py
│   ├── cifar_vgg.py
│   ├── __init__.py
│   └── small_models.py
├── notify.py
├── quant.py
├── README.md
├── requirement.txt
├── resize.py
├── torch_utils.py
└── yaml2.py
```

## TODO ##

[ ] Reorganize torch2.
    [ ] Adding base module with tensorboard tracking.
    [ ] Dataset with tensorboard add_images.
    [ ] Tensorboard with add_scalars with a dict input.
    [ ] Base dataset with mode "RGB" and "BGR".
    [ ] Base dataset with from_txt and folder.
    [ ] Burst read for all base dataset.
