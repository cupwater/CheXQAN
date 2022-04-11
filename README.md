<!--
 * @Author: Baoyun Peng
 * @Date: 2021-10-11 14:08:56
 * @LastEditTime: 2022-04-10 21:51:17
 * @Description: 
 * 
-->
# Chest X-ray image quality assessment network (CheXQAN) in PyTorch


## TODO
2022-04-09
- [x] add segmentation mask to for training;
- [x] add focal loss for multi-task training, class-wise;
- [ ] add task-mask for multitask training;
- [x] add config for multitask training;

## What？
- Provide training framework for medical related AI tasks, e.g. AI quality control, segmentation, classification;
- Provide inference tool, and CRUD operations to mysql database, e.g. insert the inference results into mysql table;
- Provide deploy framework.


## structure of this repo
```
├── augmentation
│   ├── __init__.py
│   └── medical_augment.py
├── checkpoints
│   ├── pretrained_densenet121.pth.tar
│   └── segment-unet-6v.pt
├── data
├── dataset
├── db
│   ├── __init__.py
│   ├── crud_mysql.py
│   ├── getTableInfo.py
│   └── table_schema.py
├── deploy
├── docs
├── experiments
│   └── template
│       └── config.yaml
├── inference2db.py
├── logs
├── losses
│   ├── __init__.py
│   └── loss.py
├── main.py
├── models
├── multitask_main.py
├── options.py
├── segmentations
│   ├── lung_segmentation.py
│   └── src
│       ├── data.py
│       ├── metrics.py
│       └── models.py
├── tools
└── utils
```

## How to run this repo
### training
python multitask_main.py --config experiments/template/config.yaml

### inference
python 