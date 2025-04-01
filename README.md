# CI-SSOD

The official code for our ICCV2023 paper ["Gradient-based Sampling for Class Imbalanced Semi-supervised Object Detection"](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Gradient-based_Sampling_for_Class_Imbalanced_Semi-supervised_Object_Detection_ICCV_2023_paper.html).

<!-- ## Process:
Reproducing the results. -->
<!-- 1. [✓] Submit the initial code.
2. [✓] Submit the datasets and the instructions of data preparation().
3. [✓] Submit the instructions for environment installation.
4. [] Submit the instructions for training and testing.
5. [] Reproduce the results with the current code and submit the checkpoints.
6. [] Modify the initial code for robustness. -->

## Usage:
### Data preparation

#### MS-COCO sub-task
- Download the COCO2017 dataset(including training and val images) from [this link](https://cocodataset.org/#download).
- Download the annotations for split1 and split2 for  from [Google Drive](https://drive.google.com/drive/folders/11ggu8fnimMDS8w2dcqUTLs2xtyNceuCO).
- Organize the data as the following structure(or rewrite the path in configs as you need):
```
CISSOD/
    dataset/
        coco/
            train2017/
            val2017/
            annotations/
                instances_train2017_coco_split1_label.json
                instances_train2017_coco_split1_unlabel.json
                instances_train2017_coco_split2_label.json
                instances_train2017_coco_split2_unlabel.json

```

#### MS-COCO → Object365
- Download the COCO2017 dataset(including training and instances_train2017.json) from [this link](https://cocodataset.org/#download).
- Download the Object365dataset(including training and val images) from .[this link](https://www.objects365.org/download.html)
- Download the annotations from [Google Drive](https://drive.google.com/drive/folders/11ggu8fnimMDS8w2dcqUTLs2xtyNceuCO).
- Organize the data as the following structure(or rewrite the path in configs as you need):
```
CISSOD/
    dataset/
        coco/
            train2017/
            val2017/
            annotations/
                instances_train2017.json

        Object365/
            images/
            annotations/
                zhiyuan_objv2_train_object365_label.json
                zhiyuan_objv2_train_object365_unlabel.json
                zhiyuan_objv2_val.json
           

```

#### LVIS
- Download the COCO2017 dataset(including training and val images) from [this link](https://cocodataset.org/#download).
- Download the LVIS annotations from .
- Download the annotations from [Google Drive](https://drive.google.com/drive/folders/11ggu8fnimMDS8w2dcqUTLs2xtyNceuCO).
- Organize the data as the following structure(or rewrite the path in configs as you need):
```
CISSOD/
    dataset/
        coco/
            train2017/
            val2017/
        lvis/
            annotations/
                lvis_v1_train_lvis_label.json
                lvis_v1_train_lvis_unlabel.json
                lvis_v1_val.json
```


## Installation:
You can follow the [Soft teacher](https://github.com/microsoft/SoftTeacher/) to finish the installation. Note that we do not use the wandb.
Or use the following command:
```
make install
```
Note that you may need to modify the version of torch, torchvision and mmcv-full based on your cuda version.
mmcv-full can be installed by:
```
pip install mmcv-full==1.3.12 -f https://download.openmmlab.com/mmcv/dist/YOUR_CUDA_VERSION/YOUR_TORCH_VERSION/index.html
# mmcv > 1.3.8 and <1.4.0
# pip install mmcv-full==1.3.12  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.9.0/index.html
```

## Training:
```
sh ${script} baseline/semi ${name_of_fold} ${name_of_data} ${gpu_number}
```
For example, for the split 1 of COCO on 8 GPUs on semi-supervised setting, 
```
sh tools/dist_train_partially.sh semi grs coco1 8
```
To modify the split of COCO, change the SPLIT in line 7 of ./configs/soft_teacher/ubsoft_teacher_grs_faster_rcnn_r50_caffe_fpn_coco_180k.py.
For concrete instructions of training on each setting, please refer to train.sh.

## Evaluation:
```
sh tools/dist_test.sh ${name_of_config} ${location_of_checkpoint} ${gpu_number} --eval bbox
```
For example,
```
sh tools/dist_test.sh configs/soft_teacher/ubsoft_teacher_grs_faster_rcnn_r50_caffe_fpn_coco_180k.py work_dirs/ubsoft_teacher_grs_faster_rcnn_r50_caffe_fpn_coco_180k/coco2/grs2/iter_180000.pth 4 --eval bbox
```


## Acknowledgment:
The code is heavily borrowed from [Soft teacher](https://github.com/microsoft/SoftTeacher/) and thanks for their contribution.
