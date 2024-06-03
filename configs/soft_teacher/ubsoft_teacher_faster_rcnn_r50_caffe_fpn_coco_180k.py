from configs.soft_teacher.base import  train_pipeline, test_pipeline, strong_pipeline, weak_pipeline, unsup_pipeline
_base_ = "base.py"
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
_dataset_dir = "../dataset/"
#"/root/paddlejob/workspace/env_run/output/lijiaming/dataset/"

SPLIT = 1

data = dict(
    samples_per_gpu=5,
    workers_per_gpu=5,
    train=dict(
        sup=dict(
            type="SemiCocoDataset",
            ann_file="${_dataset_dir}/coco/annotations/instances_train2017_coco_split${SPLIT}_label.json",
            img_prefix="${_dataset_dir}/coco/train2017/",
            data_split = SPLIT,
            pipeline = train_pipeline,
        ),
        unsup=dict(
            type="SemiCocoDataset",
            ann_file="${_dataset_dir}/coco/annotations/instances_train2017_coco_split${SPLIT}_unlabel.json",
            img_prefix="${_dataset_dir}/coco/train2017/",
            data_split = SPLIT,
            is_labeled=False
        ),
    ),
    val=dict(
        type='SemiCocoDataset',
        ann_file="${_dataset_dir}/coco/annotations/instances_val2017.json",
        img_prefix="${_dataset_dir}/coco/val2017/",
        data_split = SPLIT,
        pipeline=test_pipeline),
    test=dict(
        type='SemiCocoDataset',
        ann_file="${_dataset_dir}/coco/annotations/instances_val2017.json",
        img_prefix="${_dataset_dir}/coco/val2017/",
        data_split = SPLIT,
        pipeline=test_pipeline),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 4],
        )
    ),
)


fold = 1
percent = 1
evaluation = dict(type="SubModulesDistEvalHook", interval=20000)
work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)
