from configs.soft_teacher.base import  test_pipeline, strong_pipeline, weak_pipeline, unsup_pipeline
_base_ = "base.py"
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
_dataset_dir = "./dataset/"


model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=365,),)
    )



data = dict(
    samples_per_gpu=5,
    workers_per_gpu=1,
    train=dict(
        sup=dict(
            type="ConcatDataset",
            datasets = [dict(
                type = "Semi_COCO2Object365_Dataset",
                ann_file="${_dataset_dir}/coco/annotations/instances_train2017.json",
                img_prefix="${_dataset_dir}/coco/train2017/",
                is_test = False,
                pipeline = train_pipeline,
            ),
            dict(
                type = "Semi_Object365_Dataset",
                ann_file="${_dataset_dir}/Object365/zhiyuan_objv2_train_object365_label.json",
                img_prefix="${_dataset_dir}/Object365/images/train/",
                pipeline = train_pipeline,
            )]

        ),
        unsup=dict(
            type="Semi_Object365_Dataset",
            ann_file="${_dataset_dir}/Object365/zhiyuan_objv2_train_object365_unlabel.json",
            img_prefix="${_dataset_dir}/Object365/images/train/",
            is_labeled=False
        ),
    ),
    val=dict(
        type='Semi_Object365_Dataset',
        ann_file="${_dataset_dir}/Object365/zhiyuan_objv2_val.json",
        img_prefix="${_dataset_dir}/Object365/images/val/",
        pipeline=test_pipeline),
    test=dict(
        type='Semi_Object365_Dataset',
        ann_file="${_dataset_dir}/Object365/zhiyuan_objv2_val.json",
        img_prefix="${_dataset_dir}/Object365/images/val/",
        pipeline=test_pipeline),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 4],
            epoch_length=60000,
        )
    ),
)




fold = 1
percent = 1
evaluation = dict(type="SubModulesDistEvalHook", interval=40000)
lr_config = dict(step=[280000, 320000])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=360000)
work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)
