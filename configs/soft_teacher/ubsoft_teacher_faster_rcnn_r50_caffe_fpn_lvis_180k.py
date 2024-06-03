from configs.soft_teacher.base import  test_pipeline, strong_pipeline, weak_pipeline, unsup_pipeline
_base_ = "base.py"
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
_dataset_dir = "./dataset/"

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1203,
            )),
    test_cfg=dict(
        rcnn=dict(score_thr=0.05, max_per_img=100)),
)

semi_wrapper = dict(
    model="${model}",
    train_cfg=dict(
        pseudo_label_initial_score_thr=0.5,
    ),
    test_cfg=dict(inference_on="teacher"),
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Sequential",
        transforms=[
            dict(
                type="RandResize",
                img_scale=[(1333, 400), (1333, 1200)],
                multiscale_mode="range",
                keep_ratio=True,
            ),
            dict(type="RandFlip", flip_ratio=0.5),
            dict(
                type="OneOf",
                transforms=[
                    dict(type=k)
                    for k in [
                        "Identity",
                        "AutoContrast",
                        "RandEqualize",
                        "RandSolarize",
                        "RandColor",
                        "RandContrast",
                        "RandBrightness",
                        "RandSharpness",
                        "RandPosterize",
                    ]
                ],
            ),
        ],
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="sup"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
        ),
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]



data = dict(
    samples_per_gpu=5,
    workers_per_gpu=5,
    train=dict(
        sup=dict(
            type="ClassBalancedDataset",
            oversample_thr = 0.01,
            dataset = dict(
                type='Semi_LVISV1Dataset',
                ann_file='${_dataset_dir}/lvis/annotations/lvis_v1_train_lvis_label.json',
                img_prefix='${_dataset_dir}/coco',
                pipeline = train_pipeline,
            ),
        ),
        unsup=dict(
            type='Semi_LVISV1Dataset',
            ann_file='${_dataset_dir}/lvis/annotations/lvis_v1_train_lvis_unlabel.json',
            img_prefix='${_dataset_dir}/coco',
            is_labeled=False,
        ),
    ),
    val=dict(
        type='LVISV1Dataset',
        ann_file='${_dataset_dir}/lvis/annotations/lvis_v1_val.json',
        img_prefix='${_dataset_dir}/coco',
        pipeline=test_pipeline),
    test=dict(
        type='LVISV1Dataset',
        ann_file='${_dataset_dir}/lvis/annotations/lvis_v1_val.json',
        img_prefix='${_dataset_dir}/coco',
        pipeline=test_pipeline),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 4],
        )
    ),
)
find_unused_parameters=True
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
