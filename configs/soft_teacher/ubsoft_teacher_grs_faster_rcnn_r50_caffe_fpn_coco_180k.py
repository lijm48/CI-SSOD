from configs.soft_teacher.base import  test_pipeline, strong_pipeline, weak_pipeline, unsup_pipeline
_base_ = "base.py"
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
_dataset_dir = "./dataset/"
# _dataset_dir = "/root/paddlejob/workspace/env_run/output/lijiaming/dataset/"

SPLIT = 1
_pseudo_dir = "./pseudo/coco"+str(SPLIT)+"/"


model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='MyShared2FCBBoxHead',
            num_classes=80,
            loss_cls=dict(
                type='GbROptim', loss_weight=1.0, num_classes=80, focal=True, ),)),
    test_cfg=dict(
        rcnn=dict(max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    )                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
)

train_pipeline = [
    dict(type="MyLoadImageFromFile"),
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
            "type",
        ),
    ),
]



# set workers_per_gpu to 1 to reduce the memory cost
data = dict(
    samples_per_gpu=5,
    workers_per_gpu=1,
    train=dict(
        sup=dict(
            type="ClassBalancedDataset",
            oversample_thr = 0.1,
            dataset = dict(
                type = "SemiCocoDataset",
                ann_file = "${_dataset_dir}/coco/annotations/instances_train2017_coco_split${SPLIT}_label.json",
                img_prefix="${_dataset_dir}/coco/train2017/",
                data_split = SPLIT,
                pipeline = train_pipeline,

        )),
       unsup=dict(
            type = 'ClassBalancedDatasetCrS',
            split="coco${SPLIT}",
            dataset = dict(
                type="SemiCocoDataset",
                ann_file="${_dataset_dir}/coco/annotations/instances_train2017_coco_split${SPLIT}_unlabel.json",
                img_prefix="${_dataset_dir}/coco/train2017/",
                is_labeled=False,
                get_pseudo_labeled=True,
                pseudo_ann_file="${_pseudo_dir}/pseudo.json",
                pipeline=unsup_pipeline,
            ),
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
# lr_config = dict(step=[140000, 160000])
lr_config = dict(step=[180000])
runner = dict(_delete_=True, type="MyIterBasedRunner", max_iters=180000, update_iter = 16000)
work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)

semi_wrapper = dict(
    type="SoftTeacherGrS",
    model="${model}",
    train_cfg=dict(
        use_teacher_proposal=False,
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_threshold=0.9,
        cls_pseudo_threshold=0.9,
        reg_pseudo_threshold=0.02,
        jitter_times=10,
        jitter_scale=0.06,
        min_pseduo_box_size=0,
        unsup_weight=4.0,
        max_iter= runner['max_iters'],
        alpha = 0.5,
        warm_up_iter = 12000,
        GbT = True,
        pseudo_dir = _pseudo_dir,
    ),
    test_cfg=dict(inference_on="teacher"),
)

find_unused_parameters=True
