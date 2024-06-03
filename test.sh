export CUDA_VISIBLE_DEVICES=2,3,4,6


bash tools/dist_test.sh configs/soft_teacher/ubsoft_teacher_grs_faster_rcnn_r50_caffe_fpn_coco_180k.py work_dirs/ubsoft_teacher_grs_faster_rcnn_r50_caffe_fpn_coco_180k/coco2/grs2/iter_180000.pth 4 --eval bbox