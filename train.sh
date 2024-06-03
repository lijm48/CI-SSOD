# export CUDA_VISIBLE_DEVICES=5,6
# sh script baseline/semi name_of_fold name_of_data gpu_number

# baseline in coco1
# sh tools/dist_train_partially.sh baseline baseline coco1 8

# grs in coco1
sh tools/dist_train_partially.sh semi grs coco1 8

# grs in LVIS
# sh tools/dist_train_partially_lvis.sh semi grs lvis 8

# grs in Object365
# sh tools/dist_train_partially_object365.sh semi grs object365 8