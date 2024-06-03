# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np
import random
from mmdet.datasets.api_wrappers import COCO
from ssod.datasets.cate import  COCO_NOVEL_CATEGORIES

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

def main():

    root_dir = '/root/paddlejob/workspace/env_run/output/lijiaming/dataset/coco/annotations/'
    data_set = 'instances_train2017'
    
    json_file = root_dir + '{}.json'.format(data_set)
    coco = COCO(json_file)

    # assert sum([fully_labeled, Unsup, tagsU, tagsK, pointsU, pointsK, boxesEC, boxesU]) == 1.0

    # we first sample the fully label data
    novel_ids = [k["id"] for k in COCO_NOVEL_CATEGORIES if k["isthing"] == 1]
    # original statistics
    with open("../../coco/base_10percent.txt", "r") as f:  # 打开文件
        labeled_list = f.readlines() 
    with open("../../coco/novel_few.txt", "r") as f:  # 打开文件
        novel_list = f.readlines() 
    for i in range(len(labeled_list)):
        labeled_list[i] = labeled_list[i].strip('\n')
    for i in range(len(novel_list)):
        novel_list[i] = novel_list[i].strip('\n') 
    cat_ids = coco.get_cat_ids(cat_names=CLASSES)
    num_classes = len(cat_ids)

    img_ids = coco.get_img_ids()
    num_imgs = len(img_ids)


    anns = []
    imgs_all = []
    anns_all = []
    sample_ids = []
    for i in img_ids:
        info = coco.load_imgs([i])[0]
        if info['file_name'] in labeled_list:
            img_anns = coco.imgToAnns[i]
            for ann in img_anns:
                if ann['category_id'] not in cat_ids:
                    continue
                if ann['category_id'] in novel_ids:
                    continue 
                anns.append(ann) 
            sample_ids.append(i) 
        elif info['file_name'] in novel_list:    
            img_anns = coco.imgToAnns[i]
            for ann in img_anns:
                if ann['category_id'] not in cat_ids:
                    continue
 
                anns.append(ann)  
            sample_ids.append(i)         
        else:
            continue
    # import pdb;pdb.set_trace()
    # sample_ids = random.sample(img_ids, num_samples)
    sample_ids = sorted(sample_ids)
    imgs = coco.loadImgs(sample_ids)
    




    # sampled statistics
    classes = [x["category_id"] for x in anns]
    histogram = np.zeros((num_classes,), dtype=np.int)
    for i in classes:
        index = cat_ids.index(int(i))
        histogram[index] = histogram[index] + 1
    # import pdb;pdb.set_trace()
    class_ratios = histogram / np.sum(histogram)
    print("sampled class ratios: {}".format(class_ratios))
    print("each class has at least one example ", np.min(histogram) > 0)

    sample_data = {}
    sample_data['images'] = imgs
    sample_data['annotations'] = anns
    sample_data['categories'] = list(coco.cats.values())
    sample_data['info'] = coco.dataset['info']
    sample_data['licenses'] = coco.dataset['licenses']

    output_file_label = '{}{}_coco_split1_label.json'.format(root_dir, data_set)
    ## save to json
    with open(output_file_label, 'w') as f:
        print('writing to json output:', output_file_label)
        json.dump(sample_data, f, sort_keys=True)

    # next deal with unlabel (weakly label) data
    unsampled_ids = list(set(img_ids) - set(sample_ids))
    unsampled_ids = sorted(unsampled_ids)
    imgs = coco.loadImgs(unsampled_ids)



    dataset_anns_u = [coco.imgToAnns[img_id] for img_id in unsampled_ids]
    anns = [ann for img_anns in dataset_anns_u for ann in img_anns]




    imgs_all.extend(imgs)
    anns_all.extend(anns)

    classes = [x["category_id"] for x in anns]
    histogram = np.zeros((num_classes,), dtype=np.int)
    for i in classes:
        index = cat_ids.index(int(i))
        histogram[index] = histogram[index] + 1
        
    unsample_data = {}
    unsample_data['images'] = imgs_all
    unsample_data['annotations'] = anns_all
    unsample_data['categories'] = list(coco.cats.values())
    unsample_data['info'] = coco.dataset['info']
    unsample_data['licenses'] = coco.dataset['licenses']

    output_file_unlabel = '{}{}_coco_split1_unlabel.json'.format(root_dir, data_set)
    import pdb;pdb.set_trace()
    ## save to json
    with open(output_file_unlabel, 'w') as f:
        print('writing to json output:', output_file_unlabel)
        json.dump(unsample_data, f, sort_keys=True)


if __name__ == '__main__':
    main()
    print("finished!")
