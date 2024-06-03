# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np
import random
from mmdet.datasets.api_wrappers import COCO
from ssod.datasets.cate import  COCO_NOVEL_CATEGORIES

CLASSES = ('Person', 'Sneakers', 'Chair', 'Other Shoes', 'Hat', 'Car', \
'Lamp', 'Glasses', 'Bottle', 'Desk', 'Cup', 'Street Lights', 'Cabinet/shelf', \
'Handbag/Satchel', 'Bracelet', 'Plate', 'Picture/Frame', 'Helmet', 'Book', 'Gloves', \
'Storage box', 'Boat', 'Leather Shoes', 'Flower', 'Bench', 'Potted Plant', 'Bowl/Basin', \
'Flag', 'Pillow', 'Boots', 'Vase', 'Microphone', 'Necklace', 'Ring', 'SUV', 'Wine Glass', \
'Belt', 'Moniter/TV', 'Backpack', 'Umbrella', 'Traffic Light', 'Speaker', 'Watch', \
'Tie', 'Trash bin Can', 'Slippers', 'Bicycle', 'Stool', 'Barrel/bucket', 'Van', 'Couch', \
'Sandals', 'Bakset', 'Drum', 'Pen/Pencil', 'Bus', 'Wild Bird', 'High Heels', 'Motorcycle', \
'Guitar', 'Carpet', 'Cell Phone', 'Bread', 'Camera', 'Canned', 'Truck', 'Traffic cone', \
'Cymbal', 'Lifesaver', 'Towel', 'Stuffed Toy', 'Candle', 'Sailboat', 'Laptop', 'Awning', \
'Bed', 'Faucet', 'Tent', 'Horse', 'Mirror', 'Power outlet', 'Sink', 'Apple', \
'Air Conditioner', 'Knife', 'Hockey Stick', 'Paddle', 'Pickup Truck', 'Fork', \
'Traffic Sign', 'Ballon', 'Tripod', 'Dog', 'Spoon', 'Clock', 'Pot', 'Cow', 'Cake', \
'Dinning Table', 'Sheep', 'Hanger', 'Blackboard/Whiteboard', 'Napkin', 'Other Fish',\
 'Orange/Tangerine', 'Toiletry', 'Keyboard', 'Tomato', 'Lantern', 'Machinery Vehicle', \
 'Fan', 'Green Vegetables', 'Banana', 'Baseball Glove', 'Airplane', 'Mouse', 'Train', \
 'Pumpkin', 'Soccer', 'Skiboard', 'Luggage', 'Nightstand', 'Tea pot', 'Telephone', \
 'Trolley', 'Head Phone', 'Sports Car', 'Stop Sign', 'Dessert', 'Scooter', 'Stroller', \
 'Crane', 'Remote', 'Refrigerator', 'Oven', 'Lemon', 'Duck', 'Baseball Bat', 'Surveillance Camera', \
 'Cat', 'Jug', 'Broccoli', 'Piano', 'Pizza', 'Elephant', 'Skateboard', 'Surfboard', \
 'Gun', 'Skating and Skiing shoes', 'Gas stove', 'Donut', 'Bow Tie', 'Carrot', 'Toilet',\
  'Kite', 'Strawberry', 'Other Balls', 'Shovel', 'Pepper', 'Computer Box', 'Toilet Paper', \
  'Cleaning Products', 'Chopsticks', 'Microwave', 'Pigeon', 'Baseball', 'Cutting/chopping Board', \
  'Coffee Table', 'Side Table', 'Scissors', 'Marker', 'Pie', 'Ladder', 'Snowboard', 'Cookies', \
  'Radiator', 'Fire Hydrant', 'Basketball', 'Zebra', 'Grape', 'Giraffe', 'Potato', 'Sausage',\
   'Tricycle', 'Violin', 'Egg', 'Fire Extinguisher', 'Candy', 'Fire Truck', 'Billards',\
    'Converter', 'Bathtub', 'Wheelchair', 'Golf Club', 'Briefcase', 'Cucumber', 'Cigar/Cigarette ', \
    'Paint Brush', 'Pear', 'Heavy Truck', 'Hamburger', 'Extractor', 'Extention Cord', 'Tong', \
    'Tennis Racket', 'Folder', 'American Football', 'earphone', 'Mask', 'Kettle', 'Tennis', \
    'Ship', 'Swing', 'Coffee Machine', 'Slide', 'Carriage', 'Onion', 'Green beans', 'Projector', \
    'Frisbee', 'Washing Machine/Drying Machine', 'Chicken', 'Printer', 'Watermelon', 'Saxophone',\
     'Tissue', 'Toothbrush', 'Ice cream', 'Hotair ballon', 'Cello', 'French Fries', 'Scale', \
     'Trophy', 'Cabbage', 'Hot dog', 'Blender', 'Peach', 'Rice', 'Wallet/Purse', 'Volleyball', \
     'Deer', 'Goose', 'Tape', 'Tablet', 'Cosmetics', 'Trumpet', 'Pineapple', 'Golf Ball', \
     'Ambulance', 'Parking meter', 'Mango', 'Key', 'Hurdle', 'Fishing Rod', 'Medal', 'Flute', \
     'Brush', 'Penguin', 'Megaphone', 'Corn', 'Lettuce', 'Garlic', 'Swan', 'Helicopter', \
     'Green Onion', 'Sandwich', 'Nuts', 'Speed Limit Sign', 'Induction Cooker', 'Broom', \
     'Trombone', 'Plum', 'Rickshaw', 'Goldfish', 'Kiwi fruit', 'Router/modem', 'Poker Card', \
     'Toaster', 'Shrimp', 'Sushi', 'Cheese', 'Notepaper', 'Cherry', 'Pliers', 'CD', 'Pasta',\
      'Hammer', 'Cue', 'Avocado', 'Hamimelon', 'Flask', 'Mushroon', 'Screwdriver', 'Soap', \
      'Recorder', 'Bear', 'Eggplant', 'Board Eraser', 'Coconut', 'Tape Measur/ Ruler', 'Pig', \
      'Showerhead', 'Globe', 'Chips', 'Steak', 'Crosswalk Sign', 'Stapler', 'Campel', \
      'Formula 1 ', 'Pomegranate', 'Dishwasher', 'Crab', 'Hoverboard', 'Meat ball', 'Rice Cooker', \
      'Tuba', 'Calculator', 'Papaya', 'Antelope', 'Parrot', 'Seal', 'Buttefly', 'Dumbbell', \
      'Donkey', 'Lion', 'Urinal', 'Dolphin', 'Electric Drill', 'Hair Dryer', 'Egg tart', \
      'Jellyfish', 'Treadmill', 'Lighter', 'Grapefruit', 'Game board', 'Mop', 'Radish', \
      'Baozi', 'Target', 'French', 'Spring Rolls', 'Monkey', 'Rabbit', 'Pencil Case', \
      'Yak', 'Red Cabbage', 'Binoculars', 'Asparagus', 'Barbell', 'Scallop', 'Noddles', \
      'Comb', 'Dumpling', 'Oyster', 'Table Teniis paddle', 'Cosmetics Brush/Eyeliner Pencil', \
      'Chainsaw', 'Eraser', 'Lobster', 'Durian', 'Okra', 'Lipstick', 'Cosmetics Mirror', \
      'Curling', 'Table Tennis')

def main():

    root_dir = '/root/paddlejob/workspace/env_run/output/lijiaming/dataset/lvis/annotations/'
    data_set = 'lvis_v1_train'
    
    json_file = root_dir + '{}.json'.format(data_set)
    coco = COCO(json_file)

    # assert sum([fully_labeled, Unsup, tagsU, tagsK, pointsU, pointsK, boxesEC, boxesU]) == 1.0


    # original statistics
    with open("../../coco/lvis_split.txt", "r") as f:  # 打开文件
        novel_list = f.readlines() 
    for i in range(len(novel_list)):
        novel_list[i] = novel_list[i].strip('\n') 
    cat_ids = coco.get_cat_ids()
    # cat_ids = coco.get_cat_ids(cat_names=CLASSES)
    num_classes = len(cat_ids)

    img_ids = coco.get_img_ids()
    num_imgs = len(img_ids)


    anns = []
    imgs_all = []
    anns_all = []
    sample_ids = []
    histogram = np.zeros((num_classes,), dtype=np.int)
    for i in img_ids:
        info = coco.load_imgs([i])[0]
        vis = np.zeros((num_classes,), dtype=np.int)
        if str(info['id']) in novel_list:    
            img_anns = coco.imgToAnns[i]
            for ann in img_anns:
                if ann['category_id'] not in cat_ids:
                    continue
                index = cat_ids.index(int(ann['category_id']))
                if vis[index] == 0 :
                    histogram[index] += 1
                    vis[index] += 1
                anns.append(ann)  
            sample_ids.append(i)         
        else:
            continue
    # import pdb;pdb.set_trace()
    sample_ids = sorted(sample_ids)
    imgs = coco.loadImgs(sample_ids)
    




    # sampled statistics
    classes = [x["category_id"] for x in anns]
    # histogram = np.zeros((num_classes,), dtype=np.int)
    # for i in classes:
    #     index = cat_ids.index(int(i))
    #     histogram[index] = histogram[index] + 1
    class_ratios = histogram / np.sum(histogram)
    tmp2 = np.load("../../SoftTeacher-main/lvis_all.npy")
    import pdb;pdb.set_trace()
    print("sampled class ratios: {}".format(class_ratios))
    print("each class has at least one example ", np.min(histogram) > 0)

    sample_data = {}
    sample_data['images'] = imgs
    sample_data['annotations'] = anns
    sample_data['categories'] = list(coco.cats.values())
    sample_data['info'] = coco.dataset['info']
    sample_data['licenses'] = coco.dataset['licenses']

    output_file_label = '{}{}_lvis_label.json'.format(root_dir, data_set)
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

    output_file_unlabel = '{}{}_lvis_unlabel.json'.format(root_dir, data_set)
    import pdb;pdb.set_trace()
    ## save to json
    with open(output_file_unlabel, 'w') as f:
        print('writing to json output:', output_file_unlabel)
        json.dump(unsample_data, f, sort_keys=True)


if __name__ == '__main__':
    main()
    print("finished!")
