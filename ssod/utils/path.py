import os.path as osp
import os

def check_and_create_path(file_path):
    directory = osp.dirname(file_path)
    if not osp.exists(directory):
        try:
            os.makedirs(directory) 
            print(f"directory {directory} created!")
        except Exception as e:
            print(f"directory creation failed:{e}")
    # if not osp.exists(file_path):
    #     try:
    #         with open(file_path, 'w') as f:
    #             json.dump({}, f)  
    #         print(f"file {file_path} created!")
    #     except Exception as e:
    #         print(f"file creation failed:{e}")
