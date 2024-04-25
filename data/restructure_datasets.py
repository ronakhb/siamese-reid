import shutil
from shutil import copyfile
import os
import random
from math import ceil
import argparse

def move_to_gal(base_path):
    #gallery
    gallery_path = os.path.join(base_path,'bounding_box_test')
    gallery_save_path_parent = os.path.join(base_path,'market1501')
    gallery_save_path = os.path.join(base_path,'market1501/gallery')
    if not os.path.isdir(gallery_save_path):
        os.makedirs(gallery_save_path_parent,exist_ok=True)
        os.makedirs(gallery_save_path,exist_ok=True)

    for root, dirs, files in os.walk(gallery_path, topdown=True):
        for name in files:
            if not name[-3:]=='jpg':
                continue
            ID  = name.split('_')
            src_path = gallery_path + '/' + name
            dst_path = gallery_save_path + '/' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            shutil.copy(src_path, dst_path + '/' + name)

def move_outside_folder(path,mode):
    query_path = path + f'/{mode}'

    for folder in os.listdir(query_path):
        for image in os.listdir(f'{query_path}/{folder}'):
            shutil.move(f'{query_path}/{folder}/{image}', query_path)
        os.rmdir(f'{query_path}/{folder}')

def move_inside_folder(path):
    for filename in os.listdir(path):
        source_path = os.path.join(path, filename)

        # Extract the ID from the filename (assuming the ID is before the first '_')
        try:
            photo_id = int(filename.split('_')[0])
        except ValueError:
            print(f"Skipping file {filename} as it doesn't follow the expected naming convention.")
            continue

        # Create a folder based on the photo ID in the destination folder
        id_folder = os.path.join(path, str(photo_id))
        os.makedirs(id_folder, exist_ok=True)

        # Move the photo to the corresponding folder
        destination_path = os.path.join(id_folder, filename)
        shutil.move(source_path, destination_path)

def split_dataset(dataset_path):
    gallery_path = os.path.join(dataset_path,"gallery")
    query_path = os.path.join(dataset_path,"query")
    os.makedirs(query_path,exist_ok=True)
    for id in os.listdir(gallery_path):
        id_path = os.path.join(gallery_path,id)
        images = os.listdir(id_path)
        num_images = len(images)
        query_images = random.sample(images, int(2*num_images/9))
        for image in query_images:
            image_path = os.path.join(id_path,image)
            shutil.move(image_path,query_path)

def rename_dataset(dataset_path):
    gallery_path = os.path.join(dataset_path,"gallery")
    for id in os.listdir(gallery_path):
        camid = 1
        scene = 1
        id_path = os.path.join(gallery_path,id)
        images = os.listdir(id_path)
        num_images = len(images)
        switch_interval = ceil(num_images/6)
        scene_switch_interval = ceil(switch_interval/6)
        for idx,image in enumerate(os.listdir(id_path)):
            if (idx+1)%switch_interval == 0:
                camid+=1
            if (idx+1)%scene_switch_interval == 0:
                if scene == 6:
                    scene = 1
                else:
                    scene+=1
            image_path = os.path.join(id_path,image)
            image_name = image.split('_')
            rename_path = os.path.join(id_path,f'{id}_c{camid}s{scene}_{image_name[-1]}')
            if not image_name[1].startswith("cs"):
                os.rename(image_path,rename_path)

def prepare_market(download_path):

    if not os.path.isdir(download_path):
        print('please change the download_path')

    save_path = download_path + '/market1501'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    #gallery
    gallery_path = download_path + '/bounding_box_test'
    gallery_save_path = download_path + '/market1501/gallery'
    if not os.path.isdir(gallery_save_path):
        os.mkdir(gallery_save_path)

    for root, dirs, files in os.walk(gallery_path, topdown=True):
        for name in files:
            if not name[-3:]=='jpg':
                continue
            ID  = name.split('_')
            src_path = gallery_path + '/' + name
            dst_path = gallery_save_path + '/' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)
    #-----------------------------------------
    #query
    query_path = download_path + '/query'
    query_save_path = download_path + '/market1501/query'
    if not os.path.isdir(query_save_path):
        os.mkdir(query_save_path)

    for root, dirs, files in os.walk(query_path, topdown=True):
        for name in files:
            if not name[-3:]=='jpg':
                continue
            ID  = name.split('_')
            src_path = query_path + '/' + name
            dst_path = query_save_path + '/' + ID[0] 
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)

    #-----------------------------------------
    # #multi-query
    # query_path = download_path + '/gt_bbox'
    # # for dukemtmc-reid, we do not need multi-query
    # if os.path.isdir(query_path):
    #     query_save_path = download_path + '/market1501/multi-query'
    #     if not os.path.isdir(query_save_path):
    #         os.mkdir(query_save_path)

    #     for root, dirs, files in os.walk(query_path, topdown=True):
    #         for name in files:
    #             if not name[-3:]=='jpg':
    #                 continue
    #             ID  = name.split('_')
    #             src_path = query_path + '/' + name
    #             dst_path = query_save_path + '/' + ID[0]
    #             if not os.path.isdir(dst_path):
    #                 os.mkdir(dst_path)
    #             copyfile(src_path, dst_path + '/' + name)
    #---------------------------------------
    #train_all
    train_path = download_path + '/bounding_box_train'
    train_save_path = download_path + '/market1501/train_all'
    if not os.path.isdir(train_save_path):
        os.mkdir(train_save_path)

    for root, dirs, files in os.walk(train_path, topdown=True):
        for name in files:
            if not name[-3:]=='jpg':
                continue
            ID  = name.split('_')
            src_path = train_path + '/' + name
            dst_path = train_save_path + '/' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)


    #---------------------------------------
    #train_val
    train_path = download_path + '/bounding_box_train'
    train_save_path = download_path + '/market1501/train'
    val_save_path = download_path + '/market1501/val'
    if not os.path.isdir(train_save_path):
        os.mkdir(train_save_path)
        os.mkdir(val_save_path)

    for root, dirs, files in os.walk(train_path, topdown=True):
        for name in files:
            if not name[-3:]=='jpg':
                continue
            ID  = name.split('_')
            src_path = train_path + '/' + name
            dst_path = train_save_path + '/' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
                dst_path = val_save_path + '/' + ID[0]  #first image is used as val image
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)

def convert_gita_to_market(data_path):
    data_path = os.path.expanduser(data_path)
    move_to_gal(data_path)
    rename_dataset(os.path.join(data_path,"market1501"))
    split_dataset(os.path.join(data_path,"market1501"))
    move_outside_folder(os.path.join(data_path,"market1501"),"gallery")

def restructure_market(dataset_path):
    dataset_path = os.path.expanduser(dataset_path)
    prepare_market(dataset_path)
    path = os.path.join(dataset_path,"market1501")
    move_outside_folder(path,"train")
    move_outside_folder(path,"query")
    move_outside_folder(path,"gallery")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="ReID Inference")
    parser.add_argument("--mode",type=str,default="gita")
    parser.add_argument("--path",type=str,default="~/trt_pose/Dataset/GitaData")
    cfg = parser.parse_args()
    
    if cfg.mode == "gita":
        convert_gita_to_market(cfg.path)
    
    elif cfg.mode == "restructure":
        restructure_market(cfg.path)
        
    elif cfg.mode == "move_in":
        move_inside_folder(cfg.path)
        
    else:
        print(f'Please check mode. Select between "gita" or "resturcture"')