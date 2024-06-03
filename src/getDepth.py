import numpy as np
import cv2
import os
from tqdm import tqdm



def get_images(folder_path):
    dataset={}
    left_imgs_path='/cam0/data'
    right_imgs_path='/cam1/data'
    idx=0
    for filename in tqdm(sorted(os.listdir(folder_path+left_imgs_path))):
        img_l = cv2.imread(os.path.join(folder_path+left_imgs_path,filename))
        dataset[filename]={'l':[],'r':[]}
        if img_l is not None:
            dataset[filename]['l']=img_l
        img_r = cv2.imread(os.path.join(folder_path+right_imgs_path,filename))
        if img_r is not None:
            dataset[filename]['r']=img_r

    return dataset

def calculate_depth(img_l,img_r):
    # Initialize the stereo block matching object 
    bSize=5
    stereo = cv2.StereoBM_create(numDisparities=32, blockSize=bSize)

    # Compute the disparity image
    disparity = stereo.compute(img_l, img_r)

    # Normalize the image for representation
    min = disparity.min()
    max = disparity.max()
    disparity = np.uint8(255 * (disparity - min) / (max - min))

    depth_map = []
    return depth_map

if __name__=='__main__':

    CWD_PATH = os.path.abspath(os.getcwd())
    PATH=CWD_PATH+'/Dataset'
    
    print("Loading images")
    dataset_handler = get_images(PATH)
    print(len(dataset_handler.values()))
    # print(dataset_handler.keys())
    # print(dataset_handler.values())