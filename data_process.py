import pandas as pd
from os import path
from PIL import Image
import numpy as np
import math
import glob

def read_csv():
    csv_data = pd.read_csv('driver_imgs_list.csv')
    shuffle_data = csv_data.sample(frac=1).reset_index(drop=True)
    folders = shuffle_data['classname']
    files_name = shuffle_data['img']
    labels = np.array(pd.get_dummies(folders))
    paths = []
    for i in range(len(folders)):
        paths.append(path.join('train',folders[i],files_name[i]))
    return np.array(paths),labels


def get_batches(x,y,batch_size):
    length = len(x)
    n_batches = math.ceil(length / batch_size)
    for i in range(n_batches):
        idx = i*batch_size
        batch_raw_images = [Image.open(item) for item in x[idx:idx+batch_size]] 
        batch_images = [(np.array(item.resize((224,224)))/255 - 0.5) for item in batch_raw_images]
        yield batch_images,y[idx:idx+batch_size]
        
int_txt_dict = {
    0:"Normal",
    1:"Texting-Right",
    2:"Talking-Right",
    3:"Texting-Left",
    4:"Talking-Left",
    5:"Tune-Radio",
    6:"Drinking",
    7:"Reaching-Behind",
    8:"Makeup-Hair",
    9:"Talking-To-Passenger"
}

def get_test_images(batch_size):
    test_file_path = glob.glob('./test/*.jpg')
    file_list = [path.basename(each) for each in test_file_path]
    length = len(test_file_path)
    n_batches = math.ceil(length / batch_size)
    for i in range(n_batches):
        idx = i*batch_size
        batch_raw_images = [Image.open(item) for item in test_file_path[idx:idx+batch_size]]
        batch_images = [(np.array(item.resize((224,224)))/255 - 0.5) for item in batch_raw_images]
        file_names = file_list[idx:idx+batch_size]
        yield batch_images,file_names