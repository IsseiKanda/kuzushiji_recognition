import pandas as pd 
import matplotlib.pylab as plt
import numpy as np
from PIL import Image
import uuid
import os
import tqdm

##  Function for fetching image from path and returns as int32 np.array
def load_image_data(path):
    img = plt.imread(path)
    img = img.astype(np.int32)
    return img

##  Function for decoding the labels data in train.csv
def get_segmented_labels(labels):
    labels = np.array(labels.split(' '))
    count_semented_labels = int(len(labels) / 5)
    labels = labels.reshape((count_semented_labels, 5))
    segmented_labels = []
    for label in labels:
        unicode = label[0]
        x = label[1].astype(np.int32)
        y = label[2].astype(np.int32)
        w = label[3].astype(np.int32)
        h = label[4].astype(np.int32)
        segmented_labels.append({'unicode': unicode, 'x': x, 'y': y, 'w': w, 'h': h})
    return segmented_labels

## Function for extract a specific area of the img
def get_segmeneted_image(img, x, y, w, h):
    return img[y:y+h, x:x+w]

## Function to convert the RGB img to Grayscaled img
def rgb_to_grayscale(img):
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])


## Function to iterate the dataframe to get label image data
def get_label_images(dataframe):
    label_image_data = []
    for _, row in tqdm.tqdm(dataframe.iterrows(), total=dataframe.shape[0]):
        image_id = str(row['image_id'])
        image_file = 'kuzushiji-recognition/train_images/' + image_id + '.jpg'
        image = load_image_data(image_file)
        labels = row['labels']
        segmented_chars = get_segmented_labels(labels)
        for char in segmented_chars:
            x = char['x']
            y = char['y']
            w = char['w']
            h = char['h']
            char_img = get_segmeneted_image(image, x, y, w, h)
            char_img = rgb_to_grayscale(char_img)
            label_image_data.append({'image': char_img, 'unicode': char['unicode'], 'image_id': image_id})
    return label_image_data
    
## Function to save label image 
def save_segmented_labels(label_image_data):
    for data in tqdm.tqdm(label_image_data, total=len(label_image_data)):
        img = data['image']
        img = img.astype(np.uint8)
        im = Image.fromarray(img)
        if not os.path.exists('label_image/' + data['unicode']):
            os.makedirs('label_image/' + data['unicode'])
        im.save('label_image/' + data['unicode'] + '/' + str(uuid.uuid4()) + '.jpg',bitmap_format='jpg')

dataframe = pd.read_csv('kuzushiji-recognition/train.csv')
segmented_label_data = get_label_images(dataframe)
save_segmented_labels(segmented_label_data)