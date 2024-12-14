import h5py
import os
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image
import json

image_folder = "./data/tiny-imagenet-200"

class_names = sorted(os.listdir(f"{image_folder}/train"))

mapping = {}

with open(f"{image_folder}/words.txt") as file:
    data = file.read().split("\n")
    for i in data:
        k,v = i.split("\t")
        mapping[k] = v

print("Generating Training hdf5")

images = np.zeros((1_00_000, 64, 64, 3), dtype=np.uint8)
labels = np.zeros((1_00_000), dtype=np.uint8)
label_dict = {}
count = 0

pbar = tqdm(total=images.shape[0])
for idx, class_name in enumerate(class_names):
    folder = f"{image_folder}/train/{class_name}/images/*.JPEG"
    label_dict[idx] = mapping[class_name]
    for image_name in glob(folder):
        with Image.open(image_name).convert('RGB') as img:
            images[count] = np.array(img)
            labels[count] = idx
            count += 1
            pbar.update(1)
del pbar

with h5py.File("tinyimage_train.hdf5", "w") as file:
    imgs = file.create_dataset("images", data=images, dtype="uint8", compression='gzip', compression_opts=9)
    lbls = file.create_dataset("labels", data=labels, dtype="int64", compression='gzip', compression_opts=9)

print("Training hdf5 generated")


print("Generating Validation hdf5")

images = np.zeros((10_000, 64, 64, 3), dtype=np.uint8)
labels = np.zeros((10_000), dtype=np.uint8)
count = 0

class_idx = {class_name:idx for idx,class_name in enumerate(class_names)}
folder = f"{image_folder}/val/images/"
data = open(f"{image_folder}/val/val_annotations.txt").readlines()
for details in tqdm(data):
# for idx, class_name in enumerate(class_names):
    image_name, class_name, _, _, _, _ = details.split("\t")
    with Image.open(f"{image_folder}/val/images/{image_name}").convert('RGB') as img:
        images[count] = np.array(img)
        labels[count] = class_idx[class_name]
        count += 1

with h5py.File("tinyimage_val.hdf5", "w") as file:
    imgs = file.create_dataset("images", data=images, dtype="uint8", compression='gzip', compression_opts=9)
    lbls = file.create_dataset("labels", data=labels, dtype="int64", compression='gzip', compression_opts=9)

print("Validation hdf5 generated")

with open("labels.json", "w") as file:
    json_data = json.dumps(label_dict, indent=4)
    file.write(json_data)
