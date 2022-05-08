import os

import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import json
import pickle
from tqdm import tqdm
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def read_data(file_path):
    with open(file_path, 'r') as f:
        str_data = f.read()
    return json.loads(str_data)


def main(type):
    path = 'D:\\python\\cs536_option\\original_feature'
    val_data_file = os.path.join(path, 'val_data_path.json')
    train_data_file = os.path.join(path, 'train_data_path.json')
    test_data_file = os.path.join(path, 'test_data_path.json')
    current_path = os.path.abspath('.')
    data_len = 1000

    flag = type
    images = []
    count = 0
    if type == 'train':
        data = read_data(train_data_file)
    elif type == 'test':
        data = read_data(test_data_file)
    elif type == 'val':
        data = read_data(val_data_file)
    else:
        print("wrong")
    all_features = None
    for image_path in tqdm(data):
        image_rgb = Image.open(image_path)
        images.append(image_rgb)
        count += 1
        if count == data_len:
            with torch.no_grad():
                # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                inputs = processor(text=['a'], images=images, return_tensors='pt', padding=True).to(device)
                # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
                output = model(**inputs.to(device))

                # output.to('cpu')
                features = output['image_embeds'].to('cpu')
                if all_features is None:
                    all_features = features
                else:
                    all_features = torch.cat([all_features, features], dim=0)
                count = 0
                images = []

    all_features = pickle.dumps(all_features)
    with open(os.path.join(current_path, 'result', f'{flag}_feature.pkl'), 'wb') as f:
        f.write(all_features)


if __name__ == '__main__':
    main('val')
    main('test')
    main('train')
