import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import os


current_path=os.path.dirname(os.path.abspath('__file__'))

class New_DataSet(Dataset):
    def __init__(self, image_dataset, text_dataset, ids):
        self.text_data = text_dataset
        self.image_data=image_dataset
        self.ids=ids


    def __getitem__(self, item):
        return [self.image_data[item], self.text_data[item]], self.ids[item]


    def __len__(self):
        return self.text_data.shape[0]


def get_train_loader(batchsize=1000):
    # path1='D:\\python\\536_final\\features\\ingredients_embeddings_train.pkl'
    path1='D:\\python\\536_final\\features\\instructions_embeddings_train.pkl'
    # path1=os.path.join(current_path,'features', 'embeddings_train1.pkl')
    # path1='D:\\python\\536_final\\features\\embeddings_train1.pkl'
    path2=os.path.join(current_path,'result', 'train_result.pkl')
    # path2='D:\\python\\cs536_option\\result\\train_result.pkl'
    with open(path1, 'rb') as f:
        text_data=f.read()
    with open(path2, 'rb') as f:
        images_data=f.read()
    data=pickle.loads(text_data)
    text_data, ids=data[0], data[1]
    text_data = torch.from_numpy(text_data)
    images_data=pickle.loads(images_data)
    train_dataset=New_DataSet(images_data, text_data, ids)
    return DataLoader(train_dataset, batchsize)


def get_val_loader(batchsize=1000):
    # path1='D:\\python\\536_final\\features\\ingredients_embeddings_val.pkl'
    path1='D:\\python\\536_final\\features\\instructions_embeddings_val.pkl'
    # path1 = os.path.join(current_path, 'features', 'embeddings_val1.pkl')
    path2 = os.path.join(current_path, 'result', 'val_result.pkl')
    # path1='D:\\python\\536_final\\features\\embeddings_val1.pkl'
    # path2='D:\\python\\cs536_option\\result\\val_result.pkl'
    with open(path1, 'rb') as f:
        text_data=f.read()
    with open(path2, 'rb') as f:
        images_data=f.read()
    data=pickle.loads(text_data)
    text_data, ids=data[0], data[1]
    text_data=torch.from_numpy(text_data)
    images_data=pickle.loads(images_data)
    val_dataset=New_DataSet(images_data, text_data, ids)
    return DataLoader(val_dataset, batchsize)

def get_test_loader(batch_size=1000):
    if batch_size is None:
        batch_size=80
    # path1 = 'D:\\python\\536_final\\features\\ingredients_embeddings_test.pkl'
    path1 = 'D:\\python\\536_final\\features\\instructions_embeddings_test.pkl'
    # path1 = os.path.join(current_path, 'features', 'embeddings_test1.pkl')
    path2 = os.path.join(current_path, 'result', 'test_result.pkl')
    # path1 = 'D:\\python\\536_final\\features\\embeddings_test1.pkl'
    # path2 = 'D:\\python\\cs536_option\\result\\test_result.pkl'
    with open(path1, 'rb') as f:
        text_data=f.read()
    with open(path2, 'rb') as f:
        images_data=f.read()
    data=pickle.loads(text_data)
    text_data, ids=data[0], data[1]
    text_data=torch.from_numpy(text_data)
    images_data=pickle.loads(images_data)
    test_dataset=New_DataSet(images_data, text_data, ids)
    return DataLoader(test_dataset, batch_size)