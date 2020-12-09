import torch.utils.data as data
from torch.utils.data import DataLoader
import os
from PIL import Image
import torchvision.transforms as transforms

class nyu_DataSet(data.Dataset):
    def __init__(self,input_img_path,label_img_path,transform=None,max_len=10):
        self.max_l = max_len
        self.input_path = input_img_path
        self.label_path = label_img_path
        self.transform = transform
        self.input_img = os.listdir(input_img_path)
        self.label_img = os.listdir(label_img_path)

    def __len__(self):
        return min(len(self.input_img),self.max_l)
    def __getitem__(self, idx):

        name = self.input_img[idx][:-3]
        input_img_name = self.input_path+'/'+name+"jpg"
        label_img_name = self.label_path + '/' + name+"png"

        input_data = Image.open(input_img_name)
        label_data = Image.open(label_img_name)

        if self.transform:
            input_tensor = self.transform(input_data)
            label_tensor = self.transform(label_data)
        else:
            input_tensor = transforms.ToTensor()(input_data)
            label_tensor = transforms.ToTensor()(label_data)

        return input_tensor,label_tensor

def use_nyu_data(batch_s = 1,isBenchmark=False,max_len=10000):

    if isBenchmark==False:
        input_path = './nyu/nyu_images'
        label_path = './nyu/nyu_depths'
    else:
        input_path = './nyu/bench_images'
        label_path = './nyu/bench_depths'

    d = nyu_DataSet(input_path, label_path,max_len=max_len)
    data_loader = DataLoader(
        d, batch_size=batch_s, shuffle=False, num_workers=4
    )
    return data_loader

if __name__ == '__main__':
    a = use_nyu_data(batch_s=1,max_len=23,isBenchmark=False)
    for i,j in a:
        print(i.shape)
