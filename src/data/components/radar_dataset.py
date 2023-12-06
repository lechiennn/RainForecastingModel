import os
from torch.utils.data import Dataset
import torchvision
import torch
import gdal
import pandas as pd

class RadarDataset(Dataset):
    def __init__(
            self,
            dataset: str,
            sequence_length: int
        ) -> None:
        super().__init__()
        self.dataset = dataset
        self.sequence_length = sequence_length + 1 #last for output
        self.data_dir = os.path.join('data', dataset)

        dateList = []
        imageList = []
        self.sequenceList = []

        for (root, dirs, files) in os.walk(self.data_dir, topdown=True):
            if len(files) < 1: continue
            for f in files:
                dateList.append(f.split('_')[1].split('.')[0])
                imageList.append(os.path.join(root, f))

        index = pd.to_datetime(dateList, format='%Y%m%d%H%M%S').sort_values()
        series = pd.Series(imageList, index=index)
        df = pd.DataFrame({'image': series})
        df = df.asfreq(freq='H')
        

        for i in range(len(df.image) - self.sequence_length):
            sequence = df.image.iloc[i:i+self.sequence_length]

            if not sequence.isnull().any():
                self.sequenceList.append(list(sequence))

    def __len__(self):
        return len(self.sequenceList)

    def __getitem__(self, index):
        sequence = self.sequenceList[index]
        stack = []
        for image in sequence[:-1]:
            # image = os.path.join(self.data_dir, self.imageList[index+seq_index])
            image = gdal.Open(image)
            band = image.GetRasterBand(1).ReadAsArray()
            stack.append(torchvision.transforms.ToTensor()(band))
        output = sequence[-1]
        output = gdal.Open(output)
        output = output.GetRasterBand(1).ReadAsArray()

        return torch.stack(stack), torchvision.transforms.ToTensor()(output)
    
if __name__ == '__main__':

    dataset = RadarDataset('test', 5)
    print(len(dataset))
    print(dataset.sequenceList[1])

    input, output = dataset[1]
    print(input.shape)
    print(output.shape)
    # dataset = RadarDataset('/home/lechiennn/lab/thesis/RainForecastingModel/data/train', 5)
    
    # print(len(dataset.imageList))
    # input, output = dataset[1]
    # print(type(input), type(output))
    # print(input.shape, output.shape)