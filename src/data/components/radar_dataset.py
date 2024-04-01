import os
from torch.utils.data import Dataset
import torchvision
import torch
import gdal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class RadarDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            sequence_length: int
        ) -> None:
        super().__init__()

        self.sequence_length = sequence_length + 1 #last for output
        self.data_dir = data_dir

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
            ##
            # band[band == -np.inf] = 0
            # band[band < 0] = 0
            # print(np.min(band))
            ##
            stack.append(torchvision.transforms.ToTensor()(band))
        output_name = sequence[-1]
        output = gdal.Open(output_name)
        output = output.GetRasterBand(1).ReadAsArray()
        # output[output < 0] = 0
        
        return torch.stack(stack), torchvision.transforms.ToTensor()(output), output_name
    
    @staticmethod
    def plotCmap(pred, target):
        plt.close()
        pred_array = pred.cpu().numpy().squeeze()
        target_array = target.cpu().numpy().squeeze()
        norm = plt.Normalize(vmin=0, vmax=max(np.max(pred_array), np.max(target_array)))
        cmap = plt.get_cmap('viridis')
        fig, (ax1, ax2) = plt.subplots(2, 1)
        rgba_pred = cmap(norm(pred_array))
        rbga_target = cmap(norm(target_array))

        ax1.set_title('Predict')
        ax2.set_title('Target')
        
        ax1.imshow(rgba_pred)
        ax2.imshow(rbga_target)

        return fig


if __name__ == '__main__':

    dataset = RadarDataset('data/train', 5)
    print(len(dataset))
    print(dataset.sequenceList[1])

    
    # print(output.isnan().any())
    # print(input.shape)
    # print(output.shape)
    # print(output.max())
    # dataset = RadarDataset('/home/lechiennn/lab/thesis/RainForecastingModel/data/train', 5)
    
    tensor = torch.rand(4, 1, 90, 250).cuda()

    fig = RadarDataset.plotCmap(tensor, tensor)