
import torch

import numpy as np

from skimage.morphology import binary_opening, disk
from skimage.filters import threshold_otsu
import load_t
from multiprocessing import cpu_count
from torchvision.transforms import Compose

from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset

from torchvision import transforms, datasets, models
import torch.nn.functional as F


import Unet


class SimData(Dataset):
    def __init__(self, transform=None):

        original_imgs_test = "Dataset/images/"
        self.input_images_test = load_t.get_datasets_test(16, original_imgs_test) #number of images in input, path"

        self.transform = transform

    def __len__(self):
        return len(self.input_images_test)

    def __getitem__(self, idx):

        image = self.input_images_test[idx]
        image = np.float32(image)
        image = np.expand_dims(image, axis=2)

        if self.transform:
            image = self.transform(image)

        return image


kwargs = {'num_workers': 0
    ,
          'pin_memory': True} \

trans = Compose([
    transforms.ToTensor(),

])
test_dataset = SimData(transform=Compose([transforms.ToTensor()]))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, **kwargs)

inputs = next(iter(test_loader))



def load_checkpoint(filepath):
     checkpoint = torch.load(filepath,map_location='cpu')
     model= Unet.UNet(2)
     model.load_state_dict(checkpoint['state_dict'])
     for parameter in model.parameters():
         parameter.requires_grad = False

     model.eval()
     return model




model = load_checkpoint('Model.pth')
model.eval()

output = model(inputs)
output = F.sigmoid(output)
output = output.data.cpu().numpy()


len_output = len(output[:,0,0,0])
max_out = np.empty(len_output)
min_out = np.empty(len_output)

max_out = np.empty(len_output)
thresh_segm = np.empty(len_output)

for i in range((len_output)):
    max_out[i] = np.max(output[i,1])
    thresh_segm[i] = threshold_otsu(output[i, 0, :, :])

    output[i, 1, :, :] = binary_opening(output[i, 1, :, :] > 0.8 * max_out[i], disk(1)).astype(np.int)
    output[i, 0, :, :] = output[i, 0, :, :] > thresh_segm[i]


#save images as numpy array
#np.save('Results.npy', output)

