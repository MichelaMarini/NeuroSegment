
import torch

import numpy as np

from skimage.morphology import binary_opening, disk
from matplotlib import pyplot as plt

import load_t
from multiprocessing import cpu_count
from torchvision.transforms import Compose

from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset

from torchvision import transforms, datasets, models
import torch.nn.functional as F

class SimData(Dataset):
    def __init__(self, transform=None):
        original_imgs_test = "image/"  # "path of test images"
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




print("number of cpus: ", cpu_count())


kwargs = {'num_workers': 0
    ,
          'pin_memory': True} \

trans = Compose([
    transforms.ToTensor(),

])
test_dataset = SimData(transform=Compose([transforms.ToTensor()]))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, **kwargs)

inputs = next(iter(test_loader))
print('input_type:', type(inputs))
print('image_shape', inputs.shape)
#print('input_tensor_max1', torch.max(inputs[1,0,:,:]))# max
#print('input_tensor_min1', torch.min(inputs[1,0,:,:]))# min

#print('image_shape', inputs.shape)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath,map_location='cpu')
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model



#segmetation task



model = load_checkpoint('network.pth')
model.eval()

output = model(inputs)
print('output_type:', type(output))
print('max value_output', np.max(output.data.cpu().numpy()))
output = F.sigmoid(output)
output = output.data.cpu().numpy()

max1 = np.max(output)
min1 = np.min(output)
print('max_output', max1)
print('min_output', min1)
len_output = len(output[:,0,0,0])
max_out = np.empty(len_output)
min_out = np.empty(len_output)

for i in range((len_output)):
    max_out[i] = np.max(output[i])
    min_out[i] = np.min(output[i])
    output[i,0,:,:] = binary_opening(output[i,0,:,:] > 0.2 * max_out[i], disk(1)).astype(np.int)


print('value  max_output', max_out)
print('value  min_output', min_out)
np.save('segmentations.npy', output)
print('output_shape', output.shape)
max= np.max(output)
min = np.min(output)
print('max',max)
print('min',min)
plt.subplot(4, 4, 1)
plt.imshow(output[0, 0, :, :], cmap='gray')
plt.title('Segm 1')
plt.subplot(4, 4, 2)
plt.imshow(output[1, 0, :, :], cmap='gray')
plt.title('Segm 2')
plt.subplot(4, 4, 3)
plt.imshow(output[2, 0, :, :], cmap='gray')
plt.title('Segm 3')
plt.subplot(4, 4, 4)
plt.imshow(output[3, 0, :, :], cmap='gray')
plt.title('Segm 4')
plt.subplot(4, 4, 5)
plt.imshow(output[4, 0, :, :], cmap='gray')
plt.title('Segm 5')
plt.subplot(4, 4, 6)
plt.imshow(output[5, 0, :, :], cmap='gray')
plt.title('Segm 6')
plt.subplot(4, 4, 7)
plt.imshow(output[6, 0, :, :], cmap='gray')
plt.title('Segm 7')
plt.subplot(4, 4, 8)
plt.imshow(output[7, 0, :, :], cmap='gray')
plt.title('Segm 8')
plt.subplot(4, 4, 9)
plt.imshow(output[8, 0, :, :], cmap='gray')
plt.title('Segm 9')
plt.subplot(4, 4, 10)
plt.imshow(output[9, 0, :, :], cmap='gray')
plt.title('Segm 10')
plt.subplot(4, 4, 11)
plt.imshow(output[10, 0, :, :], cmap='gray')
plt.title('Segm 11')
plt.subplot(4, 4, 12)
plt.imshow(output[11, 0, :, :], cmap='gray')
plt.title('Segm 12')
plt.subplot(4, 4, 13)
plt.imshow(output[12, 0, :, :], cmap='gray')
plt.title('Segm 13')
plt.subplot(4, 4, 14)
plt.imshow(output[13, 0, :, :], cmap='gray')
plt.title('Segm 14')
plt.subplot(4, 4, 15)
plt.imshow(output[14, 0, :, :], cmap='gray')
plt.title('Segm 15')
plt.subplot(4, 4, 16)
plt.imshow(output[15, 0, :, :], cmap='gray')
plt.title('Segm 16')
plt.show()
plt.savefig(r'segmentation_9_new.png')