import os
import time
import pickle
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage import io

GPU = torch.cuda.is_available()
batch_size = 256
num_workers = 32

class Image(Dataset):
  def __init__(self, folder):
    self.folder = folder
    self.image_files = os.listdir(folder)
    self.transformer = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Scale(256),
      transforms.CenterCrop(224), 
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

  def __len__(self):
    return len(self.image_files)

  def __getitem__(self, i):
    image = io.imread(os.path.join(self.folder, self.image_files[i]))
    try:
      transformed_image = self.transformer(image)
    except Exception as e:
      print "using zeros", self.image_files[i]
      transformed_image = torch.zeros(3, 224, 224) 
    return [transformed_image, int(self.image_files[i][-16:-4])]

def get_embedding(loc, resnet_model):
  folder = "/scratch/cse/btech/cs1140485/DL_Course_Data/{}".format(loc)
  dataset = Image(folder)
  dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
  save_folder = folder + "_resnet"
  print "enter loop"
  for _, (image_tensor, image_index) in enumerate(dataloader):
    t_start = time.time()
    image_index = image_index.numpy()
    if GPU: image_tensor = image_tensor.cuda()
    output = resnet_model(Variable(image_tensor, volatile=True))
    output = (output[:, :, 0, 0]).data.cpu().numpy()
    for i in xrange(output.shape[0]):
      image_name = "COCO_{}_{:012}.jpg.pkl".format(loc, image_index[i])
      image_path = os.path.join(save_folder, image_name)
      with open(image_path, "wb") as f:
        pickle.dump(output[i], f, protocol=2)
    t_end = time.time()
    print "time", (t_end - t_start)

if __name__ == '__main__':
  resnet_model = torchvision.models.resnet152(pretrained=True)
  resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])
  if GPU: resnet_model = resnet_model.cuda()
  folder1 = "train2014"
  folder2 = "val2014"
  folder3 = "test2015"
  get_embedding(folder1, resnet_model)
  get_embedding(folder2, resnet_model)
  get_embedding(folder3, resnet_model)
  # folder = "/Users/Shreyan/Desktop/Datasets/DL_Course_Data/train2014"
  # get_embedding(folder, resnet_model)
