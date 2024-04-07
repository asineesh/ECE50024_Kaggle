import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import models,transforms
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

class CustomDataset_Test(Dataset):
  def __init__(self,imgdir_loc,cat_loc):
    """
    imgdir_loc is a string containing the location of the image directory
    cat_loc is a string containing the location of the csv files with the numbers corresponding to each celeb
    """
    super().__init__()
    #Defining the transforms to be applied to the image
    self.transform = transforms.Compose([transforms.Resize(544),
                                           transforms.CenterCrop(512),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    #Mapping each number to a name
    self.dictionary = {}
    with open(cat_loc, mode ='r') as file:
      csvFile = csv.reader(file)
      for line in csvFile:
        if line[0].isnumeric():
          self.dictionary[int(line[0])] = line[1]  #The name is mapped to a number
          
    #Saving all the image locations
    self.dataset = [] 
    for img_name in os.listdir(imgdir_loc):
        self.dataset.append(os.path.join(imgdir_loc,img_name))    

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self,index):
    img = Image.open(self.dataset[index]).convert("RGB")
    label = int(self.dataset[index].split("/")[-1].split(".")[0])
    trns_img = self.transform(img)
    return trns_img,label


def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(device)

  model = models.swin_v2_b().to(device)
  model.head = nn.Linear(1024,100).to(device)

  model.load_state_dict(torch.load("models/Model_26_82.36.pt"))
  model.eval()

  test_set = CustomDataset_Test("test","category.csv")
  test_loader = DataLoader(test_set,batch_size=8,num_workers=16)

  print(f"The test set has {len(test_set)} elements")

  predictions = []
  loop = tqdm(test_loader)
  with torch.no_grad():
    for imgs,index in loop:
      imgs = imgs.to(device)
      prediction = model(imgs) 
      pred_val = prediction.argmax(dim=1).to("cpu")
      temp = list(torch.concat([index.unsqueeze(1),pred_val.unsqueeze(1)],dim=1))
      predictions.extend(temp)
  
  final_values = [[x[0].item(),test_set.dictionary[x[1].item()]] for x in predictions]
  final_values.sort(key=lambda x:x[0])

  with open('trial.csv', mode ='w',newline='') as file:
    writer = csv.writer(file)
    for line in final_values:
      writer.writerow(line)    

  file.close()

if __name__ == "__main__":
    main()

