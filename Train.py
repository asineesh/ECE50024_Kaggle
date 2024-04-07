import torch
import torch.nn as nn
import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms,models
import csv
import os
from tqdm import tqdm
import warnings

class CustomDataset(Dataset):
  def __init__(self,imgdir_loc,label_loc,cat_loc,train_flag):
    """
    imgdir_loc is a string containing the location of the image directory
    label_loc is a string containing the location of the label csv file
    cat_loc is a string containing the location of the csv files with the numbers corresponding to each celeb
    train_flag is "train" for training "validation" for validation
    """
    super().__init__()
    #Defining the transforms to be applied to the image
    if train_flag == "train":
      self.transform = transforms.Compose([transforms.RandomResizedCrop(512),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    elif train_flag == "validation" or "curriculum":  
      self.transform = transforms.Compose([transforms.Resize(544),
                                           transforms.CenterCrop(512),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    #Mapping each celeb name to a number
    dictionary = {}
    with open(cat_loc, mode ='r') as file:
      csvFile = csv.reader(file)
      for line in csvFile:
        if line[0].isnumeric():
          dictionary[line[1]] = int(line[0])  #The name is mapped to a number

    #Saving all the image locations
    self.dataset = []
    #Saving the label for each image name
    self.labels = {}
    with open(label_loc, mode ='r') as file:
      csvFile = csv.reader(file)
      for line in csvFile:
        if line[0].isnumeric():
          self.dataset.append(os.path.join(imgdir_loc,line[1]))
          self.labels[os.path.join(imgdir_loc,line[1])] = dictionary[line[2]]  #The image name is mapped to the number corresponding to celeb name

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self,index):
    img = Image.open(self.dataset[index]).convert("RGB")
    label = self.labels[self.dataset[index]]
    trns_img = self.transform(img)
    return trns_img,label

def train(model,train_loader,criterion,optimizer,device):
  model.train()
  loop = tqdm(train_loader)
  cur_loss = 0.0
  accum = 16 #The effective batch size is b_size*16

  for i, (X,y) in enumerate(loop):
    X,y = X.to(device), y.to(device)
    y_pred = model(X)
    loss = criterion(y_pred,y)
    cur_loss += loss.item()
    loop.set_postfix(loss=cur_loss/(i+1))

    loss = loss/accum
    loss.backward()

    if ((i + 1) % accum == 0) or (i + 1 == len(train_loader)):
      optimizer.step()
      optimizer.zero_grad()


def test(model,test_loader,criterion,device):
  model.eval()
  loop = tqdm(test_loader)
  corr = 0
  cur_loss = 0.0

  with torch.no_grad():
    for i,(X,y) in enumerate(loop):
      X,y = X.to(device), y.to(device)
      y_pred = model(X)
      loss = criterion(y_pred,y)
      pred = y_pred.argmax(dim=1)

      corr += (pred==y).sum().item()
      cur_loss += loss.item()
      loop.set_postfix(loss=cur_loss/(i+1))
  
  print(f"The accuracy is {100*corr/len(test_loader.dataset)}")
  return round(100*corr/len(test_loader.dataset),2)

def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(device)

  model = models.swin_v2_b("Swin_V2_B_Weights.IMAGENET1K_V1").to(device)
  model.head = nn.Linear(1024,100).to(device)

  train_set = CustomDataset("train","main_train.csv","category.csv","train")
  test_set = CustomDataset("train","main_test.csv","category.csv","validation")

  train_dataloader = DataLoader(train_set,batch_size=8,num_workers=4,shuffle=True)
  test_dataloader = DataLoader(test_set,batch_size=8,num_workers=4,shuffle=False)

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(model.parameters(),lr=0.0001)
  epochs = 100

  print(f"The train set has {len(train_set)} elements")
  print(f"The test set has {len(test_set)} elements")

  best_acc = 0
  for epoch in range(epochs):
    warnings.filterwarnings("ignore")
    print(f"This is epoch {epoch}")
    train(model,train_dataloader,criterion,optimizer,device)
    cur_acc = test(model,test_dataloader,criterion,device)
    if cur_acc>best_acc:
      best_acc = cur_acc
      torch.save(model.state_dict(),"models/Model_"+str(epoch)+"_"+str(best_acc)+".pt")


if __name__ == "__main__":
    main()

