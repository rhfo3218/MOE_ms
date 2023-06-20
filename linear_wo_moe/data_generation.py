import torch
from torch.utils.data import Dataset
import pickle


class CustomDataset(Dataset): 
  def __init__(self, loc = '2d_syn_data.pickle'):
    with open(loc, 'rb') as f:
      data = pickle.load(f)  
    self.x_data = data['x']
    self.y_data = data['y'].reshape(-1,1)

  def __len__(self): 
    return len(self.x_data)

  def __getitem__(self, idx): 
    x = torch.FloatTensor(self.x_data[idx])
    y = torch.FloatTensor(self.y_data[idx])
    return x, y
  
  

# class CustomDataset(Dataset): 
#   def __init__(self, loc_x = '2d_syn_x.csv', loc_y='2d_syn_x.csv'):
#     self.loc_x = loc_x
#     self.loc_y = loc_y
#     self.x_data = pd.read_csv(loc_x)
#     self.y_data = pd.read_csv(loc_y)
  
#   def __len__(self): 
#     return len(self.x_data)

#   def __getitem__(self, idx):     
#     x = torch.FloatTensor(self.x_data[idx].values)
#     y = torch.FloatTensor(self.y_data[idx].values)
#     return x, y

