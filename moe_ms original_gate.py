
import torch
from torch import nn
import torch.nn.functional as F
import math
from inspect import isfunction
from linear_wo_moe.data_generation import CustomDataset
from torch.utils.data import DataLoader
import numpy as np
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class expert(nn.Module):
    # network structure
    def __init__(self, 
                expert_input_size, 
                hidden_size,
                expert_output_size,
                w,b):
        super(expert, self).__init__()
        self.layer = nn.Linear(expert_input_size, expert_output_size)
        self.layer.weight.data = torch.FloatTensor([[w]])
        self.layer.bias.data = torch.FloatTensor([b])
        # self.layer = nn.Sequential(
        #     nn.Linear(expert_input_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, expert_output_size),            
        # )        

    def forward(self, x):        
        return self.layer(x)

# Gating network

class gate(nn.Module):
    # network structure
    def __init__(self,
                expert_input_size,
                gate_hidden_size,
                expert_output_size,
                num_experts):
        super(gate, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(expert_input_size, gate_hidden_size),
            nn.ReLU(),
            nn.Linear(gate_hidden_size, gate_hidden_size),
            nn.ReLU(),
            nn.Linear(gate_hidden_size, gate_hidden_size),
            nn.ReLU(),
            nn.Linear(gate_hidden_size, num_experts),
        )

    def forward(self, x):        
        return self.layer(x)


class Moe(nn.Module):
    def __init__(self,
                expert_input_size,
                hidden_size,
                gate_hidden_size,
                expert_output_size,
                w_and_b,
                num_experts):
        super(Moe, self).__init__()
        self.expert_input_size = expert_input_size
        self.hidden_size = hidden_size
        self.expert_output_size = expert_output_size
        self.num_experts = num_experts
        self.gate_hidden_size = gate_hidden_size
        
        self.experts = nn.ModuleList([expert(expert_input_size,
                                             hidden_size,
                                             expert_output_size,
                                             w, b) for i, (w, b) in enumerate(w_and_b)])
        
        self.gate = gate(expert_input_size,
                         gate_hidden_size,
                         expert_output_size,
                         num_experts)
        self.softmax = nn.Softmax(dim=1)
    
    def for_experts(self):
        return self.experts
            
    def forward(self,x):
        exp_res = torch.cat([e(x).t() for e in self.experts]).t()
        # gate_input = torch.cat([exp_res, x], dim=1)
        weights = self.softmax(self.gate(x))
        
        out = torch.einsum("ab,ab->a",exp_res,weights).reshape(-1,self.expert_output_size)
        entropy = -torch.einsum("ab,ab->a",weights, torch.log(weights)).reshape(-1,self.expert_output_size)
        
        return out, entropy, weights, exp_res
        


num_experts = 3
expert_input_size = 1
hidden_size = 2
gate_hidden_size = 6
expert_output_size = 1
batch_size = 100
num_epochs = 50
only_for_gate_epoch = 30
lr = 1e-4
lamb = 0

with open('regression0.pickle', 'rb') as f:
    model_0 = pickle.load(f)
with open('regression1.pickle', 'rb') as f:
    model_1 = pickle.load(f)
with open('regression2.pickle', 'rb') as f:
    model_2 = pickle.load(f)

w_and_b = [model_0,model_1,model_2]

model = Moe(expert_input_size,
            hidden_size,
            gate_hidden_size,
            expert_output_size,
            w_and_b,
            num_experts).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
criterion = nn.MSELoss()

def criterion_1_2(exp_res, weights, y):
    tmp = torch.einsum("ab,ab->ab", y - exp_res, y - exp_res)
    res = torch.einsum("ab,ab->a",weights,tmp).mean()
    return res

dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

total_step = len(dataloader)
for epoch in range(num_epochs):
    
    if epoch % 2 == 1 : 
         for p in model.experts.parameters():
             p.requires_grad = False
         for p in model.gate.parameters():
             p.requires_grad = True
    else: 
         for p in model.experts.parameters():
             p.requires_grad = True
         for p in model.gate.parameters():
             p.requires_grad = False
    
    for i, (x, y) in enumerate(dataloader):  
        
        x = x.to(device)
        y = y.to(device)
        
        outputs = model(x)[0]
        entropy = model(x)[1]
        weights = model(x)[2]
        exp_res = model(x)[3]
        
        # loss = criterion(outputs, y) + lamb*torch.mean(entropy)
        loss = criterion_1_2(exp_res, weights, y) + lamb*torch.mean(entropy)
                
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


with torch.no_grad():
    outputs = []
    ys = []
    xs = []
    ws = []
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        outputs.append(model(x)[0])
        ys.append(y)
        xs.append(x)
        ws.append(torch.argmax(model(x)[2], dim=1))

xxx = torch.cat(xs).detach().cpu().numpy()
yyy = torch.cat(ys).detach().cpu().numpy()
ccc = torch.cat(ws).detach().cpu().numpy()
import matplotlib.pyplot as plt

plt.scatter(xxx, yyy, c=ccc)
plt.axis('equal')
plt.xlim([np.min(xxx), np.max(xxx)])
plt.ylim([np.min(yyy), np.max(yyy)])
plt.title(torch.abs(torch.cat(outputs) - torch.cat(ys)).mean())

for i in model.for_experts():
    a = i.get_parameter('layer.weight').detach().cpu().numpy()[0]
    b = i.get_parameter('layer.bias').detach().cpu().numpy()[0]
    xx = np.linspace(-5,20,100)
    yy = a*xx+b
    plt.plot(xx, yy, '-r')
plt.savefig('my_plot2.png')
plt.close()


