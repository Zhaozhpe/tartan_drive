from torchsummary import summary
import torch
from torch.utils.tensorboard import SummaryWriter
from util import dict_map

model_fp = '../models/all_modalities/model.cpt'
traj_fp = '/home/zhipeng/datasets/tartandrive/data/test-hard/20210910_2.pt'
traj = torch.load(traj_fp)
batch = dict_map(traj, lambda x: x.unsqueeze(0))

t=0
x0 = dict_map(batch['observation'], lambda x:x[:, t])
u = batch['action'][:, t:t+10]

model = torch.load(model_fp)
print(model)

# writer = SummaryWriter()
# # dummy_input = [torch.randn(x0.shape),torch.randn(u.shape)]
# dummy_input = x0  # Replace with the size and batch of your input
# dummy_input2 = u
# writer.add_graph(model, (dummy_input, dummy_input2))
# writer.close()