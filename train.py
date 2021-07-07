import torch
from model.yolo import Model
device = torch.device("cpu")
model = Model('./model/yolo.yaml')
model.to(device)


state_dict = torch.load('yolov3_state_dict.pt', map_location=device)

model.load_state_dict(state_dict)


for name, param in model.named_parameters():
    print(name)