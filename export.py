# ============= only used for UNet alpha ===================================
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import onnxruntime as ort
import onnx
import onnx.helper as helper

from torch.utils.data import DataLoader, random_split
from evaluate import evaluate
from utils.data_loading import BasicDataset, CarvanaDataset


from unet import UNet, UNet3, UNet_alpha

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#=========================================================================================

#               加载pytorch模型
#=========================================================================================
alpha = 0.25

state_dict_path = f'./checkpoints/UNet_alpha_alpha{alpha}/checkpoint_epoch200.pth'

model = UNet_alpha(n_channels=3, n_classes=6, bilinear=False, alpha=alpha)

state_dict = torch.load(state_dict_path)
mask_values = state_dict.pop('mask_value',[0,1])
model.load_state_dict(state_dict, False)

model_name = state_dict_path.split('/')[-2] 

#=========================================================================================

#               测试训练集精度
#=========================================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

data_path = "data/PSV_dataset"
dir_img = os.path.join(data_path,"images")
dir_mask = os.path.join(data_path,"labels")
dataset = BasicDataset(dir_img, dir_mask, 0.5)
val_percent = 0.1
n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val
train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

loader_args = dict(batch_size=1, num_workers=12, pin_memory=True)
    
# train_loader = DataLoader(train_set, shuffle=True, worker_init_fn=np.random.seed(seed) ,**loader_args)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    
train_score = evaluate(model, train_loader, device, False)
val_score = evaluate(model, val_loader, device, False)

print(f'val score: {val_score}')
print(f'train score: {train_score}')

#              计算模型参数量
#=========================================================================================

params_count = sum(p.numel() for p in model.parameters())
print(f"Number of {model_name} parameters: {params_count}")

#=========================================================================================

#               图像单元测试 .pth
#=========================================================================================


output_dir = os.path.join('test', model_name + '_result')
os.makedirs(output_dir,exist_ok=True)
print(f'output save in {output_dir}')
model.eval()

imgs_dir = 'test/img'
imgs = os.listdir(imgs_dir)

for img in imgs:
    img_path = os.path.join(imgs_dir,img)
    print(f'inferencing: {img_path}')
    input = Image.open(img_path)
    input = input.resize((512,512))
    input = np.array(input)
    input = torch.from_numpy(input)
    input = input.permute(2,0,1)
    if (input>1).any():
        input = input/255.0
    input = input.unsqueeze(0)

    time_start = time.time()
    output = model(input)
    time_end = time.time()
    print("torch inference time: %f ms" %(time_end - time_start))
    output = torch.argmax(output, dim=1)
    output = output[0].long().squeeze().numpy()
    mask = np.zeros((output.shape[0], output.shape[1]),dtype=bool)
    for i in [1,2,3,4,5]:
        mask[output==i] = True
    mask.transpose((1,0))

    result = Image.fromarray(mask)
    result.save(os.path.join(output_dir,img))
    
#=========================================================================================

#               转换为onnx模型
#=========================================================================================

input_size = (1,3,512,512)

dummy_input = torch.randn(input_size, requires_grad=True)

model.onnx_export = True

model_export_path = os.path.join('onnx_model', state_dict_path.split('/')[-2] + '.onnx')

torch.onnx.export(model,
                dummy_input,
                model_export_path,
                export_params=True,
                opset_version=10,
                do_constant_folding=True,
                input_names=['modelInput'],
                output_names=['modelOutput'])
print(" ")
print("Model has been converted to ONNX")

#=========================================================================================

#                     onnx推理
#=========================================================================================

# model = onnx.load('./unet3_nopad_argmax.onnx')
# node_info = helper.make_tensor_value_info('/outc/conv/Conv_output_0', onnx.TensorProto.FLOAT, (1, 6, 512, 512))
# model.graph.output.insert(1, node_info)
# onnx.save_model(model, "unet3_nopad_argmax_1.onnx")
session = ort.InferenceSession('./unet3_nopad_argmax.onnx')
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
# output_name = '/outc/conv/Conv_output_0'

# input = Image.open('test1.jpg')
input = Image.open('data/PSV_dataset/images/20160725-3-97.jpg')
input = input.resize((512,512))
input = np.array(input)
input = np.transpose(input,(2,0,1))
if(input>1).any():
    input = input/255.0
input = np.expand_dims(input, axis=0)
input = input.astype(np.float32)

for i in range(5):
    time_start = time.time()
    output = session.run([output_name], {input_name: input})[0]
    time_end = time.time()
    print("torch inference time: %f ms" %(time_end - time_start))

# print(output)
# output = np.argmax(output, axis=1)

output = output[0]
mask = np.zeros((output.shape[0], output.shape[1]),dtype=bool)
list_draw = [1,2,3,4,5]
for cls in list_draw:
    mask[output==cls] = True

# mask.transpose((1,0))

result = Image.fromarray(mask)
result.save("output_unet3.jpg")