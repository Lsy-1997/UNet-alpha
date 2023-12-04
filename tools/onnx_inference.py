import onnxruntime as ort
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import onnx
import onnx.helper as helper

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