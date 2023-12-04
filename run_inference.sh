# python predict.py --model checkpoints/UNet_alpha2.0_woodscape/checkpoint_epoch100.pth --input /home/cvrsg/lsy/Parking_line_detection/mmsegmentation/data/woodscape/woodscape_split/images/validation --output ./infer_result_woodscape

# python predict.py --model checkpoints/UNet_alpha0.5/checkpoint_epoch200.pth --input /home/cvrsg/lsy/Parking_line_detection/Pytorch-UNet/test/lidar-binary/img/binary/ori --output ./infer_result_lidar/binary/ori
# python predict.py --model checkpoints/UNet_alpha0.5/checkpoint_epoch200.pth --input /home/cvrsg/lsy/Parking_line_detection/Pytorch-UNet/test/lidar-binary/img/binary/dilated --output ./infer_result_lidar/binary/dilated

# python predict.py --model checkpoints/UNet_alpha0.5/checkpoint_epoch200.pth --input /home/cvrsg/lsy/Parking_line_detection/Pytorch-UNet/test/lidar-binary/img/height_diff/ori --output ./infer_result_lidar/height_diff/ori
# python predict.py --model checkpoints/UNet_alpha0.5/checkpoint_epoch200.pth --input /home/cvrsg/lsy/Parking_line_detection/Pytorch-UNet/test/lidar-binary/img/height_diff/dilated --output ./infer_result_lidar/height_diff/dilated

# python predict.py --model checkpoints/UNet_alpha0.5/checkpoint_epoch200.pth --input /home/cvrsg/lsy/Parking_line_detection/Pytorch-UNet/test/lidar-binary/img/height_diff_gamma/ori --output ./infer_result_lidar/height_diff_gamma/ori
# python predict.py --model checkpoints/UNet_alpha0.5/checkpoint_epoch200.pth --input /home/cvrsg/lsy/Parking_line_detection/Pytorch-UNet/test/lidar-binary/img/height_diff_gamma/dilated --output ./infer_result_lidar/height_diff_gamma/dilated

# python predict.py --model checkpoints/UNet_alpha0.5_Cityscapes2023-11-30/checkpoint_epoch10.pth --input /home/cvrsg/lsy/Parking_line_detection/Pytorch-UNet/test/zfzx_3.9_0.7 --output ./test/zfzx_3.9_0.7_test_result_ori_size/ --classes 34 --scale 1

python predict.py --model checkpoints/UNet_alpha0.5_Cityscapes2023-11-30/checkpoint_epoch30.pth --input /home/cvrsg/lsy/Parking_line_detection/Pytorch-UNet/test/zfzx_3.9_0.7 --output ./test/zfzx_3.9_0.7_test_result_ori_size_30/ --classes 34 --scale 1
