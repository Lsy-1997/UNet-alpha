export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
# python train.py --epochs 200 --batch-size 1 --classes 6 --data-path 'data/PSV_dataset'

# python train.py --epochs 200 --batch-size 1 --classes 6 --data-path 'data/PSV_dataset' --model FFNet

# 202305101817
# python train.py --epochs 200 --batch-size 1 --classes 6 --data-path 'data/PSV_dataset' --model UNet_alpha --alpha 0.25


# 202305102015
# python train.py --epochs 200 --batch-size 1 --classes 6 --data-path 'data/PSV_dataset' --model UNet_alpha --alpha 0.25

# 202305102143
# python train.py --epochs 200 --batch-size 1 --classes 6 --data-path 'data/PSV_dataset' --model UNet_alpha --alpha 0.5

# 202306021029
# python train.py --epochs 200 --batch-size 1 --classes 6 --data-path 'data/PSV_dataset' --model fcn

# 202311281624
python train.py --epochs 100 --batch-size 1 --classes 34 --data-path 'data/Cityscapes' --model UNet_alpha --alpha 0.5