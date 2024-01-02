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
# python train.py --epochs 100 --batch-size 1 --classes 34 --data-path 'data/Cityscapes' --model UNet_alpha --alpha 0.5

# 202312281621 同济地下车库自采数据集训练
# python train.py --epochs 5 --batch-size 1 --classes 6 --data-path 'data/tongji_parking_rgbi' --model UNet_alpha --alpha 0.5 --scale 1.0 --learning-rate 1e-4
# python train.py --epochs 5 --batch-size 1 --classes 6 --data-path 'data/tongji_parking_rgbi' --model UNet_alpha --alpha 0.5 --scale 1.0 --learning-rate 1e-3
# python train.py --epochs 20 --batch-size 1 --classes 6 --data-path 'data/PSV_dataset' --model UNet_alpha --alpha 0.5 --validation 90
# python train.py --epochs 5 --batch-size 1 --classes 6 --data-path 'data/tongji_parking_rgbi' --model UNet_alpha --alpha 0.5 --scale 1.0 --learning-rate 1e-3
# python train.py --epochs 20 --batch-size 1 --classes 6 --data-path 'data/tongji_parking_rgbi_slice' --model UNet_alpha --alpha 0.5 --scale 1.0
# python train.py --epochs 100 --batch-size 1 --classes 6 --data-path 'data/tongji_parking_rgbi_slice' --model UNet_alpha --alpha 0.5 --scale 1.0
# python train.py --epochs 100 --batch-size 1 --classes 6 --data-path 'data/tongji_parking_rgbi_slice_splitted' --model UNet_alpha --alpha 0.5 --scale 1.0
# python train.py --epochs 100 --batch-size 1 --classes 6 --data-path 'data/tongji_parking_rgbi_slice_splitted_truergbi' --model UNet_alpha --alpha 0.5 --scale 1.0 --use-intensity 0
# python train.py --epochs 100 --batch-size 1 --classes 6 --data-path 'data/tongji_parking_rgbi_slice_splitted_truergbi' --model UNet_alpha --alpha 0.5 --scale 1.0 --use-intensity 1
python train.py --epochs 100 --batch-size 1 --classes 6 --data-path 'data/tongji_parking_rgbi_slice_splitted_truergbi' --model UNet_alpha --alpha 0.5 --scale 1.0 --use-intensity 1 --intensity-upsample-filter-size 15
