# faster_rcnn_pytorch
Faster R-CNN implementation based on PyTorch higher version
# run example

## train

```
python train.py --train_root dataset/train --test_root dataset/test --save_path ./model.pt
```

## detect

```
python detect.py --weight model.pt --source dataset/detect --thickness 2
```
