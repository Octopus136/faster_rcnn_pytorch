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

# TODO

## Evaluations

1. Ensure that evaluation indicators are working properly;

## Features

1. Add test code;
2. Store the first few epochs of plots for observation during training;
3. Support for backbone replacement
