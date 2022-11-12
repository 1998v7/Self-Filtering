
# Self-Filtering

Official PyTorch Implementation of Self-Filtering. 

> Paper "Self-Filtering: A Noise-Aware Sample Selection for Label Noise with Confidence Penalization" is accepted to **ECCV 2022**.

# Training

### Hyper-parameter and settings

`k`  denotes memory bank size. It can be set as `[2,3,4]`

`T`  denotes threshold in confidence penalty. For all experiment, we set it as `0.2`

For CIFAR-10, `warm_up = 10`,`model = resnet18`

For CIFAR-100, `warm_up = 30`,`model = resnet34`

### Run

```
python main.py --dataset cifar10 --model resnet18 --batch_size 32 --lr 0.02 --warm_up 10 --num_epochs 100 --noise_mode instance --r 0.2 --k 2 --T 0.2 --gpuid 0
```

> Note that the code refers to DivideMix. 

### Contact
If you have any problem about our code, feel free to contact 1998v7@gmail.com

### Cite
If you find the code useful, please consider citing our paper:
```
@inproceedings{wei2022self,
  title={Self-Filtering: A Noise-Aware Sample Selection for Label Noise with Confidence Penalization},
  author={Wei, Qi and Sun, Haoliang and Lu, Xiankai and Yin, Yilong},
  booktitle={European Conference on Computer Vision},
  pages={516--532},
  year={2022},
  organization={Springer}
}
```
