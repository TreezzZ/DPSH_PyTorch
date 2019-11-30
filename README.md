# Feature Learning based Deep Supervised Hashing with Pairwise Labels

## REQUIREMENTS
1. pytorch
2. loguru

`pip install -r requirements.txt`

## DATASETS
1. [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
2. [NUS-WIDE](https://pan.baidu.com/s/1f9mKXE2T8XpIq8p7y8Fa6Q) Password: uhr3
3. [Imagenet100](https://pan.baidu.com/s/1Vihhd2hJ4q0FOiltPA-8_Q) Password: ynwf

## USAGE
```
usage: run.py [-h] [--dataset DATASET] [--root ROOT] [--num-query NUM_QUERY]
              [--arch ARCH] [--num-train NUM_TRAIN]
              [--code-length CODE_LENGTH] [--topk TOPK] [--gpu GPU] [--lr LR]
              [--batch-size BATCH_SIZE] [--max-iter MAX_ITER]
              [--num-workers NUM_WORKERS]
              [--evaluate-interval EVALUATE_INTERVAL] [--eta ETA]

DPSH_PyTorch

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Dataset name.
  --root ROOT           Path of dataset
  --num-query NUM_QUERY
                        Number of query data points.(default: 1000)
  --arch ARCH           CNN model name.(default: alexnet)
  --num-train NUM_TRAIN
                        Number of training data points.(default: 5000)
  --code-length CODE_LENGTH
                        Binary hash code length.(default: 12,24,32,48)
  --topk TOPK           Calculate map of top k.(default: all)
  --gpu GPU             Using gpu.(default: False)
  --lr LR               learning rate(default: 1e-5)
  --batch-size BATCH_SIZE
                        batch size(default: 128)
  --max-iter MAX_ITER   Number of iterations.(default: 150)
  --num-workers NUM_WORKERS
                        Number of loading data threads.(default: 6)
  --evaluate-interval EVALUATE_INTERVAL
                        Evaluation interval(default: 10)
  --eta ETA             Hyper-parameter.(default: 0.1)
```

## EXPERIMENTS
CNN model: Alexnet. Compute mean average precision(MAP).

cifar10: 1000 query images, 5000 training images.

nus-wide-tc21: 21 classes, 2100 query images, 10500 training images.

imagenet100: 100 classes, 5000 query images, 10000 training images.

 bits | 12 | 16 | 24 | 32 | 48 | 64 | 128
   :-:   |  :-:    |   :-:   |   :-:   |   :-:   |   :-:   |   :-:   |   :-:     
cifar10@ALL | 0.6676 | 0.7131 | 0.7118 | 0.7362 | 0.7487 | 0.7542 | 0.7565
nus-wide-tc21@5000 | 0.8091 | 0.8188 | 0.8346 | 0.8403 | 0.8450 | 0.8503 |0.8588
imagenet100@1000 | 0.1985 | 0.2497 | 0.3654 | 0.4147 | 0.4612 | 0.4950 | 0.5687
