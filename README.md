# Feature Learning based Deep Supervised Hashing with Pairwise Labels

## REQUIREMENTS
1. pytorch
2. loguru
`pip install -r requirements.txt`

## DATASETS
1. [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
2. [NUS-WIDE](https://pan.baidu.com/s/1f9mKXE2T8XpIq8p7y8Fa6Q) Password: uhr3

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
                        Binary hash code length.(default: 16,32,48,64)
  --topk TOPK           Calculate map of top k.(default: all)
  --gpu GPU             Using gpu.(default: False)
  --lr LR               learning rate(default: 1e-5)
  --batch-size BATCH_SIZE
                        batch size(default: 512)
  --max-iter MAX_ITER   Number of iterations.(default: 150)
  --num-workers NUM_WORKERS
                        Number of loading data threads.(default: 6)
  --evaluate-interval EVALUATE_INTERVAL
                        Evaluation interval(default: 10)
  --eta ETA             Hyper-parameter.(default: 10)
```

## Experiments
cifar10-5000: 1000 query images, 5000 training images.

nus-wide: 2100 query images, 10500 training images.

计算top 5000 mAP，跑3次，取平均

 bits | 12 | 24 | 32 | 48  
   :-:   |  :-:    |   :-:   |   :-:   |   :-:     
cifar10-5000 mAP | 0.7277 | 0.7649 | 0.7550 | 0.7684
