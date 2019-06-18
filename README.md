# Feature Learning based Deep Supervised Hashing with Pairwise Labels

论文[Feature Learning based Deep Supervised Hashing with Pairwise Labels](http://202.119.32.195/cache/1/03/cs.nju.edu.cn/01c07b4c0cb0161455ace83be60f9ffc/IJCAI16_DPSH.pdf)

## Requirements
1. pytorch 1.1
2. loguru

## 运行
`python run.py --dataset cifar10 --data-path <data_path> --code-length 64 `

日志记录在`logs`文件夹内

生成的hash code保存在`result`文件夹内，Tensor形式保存

## 参数说明
```
usage: run.py [-h] [--dataset DATASET] [--data-path DATA_PATH]
              [--num-query NUM_QUERY] [--num-train NUM_TRAIN]
              [--code-length CODE_LENGTH] [--model MODEL] [--multi-gpu]
              [--gpu GPU] [--lr LR] [--batch-size BATCH_SIZE]
              [--epochs EPOCHS] [--num-workers NUM_WORKERS] [--eta ETA]
              [--server SERVER] [--port PORT]

DPSH_PyTorch

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     dataset used to train (default: cifar10)
  --data-path DATA_PATH
                        path of cifar10 dataset
  --num-query NUM_QUERY
                        number of query(default: 1000)
  --num-train NUM_TRAIN
                        number of train(default: 5000)
  --code-length CODE_LENGTH
                        hyper-parameter: binary hash code length (default: 12)
  --model MODEL         CNN model(default: alexnet
  --multi-gpu           use multiple gpu
  --gpu GPU             use gpu(default: 0. -1: use cpu)
  --lr LR               learning rate(default: 1e-3)
  --batch-size BATCH_SIZE
                        batch size(default: 64)
  --epochs EPOCHS       epochs(default:64)
  --num-workers NUM_WORKERS
                        number of workers(default: 4)
  --eta ETA             hyper-parameter: regularization term (default: 50)
  --server SERVER       visdom server address
  --port PORT           visdom server port

```

## 实验
### CIFAR10
![map](Figure_1.png)

### NUS-WIDE