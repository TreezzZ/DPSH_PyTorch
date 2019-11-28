import dpsh
import os

import argparse
import torch
from loguru import logger
from data.data_loader import load_data


def run():
    args = load_config()
    logger.add(os.path.join('logs', '{}_model_{}_codelength_{}_query_{}_train_{}_topk_{}_eta_{}.log'.format(
        args.dataset,
        args.arch,
        ','.join([str(c) for c in args.code_length]),
        args.num_query,
        args.num_train,
        args.topk,
        args.eta,
        )), rotation='500 MB', level='INFO')
    logger.info(args)
    torch.backends.cudnn.benchmark = True

    # Load dataset
    train_dataloader, query_dataloader, retrieval_dataloader = load_data(
        args.dataset,
        args.root,
        args.num_query,
        args.num_train,
        args.batch_size,
        args.num_workers,
    )

    # Training
    for code_length in args.code_length:
        logger.info('[code_length:{}]'.format(code_length))
        checkpoint = dpsh.train(
            train_dataloader,
            query_dataloader,
            retrieval_dataloader,
            args.arch,
            code_length,
            args.device,
            args.eta,
            args.lr,
            args.max_iter,
            args.topk,
            args.evaluate_interval,
        )
        torch.save(checkpoint, os.path.join('checkpoints', '{}_model_{}_codelength_{}_query_{}_train_{}_topk_{}_eta_{}.pt'.format(args.dataset, args.arch, code_length, args.num_query, args.num_train, args.topk, args.eta)))
        logger.info('[code_length:{}][map:{:.4f}]'.format(code_length, checkpoint['map']))


def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='DPSH_PyTorch')
    parser.add_argument('--dataset',
                        help='Dataset name.')
    parser.add_argument('--root',
                        help='Path of dataset')
    parser.add_argument('--num-query', default=1000, type=int,
                        help='Number of query data points.(default: 1000)')
    parser.add_argument('--arch', default='alexnet', type=str,
                        help='CNN model name.(default: alexnet)')
    parser.add_argument('--num-train', default=5000, type=int,
                        help='Number of training data points.(default: 5000)')
    parser.add_argument('--code-length', default='16,32,48,64', type=str,
                        help='Binary hash code length.(default: 16,32,48,64)')
    parser.add_argument('--topk', default=-1, type=int,
                        help='Calculate map of top k.(default: all)')
    parser.add_argument('--gpu', default=None, type=int,
                        help='Using gpu.(default: False)')
    parser.add_argument('--lr', default=1e-5, type=float,
                        help='learning rate(default: 1e-5)')
    parser.add_argument('--batch-size', default=512, type=int,
                        help='batch size(default: 512)')
    parser.add_argument('--max-iter', default=100, type=int,
                        help='Number of iterations.(default: 100)')
    parser.add_argument('--num-workers', default=6, type=int,
                        help='Number of loading data threads.(default: 6)')
    parser.add_argument('--evaluate-interval', default=10, type=int,
                        help='Evaluation interval(default: 10)')
    parser.add_argument('--eta', default=0.1, type=float,
                        help='Hyper-parameter.(default: 0.1)')

    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)
        torch.cuda.set_device(args.gpu)

    # Hash code length
    args.code_length = list(map(int, args.code_length.split(',')))

    return args


if __name__ == "__main__":
    run()
