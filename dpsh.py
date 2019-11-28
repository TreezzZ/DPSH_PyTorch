import torch
import torch.optim as optim
import time

from torch.optim.lr_scheduler import CosineAnnealingLR
from models.model_loader import load_model
from loguru import logger
from models.dpsh_loss import DPSHLoss
from utils.evaluate import mean_average_precision


def train(
        train_dataloader,
        query_dataloader,
        retrieval_dataloader,
        arch,
        code_length,
        device,
        eta,
        lr,
        max_iter,
        topk,
        evaluate_interval,
):
    """
    Training model.

    Args
        train_dataloader, query_dataloader, retrieval_dataloader(torch.utils.data.dataloader.DataLoader): Data loader.
        arch(str): CNN model name.
        code_length(int): Hash code length.
        device(torch.device): GPU or CPU.
        eta(float): Hyper-parameter.
        lr(float): Learning rate.
        max_iter(int): Number of iterations.
        topk(int): Calculate map of top k.
        evaluate_interval(int): Evaluation interval.

    Returns
        checkpoint(dict): Checkpoint.
    """
    # Create model, optimizer, criterion, scheduler
    model = load_model(arch, code_length).to(device)
    criterion = DPSHLoss(eta)
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5,
        )
    scheduler = CosineAnnealingLR(optimizer, max_iter, 1e-7)

    # Initialization
    N = len(train_dataloader.dataset)
    U = torch.zeros(N, code_length).to(device)
    train_targets = train_dataloader.dataset.get_onehot_targets().to(device)

    # Training
    best_map = 0.0
    iter_time = time.time()
    for it in range(max_iter):
        model.train()
        running_loss = 0.
        iter_time = time.time()
        for data, targets, index in train_dataloader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()

            S = (targets @ train_targets.t() > 0).float()
            U_cnn = model(data)
            U[index, :] = U_cnn.data
            loss = criterion(U_cnn, U, S)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        # Evaluate
        if it % evaluate_interval == evaluate_interval-1:
            iter_time = time.time() - iter_time

            # Generate hash code and one-hot targets
            query_code = generate_code(model, query_dataloader, code_length, device)
            query_targets = query_dataloader.dataset.get_onehot_targets()
            retrieval_code = generate_code(model, retrieval_dataloader, code_length, device)
            retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets()

            # Compute map
            mAP = mean_average_precision(
                query_code.to(device),
                retrieval_code.to(device),
                query_targets.to(device),
                retrieval_targets.to(device),
                device,
                topk,
            )

            # Save checkpoint
            if best_map < mAP:
                best_map = mAP
                checkpoint = {
                    'qB': query_code,
                    'qL': query_targets,
                    'rB': retrieval_code,
                    'rL': retrieval_targets,
                    'model': model.state_dict(),
                    'map': best_map,
                }
            logger.debug('[iter:{}/{}][loss:{:.2f}][map:{:.4f}][time:{:.2f}]'.format(
                it+1,
                max_iter,
                running_loss,
                mAP,
                iter_time,
            ))
            iter_time = time.time()

    return checkpoint


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code

    Args
        dataloader(torch.utils.data.dataloader.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.

    Returns
        code(torch.Tensor, n*code_length): Hash code.
    """
    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, index in dataloader:
            data = data.to(device)
            hash_code = model(data)
            code[index, :] = hash_code.sign().cpu()

    model.train()
    return code
