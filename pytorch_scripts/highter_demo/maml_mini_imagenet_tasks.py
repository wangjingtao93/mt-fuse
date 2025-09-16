
import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication')
from tqdm import tqdm
import argparse
import time
import typing

import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import higher

from MiniImagenet import MiniImagenet

from conv_84 import Learner, Conv84

from common.data_enter import  mini_imagenet_enter


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument(
        '--k_shot', type=int, help='k shot for support set', default=5)
    argparser.add_argument(
        '--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument(
        '--task_num',
        type=int,
        help='meta batch size, namely task num',
        default=4)
    argparser.add_argument('--seed', type=int, help='random seed', default=1)

    argparser.add_argument('--datatype', type=str, default='mini_imagenet')

    args = argparser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Set up the Omniglot loader.
    device = torch.device('cuda')
    # db = OmniglotNShot(
    #     '/data1/wangjingtao/workplace/python/data/classification/omniglot/python',
    #     batchsz=args.task_num,
    #     n_way=args.n_way,
    #     k_shot=args.k_spt,
    #     k_query=args.k_qry,
    #     imgsz=28,
    #     device=device,
    # )

    # batchsz here means total episode number
    # mini = MiniImagenet('/data1/wangjingtao/workplace/python/data/meta-oct/classic/mini-imagenet/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
    #                     k_query=args.k_qry,
    #                     batchsz=1000, resize=84)
    # mini_test = MiniImagenet('/data1/wangjingtao/workplace/python/data/meta-oct/classic/mini-imagenet/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
    #                          k_query=args.k_qry,
    #                          batchsz=100, resize=84)
    
    # 我的任务组件方式
    args.project_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication'
    args.n_train_tasks = 1000
    args.n_val_tasks = 100
    args.meta_size = 4
    args.resize = 84
    final_test, meta_train, meta_test= mini_imagenet_enter(args)


    # Create a vanilla PyTorch neural network that will be
    # automatically monkey-patched by higher later.
    # Before higher, models could *not* be created like this
    # and the parameters needed to be manually updated and copied
    # for the updates.
    # net = nn.Sequential(
    #     nn.Conv2d(1, 64, 3),
    #     nn.BatchNorm2d(64, momentum=1, affine=True),
    #     nn.ReLU(inplace=True),
    #     nn.MaxPool2d(2, 2),
    #     nn.Conv2d(64, 64, 3),
    #     nn.BatchNorm2d(64, momentum=1, affine=True),
    #     nn.ReLU(inplace=True),
    #     nn.MaxPool2d(2, 2),
    #     nn.Conv2d(64, 64, 3),
    #     nn.BatchNorm2d(64, momentum=1, affine=True),
    #     nn.ReLU(inplace=True),
    #     nn.MaxPool2d(2, 2),
    #     Flatten(),
    #     nn.Linear(64, args.n_way)).to(device)

    # net = Conv84(args.n_way).to(device)
    net = Learner(args.n_way).to(device)

    # We will use Adam to (meta-)optimize the initial parameters
    # to be adapted.
    meta_opt = optim.Adam(net.parameters(), lr=1e-3)

    log = []
    for epoch in range(100):
        
        train(meta_train, net, device, meta_opt, epoch, log)

        
        test(meta_test, net, device, epoch, log)
        plot(log)

def unpack_batch(batch):

    device = torch.device('cuda')
    train_inputs, train_targets = batch[0]
    train_inputs = train_inputs.to(device=device)
    # train_targets = train_targets.to(device=device, dtype=torch.long)
    train_targets = train_targets.to(device=device)

    test_inputs, test_targets = batch[1]
    test_inputs = test_inputs.to(device=device)
    # test_targets = test_targets.to(device=device, dtype=torch.long)
    test_targets = test_targets.to(device=device)

    return train_inputs, train_targets, test_inputs, test_targets 

def train(db, net, device, meta_opt, epoch, log):

    net.train()
    # tqdm_train = tqdm(db)
    # for batch_idx, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
    for batch_idx, (x_spt, y_spt, x_qry, y_qry) in enumerate(db, 1):
        start_time = time.time()

        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        # TODO: Maybe pull this out into a separate module so it
        # doesn't have to be duplicated between `train` and `test`?

        # Initialize the inner optimizer to adapt the parameters to
        # the suppor    t set.
        n_inner_iter = 5
        inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)

        qry_losses = []
        qry_accs = []
        meta_opt.zero_grad()
        for i in range(task_num):
            # track_higher_grads = False会导致不收敛
            # with higher.innerloop_ctx(
            #     net, inner_opt, copy_initial_weights=False, track_higher_grads=False
            # ) as (fnet, diffopt):
            with higher.innerloop_ctx(
                net, inner_opt, copy_initial_weights=False
            ) as (fnet, diffopt):

            # 也会不收敛
            # with higher.innerloop_ctx(
            #         net, inner_opt, track_higher_grads=False
            # ) as (fnet, diffopt):
                # Optimize the likelihood of the support set by taking
                # gradient steps w.r.t. the model's parameters.
                # This adapts the model's meta-parameters to the task.
                # higher is able to automatically keep copies of
                # your network's parameters as they are being updated.
                for _ in range(n_inner_iter):
                    spt_logits = fnet(x_spt[i])
                    spt_loss = F.cross_entropy(spt_logits, y_spt[i].long())
                    diffopt.step(spt_loss)

                # The final set of adapted parameters will induce some
                # final loss and accuracy on the query dataset.
                # These will be used to update the model's meta-parameters.
                qry_logits = fnet(x_qry[i])
                qry_loss = F.cross_entropy(qry_logits, y_qry[i].long())
                qry_losses.append(qry_loss.detach())
                qry_acc = (qry_logits.argmax(
                    dim=1) == y_qry[i]).sum().item() / querysz
                qry_accs.append(qry_acc)

                # Update the model's meta-parameters to optimize the query
                # losses across all of the tasks sampled in this batch.
                # This unrolls through the gradient steps.
                qry_loss.backward() # 为什么要在这呢

        meta_opt.step()
        qry_losses = sum(qry_losses) / task_num
        qry_accs = 100. * sum(qry_accs) / task_num
        # i = epoch + float(batch_idx) / n_train_iter
        iter_time = time.time() - start_time
        if batch_idx % 4 == 0:
            print(
                f'[Epoch {epoch} Batch_idx {batch_idx}]  Train Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f} | Time: {iter_time:.2f}'
            )

        log.append({
            'epoch': epoch,
            'batcn_idx': batch_idx,
            'loss': qry_losses,
            'acc': qry_accs,
            'mode': 'train',
            'time': time.time(),
        })

def test(db, net, device, epoch, log):
    # Crucially in our testing procedure here, we do *not* fine-tune
    # the model during testing for simplicity.
    # Most research papers using MAML for this task do an extra
    # stage of fine-tuning here that should be added if you are
    # adapting this code for research.
    net.train()
    
    qry_losses = []
    qry_accs = []

    
    # for batch_idx, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
    # tqdm_test = tqdm(db)
    for batch_idx, (x_spt, y_spt, x_qry, y_qry)  in enumerate(db, 1):
        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        # TODO: Maybe pull this out into a separate module so it
        # doesn't have to be duplicated between `train` and `test`?
        n_inner_iter = 10
        inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)

        for i in range(task_num):
            with higher.innerloop_ctx(net, inner_opt, track_higher_grads=False) as (fnet, diffopt):
                # Optimize the likelihood of the support set by taking
                # gradient steps w.r.t. the model's parameters.
                # This adapts the model's meta-parameters to the task.
                for _ in range(n_inner_iter):
                    spt_logits = fnet(x_spt[i])
                    spt_loss = F.cross_entropy(spt_logits, y_spt[i].long())
                    diffopt.step(spt_loss)

                # The query loss and acc induced by these parameters.
                qry_logits = fnet(x_qry[i]).detach()
                qry_loss = F.cross_entropy(
                    qry_logits, y_qry[i].long(), reduction='none')
                qry_losses.append(qry_loss.detach())
                qry_accs.append(
                    (qry_logits.argmax(dim=1) == y_qry[i]).detach())

    qry_losses = torch.cat(qry_losses).mean().item()
    qry_accs = 100. * torch.cat(qry_accs).float().mean().item()
    print(
        f'[Epoch {epoch+1:.2f}] Test Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f}'
    )
    log.append({
        'epoch': epoch + 1,
        'loss': qry_losses,
        'acc': qry_accs,
        'mode': 'test',
        'time': time.time(),
    })




def plot(log):
    # Generally you should pull your plotting code out of your training
    # script but we are doing it here for brevity.
    df = pd.DataFrame(log)

    fig, ax = plt.subplots(figsize=(6, 4))
    train_df = df[df['mode'] == 'train']
    test_df = df[df['mode'] == 'test']
    ax.plot(train_df['epoch'], train_df['acc'], label='Train')
    ax.plot(test_df['epoch'], test_df['acc'], label='Test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(70, 100)
    fig.legend(ncol=2, loc='lower right')
    fig.tight_layout()
    fname = 'maml-accs.png'
    print(f'--- Plotting accuracy to {fname}')
    fig.savefig(fname)
    plt.close(fig)


# Won't need this after this PR is merged in:
# https://github.com/pytorch/pytorch/pull/22245
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


if __name__ == '__main__':
    main()
