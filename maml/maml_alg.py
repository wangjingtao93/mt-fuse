from torch import nn
import numpy as np
from tqdm import tqdm
import torch
import higher
from copy import deepcopy

from sklearn.metrics import confusion_matrix, auc, roc_curve, recall_score, precision_score, f1_score, cohen_kappa_score,accuracy_score, roc_auc_score,classification_report
from common.meta.gbml import GBML
from common.eval import classification_report_with_specificity as cr

class MAML(GBML):
    def __init__(self, args):
        super().__init__(args)
        self._init_net()
        self._init_opt()
        self.args=args

    def train(self, db, epoch):
        self.network.train()
        criterion = nn.CrossEntropyLoss()

        qry_acc_all_batch = []# 所有batch tasks的query set 的acc
        qry_loss_all_tasks = []
        tqdm_train = tqdm(db)
        for batch_idx, (x_spt, y_spt, x_qry, y_qry) in enumerate(tqdm_train, 1):
            support_x, support_y, query_x, query_y = x_spt.to(self.args.device), y_spt.to(self.args.device), x_qry.to(self.args.device), y_qry.to(self.args.device)
            task_num, setsz, c_, h, w = support_x.size()
            querysz = query_x.size(1)

            # Initialize the inner optimizer to adapt the parameters to
            # the support set.
            # 每一个epoch都实例化一个，不用从上一个epoch开始
            # 为什么呢
            inner_opt = torch.optim.SGD(self.network.parameters(), lr=self.args.inner_lr)

            qry_losses = []
            qry_acc_list=[] # 存储一个batch task，query set的acc
            self.outer_optimizer.zero_grad()
            for i in range(task_num):
                with higher.innerloop_ctx(self.network, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
                    # Optimize the likelihood of the support set by taking
                    # gradient steps w.r.t. the model's parameters.
                    # This adapts the model's meta-parameters to the task.
                    # higher is able to automatically keep copies of
                    # your network's parameters as they are being updated.
                    for _ in range(self.args.n_inner):
                        spt_logits = fnet(support_x[i])
                        spt_loss = criterion(spt_logits, support_y[i])

                        diffopt.step(spt_loss)
                    # The final set of adapted parameters will induce some
                    # final loss and accuracy on the query dataset.
                    # These will be used to update the model's meta-parameters.
                    qry_logits = fnet(query_x[i])
                    qry_loss = criterion(qry_logits, query_y[i])
                    qry_losses.append(qry_loss.detach().cpu())

                    logits = qry_logits.detach().cpu()
                    labels = query_y[i].detach().cpu()
                    _, pred = torch.max(logits.data, 1)
                    acc = accuracy_score(pred, labels)
                    qry_acc_list.append(round(acc,4))

                    # Update the model's meta-parameters to optimize the query
                    # losses across all of the tasks sampled in this batch.
                    # This unrolls through the gradient steps.
                    qry_loss.backward()  # 为什么要在这呢
            self.outer_optimizer.step()
            qry_losses = round((sum(qry_losses) / task_num).item(),4)
            # qry_dscs = 100. * sum(qry_dscs) / task_num # .* 和*有什么区别吗？没有吧
            acc_ave = round(sum(qry_acc_list) / task_num,4)

            qry_acc_all_batch.append(acc_ave)
            qry_loss_all_tasks.append(qry_losses)

            tqdm_train.set_description(f'Meta Training_Tasks Epoch {epoch}, batch_idx {batch_idx}, acc={acc_ave}, , Loss={qry_losses}')
        ave_qry_acc_all_batch = round(np.array(qry_acc_all_batch).mean(), 4)
        ave_qry_loss_all_batch = round(np.array(qry_loss_all_tasks).mean(), 4)

        # 返回：1.所有batch tasks的query set 的acc的均值. 2.最以后一个batch task的query set的acc 3.所有任务平均loss
        return ave_qry_acc_all_batch, round(qry_acc_all_batch[-1],4), ave_qry_loss_all_batch

    # with higher
    def val_withgher(self, db, epoch):
        self.network.train()
        criterion = nn.CrossEntropyLoss()

        qry_losses = []
        qry_acc_list=[] # 所有测试任务
        qry_acc_inner = [0 for i in range(self.args.n_inner_meta_test)] # 任务内循环每一次梯度更新，均值

        qry_sens_inner = [0 for i in range(self.args.n_inner_meta_test)] # 任务内循环每一次梯度更新，均值

        true_label_list = []
        pred_score_list = []

        tqdm_val = tqdm(db)
        for batch_idx, (x_spt, y_spt, x_qry, y_qry) in enumerate(tqdm_val, 1):
            support_x, support_y, query_x, query_y = x_spt.to(self.args.device), y_spt.to(self.args.device), x_qry.to(self.args.device), y_qry.to(self.args.device)
            task_num, setsz, c_, h, w = support_x.size()
            querysz = query_x.size(1)

            n_inner_iter = self.args.n_inner_meta_test
            inner_opt = torch.optim.SGD(self.network.parameters(), lr=self.args.inner_lr)
            # track_higher_grads=False,随着innerloop加深，不会增加内存使用
            for i in range(task_num):
                labels = query_y[i].detach().cpu()
                pred_score_inner = [0 for i in range(n_inner_iter)]
                with higher.innerloop_ctx( self.network, inner_opt, track_higher_grads=False) as (fnet, diffopt):

                    # Optimize the likelihood of the support set by taking
                    # gradient steps w.r.t. the model's parameters.
                    # This adapts the model's meta-parameters to the task.
                    for k in range(n_inner_iter):

                        spt_logits = fnet(support_x[i])
                        spt_loss = criterion(spt_logits, support_y[i])
                        diffopt.step(spt_loss)

                        # 是否要置成不记录梯度
                        qry_logits = fnet(query_x[i])
                        logits = qry_logits.detach().cpu()

                        pred_score_inner[k] = logits.numpy()

                        _, pred = torch.max(logits.data, 1)
                        acc = accuracy_score(labels, pred)
                        qry_acc_inner[k] += acc

                        if self.args.datatype == 'oct' or self.args.datatype =='mini_imagenet':
                            sens=recall_score(labels, pred, labels=[0,1,2], average='weighted')
                        else:
                            sens = recall_score(labels, pred, average='weighted')

                        qry_sens_inner[k] += sens

                    pred_score_list.append(pred_score_inner)
                    true_label_list.append(labels.numpy())

                    # The query loss and acc induced by these parameters.
                    # qry_logits = fnet(query_x[i]).detach() #放到循环里，记录每一次梯度更新
                    qry_loss = criterion(qry_logits, query_y[i])
                    qry_losses.append(qry_loss.detach().cpu()) 

                    qry_acc_list.append(acc)
                    # qry_sens_list.append(sens)

            tqdm_val.set_description('Meta_Val_Tasks Epoch {}, acc={:.4f}, queryloss {:.4f}'.format(epoch, np.mean(qry_acc_list), np.mean(np.array(qry_losses))))

        # acc_ave_last_inner = round(np.array(qry_acc_list).mean(),4)
        # # std = np.array(qry_acc_list).std()

        ave_qry_loss_all_batch = round(np.array(qry_losses).mean(), 4)

        ave_acc_all_task_inner = list(map(lambda x: round(x/self.args.n_val_tasks, 4),qry_acc_inner))

        best_acc =max(ave_acc_all_task_inner)
        best_acc_inner = ave_acc_all_task_inner.index(best_acc)
        pred_score_relative_acc = []
        for i in range(len(pred_score_list)):
            pred_score_relative_acc.append(pred_score_list[i][best_acc_inner])

        ave_sens_all_task_inner = list(map(lambda x: round(x/self.args.n_val_tasks, 4),qry_sens_inner))
        best_sens = max(ave_sens_all_task_inner)
        best_sens_innner = ave_sens_all_task_inner.index(best_sens)

        y_true = np.concatenate(true_label_list)
        y_score_acc = np.concatenate(pred_score_relative_acc)
        y_pred_acc = np.argmax(y_score_acc, axis=1)

        cm_acc = confusion_matrix(y_true, y_pred_acc)
        class_report_acc = cr(y_true, y_pred_acc,cm=cm_acc)

        prob_new = torch.nn.functional.softmax(torch.from_numpy(y_score_acc), dim=1)
        roc_auc_acc = roc_auc_score(y_true, prob_new, average='weighted', multi_class='ovo')


        # precision_acc = precision_score(y_true, y_pred_acc, average='weighted')
        # f1_acc = f1_score(y_true, y_pred_acc, average='weighted')
        # specificity_acc = 0

        # precision_acc = class_report_acc.loc['weighted avg', 'precision']
        # f1_acc = class_report_acc.loc['weighted avg', 'f1-score']
        # specificity_acc = class_report_acc.loc['weighted avg', 'specificity']
        # sens_acc = class_report_acc.loc['weighted avg', 'sens']



        pred_score_relative_sens = []
        if best_sens_innner == best_acc_inner:
            pred_score_relative_sens = pred_score_relative_acc
            cm_sens = cm_acc
            class_report_sens = class_report_acc
            roc_auc_sens = roc_auc_acc

        else:
            for i in range(len(pred_score_list)):
                pred_score_relative_sens.append(pred_score_list[i][best_sens_innner])
            y_score_sens = np.concatenate(pred_score_relative_sens)
            y_pred_sens = np.argmax(y_score_sens, axis=1)

            cm_sens = confusion_matrix(y_true, y_pred_sens)
            # class_report_sens = classification_report(y_true, y_pred_sens,cm=cm_sens)
            class_report_sens = cr(y_true, y_pred_sens,cm=cm_sens)
            prob_new = torch.nn.functional.softmax(torch.from_numpy(y_score_sens), dim=1)
            roc_auc_sens = roc_auc_score(y_true, prob_new, average='weighted', multi_class='ovo')


        return [ave_acc_all_task_inner, ave_sens_all_task_inner, ave_qry_loss_all_batch, cm_acc, class_report_acc, cm_sens, class_report_sens, best_acc, best_acc_inner, best_sens, best_sens_innner, round(roc_auc_acc,4), round(roc_auc_sens,4)]

    # without higher
    def val(self, db, epoch):
        net_val = deepcopy(self.network)
        for name, param in net_val.named_parameters():
            param.requires_grad  = True
        ini_weights = deepcopy(net_val.state_dict())
        criterion = nn.CrossEntropyLoss()

        qry_losses = []
        qry_acc_list=[] # 所有测试任务
        qry_acc_inner = [0 for i in range(self.args.n_inner_meta_test)] # 任务内循环每一次梯度更新，均值

        qry_sens_inner = [0 for i in range(self.args.n_inner_meta_test)] # 任务内循环每一次梯度更新，均值

        true_label_list = []
        pred_score_list = []

        tqdm_val = tqdm(db)
        for batch_idx, (x_spt, y_spt, x_qry, y_qry) in enumerate(tqdm_val, 1):
            support_x, support_y, query_x, query_y = x_spt.to(self.args.device), y_spt.to(self.args.device), x_qry.to(self.args.device), y_qry.to(self.args.device)
            task_num, setsz, c_, h, w = support_x.size()
            querysz = query_x.size(1)

            n_inner_iter = self.args.n_inner_meta_test
            inner_opt = torch.optim.SGD(net_val.parameters(), lr=self.args.inner_lr)
            # track_higher_grads=False,随着innerloop加深，不会增加内存使用
            for i in range(task_num):
                net_val.load_state_dict(deepcopy(ini_weights))# 不用deepcopy是万万不行的，否则ini_weights会随net_val发生变化
                labels = query_y[i].detach().cpu()
                pred_score_inner = [0 for i in range(n_inner_iter)]

                # Optimize the likelihood of the support set by taking
                # gradient steps w.r.t. the model's parameters.
                # This adapts the model's meta-parameters to the task.
                for k in range(n_inner_iter):
                    net_val.train()

                    spt_logits = net_val(support_x[i])
                    spt_loss = criterion(spt_logits, support_y[i])
                    inner_opt.zero_grad()
                    spt_loss.backward()
                    inner_opt.step()

                    net_val.eval()
                    with torch.no_grad():
                        qry_logits = net_val(query_x[i])
                        logits = qry_logits.detach().cpu()

                        pred_score_inner[k] = logits.numpy()

                        _, pred = torch.max(logits.data, 1)
                        acc = accuracy_score(labels, pred)
                        qry_acc_inner[k] += acc

                        if self.args.datatype == 'oct' or self.args.datatype =='mini_imagenet':
                            sens=recall_score(labels, pred, labels=[0,1,2], average='weighted')
                        else:
                            sens = recall_score(labels, pred, average='weighted')

                        qry_sens_inner[k] += sens

                pred_score_list.append(pred_score_inner)
                true_label_list.append(labels.numpy())

                # The query loss and acc induced by these parameters.
                # qry_logits = fnet(query_x[i]).detach() #放到循环里，记录每一次梯度更新
                qry_loss = criterion(qry_logits, query_y[i])
                qry_losses.append(qry_loss.detach().cpu())

                qry_acc_list.append(acc)
                # qry_sens_list.append(sens)

            tqdm_val.set_description('Meta_Val_Tasks Epoch {}, acc={:.4f}, queryloss {:.4f}'.format(epoch, np.mean(qry_acc_list), np.mean(np.array(qry_losses))))

        # acc_ave_last_inner = round(np.array(qry_acc_list).mean(),4)
        # # std = np.array(qry_acc_list).std()
        del net_val

        ave_qry_loss_all_batch = round(np.array(qry_losses).mean(), 4)

        ave_acc_all_task_inner = list(map(lambda x: round(x/self.args.n_val_tasks, 4),qry_acc_inner))

        best_acc =max(ave_acc_all_task_inner)
        best_acc_inner = ave_acc_all_task_inner.index(best_acc)
        pred_score_relative_acc = []
        for i in range(len(pred_score_list)):
            pred_score_relative_acc.append(pred_score_list[i][best_acc_inner])

        ave_sens_all_task_inner = list(map(lambda x: round(x/self.args.n_val_tasks, 4),qry_sens_inner))
        best_sens = max(ave_sens_all_task_inner)
        best_sens_innner = ave_sens_all_task_inner.index(best_sens)

        y_true = np.concatenate(true_label_list)
        y_score_acc = np.concatenate(pred_score_relative_acc)
        y_pred_acc = np.argmax(y_score_acc, axis=1)

        cm_acc = confusion_matrix(y_true, y_pred_acc)
        class_report_acc = cr(y_true, y_pred_acc,cm=cm_acc)

        prob_new = torch.nn.functional.softmax(torch.from_numpy(y_score_acc), dim=1)
        roc_auc_acc = roc_auc_score(y_true, prob_new, average='weighted', multi_class='ovo')


        # precision_acc = precision_score(y_true, y_pred_acc, average='weighted')
        # f1_acc = f1_score(y_true, y_pred_acc, average='weighted')
        # specificity_acc = 0

        # precision_acc = class_report_acc.loc['weighted avg', 'precision']
        # f1_acc = class_report_acc.loc['weighted avg', 'f1-score']
        # specificity_acc = class_report_acc.loc['weighted avg', 'specificity']
        # sens_acc = class_report_acc.loc['weighted avg', 'sens']



        pred_score_relative_sens = []
        if best_sens_innner == best_acc_inner:
            pred_score_relative_sens = pred_score_relative_acc
            cm_sens = cm_acc
            class_report_sens = class_report_acc
            roc_auc_sens = roc_auc_acc

        else:
            for i in range(len(pred_score_list)):
                pred_score_relative_sens.append(pred_score_list[i][best_sens_innner])
            y_score_sens = np.concatenate(pred_score_relative_sens)
            y_pred_sens = np.argmax(y_score_sens, axis=1)

            cm_sens = confusion_matrix(y_true, y_pred_sens)
            # class_report_sens = classification_report(y_true, y_pred_sens,cm=cm_sens)
            class_report_sens = cr(y_true, y_pred_sens,cm=cm_sens)
            prob_new = torch.nn.functional.softmax(torch.from_numpy(y_score_sens), dim=1)
            roc_auc_sens = roc_auc_score(y_true, prob_new, average='weighted', multi_class='ovo')


        return [ave_acc_all_task_inner, ave_sens_all_task_inner, ave_qry_loss_all_batch, cm_acc, class_report_acc, cm_sens, class_report_sens, best_acc, best_acc_inner, best_sens, best_sens_innner, round(roc_auc_acc,4), round(roc_auc_sens,4)]

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
        modellrnew = self.out_lr * (0.1 ** (epoch // 30))# ** 乘方
        print("lr:", modellrnew)
        for param_group in self.meta_opt.param_groups:
            param_group['lr'] = modellrnew

def unpack_batch(batch):

    device = torch.device('cuda')
    train_inputs, train_targets = batch[0]
    train_inputs = train_inputs.to(device=device)
    train_targets = train_targets.to(device=device, dtype=torch.long)

    test_inputs, test_targets = batch[1]
    test_inputs = test_inputs.to(device=device)
    test_targets = test_targets.to(device=device, dtype=torch.long)

    return train_inputs, train_targets, test_inputs, test_targets 