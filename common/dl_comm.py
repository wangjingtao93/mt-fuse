import os
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, auc, recall_score, precision_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from collections import OrderedDict

from common.eval import classification_report_with_specificity as cr
from common.net_enter.net_enter import net_enter



class dl_comm():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.imagenet_pre_path = '/data1/wangjingtao/workplace/python/pycharm_remote/result/meta-learning-classfication/result/result_20231010_sub10/pre_train/imagenet'

        self.fig_path = os.path.join(args.store_dir, 'figures')

        return None
    def _init_net(self):

        self.model = net_enter(self.args)
        # print(self.model)
        self.model.train()
        # self.model.cuda()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def _init_opt(self):
        self.modellr = 1e-4
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.modellr)


    def train(self, train_loader, epoch):
        self.model.train()
        sum_loss = 0
        total_num = len(train_loader.dataset)
        print(f'train_dataset len: {total_num}', f'train_loader len: {len(train_loader)}')
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data).to(self.device), Variable(target).to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print_loss = loss.data.item()
            sum_loss += print_loss
            if (batch_idx + 1) % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                        100. * (batch_idx + 1) / len(train_loader), loss.item()))

        ave_loss = sum_loss / len(train_loader)
        print('epoch:{},loss:{}'.format(epoch, ave_loss))
        return ave_loss


    def val(self, val_loader):
        self.model.eval()
        res = OrderedDict()
        val_loss = 0

        true_label_list = []
        pred_score_list = []

        total_num = len(val_loader.dataset)
        print(f'val_dataset len: {total_num}' , f'val_loader len: {len(val_loader)}')
        with torch.no_grad():
            for data, target in val_loader:
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                pred_score_list.append(output.data.cpu().detach().numpy())
                true_label_list.append(target.cpu().detach().numpy())

                print_loss = loss.data.item()
                val_loss += print_loss

            avgloss = val_loss / len(val_loader)

        y_true = np.concatenate(true_label_list)
        y_score = np.concatenate(pred_score_list)
        y_pred = np.argmax(y_score, axis=1)

        accuracy = accuracy_score(y_true,y_pred)
        # precision = precision_score(y_true, y_pred, average='weighted')
        # recall=recall_score(y_true, y_pred, average='weighted')
        # f1 = f1_score(y_true, y_pred, average='weighted')

        kappa = cohen_kappa_score(y_true, y_pred)

        if self.args.num_classes > 2:
            # prob_new = torch.nn.functional.softmax(torch.from_numpy(y_score), dim=1)
            # roc_auc = roc_auc_score(y_true, prob_new, average='weighted', multi_class='ovo')
            # y_true_bin = label_binarize(y_true, classes=np.arange(self.args.num_classes))
            # fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
        else:
            fpr, tpr, thresholds = roc_curve(y_true, y_score[:, 1])
            roc_auc = auc(fpr, tpr)

        # 特定类别，罕见病类别[0,1,2]
        spec_cls_sens= []
        if len(spec_cls_sens) != 0:
            sensitivity = recall_score(y_true, y_pred, labels=[0,1,2], average='weighted')
        else:
            sensitivity =  accuracy

        cm = confusion_matrix(y_true, y_pred)
        # class_report = classification_report(y_true, y_pred)
        class_report = cr(y_true, y_pred, cm=cm)
        precision = float(class_report.loc['weighted avg', 'precision'])
        recall = float(class_report.loc['weighted avg', 'recall'])
        f1 = float(class_report.loc['weighted avg', 'f1-score'])
        specificity = float(class_report.loc['weighted avg', 'specificity'])

        print('val++++++++')
        # print("accuracy: ", "%.4f"%accuracy)
        print("auc: ","%.4f"%roc_auc)
        # print("precision: ","%.4f"%precision)
        # print("recall: ","%.4f"%recall)
        # print("f1: ","%.4f"%f1)
        # print("specificity: ", "%.4f"%specificity)
        print("sensitivity: ", "%.4f"%sensitivity)
        print("average_loss: ","%.4f"%avgloss)
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", class_report)

        res['loss'] = round(avgloss,4)
        res['acc'] = round(accuracy,4)
        res['auc'] = round(roc_auc,4)
        res['prec'] = round(precision,4)
        res['recall'] =  round(recall,4)
        res['f1'] = round(f1,4)
        res['ka'] = round(kappa,4)
        res['sens'] = round(sensitivity,4)
        res['spec'] = round(specificity,4)
        res['cm'] = cm
        res['report'] = class_report
        res['y_true'] = y_true
        res['y_score'] = np.round(y_score, 6)
        res['fpr'] = fpr
        res['tpr'] = tpr

        self.model.train()
        return  res


    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
        self.modellrnew = self.modellr * (0.1 ** (epoch // 10))# ** 乘方
        print("lr:", self.modellrnew)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.modellrnew    


