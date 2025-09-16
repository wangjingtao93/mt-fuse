from torch import nn 
import numpy as np
from tqdm import tqdm
import torch
import higher
import torch.optim as optim
import common.eval as evl
# from common.loss_function import KLDivLoss

from copy import deepcopy
from sklearn.metrics import confusion_matrix, auc, roc_curve, recall_score, precision_score, f1_score, cohen_kappa_score,accuracy_score, roc_auc_score,classification_report

import torchvision.models
from model.transformer.vit_model import vit_base_patch16_224 

# for imaml
from imaml.utils.hessianfree import HessianFree
from common.meta.gbml import GBML
from imaml.utils.utils import get_accuracy, apply_grad, mix_grad, grad_to_cos, loss_to_ent
import torch.nn.functional as F

# import keras.backend as K
# import keras

class iMAML(GBML):

    def __init__(self, args):
        super().__init__(args)
        self._init_net()
        self._init_opt()
        self.lamb = 100.0
        self.n_cg = args.cg_steps
        self.version = args.version

        if self.version == 'HF':
            self.inner_optimizer = HessianFree(cg_max_iter=3,)
        return None

    @torch.enable_grad()
    def inner_loop(self, fmodel, diffopt, train_input, train_target):

        train_logit = fmodel(train_input)
        inner_loss = F.cross_entropy(train_logit, train_target)
        diffopt.step(inner_loss)

        return None

    @torch.no_grad()
    def cg(self, in_grad, outer_grad, params):
        x = outer_grad.clone().detach()
        r = outer_grad.clone().detach() - self.hv_prod(in_grad, x, params)
        p = r.clone().detach()
        for i in range(self.n_cg):
            Ap = self.hv_prod(in_grad, p, params)
            alpha = (r @ r)/(p @ Ap)
            x = x + alpha * p
            r_new = r - alpha * Ap
            beta = (r_new @ r_new)/(r @ r)
            p = r_new + beta * p
            r = r_new.clone().detach()
        return self.vec_to_grad(x)
    
    def vec_to_grad(self, vec):
        pointer = 0
        res = []
        for param in self.network.parameters():
            num_param = param.numel()
            res.append(vec[pointer:pointer+num_param].view_as(param).data)
            pointer += num_param
        return res

    @torch.enable_grad()
    def hv_prod(self, in_grad, x, params):
        hv = torch.autograd.grad(in_grad, params, retain_graph=True, grad_outputs=x)
        
        # 会报错
        # hv = torch.nn.utils.parameters_to_vector(hv).detach()
        # 所以用你
        vec = []
        for param in hv:
            vec.append(param.reshape(-1))
        hv = torch.cat(vec).detach()


        # precondition with identity matrix
        return hv/self.lamb + x

    def outer_loop(self, batch, is_train):
        
        train_inputs, train_targets, test_inputs, test_targets = self.unpack_batch(batch)

        loss_log = 0
        acc_log = 0
        grad_list = []
        loss_list = []

        # [batch_task]
        for (train_input, train_target, test_input, test_target) in zip(train_inputs, train_targets, test_inputs, test_targets):

            with higher.innerloop_ctx(self.network, self.inner_optimizer, track_higher_grads=False) as (fmodel, diffopt):

                for step in range(self.args.n_inner):
                    self.inner_loop(fmodel, diffopt, train_input, train_target)
                
                train_logit = fmodel(train_input)
                in_loss = F.cross_entropy(train_logit, train_target)

                test_logit = fmodel(test_input)
                outer_loss = F.cross_entropy(test_logit, test_target)
                loss_log += outer_loss.item()/self.args.meta_size

                with torch.no_grad():
                    acc_log += get_accuracy(test_logit, test_target).item()/self.args.meta_size
            
                if is_train:
                    params = list(fmodel.parameters(time=-1))
                    # for i in params:
                    #     if not i.requires_grad():
                    #         print('aaaaaaa')
                    in_grad = torch.nn.utils.parameters_to_vector(torch.autograd.grad(in_loss, params, create_graph=True))
                    outer_grad = torch.nn.utils.parameters_to_vector(torch.autograd.grad(outer_loss, params))
                    implicit_grad = self.cg(in_grad, outer_grad, params)
                    grad_list.append(implicit_grad)
                    loss_list.append(outer_loss.item())

        if is_train:
            self.outer_optimizer.zero_grad()
            weight = torch.ones(len(grad_list))
            weight = weight / torch.sum(weight)
            grad = mix_grad(grad_list, weight)
            grad_log = apply_grad(self.network, grad)
            self.outer_optimizer.step()

            return loss_log, acc_log, grad_log
        else:
            return loss_log, acc_log


    def outer_loop_bayes(self, batch, bayes_choice_s, bayes_choice_q, is_train):

        train_inputs, train_targets, test_inputs, test_targets = self.unpack_batch(batch)

        device = torch.device('cuda')
        target_s_x, target_s_y, target_q_x, target_q_y = bayes_choice_s[0].to(device=device), bayes_choice_s[1].to(device=device, dtype=torch.long), bayes_choice_q[0].to(device=device), bayes_choice_q[1].to(device=device, dtype=torch.long), 

        loss_log = 0
        acc_log = 0
        grad_list = []
        loss_list = []


        train_inputs = torch.concat((target_s_x, train_inputs), dim=0)
        train_targets =  torch.concat((target_s_y, train_targets), dim=0)
        test_inputs = torch.concat((target_q_x, test_inputs), dim=0)
        test_targets = torch.concat((target_q_y, test_targets), dim=0)

        bayes_s_p = torch.zeros_like(target_s_y[0])
        bayes_q_p = torch.zeros_like(target_q_y[0])
        index = 0

        for (train_input, train_target, test_input, test_target) in zip(train_inputs, train_targets, test_inputs, test_targets):
            index += 1
            with higher.innerloop_ctx(self.network, self.inner_optimizer, track_higher_grads=False) as (fmodel, diffopt):
                # 内循环
                for step in range(self.args.n_inner):
                    self.inner_loop(fmodel, diffopt, train_input, train_target)

                if index == 1:
                    bayes_s_p = fmodel(train_input)
                    bayes_q_p = fmodel(test_input)# query 也参与梯度更新
                    continue

                train_logit=fmodel(train_input)
                # in_loss = F.cross_entropy(train_logit, train_target)
                # 多目标损失函数--贝叶斯损失函数 交叉熵损失函数，KL散度损失函数
                in_loss = self.continualLearningLoss(train_target, train_logit, target_s_y[0], bayes_s_p)

                test_logit = fmodel(test_input)# query 也参与梯度更新
                # outer_loss = F.cross_entropy(test_logit, test_target)
                outer_loss =  self.continualLearningLoss(test_target, test_logit, target_q_y[0], bayes_q_p)



                loss_log += outer_loss.item()/(self.args.meta_size + 1)

                with torch.no_grad():
                    acc_log += get_accuracy(test_logit, test_target).item()/(self.args.meta_size + 1)

                if is_train:
                    params = list(fmodel.parameters(time=-1))
                    in_grad = torch.nn.utils.parameters_to_vector(torch.autograd.grad(in_loss, params, create_graph=True))
                    outer_grad = torch.nn.utils.parameters_to_vector(torch.autograd.grad(outer_loss, params))
                    implicit_grad = self.cg(in_grad, outer_grad, params)
                    grad_list.append(implicit_grad)
                    loss_list.append(outer_loss.item())

        if is_train:
            self.outer_optimizer.zero_grad()
            weight = torch.ones(len(grad_list))
            weight = weight / torch.sum(weight)
            grad = mix_grad(grad_list, weight)
            grad_log = apply_grad(self.network, grad)
            self.outer_optimizer.step()

            return loss_log, acc_log, grad_log
        else:
            return loss_log, acc_log

    # def mutualDistillationLoss(self, yTrue, yPred, oldClasses, newClasses):
    def mutualDistillationLoss(self, yOldT, yOldP, yNewT, yNewP):


        # yOldP = yPred[:,:oldClasses]
        # yOldT = yTrue[:,:oldClasses]
        # yNewP = yPred[:,newClasses:]
        # yNewT = yTrue[:,newClasses:]

        # print(yNewP)
        # print(yOldP)
        # yop = yOldP
        # yot = yOldT
        # ynp = yNewP
        # ynt = yNewT

        # n1 = float(yOldP.get_shape().as_list()[1])
        # n2 = float(yNewP.get_shape().as_list()[1])
        n1 = float(yOldP.size()[0])
        n2 = float(yNewP.size()[0])

        pOld = n1/(n1+n2) # 其实就是0.5
        pNew = n2/(n1+n2)

        yOldP = (yOldP * yNewP) / yNewP
        # m1 = K.mean(yOldP)
        # s1 = K.std(yOldP)
        m1 = torch.mean(yOldP)
        s1 = torch.std(yOldP)
        # like1 = (1./(np.sqrt(2.*3.1415)*s1))*K.exp(-0.5*((yOldP-m1)**2./(s1**2.)))
        like1 = (1./(np.sqrt(2.*3.1415)*s1))*torch.exp(-0.5*((yOldP-m1)**2./(s1**2.)))

        yOldNewP = like1 * pOld

        # return K.mean(K.categorical_crossentropy(yOldT,yOldNewP))
        return torch.mean(F.cross_entropy(yOldNewP, yOldT))

    # def continualLearningLoss(self, yTrue,yPred, iteration, oldClasses,temperature):
    def continualLearningLoss(self, yOldT, yOldP, yNewT, yNewP):
        a = 0.25
        # b = 0.45
        b = 0.20
        c = 0.30

        temperature = 1.65

        yOldP = yOldP/ temperature
        yNewP = yNewP/ temperature

        # categorical_crossentropy输入的是真实标签的one-hot编码，cross_entropy输入的是真实标签的索引0,1,2,3
        # 很奇怪，categorical_crossentropy的入参应该是y_true,y_pred
        # 这个地方为什么入参反过来了
        # loss_1 = a * K.categorical_crossentropy(yOldP,yOldT,from_logits=True)
        loss_1 = a * F.cross_entropy(yOldP, yOldT)
        # loss_2 = c * keras.losses.kullback_leibler_divergence(yNewT,yNewP,from_logits=True)
        # loss_2 = nn.KLDivLoss(yNewP,F.one_hot(yNewT, 4)) # 报错
        loss_2 = F.kl_div(F.log_softmax(yNewP, dim=1),F.one_hot(yNewT, 4).float(), reduction = 'batchmean')
        loss_bayes = b * self.mutualDistillationLoss(yOldT, yOldP, yNewT, yNewP)

        return loss_1 + loss_2 + loss_bayes

        # if iteration == 0:
        #     return K.categorical_crossentropy(yTrue,yPred,from_logits=True)
        # else:
        #     newClasses = 1
        #     if iteration == adaptationIteration:
        #         newClasses = 2
        #     total = oldClasses
        #     oldClasses = oldClasses - newClasses    
            
        #     yOldP = yPred[:,:oldClasses]/ temperature
        #     yOldT = yTrue[:,:oldClasses]
        #     yNewP = yPred[:,newClasses:]/ temperature
        #     yNewT = yTrue[:,newClasses:]
            
        #     return (a * K.categorical_crossentropy(yOldP,yOldT,from_logits=True)) + (b * mutualDistillationLoss(yTrue, yPred, oldClasses, newClasses)) + (c * keras.losses.kullback_leibler_divergence(yNewT,yNewP,from_logits=True))


