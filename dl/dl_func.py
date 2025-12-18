import os
from copy import deepcopy
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

from common.dl_comm import dl_comm
from common.dataloader import *
import common.utils as utils

from dl.load_for_meta import load_for_meta
import common.record_files as rec


# 测试任务包含S Q 和final test需要将S和Q重新组batch,S作为train data, Q 作为val data,
def trainer(args, sppport_all_task, query_all_task, final_test_task, meta_epoch):
    # 创建一个csv文件记录测试任务的指标
    dir_meta_epoch = os.path.join(args.store_dir,'meta_epoch')
    utils.mkdir(dir_meta_epoch)
    metric_dir = os.path.join(dir_meta_epoch,'metric_' + str(meta_epoch) + '.csv')
    rec.ceate_files(metric_dir)

    # 创建一个txt文件记录混淆矩阵
    utils.mkdir( os.path.join(dir_meta_epoch,'cm'))
    test_cm_res_dir = os.path.join(dir_meta_epoch,'cm', 'cm_test_' + str(meta_epoch) + '.txt')
    val_cm_res_dir = os.path.join(dir_meta_epoch,'cm', 'cm_val_' + str(meta_epoch) + '.txt')

    task_num = len(sppport_all_task) # 多构建几个任务，结果会更准确
    val_acc_all_epoch = [0] * args.n_epoch # 记录所有测试任务每个epoch的val 的acc

    val_values_all_task = {}
    test_values_all_task = {}

    for task_idx in sppport_all_task.keys():
        # 创建 网络 对象
        dl_ob = dl_comm(args)
        dl_ob._init_net()
        dl_ob._init_opt()
        if args.alg == 'dl' and args.load != '':
            dl_ob.model.load_state_dict(torch.load(args.load))
        elif args.alg == 'reptile' or args.alg =='maml' or args.alg == 'imaml':
            dl_ob.model.load_state_dict(load_for_meta(args,meta_epoch))

        # writer = SummaryWriter('{}/tensorboard_log/{}_meta_epoch/{}_task_id'.format(args.store_dir,meta_epoch, task_idx))

        train_loader = sppport_all_task[task_idx]
        val_loader = query_all_task[task_idx]
        test_loader = final_test_task[task_idx]

        # for val acc
        best_val = 0.0
        best_val_epoch = 0
        best_final_val_state_dict = deepcopy(dl_ob.model.state_dict()) # 不用deepcopy是万万不行的，否则会随dl_ob.model发生变化
        best_cm_val = []

        for epoch in range(args.n_epoch):
            dl_ob.adjust_learning_rate(epoch)
            print('task_idx--------: ', task_idx)
            train_loss = dl_ob.train(train_loader, epoch)
            res_val = dl_ob.val(val_loader)

            val_acc_all_epoch[epoch] += res_val['acc']

            # 根据acc选取最优模型
            is_val_set_best = res_val['acc'] > best_val
            best_val = max(best_val, res_val['acc'])
            if is_val_set_best:
                best_val_epoch = epoch
                val_values_all_task[task_idx] = [res_val['loss'], res_val['acc'], res_val['auc'], res_val['prec'], res_val['recall'],
                                                  res_val['f1'], res_val['ka'], res_val['sens'], res_val['spec']] + [best_val, best_val_epoch]
                best_final_val_state_dict = deepcopy(dl_ob.model.state_dict())
                best_cm_val = [res_val['cm'], res_val['report']]
                if args.is_save_val_net:
                    taskpth_store_dir = os.path.join(dir_meta_epoch, f'taskid_{task_idx}')
                    utils.mkdir(taskpth_store_dir)
                    torch.save(best_final_val_state_dict,os.path.join(taskpth_store_dir,f'best_model_for_valset_{meta_epoch}.pth' ))

            # 落盘
            rec.write_result(metric_dir, res_val, task_idx, epoch, best_val, best_val_epoch)

        #+++++++++++++++++for test++++++++++++++begin
        dl_ob.model.load_state_dict(best_final_val_state_dict)
        res_test = dl_ob.val(test_loader)
        test_values_all_task[task_idx] = [res_test['loss'], res_test['acc'], res_test['auc'], res_test['prec'], res_test['recall'],
                                           res_test['f1'],  res_test['ka'], res_test['sens'], res_test['spec']] + [best_val, best_val_epoch]

        # 落盘
        rec.write_result(metric_dir, res_test, 'finaltest', ' ', best_val, best_val_epoch)
        rec.write_cm(test_cm_res_dir, task_idx,  best_val_epoch, res_test['cm'], res_test['report'])
        rec.write_cm(val_cm_res_dir, task_idx, best_val_epoch, best_cm_val[0], best_cm_val[1])


        # 保存y_true和y_score, fpr和tpr
        y_score_dir = os.path.join(dir_meta_epoch,'cm')
        rec.save_model_output(res_test['y_true'], res_test['y_score'], y_score_dir,task_idx, meta_epoch)
        rec.save_fpr_tpr(res_test['fpr'], res_test['tpr'],y_score_dir, task_idx, meta_epoch)

    all_tasks_ave = rec.write_result_sum(metric_dir, val_values_all_task, test_values_all_task)


    return [round(item / task_num, 4) for item in val_acc_all_epoch], all_tasks_ave



def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)



def load_model_output(filename):
    """
    从CSV文件加载AUC计算数据

    参数:
    filename - CSV文件名

    返回:
    y_true, y_pred_score
    """
    df = pd.read_csv(filename)

    # 提取真实标签
    y_true = df['true_label'].values

    # 提取所有概率列
    prob_columns = [col for col in df.columns if col.startswith('prob_class_')]
    y_pred_score = df[prob_columns].values

    print(f"从 {filename} 成功加载数据")
    print(f"样本数: {len(y_true)}, 类别数: {y_pred_score.shape[1]}")

    return y_true, y_pred_score


