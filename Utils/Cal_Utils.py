import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix, accuracy_score, recall_score, precision_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# accracy, precision, TPR, TNR, FNR, FPR
def cal_rate(result, thres):
    all_number = len(result[0])
    # print all_number
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for item in range(all_number):
        disease = result[0][item]
        if disease >= thres:
            disease = 1
        if disease == 1:
            if result[1][item] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if result[1][item] == 0:
                TN += 1
            else:
                FN += 1
    # print TP+FP+TN+FN
    accracy = float(TP+FP) / float(all_number)
    if TP+FP == 0:
        precision = 0
    else:
        precision = float(TP) / float(TP+FP)
    TPR = float(TP) / float(TP+FN)
    TNR = float(TN) / float(FP+TN)
    FNR = float(FN) / float(TP+FN)
    FPR = float(FP) / float(FP+TN)
    # print accracy, precision, TPR, TNR, FNR, FPR
    return accracy, precision, TPR, TNR, FNR, FPR



# 获取混淆矩阵
def Get_Confusion_Matrix(labels, preds, tag):
    confusion = confusion_matrix(labels, preds, normalize='true')  # 官方提供的示例代码
    # cm_display = ConfusionMatrixDisplay(confusion).plot()
    FN = confusion[1][0]
    TN = confusion[0][0]
    TP = confusion[1][1]
    FP = confusion[0][1]
    plt.bar(['False Negative', 'True Negative', 'True Positive', 'False Positive'], [FN, TN, TP, FP])
    plt.savefig('Bar ' + tag + '.jpg')
    plt.clf()

    sns.heatmap(confusion, annot=True, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.ylabel("Label")
    plt.xlabel("Predicted")
    # plt.show()
    plt.savefig('Confusion Matrix Heatmap ' + tag + '.jpg')
    plt.clf()

# 计算全套指标
def Cal_TP_FP_TN_FN_Sen_Spec_PPV_NPV_F1(labels, preds):
    TP = 0
    for i in range(0, len(labels)):
        if labels[i] == preds[i] and labels[i] == 1:
            TP += 1
    print("True Positive: ", TP)

    FP = 0
    for i in range(0, len(labels)):
        if labels[i] == 0 and preds[i] == 1:
            FP += 1
    print("False Positive: ", FP)

    TN = 0
    for i in range(0, len(labels)):
        if labels[i] == preds[i] and labels[i] == 0:
            TN += 1
    print("True Negative: ", TN)

    FN = 0
    for i in range(0, len(labels)):
        if labels[i] == 1 and preds[i] == 0:
            FN += 1
    print("False Negative: ", FN)

    print('acc:')
    print(accuracy_score(labels, preds) * 100)

    print('recall:')
    print(recall_score(labels, preds))

    print('precision:')
    print(precision_score(labels, preds) * 100)

    print('specificity:')
    print(TN / (TN + FP))

    print('PPV:')
    print(TP / (TP + FP) * 100)

    print('NPV:')
    print(TN / (TN + FN) * 100)

    print('F1 score:')
    print(f1_score(labels, preds))

# 获取ROC曲线
def Get_ROC_Curve(scores, label):
    plt.rc('font', family='Times New Roman')
    my_colors = ["#1EB2A6", "#ffc4a3", "#e2979c", "#F67575"]
    th = 36  # 分类阈值
    # 定义函数
    lw = 1.5  # 线条粗细
    ticks_size = 12  # 坐标点标注大小
    label_size = 16  # xylabel大小
    title_size = 18  # 标题大小
    legend_size = 14  # 图例大小
    legend_sit = (0.85, 0.3)  # 图例坐标
    figsize = (5, 5)  # 曲线图大小
    title = 'ROC'
    fpr, tpr, threshold = roc_curve(label, scores)
    roc_auc = roc_auc_score(label, scores)  # 计算auc的值
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color=my_colors[3], lw=lw,
             label='AUC: %0.4f' % roc_auc)  # FPR为横坐标，TPR为纵坐标
    plt.plot([0, 1], [0, 1], color=my_colors[0], lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('1 - Sepcificity', size=label_size)
    plt.ylabel('Sensitivity', size=label_size)
    plt.yticks(fontsize=ticks_size)
    plt.xticks(fontsize=ticks_size)
    plt.title(title, size=title_size)
    # plt.grid(True)
    plt.legend(bbox_to_anchor=legend_sit,
               fontsize=legend_size,
               borderaxespad=0.)
    plt.savefig('ROC_Curve.jpg')
    # plt.show()
    plt.clf()

def Get_ROC_Curve_multimodel(scores, label):
    plt.rc('font', family='Times New Roman')
    my_colors = ["#1EB2A6", "#ffc4a3", "#e2979c", "#F67575"]
    model_colors = ["#4a86e8", "#6aa84f", "#ff9900"]
    model_name = ['ResNet10', 'ResNet18', 'ResNet34']

    lw = 1.5  # 线条粗细
    ticks_size = 12  # 坐标点标注大小
    label_size = 14  # xylabel大小
    title_size = 18  # 标题大小
    legend_size = 10  # 图例大小
    legend_sit = (0.4, 0.2)  # 图例坐标
    figsize = (5, 5)  # 曲线图大小
    title = 'ROC'
    plt.figure(figsize=figsize)
    for idx, (gt, out) in enumerate(zip(label, scores)):
        fpr, tpr, threshold = roc_curve(gt, out)
        roc_auc = roc_auc_score(gt, out)  # 计算auc的值
        plt.plot(fpr, tpr, color=model_colors[idx], lw=lw,
                 label=model_name[idx] + ' AUC: %0.4f' % roc_auc)  # FPR为横坐标，TPR为纵坐标
    plt.plot([0, 1], [0, 1], color=my_colors[0], lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('1 - Sepcificity', size=label_size)
    plt.ylabel('Sensitivity', size=label_size)
    plt.yticks(fontsize=ticks_size)
    plt.xticks(fontsize=ticks_size)
    plt.title(title, size=title_size)
    # plt.grid(True)
    plt.legend(bbox_to_anchor=legend_sit,
               fontsize=legend_size,
               borderaxespad=0.)
    # 设置大小
    plt.savefig('ROC_Curve.jpg')
    # plt.show()
    plt.clf()

# 获取PR曲线
def Get_PR_Fig(score, label_onehot):
    num_class = 2

    # 调用sklearn库，计算每个类别对应的precision和recall
    precision_dict = dict()
    recall_dict = dict()
    average_precision_dict = dict()
    for i in range(num_class):
        precision_dict[i], recall_dict[i], _ = precision_recall_curve(label_onehot[:, i], score[:, i])
        average_precision_dict[i] = average_precision_score(label_onehot[:, i], score[:, i])
        print(precision_dict[i].shape, recall_dict[i].shape, average_precision_dict[i])

    # micro
    precision_dict["micro"], recall_dict["micro"], _ = precision_recall_curve(label_onehot.ravel(),
                                                                              score.ravel())
    average_precision_dict["micro"] = average_precision_score(label_onehot, score, average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision_dict["micro"]))

    # 绘制所有类别平均的pr曲线
    plt.figure()
    plt.step(recall_dict['micro'], precision_dict['micro'], where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
            .format(average_precision_dict["micro"]))
    plt.savefig("pr_curve.jpg")
    # plt.show()
    plt.clf()

# 计算dice系数
def alculate_dice_coefficient(pred, target):
    smooth = 1.
    num = pred.size(0) #获取样本量
    print(pred.shape)
    m1 = pred.view(num, -1)  # 按照样本量展平
    m2 = target.view(num, -1)  # 按照样本量展平
    intersection = (m1 * m2).sum() # 交集元素数量
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

if __name__ == '__main__':
    cal_rate(None, None)
    # prob是样本正确率的array，label则是样本label的array
    threshold_vaule = sorted(prob)
    threshold_num = len(threshold_vaule)
    accracy_array = np.zeros(threshold_num)
    precision_array = np.zeros(threshold_num)
    TPR_array = np.zeros(threshold_num)
    TNR_array = np.zeros(threshold_num)
    FNR_array = np.zeros(threshold_num)
    FPR_array = np.zeros(threshold_num)
    # calculate all the rates
    for thres in range(threshold_num):
        accracy, precision, TPR, TNR, FNR, FPR = cal_rate((prob, label), threshold_vaule[thres])
        accracy_array[thres] = accracy
        precision_array[thres] = precision
        TPR_array[thres] = TPR
        TNR_array[thres] = TNR
        FNR_array[thres] = FNR
        FPR_array[thres] = FPR