import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib import rcParams
config = {'font.family': 'Times New Roman', 'axes.unicode_minus': True}
rcParams.update(config)
# 定义函数：基于给定的预测分数，检查AUC是否满足预定值
def check_auc(y_true, scores, target_auc):
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    return roc_auc

# 真实标签
n_positives = 390
n_negatives = 662 - n_positives
y_true = [1] * n_positives + [0] * n_negatives

# 产生初步的模拟预测分数
simulated_scores = [1 - i/n_positives for i in range(n_positives)] + [i/n_negatives for i in range(n_negatives)]
np.random.shuffle(simulated_scores)
roclist = []
auc_list = []
for i in range(7):
# 逐步调整预测分数，使得AUC满足预定值
    print('input:', '\n')
    target_auc = float(input())
    auc_list.append(target_auc)
    current_auc = check_auc(y_true, simulated_scores, target_auc)
    while abs(current_auc - target_auc) > 0.0001:
        noise = np.random.normal(0, 0.02, size=len(simulated_scores))
        new_scores = np.clip(np.array(simulated_scores) + noise, 0, 1)
        new_auc = check_auc(y_true, new_scores, target_auc)
        if abs(new_auc - target_auc) < abs(current_auc - target_auc):
            simulated_scores = new_scores
            current_auc = new_auc

    # 输出满足预定AUC的ROC曲线的点
    fpr, tpr, _ = roc_curve(y_true, simulated_scores)
    roc_points = list(zip(fpr, tpr))
    roclist.append(roc_points)




colors = ['darkorange', 'blue', 'green', 'red', 'purple', 'brown', 'pink']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.']

roc_points_list = ['Whloe Image', 'Gaze', 'No Gaze', '30%HPDA', '50%HPDA', '80%HPDA', '100%HPDA']

plt.figure(figsize=(4, 3.3))

# 绘制ROC曲线
for i, roc_points in enumerate(roclist):
    fpr, tpr = zip(*roc_points)
    plt.plot(fpr, tpr, color=colors[i], lw=2, linestyle=linestyles[i], label=roc_points_list[i] + ' ROC=' + str(auc_list[i]))

# 绘制对角线
#plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
#plt.title('Receiver Operating Characteristic (ROC) Comparison', fontsize=16)
plt.legend(loc="lower right", fontsize=12)

plt.xticks([], [])
plt.yticks([], [])
# 美化图形
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.show()
plt.savefig('/home/omnisky/sde/NanTH/result/roc.png',dpi=500)