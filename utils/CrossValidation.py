import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from scipy.interpolate import interp1d
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
from datetime import datetime
import os

class CrossValidationPlot:
    def __init__(self, model):
        self.model = model
        self.cv = 5  # 五折交叉验证
        self.shuffle_split = KFold(n_splits=self.cv, shuffle=True, random_state=42)  # 打乱数据
        self.rgb_values = [(207, 67, 62), (244, 111, 67), (251, 221, 133), (128, 166, 226), (64, 57, 144), (108, 145, 194)]
    def train_and_compute_metrics(self, X, y):
        metrics_list = []
        all_y_test = []
        all_y_pred = []
        all_fpr = []
        all_tpr = []
        all_precision = []
        all_recall = []

        for i, (train_index, test_index) in enumerate(self.shuffle_split.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict_proba(X_test)[:, 1]

            # 计算指标值
            mcc = matthews_corrcoef(y_test, y_pred.round())
            acc = accuracy_score(y_test, y_pred.round())
            auc_score = roc_auc_score(y_test, y_pred)
            precision, recall, _ = precision_recall_curve(y_test, y_pred)
            aupr = auc(recall, precision)
            f1 = f1_score(y_test, y_pred.round())

            # 保存指标值到列表
            metrics_list.append({'Fold': i+1, 'MCC': mcc, 'ACC': acc, 'AUC': auc_score, 'AUPR': aupr, 'F1': f1})

            all_y_test.extend(y_test)
            all_y_pred.extend(y_pred)
            all_fpr.append(roc_curve(y_test, y_pred)[0])
            all_tpr.append(roc_curve(y_test, y_pred)[1])
            all_precision.append(precision)
            all_recall.append(recall)

        # 计算平均指标值
        metrics_df = pd.DataFrame(metrics_list)
        mean_metrics = metrics_df.mean()
        mean_metrics['Fold'] = 'Mean'
        std_metrics = metrics_df.std()
        std_metrics['Fold'] = 'Std'
        metrics_df = metrics_df._append(mean_metrics, ignore_index=True)
        metrics_df = metrics_df._append(std_metrics, ignore_index=True)

        # 保存指标值到CSV文件
        subfolder_evaluations = 'evaluations'

        # 保存DataFrame到子文件夹
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        folder_path = './' + subfolder_evaluations + '/'  # 子文件夹路径
        file_path = folder_path + f'evaluations_{timestamp}.csv'  # 保存文件路径

        # 确保子文件夹存在
        import os
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 保存DataFrame到文件
        metrics_df.to_csv(file_path, index=False)

        print("Metrics have been saved to", file_path)


        # 绘制AUC和AUPR曲线
        self.plot_auc_curve(all_fpr, all_tpr)
        self.plot_aupr_curve(all_recall, all_precision)


    def plot_auc_curve(self, all_fpr, all_tpr):
        plt.figure()
        normalized_rgb_values = []
        rgb_values = self.rgb_values
        for rgb in rgb_values:
            normalized_rgb = tuple(value / 255.0 for value in rgb)
            normalized_rgb_values.append(normalized_rgb)

        interp_fpr = np.linspace(0, 1, 100)
        interp_tpr = np.zeros_like(interp_fpr)

        for i in range(self.cv):
            fpr = all_fpr[i]
            tpr = all_tpr[i]
            tpr[0] = 0.0
            auc_value = auc(all_fpr[i], all_tpr[i])
            interp_tpr += np.interp(interp_fpr, fpr, tpr)
            plt.plot(fpr, tpr, label=f'Fold {i+1} (AREA={auc_value:.4f})',
                     color=normalized_rgb_values[i+1], linestyle=':')

        interp_tpr /= self.cv
        mean_fpr = interp_fpr
        mean_tpr = interp_tpr
        mean_tpr[0] = 0.0
        mean_auc = np.mean([auc(all_fpr[i], all_tpr[i]) for i in range(self.cv)])
        plt.plot(mean_fpr, mean_tpr, label=f'Mean   (AREA={mean_auc:.4f})',
                 color=normalized_rgb_values[0], linestyle='-')
        plt.plot([0, 1], [0, 1], 'k--')
        legend_font = FontProperties(family='Arial Monospaced MT', size=11)
        title_font = FontProperties(family='Arial', size=14)
        line = plt.gca().lines[5]  # 获取第一根线的对象
        line.set_linewidth(2)
        line.set_zorder(10)
        handles, labels = plt.gca().get_legend_handles_labels()

        # 调整图例顺序
        order = [5, 0, 1, 2, 3, 4]  # 根据需要设置特定曲线的索引顺序
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Curve (ROC)', fontproperties=title_font, pad=13)

        # 保存 AUC 曲线图
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='lower right', prop=legend_font)


        subfolder_images = 'images'
        if not os.path.exists(subfolder_images):
            os.makedirs(subfolder_images)
            # 保存图表到子文件夹
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = os.path.join(subfolder_images, f'auc_{timestamp}.eps')
        plt.savefig(filename)
        print("ROC plot has been saved to", filename)
        plt.show()


    def plot_aupr_curve(self, all_recall, all_precision):
        plt.figure()

        normalized_rgb_values = []
        rgb_values = self.rgb_values
        for rgb in rgb_values:
            normalized_rgb = tuple(value / 255.0 for value in rgb)
            normalized_rgb_values.append(normalized_rgb)

        for i in range(self.cv):
            recall = all_recall[i]
            precision = all_precision[i]
            precision[0] = 0.
            aupr = np.trapz(precision, recall)
            plt.plot(recall, precision, label=f'Fold {i + 1} (AREA={abs(aupr):.4f})',
                     color=normalized_rgb_values[i+1], linestyle=':')

        # 计算平均曲线和平均AUPR值
        max_length = max([len(recall) for recall in all_recall])
        mean_recall = np.linspace(0, 1, max_length)
        mean_precision = np.zeros(max_length)
        for i in range(self.cv):
            f = interp1d(all_recall[i], all_precision[i], kind='linear')
            mean_precision += f(mean_recall)
        mean_precision /= self.cv
        mean_precision[-1] = 0.

        mean_aupr = np.trapz(mean_precision, mean_recall)

        plt.plot(mean_recall, mean_precision, label=f'Mean   (AREA={mean_aupr:.4f})',
                 color=normalized_rgb_values[0], linestyle='-')
        plt.plot([0, 1], [1, 0], 'k--')
        line = plt.gca().lines[5]  # 获取第一根线的对象
        line.set_linewidth(2)
        line.set_zorder(10)
        handles, labels = plt.gca().get_legend_handles_labels()

        # 调整图例顺序
        order = [5, 0, 1, 2, 3, 4]  # 根据需要设置特定曲线的索引顺序
        legend_font = FontProperties(family='Arial Monospaced MT', size=11)
        title_font = FontProperties(family='Arial', size=14)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (PRC)', fontproperties=title_font, pad=13)

        # 保存 AUPR 曲线图
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='lower left', prop=legend_font)
        subfolder = 'images'
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        # 保存图表到子文件夹

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = os.path.join(subfolder, f'aupr_{timestamp}.eps')
        plt.savefig(filename)
        print("PRC plot has been saved to", filename)
        plt.show()

