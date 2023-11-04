import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer, fbeta_score
import torch
from torch.utils.tensorboard import SummaryWriter
from deepod.models.dsvdd import DeepSVDD
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import time
from utils.CrossValidation import CrossValidationPlot
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter('./run_deepsvdd')

def deepod_dsvdd(kcg_intersec, encoded_embedding, EID_to_index, Maybe_cancer_genes_list):
    cancer_genes = kcg_intersec[0].tolist()

    # Determine the row numbers corresponding to the required genes
    required_indices = [EID_to_index[gene] for gene in cancer_genes]
    encoded_embedding = encoded_embedding.cpu()

    # pca = PCA(n_components='mle')
    # encoded_embedding = pca.fit_transform(encoded_embedding)

    # Extract the required feature vectors from encoded_embedding
    cancer_gene_vectors = encoded_embedding[required_indices, :]

    # Create a DataFrame containing genes and their corresponding feature vectors
    cancer_gene_vectors_df = pd.DataFrame(cancer_gene_vectors, index=cancer_genes)

    # Get the list of unwanted genes
    all_genes = list(EID_to_index.keys())
    unlabeled_genes = list(set(all_genes) - set(cancer_genes))

    # Determine the row numbers corresponding to the unwanted genes
    unwanted_indices = [EID_to_index[gene] for gene in unlabeled_genes]

    # Extract the unwanted feature vectors from encoded_embedding
    unlabeled_gene_vectors = encoded_embedding[unwanted_indices, :]

    # Create a DataFrame containing unwanted genes and their corresponding feature vectors
    unlabeled_gene_vectors_df = pd.DataFrame(unlabeled_gene_vectors, index=unlabeled_genes)


    x_kcg, x_unlabeled = torch.Tensor(cancer_gene_vectors_df.to_numpy()), torch.Tensor(
        unlabeled_gene_vectors_df.to_numpy())


    clf = DeepSVDD(lr=1e-5, verbose=3, epochs=200) #使用早停策略，保证得到的AUC较高
    clf.fit(x_kcg.numpy(), y=None)
    scores = clf.decision_function(x_unlabeled.numpy())
    gene_scores = dict(zip(unlabeled_gene_vectors_df.index, scores))
    # 找到分数最高的前len(cancer_gene_vectors_df)个样本，赋值给uncancer_gene_vectors_df
    gene_scores_sorted = sorted(gene_scores.items(), key=lambda x: x[1], reverse=True)
    #结合从数据库中找到的数据，我们删除可能为癌症基因作高异常分数基因

    top_uncancer_samples = []
    genes_num__count = 0
    for geneAndscore in gene_scores_sorted:
        if geneAndscore[0] not in Maybe_cancer_genes_list:
            top_uncancer_samples.append(geneAndscore)
            genes_num__count += 1

        if genes_num__count == len(cancer_gene_vectors_df):
            break

    uncancer_gene_vectors_df = unlabeled_gene_vectors_df.loc[[sample_id for sample_id, _ in top_uncancer_samples]]

    # 2. 将cancer_gene_vectors_df和uncancer_gene_vectors_df拼接在一起
    cancer_gene_vectors_df['label'] = 1  # 正样本标签为1
    uncancer_gene_vectors_df['label'] = 0  # 负样本标签为0
    all_labeled_data_df = pd.concat([cancer_gene_vectors_df, uncancer_gene_vectors_df])

    # 3.准备特征和标签
    X = all_labeled_data_df.drop(columns=['label'])
    y = all_labeled_data_df['label']

    bst = XGBClassifier(max_depth=3,
                        learning_rate=0.1,
                        n_estimators=100,  # 使用多少个弱分类器
                        objective='binary:logistic',
                        booster='gbtree',
                        gamma=0,
                        min_child_weight=1,
                        max_delta_step=0,
                        subsample=0.8,
                        colsample_bytree=1,
                        reg_alpha=0,
                        reg_lambda=1,
                        seed=0  # 随机数种子
                        )

    # FIXME 这里画图的时候会报错
    ## 计算各个指标
    CVPlot = CrossValidationPlot(bst)
    CVPlot.train_and_compute_metrics(X, y)

    ###得到各个基因是癌症基因的概率
    uncancer_index = uncancer_gene_vectors_df.index
    waiting_for_predict_gene_vectors_df = unlabeled_gene_vectors_df[~unlabeled_gene_vectors_df.index.isin(uncancer_index)]
    #
    # ### 预测癌症基因并保存
    bst.fit(X, y)

    cancer_genes_proba = bst.predict_proba(waiting_for_predict_gene_vectors_df)
    cgene_proba_df = pd.DataFrame(cancer_genes_proba[:,1], index=waiting_for_predict_gene_vectors_df.index, columns=["Cancer Gene Proba"])


    return cancer_genes_proba
