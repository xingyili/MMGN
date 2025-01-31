import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer, fbeta_score
import torch
from deepod.models.tabular import DeepSVDD
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import time
from utils.CrossValidation import CrossValidationPlot
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def deepod_dsvdd(kcg_intersec, encoded_embedding, EID_to_index, Maybe_cancer_genes_list):
    cancer_genes = kcg_intersec[0].tolist()

    # 1. Split cancer genes into 10% for DSVDD and 90% for XGBoost
    from sklearn.model_selection import train_test_split
    train_genes, remaining_genes = train_test_split(cancer_genes, test_size=0.9, random_state=42)

    # 2. Prepare DSVDD training data (10% cancer genes)
    train_indices = [EID_to_index[gene] for gene in train_genes]
    encoded_embedding = encoded_embedding.cpu()
    x_kcg = torch.Tensor(encoded_embedding[train_indices, :].numpy())

    # 3. Prepare unlabeled data (non-cancer genes)
    all_genes = list(EID_to_index.keys())
    unlabeled_genes = list(set(all_genes) - set(cancer_genes))
    unwanted_indices = [EID_to_index[gene] for gene in unlabeled_genes]
    x_unlabeled = torch.Tensor(encoded_embedding[unwanted_indices, :].numpy())

    # 4. Train DeepSVDD and find negative samples
    clf = DeepSVDD(lr=1e-5, verbose=3, epochs=200)
    clf.fit(x_kcg.numpy())

    # 5. Select negative samples matching remaining_genes count
    scores = clf.decision_function(x_unlabeled.numpy())
    gene_scores = dict(zip(unlabeled_genes, scores))

    # Filter and select top negatives
    target_negative_count = len(remaining_genes)
    filtered_scores = [(g, s) for g, s in gene_scores.items() if g not in Maybe_cancer_genes_list]
    filtered_scores_sorted = sorted(filtered_scores, key=lambda x: x[1], reverse=True)
    selected_negatives = [g for g, _ in filtered_scores_sorted[:target_negative_count]]

    # 6. Prepare XGBoost dataset
    # Positive samples (remaining 90% cancer genes)
    remaining_indices = [EID_to_index[gene] for gene in remaining_genes]
    x_positive = encoded_embedding[remaining_indices, :].numpy()
    pos_df = pd.DataFrame(x_positive, index=remaining_genes)
    pos_df['label'] = 1

    # Negative samples (selected non-cancer genes)
    neg_indices = [EID_to_index[gene] for gene in selected_negatives]
    x_negative = encoded_embedding[neg_indices, :].numpy()
    neg_df = pd.DataFrame(x_negative, index=selected_negatives)
    neg_df['label'] = 0

    # 7. Combine and train XGBoost
    full_df = pd.concat([pos_df, neg_df])
    X = full_df.drop(columns=['label'])
    y = full_df['label']

    bst = XGBClassifier(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        objective='binary:logistic',
        booster='gbtree',
        gamma=0,
        seed=0
    )

    # 8. Cross-validation and prediction
    CVPlot = CrossValidationPlot(bst)
    CVPlot.train_and_compute_metrics(X, y)

    # 9. Predict on remaining unlabeled genes
    predict_genes = list(set(unlabeled_genes) - set(selected_negatives))
    predict_indices = [EID_to_index[gene] for gene in predict_genes]
    x_predict = encoded_embedding[predict_indices, :].numpy()

    bst.fit(X, y)
    proba = bst.predict_proba(x_predict)[:, 1]

    return pd.Series(proba, index=predict_genes)
