# MMGN

## **Integration of Multiplex network and Multiomics data using Graph neural networks combined with Negative sample generation to identify cancer driver gene for pan-cancer analysis**

## Description

Identifying cancer driver genes has paramount significance in elucidating the intricate mechanisms underlying cancer development, progression, and therapeutic interventions. The integration of biological networks and multiomics data using graph neural network models for discovering drivers have recently garnered notable attention in cancerbiology. However, many models primarily focus on individual networks, inevitably overlooking the incompleteness and noise of interactions. Moreover, samples with imbalanced classes hampers the performance of models. In this study, we propose a novel method MMGN that integrates multiplex network and multiomics pan-cancer data using graph neural networks combined with negative sample generationto discover cancer driver genes, which not noly enhances gene feature learning based on the mutual information and the consensus regularizer, but also achieves a balanced classes of positive and negative samples for model training. Experimental results indicate that the predicted cancer driver genes by MMGN are closely associated with known cancer driver genes, demonstrating a more reliable biological interpretability. We believe MMGN can provide new prospects in precision oncology and may find broader applications in predicting biomarkers for other intricate diseases.&#x20;



## Getting Started

### Special dependencies

*   Pytorch xxx
*   Pytorch geometric xxx
*   Sklearn xxx
*   Deepod xxx

### Overview

The codes are organized as follows:&#x20;

*   `main.py`: the main script of MMGN.
*   `embedding.py`: train the model to obtain embedding vectors.
*   `indentify_cancer_gene.py`: identify predicted cancer genes(PCGs).
*   `utils/dmgimodel.py`: the implementation of DMGI.&#x20;
*   `utils/datahandle.py`: the script of data input and output.
*   `utils/CrossValidation.py`: the script of cross validation.

### Input file

*   The input files are located in the `processed_dataset/six_net` directory.
*   To fit the input format of Pytorch geometric, the nodes in the network are represented as indices. The `mapping_dict_EIDtoIndex.pickle` can be used to convert between indices and Entrez IDs.
*   &#x20;`Feature_for_pyG.csv` contains the features of genes (nodes), including 16 types of cancer and multi-omics data such as expression, methylation, and mutation.
*   &#x20;`kcg_intersec.csv` contains the known cancer genes (positive examples).
*   &#x20;`Merge_maybe_cancergene.txt` contains the gene set used for filtering negative samples.

### Output file

*   The generated vectors of genes are saved in the directory  `saved_embedding`.
*   The AUC and AUPRC plots are located in the `images` directory.
*   The `evaluations` directory contains the evaluation results of the model's  cross-validation.

### How to run

`python main.py`

## Version History

*   0.1
    *   Initial Release

## Acknowledgments

We referred to the code repository of the following repository:&#x20;

*   [pyg-team/pytorch\_geometric](https://github.com/pyg-team/pytorch_geometric)
*   [xuhongzuo/DeepOD](https://github.com/xuhongzuo/DeepOD)
