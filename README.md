# GTCC-MDA

**GTCC-MDA: GNN–Graph Transformer Collaborative Contrastive Learning for Metabolite–Drug Association Prediction**

## 🏠 Overview

GTCC-MDA is a dual-branch graph learning framework for predicting potential metabolite–drug associations. It jointly models local similarity structures and global association dependencies through collaborative contrastive learning.

The local branch employs GCNII to encode higher-order structural information from fused and sparsified metabolite and drug similarity networks. The global branch uses a graph Transformer with Laplacian positional encoding and multi-hop association contexts to capture long-range dependencies in the metabolite–drug association network. Bidirectional multi-positive contrastive learning aligns the two representation spaces, while branch-specific decoders and learnable late fusion integrate their complementary predictions.

<!-- Add the framework figure after uploading a PNG file:
![Overview of GTCC-MDA](figures/GTCC-MDA_framework.png)
-->

## ✨ Main features

```text
- Dual-branch architecture for local and global structural representation learning
- GCNII-based encoding of metabolite and drug similarity networks
- Graph Transformer-based encoding of metabolite–drug association contexts
- Laplacian positional encoding for preserving graph structural information
- Multi-hop neighborhood aggregation with hop-level attention
- Bidirectional multi-positive contrastive learning
- Branch-specific PairMLP decoders
- Learnable softmax-based late fusion
- Five-fold cross-validation on two independent benchmark datasets
```

## 🛠️ Dependencies

```text
- Python == 3.13.14
- PyTorch == 2.12.1
- PyTorch Geometric == 2.8.0
- CUDA == 13.0
- cuDNN == 9.2.0
- NumPy == 2.5.0
- pandas == 3.0.4
- SciPy == 1.18.0
- scikit-learn == 1.9.0
- RDKit
- NetworkX
- Matplotlib
```

Install the required packages using:

```bash
pip install -r requirements.txt
```

## 🗃️ Datasets

Two independent benchmark datasets are provided in the `Dataset1` and `Dataset2` directories.

```text
Dataset 1
- Source: DrugBank 6.0 pharmaco-metabolomics module
- Number of metabolites: 415
- Number of drugs: 649
- Number of known associations: 2,204

Dataset 2
- Source: Human Metabolome Database 5.0
- Number of metabolites: 428
- Number of drugs: 583
- Number of known associations: 1,861
```

Each dataset contains the following processed files:

```text
association_matrix.csv
    Binary metabolite–drug association matrix

Drug_Fingerprint_sim.csv
    Molecular-fingerprint similarity matrix for drugs

Drug_Gaussian_Simi.csv
    Gaussian interaction-profile similarity matrix for drugs

Drug_SNF_Similarity.csv
    Fused drug similarity matrix generated using similarity network fusion

Drug_SNF_MutualKNN_Network.csv
    Sparsified drug similarity network

Drug_mol2vec.csv
    Mol2vec representations of drugs

Metabolite_Fingerprint_sim.csv
    Molecular-fingerprint similarity matrix for metabolites

Metabolite_Gaussian_Simi.csv
    Gaussian interaction-profile similarity matrix for metabolites

Metabolite_SNF_Similarity.csv
    Fused metabolite similarity matrix generated using similarity network fusion

Metabolite_SNF_MutualKNN_Network.csv
    Sparsified metabolite similarity network

Metabolite_mol2vec.csv
    Mol2vec representations of metabolites

model_300dim.pkl
    Pretrained 300-dimensional Mol2vec model
```

## 📁 Project structure

```text
GTCC-MDA/
├── Dataset1/                    Processed files for Dataset 1
├── Dataset2/                    Processed files for Dataset 2
├── graphormer_dataset1/         Model and training code for Dataset 1
├── graphormer_dataset2/         Model and training code for Dataset 2
├── result_dataset1/             Experimental results for Dataset 1
├── result_dataset2/             Experimental results for Dataset 2
├── figures/                     Framework and result figures
├── requirements.txt             Python dependencies
├── LICENSE                      License information
└── README.md                    Project documentation
```

## ⚙️ Main model options

```text
--seed                  int     Random seed                              Default: 3407
--epochs                int     Maximum number of training epochs        Default: 1000
--peak_lr               float   Peak learning rate                       Default: 0.001
--end_lr                float   Final learning rate                      Default: 0.0001
--weight_decay          float   Weight-decay coefficient                 Default: 5e-4
--dropout               float   Dropout rate                             Default: 0.2
--attention_dropout     float   Attention dropout rate                   Default: 0.2

--node_input            int     Initial node feature dimension           Default: 64
--node_hidden           int     Graph Transformer hidden dimension       Default: 128
--node_output           int     Output embedding dimension               Default: 64
--graphformer_layers    int     Number of graph Transformer layers       Default: 1
--n_heads               int     Number of attention heads                Default: 8
--hops                   int     Number of neighborhood hops              Default: 2
--pe_dim                 int     Laplacian positional encoding dimension  Default: 15

--GCNII_layers          int     Number of GCNII layers                   Default: 20
--gcn_hidden            int     GCNII hidden dimension                   Default: 64
--gcnii_alpha           float   Initial residual coefficient             Default: 0.1
--gcnii_theta           float   Identity-mapping coefficient             Default: 1.0

--contrast_dim          int     Contrastive projection dimension         Default: 64
--contrast_temperature  float   Contrastive temperature                  Default: 0.2
--contrast_balance      float   Bidirectional contrastive balance        Default: 0.5
--contrast_weight       float   Contrastive-loss weight                  Default: 0.5
--aux_weight            float   Auxiliary classification-loss weight     Default: 0.7

--decoder_hidden        int     PairMLP hidden dimension                 Default: 128
--k_folds               int     Number of cross-validation folds         Default: 5
```

## 🎯 How to run

Run all commands from the root directory of the project.

### Dataset 1

```bash
python graphormer_dataset1/train.py
```

### Dataset 2

```bash
python graphormer_dataset2/train.py
```

### Custom parameter settings

```bash
python graphormer_dataset1/train.py \
    --epochs 1000 \
    --hops 2 \
    --GCNII_layers 20 \
    --n_heads 8 \
    --node_output 64 \
    --contrast_weight 0.5 \
    --dropout 0.2
```

On Windows PowerShell, the same command can be written as:

```powershell
python graphormer_dataset1/train.py `
    --epochs 1000 `
    --hops 2 `
    --GCNII_layers 20 `
    --n_heads 8 `
    --node_output 64 `
    --contrast_weight 0.5 `
    --dropout 0.2
```

## 📊 Output files

The results for the two datasets are saved to `result_dataset1` and `result_dataset2`, respectively.

```text
metrics.csv
    Fold-specific and average evaluation metrics

test_auc.pdf
    Receiver operating characteristic curves

test_prc.pdf
    Precision–recall curves

fusion_weights.csv
    Learned local- and global-branch fusion weights

fusion_weight_history.csv
    Fusion-weight dynamics during training

branch_contributions.csv
    Sample-level weighted logit contributions from the two branches

fprs_1.csv and tprs_1.csv
    Data used to construct the ROC curves

precisions_1.csv and recalls_1.csv
    Data used to construct the precision–recall curves
```

## 🧪 Evaluation metrics

```text
- Accuracy
- Precision
- Recall
- F1-score
- Area under the receiver operating characteristic curve (AUC)
- Area under the precision–recall curve (AUPR)
```

All reported results are obtained using five-fold cross-validation and are presented as the mean ± standard deviation across the five folds.

## 📝 Citation

If you use GTCC-MDA or the datasets provided in this repository, please cite the corresponding paper:

```text
GTCC-MDA: GNN–Graph Transformer Collaborative Contrastive Learning
for Metabolite–Drug Association Prediction.

Citation information will be updated after publication.
```

## 📧 Contact

```text
Wenzhi Liu
School of Mathematics and Computer Science
Northwest Minzu University
Lanzhou, Gansu 730030, China

Email: 194202835@xbmu.edu.cn
```

## 📄 License

This project is distributed under the MIT License. See the `LICENSE` file for details.

Parts of the graph Transformer implementation were adapted from  
[graphormer-pyg](https://github.com/leffff/graphormer-pyg). Please retain the corresponding license and attribution information when redistributing the code.



# MAHN
Predicting disease-metabolite associations based on the metapath aggregation of tripartite heterogeneous networks

## 🏠 Overview
![image](https://github.com/Lwz1225/MAHN/assets/127914409/ddd7ad49-8a8c-4f67-8287-d5900db5f0c7)


## 🛠️ Dependecies
```
- Python == 3.9
- pytorch == 1.12.1
- dgl == 1.1.1
- numpy == 1.22.4+mkl
- pandas == 1.4.4
```

## 🗓️ Dataset
```
- disease-metabolite associations: association_DME.xlsx
- disease-microbe associations: association_DMI.xlsx
- microbe-metabolite associations: association_MIME.xlsx
- disease semantic networks based on metapath DMED and DMID: A_DME_D.xlsx and A_DMI_D.xlsx
- metabolite semantic networks based on metapath MEDME and MEMIME: A_DME_ME.xlsx and A_MIME_ME.xlsx 
- disease Gaussian kernel similarity: disease_Gaussian_Simi.xlsx
- disease semantic similarity: disease_Semantic_simi.xlsx
- metabolite functional similarity: metabolite_func_simi.xlsx
- metabolite Gaussian kernel similarity: metabolite_Gaussian_Simi.xlsx
- microbe Gaussian kernel similarities: microbe_Gaussian_Simi_1.xlsx and microbe_Gaussian_Simi_2.xlsx 
```

## 🛠️ Model options
```
--epochs           int     Number of training epochs.                 Default is 1000.
--attn_size        int     Dimension of attention.                    Default is 64.
--attn_heads       int     Number of attention heads.                 Default is 6.
--out_dim          int     Output dimension after feature extraction  Default is 64.
--sampling number  int     enhanced GraphSAGE sampling number         Default is 50.
--dropout          float   Dropout rate                               Default is 0.2.
--slope            float   Slope                                      Default is 0.2.
--lr               float   Learning rate                              Default is 0.001.
--wd               float   weight decay                               Default is 5e-3.

```

## 🎯 How to run?
```
1、Loading various associations and similarities in the data folder
2、Running main.py in the my_code folder calls train.py, model.py, layers.py and utils.py to get the experimental results

```
