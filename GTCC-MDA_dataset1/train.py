from __future__ import division, print_function

import argparse
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import metrics

try:
    from graphormer_dataset1.model import *
    from graphormer_dataset1.metric import *
except ModuleNotFoundError:
    from model import *
    from metric import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VERSION_NAME = Path(__file__).resolve().parent.name.replace("graphormer", "")

parser = argparse.ArgumentParser()

# Training parameters.
parser.add_argument("--seed", type=int, default=3407)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument(
    "--lr", type=float, default=None,
    help="optional alias that overrides peak_lr when explicitly set")
parser.add_argument("--weight_decay", type=float, default=5e-4) # 5e-4
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--attention_dropout", type=float, default=0.2)
parser.add_argument("--tot_updates", type=int, default=None)
parser.add_argument("--warmup_updates", type=int, default=None)
parser.add_argument("--warmup_ratio", type=float, default=0.05)
parser.add_argument("--peak_lr", type=float, default=0.001)
parser.add_argument("--end_lr", type=float, default=0.0001)
parser.add_argument("--lr_power", type=float, default=1.0)
parser.add_argument("--min_delta", type=float, default=1e-4)
parser.add_argument("--metric_threshold_steps", type=int, default=1000)

# Model parameters.
parser.add_argument("--pe_dim", type=int, default=15)
parser.add_argument("--hops", type=int, default=2)
parser.add_argument("--graphformer_layers", type=int, default=1)
parser.add_argument("--n_heads", type=int, default=8)
parser.add_argument("--node_input", type=int, default=64)
parser.add_argument("--node_hidden", type=int, default=128)
parser.add_argument("--node_output", type=int, default=64)
parser.add_argument("--ffn_dim", type=int, default=256)
parser.add_argument("--GCNII_layers", type=int, default=20)
parser.add_argument("--gcn_hidden", type=int, default=64)
parser.add_argument("--gcnii_alpha", type=float, default=0.1)
parser.add_argument("--gcnii_theta", type=float, default=1.0)
parser.add_argument("--contrast_weight", type=float, default=0.5)
parser.add_argument("--contrast_temperature", type=float, default=0.2)
parser.add_argument("--contrast_balance", type=float, default=0.5)
parser.add_argument("--contrast_dim", type=int, default=64)
parser.add_argument(
    "--aux_weight", type=float, default=0.7,
    help="total weight of local/global auxiliary BCE losses")
parser.add_argument("--decoder_hidden", type=int, default=128)
parser.add_argument("--k_folds", type=int, default=5)
parser.add_argument(
    "--dataset_dir", type=Path, default=PROJECT_ROOT / "Dataset1")
parser.add_argument(
    "--result_dir",
    type=Path,
    default=PROJECT_ROOT / f"result{VERSION_NAME}",
)

args = parser.parse_args()
if args.lr is not None:
    args.peak_lr = args.lr
if args.tot_updates is None:
    args.tot_updates = args.epochs
if args.warmup_updates is None:
    args.warmup_updates = int(args.tot_updates * args.warmup_ratio)

if args.epochs <= 0 or args.tot_updates <= 0:
    parser.error("epochs and tot_updates must be positive")
if not 0 <= args.warmup_ratio < 1:
    parser.error("warmup_ratio must be in [0, 1)")
if not 0 <= args.warmup_updates < args.tot_updates:
    parser.error(
        "warmup_updates must satisfy 0 <= warmup_updates < tot_updates")
if args.node_input <= 0 or args.gcn_hidden <= 0:
    parser.error("node_input and gcn_hidden must be positive")
if args.node_output <= 0 or args.ffn_dim <= 0:
    parser.error("node_output and ffn_dim must be positive")
if args.decoder_hidden <= 0 or args.contrast_dim <= 0:
    parser.error("decoder_hidden and contrast_dim must be positive")
if (args.node_hidden <= 0 or args.n_heads <= 0
        or args.node_hidden % args.n_heads != 0):
    parser.error("node_hidden must be positive and divisible by n_heads")
if not 0 <= args.dropout < 1 or not 0 <= args.attention_dropout < 1:
    parser.error("dropout values must be in [0, 1)")
if not 0 <= args.aux_weight <= 1:
    parser.error("aux_weight must be in [0, 1]")
if not 0 <= args.contrast_weight <= 1:
    parser.error("contrast_weight must be in [0, 1]")
if (args.contrast_temperature <= 0
        or not 0 <= args.contrast_balance <= 1):
    parser.error(
        "contrast_temperature must be positive and "
        "contrast_balance in [0, 1]")
if not 0 < args.gcnii_alpha <= 1 or args.gcnii_theta <= 0:
    parser.error(
        "gcnii_alpha must be in (0, 1] and gcnii_theta positive")
if args.pe_dim < 0 or args.hops < 0:
    parser.error("pe_dim and hops must be non-negative")
if args.GCNII_layers < 0 or args.graphformer_layers <= 0:
    parser.error(
        "GCNII_layers must be non-negative and graphformer_layers positive")
if args.k_folds < 2:
    parser.error("k_folds must be at least 2")
if args.peak_lr <= 0 or args.end_lr < 0 or args.end_lr > args.peak_lr:
    parser.error("learning rates must satisfy 0 <= end_lr <= peak_lr")
if args.lr_power <= 0 or args.min_delta < 0:
    parser.error("lr_power must be positive and min_delta non-negative")
if args.metric_threshold_steps < 2:
    parser.error("metric_threshold_steps must be at least 2")

RESULT_DIR = args.result_dir
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def optional_loss_value(loss):
    return float("nan") if loss is None else loss.item()


def save_weight_plot(weight_frame, output_dir):
    means = weight_frame[["local_weight", "global_weight"]].mean()
    stds = weight_frame[["local_weight", "global_weight"]].std(ddof=0)
    fig, axis = plt.subplots(figsize=(5, 4))
    axis.bar(
        ["Local (GCNII)", "Global (Graphormer)"],
        means.values,
        yerr=stds.values,
        capsize=5,
        color=["#4C78A8", "#F58518"],
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Softmax fusion weight")
    axis.set_title("Branch fusion weights across folds")
    fig.tight_layout()
    fig.savefig(output_dir / "fusion_weights.png", dpi=300)
    fig.savefig(output_dir / "fusion_weights.pdf", bbox_inches="tight")
    plt.close(fig)


def save_weight_history(history_rows, best_rows, output_dir):
    """Persist per-epoch fusion weights and mark each fold's best epoch.

    The table is rewritten after every completed fold so a long training run
    still leaves a valid partial history if it is interrupted later.
    """
    if not history_rows:
        return
    history_frame = pd.DataFrame(history_rows)
    best_epoch_by_fold = {
        int(row["fold"]): int(row["best_epoch"])
        for row in best_rows
    }
    selected_epoch = history_frame["fold"].map(best_epoch_by_fold)
    history_frame["is_best_epoch"] = (
        history_frame["epoch"].eq(selected_epoch).fillna(False)
    )
    history_frame.to_csv(
        output_dir / "fusion_weight_history.csv",
        index=False,
    )


print("args", args)
print(
    "variant",
    {
        "local": MODEL_USES_LOCAL,
        "global": MODEL_USES_GLOBAL,
        "contrast": MODEL_USES_CONTRAST,
    },
)
set_global_seed(args.seed)

Adj, Drug_adj, Meta_adj, feature, random_index, k_folds = load_data(
    args.seed,
    args.node_input,
    k_folds=args.k_folds,
    dataset_dir=args.dataset_dir,
)
negative_edge_pool = torch.as_tensor(
    np.vstack(np.where(np.asarray(Adj) < 1)),
    dtype=torch.long,
    device=device,
)
total_validation_negatives = sum(len(fold) for fold in random_index)
if total_validation_negatives > negative_edge_pool.size(1):
    raise ValueError("not enough true negatives for disjoint validation folds")
validation_negative_generator = torch.Generator(device=device)
validation_negative_generator.manual_seed(args.seed + 100_000)
validation_negative_order = torch.randperm(
    negative_edge_pool.size(1),
    generator=validation_negative_generator,
    device=device,
)
fold_validation_negative_indices = []
negative_offset = 0
for fold in random_index:
    next_offset = negative_offset + len(fold)
    fold_validation_negative_indices.append(
        validation_negative_order[negative_offset:next_offset])
    negative_offset = next_offset

auc_result = []
acc_result = []
pre_result = []
recall_result = []
f1_result = []
prc_result = []
fprs = []
tprs = []
precisions = []
recalls = []
fold_weight_rows = []
weight_history_rows = []
contribution_frames = []

print("seed=%d, evaluating metabolite-drug links...." % args.seed)
for k in range(k_folds):
    print("------this is %dth cross validation------" % (k + 1))
    negative_generator = torch.Generator(device=device)
    negative_generator.manual_seed(args.seed + k)

    Or_train = np.matrix(Adj, copy=True)
    val_pos_edge_index = torch.tensor(
        np.array(random_index[k]).T,
        dtype=torch.long,
        device=device,
    )
    val_negative_indices = fold_validation_negative_indices[k]
    val_neg_edge_index = negative_edge_pool[:, val_negative_indices]
    train_negative_mask = torch.ones(
        negative_edge_pool.size(1), dtype=torch.bool, device=device)
    train_negative_mask[val_negative_indices] = False
    fold_train_negative_pool = negative_edge_pool[:, train_negative_mask]

    Or_train[tuple(np.array(random_index[k]).T)] = 0
    train_pos_edge_index = torch.tensor(
        np.asmatrix(np.where(Or_train > 0)),
        dtype=torch.long,
        device=device,
    )

    processed_features = None
    if MODEL_USES_GLOBAL:
        or_adj = constructNet(torch.tensor(Or_train)).to(device)
        lpe = laplacian_positional_encoding(
            or_adj, args.pe_dim).to(device)
        global_features = torch.cat((feature, lpe), dim=1)
        processed_features = re_features(
            or_adj, global_features, args.hops).to(device)
        model_input_dim = global_features.shape[1]
    else:
        model_input_dim = args.node_input + args.pe_dim

    meta_data = None
    drug_data = None
    if MODEL_USES_LOCAL:
        meta_network = torch.nonzero(
            Meta_adj, as_tuple=False).t().contiguous()
        meta_data = Data(
            x=feature[:Adj.shape[0]],
            edge_index=meta_network,
            edge_weight=Meta_adj[
                meta_network[0], meta_network[1]
            ].to(dtype=feature.dtype),
        )
        drug_network = torch.nonzero(
            Drug_adj, as_tuple=False).t().contiguous()
        drug_data = Data(
            x=feature[Adj.shape[0]:],
            edge_index=drug_network,
            edge_weight=Drug_adj[
                drug_network[0], drug_network[1]
            ].to(dtype=feature.dtype),
        )

    positive_mask = None
    candidate_mask = None
    if MODEL_USES_CONTRAST:
        meta_positive = Meta_adj.bool() | Meta_adj.bool().t()
        drug_positive = Drug_adj.bool() | Drug_adj.bool().t()
        positive_mask = torch.block_diag(
            meta_positive, drug_positive).to(device)
        positive_mask.fill_diagonal_(True)
        candidate_mask = torch.block_diag(
            torch.ones_like(meta_positive, dtype=torch.bool),
            torch.ones_like(drug_positive, dtype=torch.bool),
        ).to(device)

    model = TransformerModel(
        hops=args.hops,
        output_dim=args.node_output,
        node_feature_dim=args.node_input,
        input_dim=model_input_dim,
        pe_dim=args.pe_dim,
        num_drug=Adj.shape[1],
        num_meta=Adj.shape[0],
        graphformer_layers=args.graphformer_layers,
        num_heads=args.n_heads,
        hidden_dim=args.node_hidden,
        ffn_dim=args.ffn_dim,
        dropout_rate=args.dropout,
        attention_dropout_rate=args.attention_dropout,
        GCNII_layers=args.GCNII_layers,
        gcn_hidden_dim=args.gcn_hidden,
        gcnii_alpha=args.gcnii_alpha,
        gcnii_theta=args.gcnii_theta,
        contrast_dim=args.contrast_dim,
        contrast_temperature=args.contrast_temperature,
        contrast_balance=args.contrast_balance,
        decoder_hidden_dim=args.decoder_hidden,
    ).to(device)
    print("total params:", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.peak_lr,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = PolynomialDecayLR(
        optimizer,
        warmup_updates=args.warmup_updates,
        tot_updates=args.tot_updates,
        lr=args.peak_lr,
        end_lr=args.end_lr,
        power=args.lr_power,
    )
    criterion = F.binary_cross_entropy_with_logits

    best_epoch = 0
    best_auc = -np.inf
    best_prc = -np.inf
    best_metrics = None
    best_tpr = None
    best_fpr = None
    best_recall = None
    best_precision = None
    best_weights = None
    best_contributions = None

    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        optimizer.zero_grad()

        sampled_negative_indices = torch.randperm(
            fold_train_negative_pool.size(1),
            generator=negative_generator,
            device=device,
        )[:train_pos_edge_index.size(1)]
        train_neg_edge_index = fold_train_negative_pool[
            :, sampled_negative_indices]
        train_edge_index = torch.cat(
            [train_pos_edge_index, train_neg_edge_index], dim=1)
        train_labels = get_link_labels(
            train_pos_edge_index, train_neg_edge_index).to(device)

        train_details = model(
            processed_features,
            meta_data,
            drug_data,
            pair_edge_index=train_edge_index,
            positive_mask=positive_mask,
            candidate_mask=candidate_mask,
            return_details=True,
        )
        train_logits = train_details["fused_logits"]
        loss_fused = criterion(train_logits, train_labels)
        loss_local = (
            None
            if train_details["local_logits"] is None
            else criterion(train_details["local_logits"], train_labels)
        )
        loss_global = (
            None
            if train_details["global_logits"] is None
            else criterion(train_details["global_logits"], train_labels)
        )

        if MODEL_USES_LOCAL and MODEL_USES_GLOBAL:
            loss_classification = (
                (1.0 - args.aux_weight) * loss_fused
                + 0.5 * args.aux_weight * (loss_local + loss_global)
            )
        else:
            loss_classification = loss_fused

        loss_contrast = train_details["contrastive_loss"]
        if MODEL_USES_CONTRAST:
            loss_train = (
                (1.0 - args.contrast_weight) * loss_classification
                + args.contrast_weight * loss_contrast
            )
        else:
            loss_train = loss_classification

        loss_train.backward()
        optimizer.step()
        lr_scheduler.step()

        model.eval()
        with torch.no_grad():
            val_edge_index = torch.cat(
                [val_pos_edge_index, val_neg_edge_index], dim=1)
            val_details = model(
                processed_features,
                meta_data,
                drug_data,
                pair_edge_index=val_edge_index,
                return_details=True,
            )
            val_logits = val_details["fused_logits"]
            train_scores = torch.sigmoid(
                train_logits).detach().cpu().numpy()
            train_auc = metrics.roc_auc_score(
                train_labels.detach().cpu().numpy(),
                train_scores,
            )

            score_val = torch.sigmoid(
                val_logits).detach().cpu().numpy()
            label_val = get_link_labels(
                val_pos_edge_index, val_neg_edge_index).cpu().numpy()
            metric_tmp = get_metrics_fast(
                label_val,
                score_val,
                threshold_steps=args.metric_threshold_steps,
            )
            fpr, tpr, _ = metrics.roc_curve(label_val, score_val)
            precision, recall, _ = metrics.precision_recall_curve(
                label_val, score_val)
            val_auc = metrics.auc(fpr, tpr)
            val_prc = metrics.auc(recall, precision)
            weights = val_details["fusion_weights"].detach().cpu().numpy()
            weight_history_rows.append({
                "fold": k + 1,
                "epoch": epoch + 1,
                "local_weight": float(weights[0]),
                "global_weight": float(weights[1]),
                "train_loss": float(loss_train.item()),
                "val_auc": float(val_auc),
                "val_aupr": float(val_prc),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            })

            print(
                "Epoch:", epoch + 1,
                "Train Loss: %.4f" % loss_train.item(),
                "Fused BCE: %.4f" % loss_fused.item(),
                "Local BCE: %.4f" % optional_loss_value(loss_local),
                "Global BCE: %.4f" % optional_loss_value(loss_global),
                "Contrast: %.4f" % optional_loss_value(loss_contrast),
                "Weights: [%.4f, %.4f]" % (weights[0], weights[1]),
                "Acc: %.4f" % metric_tmp[0],
                "F1: %.4f" % metric_tmp[3],
                "Train AUC: %.4f" % train_auc,
                "Val AUC: %.4f" % val_auc,
                "Val PRC: %.4f" % val_prc,
                "Time: %.2f" % (time.time() - start),
            )

            prc_improved = val_prc > best_prc + args.min_delta
            prc_tied = abs(val_prc - best_prc) <= args.min_delta
            auc_improved = val_auc > best_auc + args.min_delta
            if prc_improved or (prc_tied and auc_improved):
                best_epoch = epoch + 1
                best_metrics = metric_tmp
                best_auc = val_auc
                best_prc = val_prc
                best_tpr = tpr
                best_fpr = fpr
                best_recall = recall
                best_precision = precision
                best_weights = weights.copy()

                local_logits = val_details["local_logits"]
                global_logits = val_details["global_logits"]
                if local_logits is None:
                    local_logits = torch.zeros_like(val_logits)
                if global_logits is None:
                    global_logits = torch.zeros_like(val_logits)
                local_np = local_logits.detach().cpu().numpy()
                global_np = global_logits.detach().cpu().numpy()
                edge_np = val_edge_index.detach().cpu().numpy()
                best_contributions = pd.DataFrame({
                    "fold": k + 1,
                    "meta_index": edge_np[0],
                    "drug_index": edge_np[1],
                    "label": label_val,
                    "local_weight": best_weights[0],
                    "global_weight": best_weights[1],
                    "local_logit": local_np,
                    "global_logit": global_np,
                    "local_contribution": best_weights[0] * local_np,
                    "global_contribution": best_weights[1] * global_np,
                    "fused_logit": val_logits.detach().cpu().numpy(),
                    "probability": score_val,
                })

    print(
        "Fold:", k + 1,
        "Best Epoch:", best_epoch,
        "Val Acc: %.4f" % best_metrics[0],
        "Val Pre: %.4f" % best_metrics[1],
        "Val Recall: %.4f" % best_metrics[2],
        "Val F1: %.4f" % best_metrics[3],
        "Val AUC: %.4f" % best_auc,
        "Val PRC: %.4f" % best_prc,
        "Weights: [%.4f, %.4f]" % (
            best_weights[0], best_weights[1]),
    )

    acc_result.append(best_metrics[0])
    pre_result.append(best_metrics[1])
    recall_result.append(best_metrics[2])
    f1_result.append(best_metrics[3])
    auc_result.append(round(best_auc, 4))
    prc_result.append(round(best_prc, 4))
    fprs.append(best_fpr)
    tprs.append(best_tpr)
    recalls.append(best_recall)
    precisions.append(best_precision)
    fold_weight_rows.append({
        "fold": k + 1,
        "best_epoch": best_epoch,
        "local_weight": best_weights[0],
        "global_weight": best_weights[1],
    })
    contribution_frames.append(best_contributions)
    save_weight_history(weight_history_rows, fold_weight_rows, RESULT_DIR)

print("## Training Finished !")
print("args", args)
print("Acc", acc_result)
print("Pre", pre_result)
print("Recall", recall_result)
print("F1", f1_result)
print("Auc", auc_result)
print("Prc", prc_result)
print(
    "AUC mean: %.4f, standard deviation: %.4f\n"
    "Accuracy mean: %.4f, standard deviation: %.4f\n"
    "Precision mean: %.4f, standard deviation: %.4f\n"
    "Recall mean: %.4f, standard deviation: %.4f\n"
    "F1-score mean: %.4f, standard deviation: %.4f\n"
    "PRC mean: %.4f, standard deviation: %.4f"
    % (
        np.mean(auc_result), np.std(auc_result),
        np.mean(acc_result), np.std(acc_result),
        np.mean(pre_result), np.std(pre_result),
        np.mean(recall_result), np.std(recall_result),
        np.mean(f1_result), np.std(f1_result),
        np.mean(prc_result), np.std(prc_result),
    )
)

pd.DataFrame(recalls).to_csv(
    RESULT_DIR / "recalls_1.csv", index=False)
pd.DataFrame(precisions).to_csv(
    RESULT_DIR / "precisions_1.csv", index=False)
pd.DataFrame(fprs).to_csv(
    RESULT_DIR / "fprs_1.csv", index=False)
pd.DataFrame(tprs).to_csv(
    RESULT_DIR / "tprs_1.csv", index=False)

weight_frame = pd.DataFrame(fold_weight_rows)
weight_frame.to_csv(RESULT_DIR / "fusion_weights.csv", index=False)
save_weight_history(weight_history_rows, fold_weight_rows, RESULT_DIR)
pd.concat(contribution_frames, ignore_index=True).to_csv(
    RESULT_DIR / "branch_contributions.csv", index=False)
save_weight_plot(weight_frame, RESULT_DIR)

plot_auc_curves(
    fprs, tprs, auc_result, directory=RESULT_DIR, name="test_auc")
plot_prc_curves(
    precisions, recalls, prc_result,
    directory=RESULT_DIR, name="test_prc")
