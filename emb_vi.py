import torch

# for tsne
import pandas as pd
import numpy as np
from sklearn import manifold

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import seaborn as sns

import os
import argparse
from utilities import *
from torch_geometric import seed_everything
import colorcet as cc


def visualize(tasks, memory_bank, encoder, result_file_name, device):
    for k, task in enumerate(tasks):
        replayed_graph = memory_bank[k]
        emb_origin = encoder.encode(task.x.to(device), task.adj_t.to(device)).detach().cpu()
        emb_replay = encoder.encode(replayed_graph.x.to(device), replayed_graph.adj_t.to(device)).detach().cpu()

        tsne = manifold.TSNE(n_components=2, perplexity=50, random_state=1024, n_iter=2000, verbose=1)

        feat_2d = tsne.fit_transform(torch.cat((emb_origin, emb_replay), 0))

        feat_2d_origin = feat_2d[:emb_origin.shape[0], :]
        feat_2d_replay = feat_2d[emb_origin.shape[0]:, :]
        
        origin_df = pd.DataFrame(columns=["CP1", "CP2", "labels"], data=np.column_stack((feat_2d_origin, task.y.view(-1, 1))))
        replay_df = pd.DataFrame(columns=["CP1", "CP2", "labels"], data=np.column_stack((feat_2d_replay, replayed_graph.y.view(-1, 1))))
        
        colors_origin = ["#9DB49A", "#F8A68F"] 
        colors_replay = ["#59CD90", "#EE6352"]

        plt.figure()
        for i, cls in enumerate(task.classes):
            ax = sns.scatterplot(data=origin_df.loc[origin_df["labels"] == cls], x="CP1", y="CP2", label=f"class {cls} (Origin)", legend=False, color=colors_origin[i], s=60)
        for i, cls in enumerate(task.classes):
            ax = sns.scatterplot(data=replay_df.loc[replay_df["labels"] == cls], x="CP1", y="CP2", label=f"class {cls} (Replay)", legend=False, color=colors_replay[i], s=240, marker="*")
        ax.set(xlabel=None)
        ax.set(ylabel=None)
        plt.xticks([])
        plt.yticks([])
        font = font_manager.FontProperties(style='normal', size=12)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.175), ncol=2, prop=font)
        figure_name = os.path.join("./visualization/visualized_data", result_file_name + f"_task_{k}.png")
        plt.savefig(figure_name, format='png', dpi=300)
        plt.close()
        print(figure_name)
        if k == 5:
            break
    
def visualize_emb(memory_bank, embeds, result_file_name):
    n_tasks = 35
    tsne = manifold.TSNE(n_components=2, perplexity=50, random_state=1024, n_iter=2000, verbose=1)
    feat_2d = tsne.fit_transform(torch.cat(embeds[:n_tasks], 0))
    labels = []
    for memory in memory_bank[:n_tasks]:
        labels.append(memory.y.view(-1, 1))
    labels = torch.cat(labels, 0).squeeze_()
    emb_df = pd.DataFrame(columns=["CP1", "CP2", "labels"], data=np.column_stack((feat_2d, labels.view(-1, 1))))
    plt.figure(figsize=(18, 10))
    palette = sns.color_palette(cc.glasbey, n_colors=n_tasks*2)
    ax = sns.scatterplot(data=emb_df, x="CP1", y="CP2", hue="labels", palette=palette)
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    plt.xticks([])
    plt.yticks([])
    plt.legend(ncol=4, bbox_to_anchor=(1, 1))
    figure_name = os.path.join("./visualization/visualized_data", result_file_name + f"_embeddings.png")
    plt.subplots_adjust(right=0.7)
    plt.savefig(figure_name, format='png', dpi=300)
    plt.close()
    print(figure_name)
            

def main():
    parser = argparse.ArgumentParser()
    # Arguments for datasets.
    parser.add_argument('--data-dir', type=str, default="./data")
    parser.add_argument('--dataset-name', type=str, default="corafull")
    parser.add_argument('--cls-per-task', type=int, default=2)

    # Argumnets for CGL methods.
    # parser.add_argument('--tim', action=argparse.BooleanOptionalAction)
    parser.add_argument('--cgl-method', type=str, default="cgm")
    parser.add_argument('--cls-epoch', type=int, default=200)
    parser.add_argument('--budget', type=int, default=2)
    parser.add_argument('--m-update', type=str, default="all")
    parser.add_argument('--cgm-args', type=str, default="{}")

    # Others
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--result-path', type=str, default="./results")
    parser.add_argument('--data-type', type=str, default="graph")

    args = parser.parse_args()
    print(args)

    seed_everything(args.seed)

    result_file_name = get_result_file_name(args)
    memory_bank_file_name = os.path.join(args.result_path, "memory_bank" , result_file_name)
    emb_file_name = os.path.join(args.result_path, "embedding" , result_file_name)
    
    if os.path.exists(memory_bank_file_name):
        memory_bank = torch.load(memory_bank_file_name)
        if args.data_type == "graph":
            dataset = get_dataset(args)
            task_file = os.path.join(args.data_dir, "streaming", f"{args.dataset_name}.streaming")
            tasks = torch.load(task_file)
            encoder = get_backbone_model(dataset, tasks, args)
            visualize(tasks, memory_bank, encoder, result_file_name, args.device)
        elif args.data_type == "embedding":
            embs = torch.load(emb_file_name)
            visualize_emb(memory_bank, embs, result_file_name)
    else:
        print("The memory bank file does not exist!")

if __name__ == '__main__':
    main()