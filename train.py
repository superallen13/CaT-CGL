# For graph learning
import torch
from torch_geometric import seed_everything

# Utility
import os
import argparse
import numpy as np
from utilities import *
from data_stream import Streaming
from torch_geometric.data import Batch
from backbones.gnn import train_node_classifier, train_node_classifier_batch, eval_node_classifier


def evaluate(args, dataset, data_stream, memory_banks):
    APs = []
    AFs = []
    for i in range(args.repeat):
        memory_bank = memory_banks[i]
        # Initialize the performance matrix.
        performace_matrix = torch.zeros(len(memory_bank), len(memory_bank))
        model = get_backbone_model(dataset, data_stream, args)
        cgl_model = get_cgl_model(model, data_stream, args)
        tasks = cgl_model.tasks

        for k in range(len(memory_bank)):
            opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

            # train
            if args.dataset_name == "products" and args.cgl_method == "joint":
                max_cls = torch.unique(memory_bank[k].y)[-1]
                batches = memory_bank[:k+1]
                for data in batches:
                    data.to(args.device)
                model = train_node_classifier_batch(model, batches, opt, n_epoch=args.cls_epoch, incremental_cls=(0, max_cls+1))
            else:
                if args.tim:
                    if args.m_update == "all":
                        replayed_graphs = Batch.from_data_list(memory_bank[:k+1])
                    elif args.m_update == "onlyCurrent":
                        replayed_graphs = Batch.from_data_list([memory_bank[-1]])
                else:
                    replayed_graphs = Batch.from_data_list(memory_bank[:k] + [tasks[k]])
        
                replayed_graphs.to(args.device, "x", "y", "adj_t")
                max_cls = torch.unique(replayed_graphs.y)[-1]
                model = train_node_classifier(model, replayed_graphs, opt, n_epoch=args.cls_epoch, incremental_cls=(0, max_cls+1))

            # Test the model from task 0 to task k
            accs = []
            AF = 0
            for k_ in range(k + 1):
                task_ = tasks[k_].to(args.device, "x", "y", "adj_t")
                if args.IL == "classIL":
                    acc = eval_node_classifier(model, task_, incremental_cls=(0, max_cls+1)) * 100
                else:
                    max_cls = torch.unique(task_.y)[-1]
                    acc = eval_node_classifier(model, task_, incremental_cls=(max_cls+1-data_stream.cls_per_task, max_cls+1)) * 100
                accs.append(acc)
                task_.to("cpu")
                print(f"T{k_} {acc:.2f}", end="|")
                performace_matrix[k, k_] = acc
            AP = sum(accs) / len(accs)
            print(f"AP: {AP:.2f}", end=", ")
            for t in range(k):
                AF += performace_matrix[k, t] - performace_matrix[t, t]
            AF = AF / k if k != 0 else AF
            print(f"AF: {AF:.2f}")
        APs.append(AP)
        AFs.append(AF)
    print(f"AP: {np.mean(APs):.1f}±{np.std(APs, ddof=1):.1f}")
    print(f"AF: {np.mean(AFs):.1f}±{np.std(AFs, ddof=1):.1f}")

def main():
    parser = argparse.ArgumentParser()
    # Arguments for data.
    parser.add_argument('--dataset-name', type=str, default="corafull")
    parser.add_argument('--cls-per-task', type=int, default=2)
    parser.add_argument('--data-dir', type=str, default="./data")
    parser.add_argument('--result-path', type=str, default="./results")

    # Argumnets for CGL methods.
    parser.add_argument('--tim', action='store_true')
    parser.add_argument('--cgl-method', type=str, default="cgm")
    parser.add_argument('--cls-epoch', type=int, default=200)
    parser.add_argument('--budget', type=int, default=2)
    parser.add_argument('--m-update', type=str, default="all")
    parser.add_argument('--cgm-args', type=str, default="{}")
    parser.add_argument('--IL', type=str, default="classIL")

    # Others
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--rewrite', action='store_true')

    args = parser.parse_args()
    print(args)

    seed_everything(args.seed)

    # Get file names.
    result_file_name = get_result_file_name(args)
    memory_bank_file_name = os.path.join(args.result_path, "memory_bank" , result_file_name)
    task_file = os.path.join(args.data_dir, "streaming", f"{args.dataset_name}.streaming")

    dataset = get_dataset(args)
    if os.path.exists(task_file):
        data_stream = torch.load(task_file)
    else:
        data_stream = Streaming(args.cls_per_task, dataset)
        torch.save(data_stream, task_file)

    # Get memory banks.
    memory_banks = []
    for i in range(args.repeat):
        if os.path.exists(memory_bank_file_name + f"_repeat_{i}") and not args.rewrite:
            memory_bank = torch.load(memory_bank_file_name + f"_repeat_{i}")
            memory_banks.append(memory_bank)  # load the memory bank from the file.
        else:
            model = get_backbone_model(dataset, data_stream, args)
            cgl_model = get_cgl_model(model, data_stream, args)

            memory_bank, performace_matrix = cgl_model.observer()
            memory_banks.append(memory_bank)
            torch.save(memory_bank, memory_bank_file_name + f"_repeat_{i}")
    
    evaluate(args, dataset, data_stream, memory_banks)


if __name__ == '__main__':
    main()