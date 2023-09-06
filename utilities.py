from torch_geometric.datasets import CoraFull, Reddit
from ogb.nodeproppred import PygNodePropPredDataset

from backbones.gcn import GCN
from methods.joint import Joint
from methods.ergnn import ERGNN
from methods.ssm import SSM
from methods.cgm import CGM


def get_result_file_name(args):
    if args.cgl_method == "joint":
        result_name = f""
    elif args.cgl_method == "ergnn":
        result_name = f"_MFPlus"
    elif args.cgl_method == "ssm":
        result_name = f"_random"
    elif "cgm" in args.cgl_method:
        cgm_args = eval(args.cgm_args)
        result_name = f"_{cgm_args['feat_init']}_feat_{cgm_args['feat_lr']}_{cgm_args['n_encoders']}_{cgm_args['n_layers']}_layer_{cgm_args['hid_dim']}_GCN_hop_{cgm_args['hop']}"
    return f"{args.dataset_name}_{args.budget}_{args.cgl_method}" + result_name

def print_performance_matrix(performace_matrix, m_update):
    for k in range(performace_matrix.shape[0]):
        accs = []
        for k_ in range(k + 1):
            acc = performace_matrix[k, k_]
            accs.append(acc)
            if m_update == "all":
                print(f"T{k_} {acc:.2f}",end="|")
            elif m_update == "onlyCurrent":
                if k == k_:
                    print(f"T{k_} {acc:.2f}",end="|")
        print(f"AP: {sum(accs) / len(accs):.2f}")

def get_dataset(args):
    if args.dataset_name == "corafull":
        dataset = CoraFull(args.data_dir)
    elif args.dataset_name == "arxiv":
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=args.data_dir)
        data = dataset[0]
        data.y.squeeze_()
    elif args.dataset_name == "reddit":
        dataset = Reddit(args.data_dir)
    elif args.dataset_name == "products":
        dataset = PygNodePropPredDataset(name="ogbn-products", root=args.data_dir)
        dataset.data.y.squeeze_()
    return dataset
    
def get_backbone_model(dataset, data_stream, args):
    if args.IL == "classIL":
        model = GCN(dataset.num_features, 256, data_stream.n_tasks * args.cls_per_task, 2).to(args.device)
    else:
        model = GCN(dataset.num_features, 256, args.cls_per_task, 2).to(args.device)
    return model

def get_cgl_model(model, data_stream, args):
    if args.cgl_method == "joint":
        cgl_model = Joint(model, data_stream.tasks, None, args.m_update, args.device)
    elif args.cgl_method == "ergnn":
        cgl_model = ERGNN(model, data_stream.tasks, args.budget, args.m_update, args.device)
    elif args.cgl_method == "ssm":
        cgl_model = SSM(model, data_stream.tasks, args.budget, args.m_update, args.device)
    elif args.cgl_method == "cgm":
        cgl_model = CGM(model, data_stream.tasks, args.budget, args.m_update, args.device, eval(args.cgm_args))
    return cgl_model