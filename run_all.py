import torch, random
from model_def import SchedulingDataset
from train_gnn import train
# from accuracy import evaluate
from accuracy2 import evaluate
from CplexModel import evaluate_solution


# DATA_PATH = "TrainingDataset_aveY/data_storage_var"
DATA_PATH = "Evaluation_Dataset/data_storage_test_4days"
MODEL_PATH = "scheduling_gnn_var.pth"
RESULT_STORE_PATH = "simulation_results_150data_4days_soft_fixing_evaluate.pkl"

def get_split_data(dataset, train_ratio=0.05, seed=42):
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    
    train_set, test_set = torch.utils.data.random_split(
        dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    return train_set, test_set

def detect_all_edge_dims(data):
    edge_dims_dict = {}
    for etype in data.edge_types:
        attr = getattr(data[etype], 'edge_attr', None)
        # Store the actual dimension, or 0/None if no attributes exist
        edge_dims_dict[etype] = attr.shape[-1] if attr is not None else None
    return edge_dims_dict


full_dataset = SchedulingDataset(DATA_PATH)[:3000]
# full_dataset = SchedulingDataset(DATA_PATH)
train_set, test_set = get_split_data(full_dataset)

sample_data = full_dataset[0]
feat_dims = {
    'setup': sample_data['var_setup'].x.shape[1],
    'startup': sample_data['var_startup'].x.shape[1],
    'product': sample_data['var_prod'].x.shape[1],
    'inventory': sample_data['var_invent'].x.shape[1],
    'lostsales': sample_data['var_lost'].x.shape[1],
    'con_invent_balance': sample_data['con_invent_balance'].x.shape[1],
    'con_prod_allow': sample_data['con_prod_allow'].x.shape[1],
    'con_capacity': sample_data['con_capacity'].x.shape[1],
    'con_machine': sample_data['con_machine'].x.shape[1],
    'con_startup': sample_data['con_startup'].x.shape[1]
}
# print(f"Total Edges: {sample_data.raw_r.size(0)}")
# print(f"Max Hour Index: {sample_data['var_setup'].raw_r.max()}")
# print(f"Hour Index: {sample_data['var_setup'].raw_r}")
# print(f"Max T Index: {sample_data.raw_t.max()}")
# print(f"Max Product Index: {sample_data.raw_j.max()}")



print(f"Detected Dimensions: {feat_dims}")

e_dim = detect_all_edge_dims(sample_data)
# print(f'Detected edge dimensions: {e_dim}')


# train(train_set, test_set[:50], MODEL_PATH, feat_dims, e_dim)
# print('Training phase done.')
# evaluate(test_set, MODEL_PATH, feat_dims, e_dim)


# evaluate(train_set, MODEL_PATH, feat_dims, e_dim)

# MODEL_PATH = ['var_4800data_bce/checkpoint_epoch_19.pth','var_2400data_bce/checkpoint_epoch_19.pth', 'var_150data_bce/checkpoint_epoch_19.pth', 
#             'var_150data_spo/checkpoint_epoch_49.pth', 'var_150data_hybrid/checkpoint_epoch_49.pth','var_150data_fy/checkpoint_epoch_49.pth',
#             'var_150data_spo_moving_ave_3/checkpoint_epoch_49.pth', 'var_150data_spo_moving_ave_5/checkpoint_epoch_49.pth']
# X_LABELS = ['4800data_bce_epoch_20','2400data_bce_epoch_20', '150data_bce_epoch_20',
#             '150data_spo_epoch_50', '150data_hybrid_epoch_50', '150data_fy_epoch_50',
#             '150data_spo_moving_ave_3', '150data_spo_moving_ave_5']

# MODEL_PATH = ['var_150data_spo_CE_pen/checkpoint_epoch_49.pth']
MODEL_PATH = ['var_150data_bce/checkpoint_epoch_19.pth']
X_LABELS = ['150data_soft_fixing_epoch_20']

# MODEL_PATH = ['var_150data_spo_moving_ave_3/checkpoint_epoch_49.pth', 'var_150data_spo_moving_ave_5/checkpoint_epoch_49.pth']
# X_LABELS = ['150data_spo_moving_ave_3', '150data_spo_moving_ave_5']
# MODEL_PATH = ['var_2400data_spo/checkpoint_epoch_49.pth', 'var_2400data_spo/checkpoint_epoch_99.pth']
# X_LABELS = ['2400data_spo_epoch_50', '2400data_spo_epoch_100']

assert len(MODEL_PATH) == len(X_LABELS)
validation_dataset = SchedulingDataset(DATA_PATH)[:40]
evaluate(validation_dataset, MODEL_PATH, RESULT_STORE_PATH, X_LABELS, feat_dims, e_dim)


# # see how the data looks like
# data = train_set[0]
# print(len(train_set))
# print("--- Graph Summary ---")
# print(f"Number of nodes: {data.num_nodes}")
# print(f"Number of edges: {data.num_edges}")
# print(f"Features per node: {data.num_node_features}") # This should match your model's input_dim
# print(f"Is directed: {data.is_directed()}")

# print("\n--- Tensor Shapes ---")
# # print(f"Node Features (x): {data.x.shape}")   # [Nodes, Features]
# # print(f"Edge Index: {data.edge_index.shape}")  # [2, Edges]
# print(f"Labels (y): {data['var_setup'].y.shape}")
# print(f"Labels (x): {data['var_startup'].x.shape}")
# print(f"\nTotal Setups in this scenario: {data['var_setup'].y.sum().item()}")

# print(f"First 5 Node Feature Vectors: {data.x[:5]}")
# print(f"\nFirst 5 Labels (Setups): {data.y[:5]}")


