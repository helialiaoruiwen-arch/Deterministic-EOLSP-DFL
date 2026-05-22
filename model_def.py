import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, GATv2Conv, GINEConv, BatchNorm, LayerNorm
from torch_geometric.data import Dataset
from torch_geometric.utils import softmax
from CplexModel import solve_with_penalization_in_obj, evaluate_solution
import numpy as np
# import torch_scatter
import os
import matplotlib.pyplot as plt

HEADS = 4
EDGE_ATTR_SIZE = 6
class SchedulerForward(torch.nn.Module):
    def __init__(self, hidden_channels, feat_dims, edge_dims_dict):
        super().__init__()
        self.hidden_channels = hidden_channels

        # define the encoder
        self.setup_encoder = torch.nn.Linear(feat_dims['setup'], hidden_channels)
        self.startup_encoder = torch.nn.Linear(feat_dims['startup'], hidden_channels)
        self.prod_encoder = torch.nn.Linear(feat_dims['product'], hidden_channels)
        self.invent_encoder = torch.nn.Linear(feat_dims['inventory'], hidden_channels)
        self.lost_encoder = torch.nn.Linear(feat_dims['lostsales'], hidden_channels)

        self.con_invent_encoder = torch.nn.Linear(feat_dims['con_invent_balance'], hidden_channels)
        self.con_prod_allow_encoder = torch.nn.Linear(feat_dims['con_prod_allow'], hidden_channels)
        self.con_capacity_encoder = torch.nn.Linear(feat_dims['con_capacity'], hidden_channels)
        self.con_machine_encoder = torch.nn.Linear(feat_dims['con_machine'], hidden_channels)
        self.con_startup_encoder = torch.nn.Linear(feat_dims['con_startup'], hidden_channels)

        # define the decoder
        self.setup_decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels // 2, 1, bias=False) 
        )
            

        # Define the node types
        self.node_types = ['var_setup', 'var_startup', 'var_prod', 'var_invent', 'var_lost',
                            'con_invent_balance', 'con_prod_allow', 'con_capacity', 'con_machine', 'con_startup']
        # Define the edge types once for reuse
        self.base_edges = [
            ('var_invent', 'to', 'con_invent_balance'),
            ('var_lost', 'to', 'con_invent_balance'),
            ('var_prod', 'to', 'con_invent_balance'),
            ('var_setup', 'restrict', 'con_prod_allow'),
            ('var_prod', 'allowed', 'con_prod_allow'),
            ('var_prod', 'limited', 'con_capacity'),
            ('var_setup', 'only', 'con_machine'),
            ('var_setup', 'relate', 'con_startup'),
            ('var_startup', 'relate', 'con_startup'),
            # ('var_setup', 'conflicts_with', 'var_setup')
        ]

        # Create the reverse edges (Constraint -> Variable)
        self.rev_edges = [(dst, f'rev_{rel}', src) for src, rel, dst in self.base_edges]

        self.all_edge_types = self.base_edges + self.rev_edges
        self.edge_dim = edge_dims_dict


        def create_gine_mlp():
            return torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels)
            )
        
        gat_edges = {edge: GATv2Conv(hidden_channels, hidden_channels, edge_dim=edge_dims_dict.get(edge), add_self_loops=False) for edge in self.all_edge_types}
        # conv1 hidden_channels -> hidden_channels
        self.conv1 = HeteroConv({
            **gat_edges,
            # ('var_setup', 'conflicts_with', 'var_setup'): SAGEConv(hidden_channels, hidden_channels),
        }, aggr='sum')
        
        # conv2 hidden_channels -> hidden_channels
        self.conv2 = HeteroConv({
            **gat_edges,
            # ('var_setup', 'conflicts_with', 'var_setup'): SAGEConv(hidden_channels, hidden_channels)
        }, aggr='sum')

        # self.conv2 = HeteroConv({
        #     edge: GINEConv(create_gine_mlp(), edge_dim=edge_dims_dict.get(edge)) # Use 1 for your coefficients
        #     for edge in self.all_edge_types
        # }, aggr='sum')

        # conv3 hidden_channels -> hidden_channels
        self.conv3 = HeteroConv({
            **gat_edges,
            # ('var_setup', 'conflicts_with', 'var_setup'): SAGEConv(hidden_channels, hidden_channels)
        }, aggr='sum')

        # self.conv3 = HeteroConv({
        #     edge: GINEConv(create_gine_mlp(), edge_dim=edge_dims_dict.get(edge)) # Use 1 for your coefficients
        #     for edge in self.all_edge_types
        # }, aggr='sum')

        # conv4 hidden_channels -> hidden_channels
        self.conv4 = HeteroConv({
            **gat_edges,
            # ('var_setup', 'conflicts_with', 'var_setup'): SAGEConv(hidden_channels, hidden_channels)
        }, aggr='sum')

        # self.conv4 = HeteroConv({
        #     edge: GINEConv(create_gine_mlp(), edge_dim=edge_dims_dict.get(edge)) # Use 1 for your coefficients
        #     for edge in self.all_edge_types
        # }, aggr='sum')

        # conv5 hidden_channels -> hidden_channels
        self.conv5 = HeteroConv({
            **gat_edges,
            # ('var_setup', 'conflicts_with', 'var_setup'): SAGEConv(hidden_channels, hidden_channels)
        }, aggr='sum')

        # self.conv5 = HeteroConv({
        #     edge: GINEConv(create_gine_mlp(), edge_dim=edge_dims_dict.get(edge)) # Use 1 for your coefficients
        #     for edge in self.all_edge_types
        # }, aggr='sum')

        # conv6 hidden_channels -> hidden_channels
        self.conv6 = HeteroConv({
            **gat_edges,
            # ('var_setup', 'conflicts_with', 'var_setup'): SAGEConv(hidden_channels, hidden_channels)
        }, aggr='sum')

        # self.conv6 = HeteroConv({
        #     edge: GINEConv(create_gine_mlp(), edge_dim=edge_dims_dict.get(edge)) # Use 1 for your coefficients
        #     for edge in self.all_edge_types
        # }, aggr='sum')

        # conv7 hidden_channels -> hidden_channels
        self.conv7 = HeteroConv({
            **gat_edges,
            # ('var_setup', 'conflicts_with', 'var_setup'): SAGEConv(hidden_channels, hidden_channels)
        }, aggr='sum')

        # conv8 hidden_channels -> hidden_channels
        self.conv8 = HeteroConv({
            **gat_edges,
            # ('var_setup', 'conflicts_with', 'var_setup'): SAGEConv(hidden_channels, hidden_channels)
        }, aggr='sum')

        self.convs = torch.nn.ModuleList([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6])

        # normalize the features
        self.norm1 = torch.nn.ModuleDict({
            node_type: LayerNorm(hidden_channels) for node_type in self.node_types
        })
        self.norm2 = torch.nn.ModuleDict({
            node_type: LayerNorm(hidden_channels) for node_type in self.node_types
        })
        self.norm3 = torch.nn.ModuleDict({
            node_type: LayerNorm(hidden_channels) for node_type in self.node_types
        })

    def forward(self, data):
        # 1. Encode Variable Nodes
        # Map your custom encoder names to the actual node type keys in data
        x_dict = {
            'var_setup': self.setup_encoder(data['var_setup'].x),
            'var_startup': self.startup_encoder(data['var_startup'].x),
            'var_prod': self.prod_encoder(data['var_prod'].x),
            'var_invent': self.invent_encoder(data['var_invent'].x),
            'var_lost': self.lost_encoder(data['var_lost'].x),
            'con_invent_balance': self.con_invent_encoder(data['con_invent_balance'].x),
            'con_prod_allow': self.con_prod_allow_encoder(data['con_prod_allow'].x),
            'con_capacity': self.con_capacity_encoder(data['con_capacity'].x),
            'con_machine': self.con_machine_encoder(data['con_machine'].x),
            'con_startup': self.con_startup_encoder(data['con_startup'].x),
        }

        # print(self.edge_dim)
        edge_attr_dict = {}
        for etype in data.edge_types:
            attr = getattr(data[etype], 'edge_attr', None)
            if attr is not None:
                edge_attr_dict[etype] = attr.float()
            else:
                # Fallback: If an edge exists but has no attributes, give it 1.0s
                # print(etype)
                # num_edges = data[etype].edge_index.size(1)
                # edge_attr_dict[etype] = torch.ones((num_edges, 1), 
                #                                 device=data['var_setup'].x.device)
                pass


        # Message Passing (HeteroConv)
        # We pass the encoded features and the edge information through the layers
        for conv in self.convs:
            # If your conv layers support edge_attr (like GATConv or GINEConv), pass them here
            # Otherwise, simple layers like SAGEConv only take x_dict and edge_index_dict

            x_dict = conv(x_dict, data.edge_index_dict, edge_attr_dict=edge_attr_dict)

            h_old = x_dict
            x_dict = conv(x_dict, data.edge_index_dict, edge_attr_dict)
            x_dict = {key: torch.relu(x + h_old[key]) for key, x in x_dict.items()} # Residual
            
            # # Apply activation and dropout between layers
            # x_dict = {key: torch.relu(x) for key, x in x_dict.items()}



        # --- SETUP NODE PREDICTION ---
        # Decode the hidden states to raw scores (logits)
        setup_logits = self.setup_decoder(x_dict['var_setup'])

        # --- SOFTMAX (Competition Logic) ---
        # Ensure slot_index is unique across the whole batch
        batch_idx = data['var_setup'].batch
        num_micro = data['var_setup'].raw_r.max().item() + 1
        slot_index = (batch_idx * num_micro) + data['var_setup'].raw_r.long()
        
        Temp = 10.0 
        y_hat = softmax(setup_logits/Temp, slot_index)

        y_hat = y_hat.flatten()


        # y_hat = (setup_logits - setup_logits.mean()) / (setup_logits.std() + 1e-6)

        # y_hat = torch.sigmoid(setup_logits / 10.0)

        # # --- NATIVE SCATTER (Daily Sum) ---
        # num_macro = data.raw_t.max().item() + 1

        # daily_prod = torch.zeros(int(num_macro), device=y_hat.device, dtype=torch.float32)

        # daily_prod.index_add_(0, data.raw_t, y_hat)

        return y_hat

    

class SchedulingDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        # Filter for .pt files and sort to ensure consistent indexing
        self.file_names = sorted([f for f in os.listdir(root_dir) if f.endswith('.pt')])

    def len(self):
        return len(self.file_names)

    def get(self, idx):
        path = os.path.join(self.root_dir, self.file_names[idx])
        data = torch.load(path, weights_only=False)

        return data

class SPOPlusFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred_theta, true_y, scenario, epoch, i):
        theta = pred_theta.detach().cpu().numpy()
        y_true = true_y.detach().cpu().numpy()
        # Calculate SPO costs: 2*w - w_true
        spo_theta = 2 * theta - y_true
        
        # # CLAMPING: We clip the theta be between 0 and 1
        spo_theta_safe = np.clip(spo_theta, 0.0, 1.0)

        # we normalize theta in [-1,2] to [0,1] using min max
        # spo_theta_safe = (spo_theta + 1)/3.0
        
        # y*(2w - w_true)
        spo_obj, spo_y, _ = solve_with_penalization_in_obj(scenario, spo_theta_safe)
        # eval_obj, _ = evaluate_solution(scenario, spo_y)

        y_spo_tensor = torch.from_numpy(spo_y).to(pred_theta.device)
        y_spo_tensor_reshape = y_spo_tensor.reshape(scenario.J+1, scenario.R+1)[:,1:]
        
        # _, y_theta, _ = solve_with_penalization_in_obj(scenario, theta)
        # y_theta_tensor = torch.from_numpy(y_theta).to(pred_theta.device)
        # y_theta_tensor = y_theta_tensor.reshape(scenario.J+1, scenario.R+1)[:,1:]
        # eval_obj, _ = evaluate_solution(scenario, y_theta)

        # hamming_dist = torch.sum(torch.abs(true_y - y_theta_tensor))
        # print(f"Discrepancy in decisions: {hamming_dist.item()}")
        
        np.set_printoptions(precision=2, suppress=True)
        # if i == 0:
        #     print('theta', np.array(theta))
        #     # print('y_theta', y_theta_tensor)
        #     print('spo_theta_safe', np.array(spo_theta_safe))
        #     print('y_spo', y_spo_tensor_reshape)
        #     print('y_true', y_true)
        #     # print('grad', (1 - 2 * (2 * y_spo_tensor_reshape - true_y)) * 1000)
        #     print('true loss', eval_obj-scenario.obj)
        #     print('spo obj, true obj', spo_obj, scenario.obj)

        pred_y = torch.from_numpy(theta).to(pred_theta.device)
        ctx.save_for_backward(y_spo_tensor_reshape, true_y, pred_y)
        
        # Theoretical SPO+ loss value
        spo_loss = spo_obj - scenario.obj

        loss = spo_loss

        # bce = torch.nn.BCELoss()
        # alpha = 1.0
        # bce_loss = bce(pred_y, true_y)
        # loss = alpha*spo_loss + (1-alpha)*bce_loss*100

        torch.set_printoptions(precision=2, sci_mode=False)
        # if loss < -0.001:
        #     print('------loss------', loss, spo_obj, scenario.obj)
        #     print('theta', pred_theta)
        #     print('y_theta', y_theta_tensor)
        #     print('spo_theta', torch.tensor(spo_theta))
        #     print('y_spo', y_spo_tensor_reshape)
        #     print('y_true', true_y)
        #     print('grad', (true_y - y_spo_tensor_reshape) * 1000)
        #     print('true loss', eval_obj-scenario.obj)
        loss_gap = loss / scenario.obj
        return torch.tensor(loss, device=pred_theta.device, dtype=pred_theta.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        y_spo, y_true, y_pred = ctx.saved_tensors
        # Gradient: 2 * (y_true - y_spo)
        # grad = ( - 2 * (2 * y_spo - y_true)) * 1000
        spo_grad = (- 4 * (y_true - y_spo)) * 100
        # print(grad)

        # bce_grad = calculate_bce_gradient(y_pred, y_true)
        # clipped_bce = torch.clamp(bce_grad*100, min=-400, max=400)

        grad = spo_grad

        # alpha = 1.0
        # grad = alpha* spo_grad + (1-alpha)*clipped_bce

        # print('spo grad', spo_grad)
        # print('bce grad', bce_grad)
        # print('grad', grad)
        # print('pred y', y_pred)
        # print('true y', y_true)
        
        # hamming_dist = torch.sum(torch.abs(y_true - y_spo))
        # print(f"Discrepancy in decisions: {hamming_dist.item()}")
        return grad * grad_output, None, None, None, None


class FYFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred_theta, true_y, scenario, epoch, i):
        theta = pred_theta.detach().cpu().numpy()
        y_true = true_y.detach().cpu().numpy()
        
        sigma = 0.02
        noise = np.random.normal(0, sigma, size=theta.shape)
        # gumbel noise
        # noise = np.random.gumbel(0, sigma, size=theta.shape)
        fy_theta = theta + noise
        
        np.set_printoptions(precision=2, suppress=True)
        # print(theta)
        # print(fy_theta)

        # # CLAMPING: We clip the theta be between 0 and 1
        fy_theta_safe = np.clip(fy_theta, 0.0, 1.0)

        # theta_flat = np.array(theta).flatten()
        # fy_theta_flat = np.array(fy_theta_safe).flatten()
        # noise_flat = np.array(noise).flatten()
        # plt.figure(figsize=(10, 5))
        # plt.plot(theta_flat, label='Original Theta', color='blue', alpha=0.7)
        # plt.plot(fy_theta_flat, label='Theta + White Noise', color='red', alpha=0.5, linestyle='--')
        # plt.show()
        
        fy_obj, fy_y, _ = solve_with_penalization_in_obj(scenario, fy_theta_safe)
        # eval_obj, _ = evaluate_solution(scenario, spo_y)

        y_fy_tensor = torch.from_numpy(fy_y).to(pred_theta.device)
        y_fy_tensor_reshape = y_fy_tensor.reshape(scenario.J+1, scenario.R+1)[:,1:]
        
        # _, y_theta, _ = solve_with_penalization_in_obj(scenario, theta)
        # y_theta_tensor = torch.from_numpy(y_theta).to(pred_theta.device)
        # y_theta_tensor = y_theta_tensor.reshape(scenario.J+1, scenario.R+1)[:,1:]
        # eval_obj, _ = evaluate_solution(scenario, y_theta)

        # hamming_dist = torch.sum(torch.abs(true_y - y_theta_tensor))
        # print(f"Discrepancy in decisions: {hamming_dist.item()}")
        
        np.set_printoptions(precision=2, suppress=True)
        # if i == 0:
        #     print('theta', np.array(theta))
        #     # print('y_theta', y_theta_tensor)
        #     print('spo_theta_safe', np.array(spo_theta_safe))
        #     print('y_spo', y_spo_tensor_reshape)
        #     print('y_true', y_true)
        #     # print('grad', (1 - 2 * (2 * y_spo_tensor_reshape - true_y)) * 1000)
        #     print('true loss', eval_obj-scenario.obj)
        #     print('spo obj, true obj', spo_obj, scenario.obj)

        pred_y = torch.from_numpy(theta).to(pred_theta.device)
        ctx.save_for_backward(y_fy_tensor_reshape, true_y, pred_y)
        
        # Theoretical SPO+ loss value
        fy_loss = fy_obj - scenario.obj

        loss = fy_loss

        # bce = torch.nn.BCELoss()
        # alpha = 1.0
        # bce_loss = bce(pred_y, true_y)
        # loss = alpha*spo_loss + (1-alpha)*bce_loss*100

        torch.set_printoptions(precision=2, sci_mode=False)
        # if loss < -0.001:
        #     print('------loss------', loss, spo_obj, scenario.obj)
        #     print('theta', pred_theta)
        #     print('y_theta', y_theta_tensor)
        #     print('spo_theta', torch.tensor(spo_theta))
        #     print('y_spo', y_spo_tensor_reshape)
        #     print('y_true', true_y)
        #     print('grad', (true_y - y_spo_tensor_reshape) * 1000)
        #     print('true loss', eval_obj-scenario.obj)
        loss_gap = loss / scenario.obj
        return torch.tensor(loss, device=pred_theta.device, dtype=pred_theta.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        y_fy, y_true, y_pred = ctx.saved_tensors
        # Gradient: 2 * (y_true - y_spo)
        # grad = ( - 2 * (2 * y_spo - y_true)) * 1000
        fy_grad = (- 2 * (y_true - y_fy)) * 100
        # print(grad)

        # bce_grad = calculate_bce_gradient(y_pred, y_true)
        # clipped_bce = torch.clamp(bce_grad*100, min=-400, max=400)

        grad = fy_grad

        # alpha = 1.0
        # grad = alpha* spo_grad + (1-alpha)*clipped_bce

        # print('spo grad', spo_grad)
        # print('bce grad', bce_grad)
        # print('grad', grad)
        # print('pred y', y_pred)
        # print('true y', y_true)
        
        # hamming_dist = torch.sum(torch.abs(y_true - y_spo))
        # print(f"Discrepancy in decisions: {hamming_dist.item()}")
        return grad * grad_output, None, None, None, None


def calculate_bce_gradient(pred_p, true_y, epsilon=1e-7):
    # Clamp pred_p to avoid division by zero (numerical stability)
    pred_p = torch.clamp(pred_p, epsilon, 1.0 - epsilon)
    
    # Gradient formula: (pred - target) / (pred * (1 - pred))
    grad = (pred_p - true_y) / (pred_p * (1 - pred_p))
    
    return grad