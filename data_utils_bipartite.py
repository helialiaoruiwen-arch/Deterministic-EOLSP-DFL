'''This uses the bipartite graph to define the variables and constraints'''
import torch
from torch_geometric.data import Data, HeteroData
from CplexModel import CplexOptimizer
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
import numpy as np
import os, shutil, re
from types import SimpleNamespace

R_per_T = 8
MAX_PRODUCTS = 6

def get_node_id(j, r, R):
    """Calculates the flat index for a product-time pair."""
    return (j * R) + (r - 1)

def initial_param(num_products, num_days):
    J = num_products
    T = 2*num_days
    R = T*R_per_T
    return J,T,R

def create_instance(scenario, opt_sol, relax_sol, num_products, num_days):
    data = HeteroData()

    """
    Converts CPLEX scenario and solution into a PyG Data object.
    Uses scenario-specific max-scaling for normalization.
    """

    J,num_macro,R = initial_param(num_products, num_days)
    
    
    # --- Calculate Scenario-Specific Maximums ---
    # Flatten demand list of lists to find max
    flat_demand = [val for sublist in scenario.D for val in sublist]
    max_d = max(flat_demand) if max(flat_demand) > 0 else 1.0
    max_initS = max(scenario.initial_stock) if max(scenario.initial_stock) > 0 else 1.0
    max_sc = max(scenario.startup_cost) if max(scenario.startup_cost) > 0 else 1.0
    max_hc = max(scenario.holding_cost) if max(scenario.holding_cost) > 0 else 1.0
    max_lc = max(scenario.lost_cost) if max(scenario.lost_cost) > 0 else 1.0

    max_es = max(scenario.energy_startup) if max(scenario.energy_startup) > 0 else 1.0
    max_ep = max(scenario.energy_product) if max(scenario.energy_product) > 0 else 1.0
    max_p = max(scenario.energy_purchase_price) if max(scenario.energy_purchase_price) > 0 else 1.0
    max_pv = max(scenario.PV) if max(scenario.PV) > 0 else 1.0
    b_cap = scenario.battery_cap if scenario.battery_cap > 0 else 1.0
    b_clim = scenario.charge_lim if scenario.charge_lim > 0 else 1.0
    b_dlim = scenario.discharge_lim if scenario.discharge_lim > 0 else 1.0

    relax_X = relax_sol.X.reshape(J+1, R)
    relax_Y = relax_sol.Y.reshape(J+1, R+1)


    # --- create variable nodes ---
    # startup node
    startup_features = []
    for j in range(J+1):
        for r in range(R):
            t_idx = (r - 1) // (R // len(scenario.D[0]))
            startup_features.append(
                [r / R,
                t_idx / num_macro,
                scenario.startup_cost[j] / max_sc,
                scenario.energy_startup[j] / max_es,
                scenario.initial_stock[j] / scenario.capacity,
                scenario.D[j][t_idx] / scenario.capacity,
                sum(scenario.D[j][:t_idx+1]) / (scenario.capacity*num_macro),
                sum(scenario.D[j][t_idx+1:]) / (scenario.capacity*num_macro),
                scenario.energy_purchase_price[r] / max_p,
                scenario.PV[r] / max_pv,
                relax_X[j][r]
                ]
            )
    data['var_startup'].x = torch.tensor(startup_features, dtype=torch.float)

    # setup node
    setup_features = []
    for j in range(J+1):
        for r in range(R):
            t_idx = (r - 1) // (R // len(scenario.D[0]))
            setup_features.append(
                [r/R,
                t_idx/num_macro,
                scenario.initial_stock[j] / scenario.capacity,
                scenario.D[j][t_idx] / scenario.capacity,
                sum(scenario.D[j][:t_idx+1]) / (scenario.capacity*num_macro),
                sum(scenario.D[j][t_idx+1:]) / (scenario.capacity*num_macro),
                scenario.energy_purchase_price[r] / max_p,
                scenario.PV[r] / max_pv,
                relax_Y[j][r+1]
                ]
            )
    data['var_setup'].x = torch.tensor(setup_features, dtype=torch.float)

    # production node
    prod_features = []
    for j in range(J):
        for r in range(R):
            t_idx = (r - 1) // (R // len(scenario.D[0]))
            prod_features.append(
                [r / R,
                t_idx / num_macro,
                scenario.energy_product[j] / max_ep,
                scenario.capacity_unit[j] / max(scenario.capacity_unit),
                scenario.initial_stock[j] / scenario.capacity,
                scenario.D[j][t_idx] / scenario.capacity,
                sum(scenario.D[j][:t_idx+1]) / (scenario.capacity*num_macro),
                sum(scenario.D[j][t_idx+1:]) / (scenario.capacity*num_macro),
                scenario.energy_purchase_price[r] / max_p,
                scenario.PV[r] / max_pv
                ]
            )
    data['var_prod'].x = torch.tensor(prod_features, dtype=torch.float)

    # inventory node
    invent_features = []
    for j in range(J+1):
        for t in range(num_macro):
            invent_features.append(
                [t / num_macro,
                scenario.holding_cost[j] / max_hc,
                scenario.initial_stock[j] / scenario.capacity,
                scenario.D[j][t] / scenario.capacity,
                sum(scenario.D[j][:t+1]) / (scenario.capacity*num_macro),
                sum(scenario.D[j][t+1:]) / (scenario.capacity*num_macro),
                ]
            )
    data['var_invent'].x = torch.tensor(invent_features, dtype=torch.float)

    # lost sales node
    lost_features = []
    for j in range(J+1):
        for t in range(num_macro):
            lost_features.append(
                [t / num_macro,
                scenario.lost_cost[j] / max_lc,
                scenario.initial_stock[j] / scenario.capacity,
                scenario.D[j][t] / scenario.capacity,
                sum(scenario.D[j][:t+1]) / (scenario.capacity*num_macro),
                sum(scenario.D[j][t+1:]) / (scenario.capacity*num_macro),
                ]
            )
    data['var_lost'].x = torch.tensor(lost_features, dtype=torch.float)


    # --- create constraint nodes and edges ---

    # -- inventory balance constraint: I_{j,t} - I_{j,t-1} - \sum_{r in R_t}Q_{j,r} - L_{j,t} = -d_{j,t} --
    # create constraint node features
    # 1. Standard Demand RHS for all R slots
    # rhs_base = (-torch.tensor(scenario.D[:-1], dtype=torch.float) / scenario.capacity).flatten()
    rhs_base = (-torch.tensor(scenario.D[:-1], dtype=torch.float) / scenario.capacity).flatten()
    # print(rhs_base)

    # 2. Boundary Mask: Tells the GNN "This node has a special start condition"
    boundary_mask = torch.zeros_like(rhs_base)
    boundary_mask[::num_macro] = 1.0 # Marks r=0 for every product
    # print(boundary_mask)

    # 3. Initial Stock Feature: Only exists at t=0
    initial_stock_feat = torch.zeros_like(rhs_base)
    for j in range(J):
        initial_stock_feat[j * num_macro] = scenario.initial_stock[j] / scenario.capacity

    # print(initial_stock_feat)

    # dual_base = torch.tensor(relax_sol.con_invent_balance, dtype=torch.float)
    dual_base = norm_dual_values(relax_sol.con_invent_balance)
    # print(dual_base)

    # Combine them into the Constraint Node Features
    # [RHS, Is_Boundary, Initial_Value]
    data['con_invent_balance'].x = torch.cat([
        torch.stack([rhs_base, boundary_mask, initial_stock_feat, dual_base], dim=1),
        torch.stack([-rhs_base, boundary_mask, -initial_stock_feat, dual_base], dim=1) # The "Lower" nodes
    ], dim=0)
    # data['con_invent_balance'].x = torch.cat([
    #     torch.stack([rhs_base, boundary_mask, initial_stock_feat], dim=1),
    #     torch.stack([-rhs_base, boundary_mask, -initial_stock_feat], dim=1) # The "Lower" nodes
    # ], dim=0)

    # print(torch.stack([rhs_base, boundary_mask, initial_stock_feat], dim=1))
    # print(data['con_invent_balance'].x)

    # create the edge between variable nodes and constraint nodes
    # Since it's an equality, we need to construct dual edges for <= and >=
    edge_invent_index = []
    edge_invent_attr = []

    edge_lost_index = []
    edge_lost_attr = []

    edge_prod_index = []
    edge_prod_attr = []


    for j in range(J):
        for t in range(num_macro):
            # Index of the 'Upper' constraint node for this (j, t)
            c_idx_upper = (j * num_macro) + t
            # Index of the 'Lower' constraint node
            c_idx_lower = c_idx_upper + (J * num_macro)
            
            # --- 1. Current Inventory: +I_{j,t} ---
            inv_idx_now = (j * num_macro) + t
            # Upper: coeff +1 | Lower: coeff -1
            edge_invent_index.extend([[inv_idx_now, c_idx_upper], [inv_idx_now, c_idx_lower]])
            edge_invent_attr.extend([[1.0], [-1.0]])
            
            # --- 2. Previous Inventory: -I_{j,t-1} ---
            if t > 0:
                inv_idx_prev = (j * num_macro) + (t - 1)
                edge_invent_index.extend([[inv_idx_prev, c_idx_upper], [inv_idx_prev, c_idx_lower]])
                edge_invent_attr.extend([[-1.0], [1.0]])
            # Note: If t=0, I_{j,t-1} is Initial Stock (already in your RHS feature!)

            # --- 3. Lost sales: -L_{j,t} ---
            lost_idx_now = (j * num_macro) + t
            edge_lost_index.extend([[lost_idx_now, c_idx_upper], [lost_idx_now, c_idx_lower]])
            edge_lost_attr.extend([[-1.0], [1.0]])
    

            # --- 4. Production Sum: -Sum(L_{j,r}) ---
            # The range is 8*(t) to 8*(t+1) exclusive
            for r in range(t * R_per_T, (t + 1) * R_per_T):
                prod_idx = (j * R) + r
                # Upper: coeff -1 | Lower: coeff +1
                edge_prod_index.extend([[prod_idx, c_idx_upper], [prod_idx, c_idx_lower]])
                edge_prod_attr.extend([[-1.0], [1.0]])

    # Convert to HeteroData format
    data['var_invent', 'to', 'con_invent_balance'].edge_index = torch.tensor(edge_invent_index).t().contiguous()
    data['var_invent', 'to', 'con_invent_balance'].edge_attr = torch.tensor(edge_invent_attr, dtype=torch.float)

    data['var_lost', 'to', 'con_invent_balance'].edge_index = torch.tensor(edge_lost_index).t().contiguous()
    data['var_lost', 'to', 'con_invent_balance'].edge_attr = torch.tensor(edge_lost_attr, dtype=torch.float)

    data['var_prod', 'to', 'con_invent_balance'].edge_index = torch.tensor(edge_prod_index).t().contiguous()
    data['var_prod', 'to', 'con_invent_balance'].edge_attr = torch.tensor(edge_prod_attr, dtype=torch.float)

    
    # -- production constraint: k_j Q_{j,r} <= l(Y_{j,r}+Y_{j,r-1}) --
    # define constraint node features
    rhs_values = torch.zeros((J * R, 1))
    # dual_values = torch.tensor(relax_sol.con_products_constraint, dtype=torch.float)
    dual_values = norm_dual_values(relax_sol.con_products_constraint)

    for j in range(J):
        # For r=0, the RHS is influenced by the fixed initial state
        rhs_values[j * R, 0] = scenario.initial_setup[j]

    data['con_prod_allow'].x = torch.cat([rhs_values, dual_values.unsqueeze(1)], dim = 1)
    # data['con_prod_allow'].x = rhs_values

    # construct edges
    edge_index_qty = []
    edge_attr_qty = []

    edge_index_setup = []
    edge_attr_setup = []
    for j in range(J):
        for r in range(R):
            c_idx = (j * R) + r # One constraint node per (product, slot)
            
            # 1. Quantity Node (Q_jr)
            qty_idx = (j * R) + r
            edge_index_qty.append([qty_idx, c_idx])
            edge_attr_qty.append([scenario.capacity_unit[j] / scenario.length_microperiod])
            
            # 2. Current Setup Node (Y_jr)
            setup_idx_now = (j * R) + r
            edge_index_setup.append([setup_idx_now, c_idx])
            edge_attr_setup.append([-1.0])
            
            # 3. Previous Setup Node (Y_j,r-1)
            if r > 0:
                setup_idx_prev = (j * R) + (r - 1)
                edge_index_setup.append([setup_idx_prev, c_idx])
                edge_attr_setup.append([-1.0])

    data['var_setup', 'restrict', 'con_prod_allow'].edge_index = torch.tensor(edge_index_setup).t().contiguous()
    data['var_setup', 'restrict', 'con_prod_allow'].edge_attr = torch.tensor(edge_attr_setup, dtype=torch.float)

    data['var_prod', 'allowed', 'con_prod_allow'].edge_index = torch.tensor(edge_index_qty).t().contiguous()
    data['var_prod', 'allowed', 'con_prod_allow'].edge_attr = torch.tensor(edge_attr_qty, dtype=torch.float)





    # -- capacity limit constraint: \sum_{j} k_j Q_{j,r} <= l --
    rhs_values = torch.ones((R,1), dtype=torch.float)
    # dual_values = torch.tensor(relax_sol.con_capacity_lim, dtype=torch.float)
    dual_values = norm_dual_values(relax_sol.con_capacity_lim)

    data['con_capacity'].x = torch.cat([rhs_values, dual_values.unsqueeze(1)], dim = 1)
    # data['con_capacity'].x = rhs_values

    edge_index_qty = []
    edge_attr_qty = []

    for r in range(R):
        r_idx = r
        for j in range(J):
            qty_idx = j*R + r
            edge_index_qty.append([qty_idx, r_idx])
            edge_attr_qty.append([scenario.capacity_unit[j]/scenario.length_microperiod])  # normalize

    data['var_prod', 'limited', 'con_capacity'].edge_index = torch.tensor(edge_index_qty).t().contiguous()
    data['var_prod', 'limited', 'con_capacity'].edge_attr = torch.tensor(edge_attr_qty, dtype=torch.float)



    # -- machine only setup for one product at a time: \sum_{j} Y_{j,r} = 1 [J+1] --
    rhs_values = torch.ones((R,1), dtype=torch.float)
    # dual_values = torch.tensor(relax_sol.con_setup_constraint, dtype=torch.float)
    dual_values = norm_dual_values(relax_sol.con_setup_constraint)

    forw = torch.cat([rhs_values, dual_values.unsqueeze(1)], dim = 1)
    inver = torch.cat([-rhs_values, dual_values.unsqueeze(1)], dim = 1)

    data['con_machine'].x = torch.cat([forw, inver], dim=0)

    # data['con_machine'].x = torch.cat([rhs_values, -rhs_values], dim=0)

    edge_index_setup = []
    edge_attr_setup = []

    for r in range(R):
        r_idx_upper = r
        r_idx_lower = R + r
        for j in range(J+1):
            setup_idx = j*R + r
            edge_index_setup.extend([[setup_idx, r_idx_upper], [setup_idx, r_idx_lower]])
            edge_attr_setup.extend([[1.0], [-1.0]])

    data['var_setup', 'only', 'con_machine'].edge_index = torch.tensor(edge_index_setup).t().contiguous()
    data['var_setup', 'only', 'con_machine'].edge_attr = torch.tensor(edge_attr_setup, dtype=torch.float)


    # -- Product-to-Product Conflict Edges (Intra-slot) --
    edge_index_conflict = []
    
    for r in range(R):
        # For each time slot, every product j conflicts with every other product k
        for j in range(J + 1):
            for k in range(j + 1, J + 1):
                idx_j = (j * R) + r
                idx_k = (k * R) + r
                
                # Add bidirectional edges for the clique in slot r
                edge_index_conflict.append([idx_j, idx_k])
                edge_index_conflict.append([idx_k, idx_j])

    data['var_setup', 'conflicts_with', 'var_setup'].edge_index = \
        torch.tensor(edge_index_conflict).t().contiguous()

    num_conflicts = data['var_setup', 'conflicts_with', 'var_setup'].edge_index.size(1)
    data['var_setup', 'conflicts_with', 'var_setup'].edge_attr = torch.ones((num_conflicts, 1))


    # -- setup startup relation constraint: X_{j,r} >= Y_{j,r}-Y_{j,r-1} --
    # define the constraint features: righthand side constant
    setup_mask = torch.zeros((J + 1, R), dtype=torch.float)
    initial_vals = torch.tensor(scenario.initial_setup, dtype=torch.float) # [J+1]
    setup_mask[:, 0] = initial_vals

    # dual_values = torch.tensor(relax_sol.con_relat_startup_setup, dtype=torch.float)
    dual_values = norm_dual_values(relax_sol.con_relat_startup_setup)
    
    # data['con_startup'].x = setup_mask.view(-1,1)
    data['con_startup'].x = torch.cat([setup_mask.view(-1,1), dual_values.unsqueeze(1)], dim = 1)
    # print(data['con_startup'].x)

    # define the edge and attribute of the edges
    edge_index_setup_to_con = []
    edge_attr_setup_to_con = []

    edge_index_startup_to_con = []
    edge_attr_startup_to_con = []

    for j in range(J+1):
        for r in range(R):
            # The unique index for the constraint node for this (j, r) pair
            c_idx = (j * R) + r
            
            # 1. Connection to CURRENT setup: setup[j, r]
            s_idx_now = (j * R) + r
            edge_index_setup_to_con.append([s_idx_now, c_idx])
            edge_attr_setup_to_con.append([1.0]) # Coefficient +1
            
            # 2. Connection to PREVIOUS setup: setup[j, r-1]
            if r > 0:
                s_idx_prev = (j * R) + (r - 1)
                edge_index_setup_to_con.append([s_idx_prev, c_idx])
                edge_attr_setup_to_con.append([-1.0]) # Coefficient -1
                
            # 3. Connection to STARTUP variable: startup[j, r]
            st_idx = (j * R) + r
            edge_index_startup_to_con.append([st_idx, c_idx])
            edge_attr_startup_to_con.append([-1.0]) # Moving startup to other side: -startup <= ...

    data['var_setup', 'relate', 'con_startup'].edge_index = torch.tensor(edge_index_setup_to_con).t().contiguous()
    data['var_setup', 'relate', 'con_startup'].edge_attr = torch.tensor(edge_attr_setup_to_con, dtype=torch.float)

    data['var_startup', 'relate', 'con_startup'].edge_index = torch.tensor(edge_index_startup_to_con).t().contiguous()
    data['var_startup', 'relate', 'con_startup'].edge_attr = torch.tensor(edge_attr_startup_to_con, dtype=torch.float)


    node_j = torch.arange(J + 1).repeat_interleave(R)
    node_r = torch.arange(R).repeat(J + 1)

    data['var_setup'].raw_j = node_j
    data['var_setup'].raw_r = node_r

    # node_j = torch.arange(J + 1).repeat_interleave(num_macro)
    node_t = torch.arange(num_macro).repeat(J + 1)

    data['var_invent'].raw_t = node_t

    # --- Build Targets (Y) ---
    y_raw = np.array(opt_sol.Y).reshape(J+1, R+1)
    y_raw_flat = torch.from_numpy(y_raw[:,1:]).flatten()

    # restrict the values of y to stricitly between 0 and 1
    y_raw_flat = torch.clamp(y_raw_flat, min=0.0, max=1.0)
    
    data['var_setup'].y = y_raw_flat

    # # store the raw data for later calculation
    data.scenario = SimpleNamespace(
        J = J,
        R = R,
        num_macro = num_macro,
        days = num_days,

        D = scenario.D,
        PV = scenario.PV,
        initial_stock = scenario.initial_stock,
        initial_setup = scenario.initial_setup,

        startup_cost = scenario.startup_cost,
        holding_cost = scenario.holding_cost,
        lost_cost = scenario.lost_cost,

        capacity = scenario.capacity,
        capacity_unit = scenario.capacity_unit,
        energy_startup = scenario.energy_startup,
        energy_product = scenario.energy_product,

        energy_purchase_price = scenario.energy_purchase_price,

        length_microperiod = scenario.length_microperiod,

        battery_cap = scenario.battery_cap,
        charge_lim = scenario.charge_lim,
        discharge_lim = scenario.discharge_lim,

        obj = opt_sol.obj,
        gap = opt_sol.mip_gap,
        LB = opt_sol.best_LB,
        resolution_time = opt_sol.resolution_time
    )


    # Add reverse edges so information can flow back 
    data = T.ToUndirected()(data)

    

    return data


def norm_dual_values(duals_list):
    duals = torch.tensor(duals_list, dtype=torch.float32)

    # Apply Signed Log Scaling
    norm_duals = torch.sign(duals) * torch.log1p(torch.abs(duals))

    # Optional: Further standardize the log-scaled values
    if norm_duals.std() > 0:
        norm_duals = (norm_duals - norm_duals.mean()) / (norm_duals.std() + 1e-6)

    return norm_duals


def generate_complex_samples(gen_list, folder_path, add=False):
    total_data_pairs = [] # Store pairs of (input, result)
    
    for days, J, n_samples in gen_list:
        opt = CplexOptimizer(days, J)
        try:
            inputs, results, relax_values = opt.generate(n_samples)
            # Store the specific J and R with the data so we don't lose them
            for inp, res, relax in zip(inputs, results, relax_values):
                total_data_pairs.append((inp, res, relax, opt.J, days))
        finally:
            opt.terminate()

    next_idx = 0
    if add==False:
        clear_folder(folder_path)

    if add==True:
        next_idx = get_max_graph_index(folder_path)+1

    for i, (scenario, opt_sol, relax_sol, actual_J, actual_R) in enumerate(total_data_pairs):
        # USE the specific J and R for THIS graph instance
        graph_data = create_instance(scenario, opt_sol, relax_sol, actual_J, actual_R)
        
        torch.save(graph_data, f"{folder_path}/graph_{next_idx+i}.pt")

    print(f"Successfully saved {len(total_data_pairs)} graphs of varying sizes.")


def clear_folder(folder_path):
    """Safely clears all files in a folder without deleting the folder itself."""
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) or os.path.is_link(file_path):
                os.unlink(file_path) # Deletes the file
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path) # Deletes subfolders
        print(f"Directory cleared: {folder_path}")
    else:
        os.makedirs(folder_path)
        print(f"Directory created: {folder_path}")

def get_max_graph_index(folder_path):
    if not os.path.exists(folder_path):
        print('Path not exists.')
        return -1
        
    
    indices = []
    # Loop through all files in the directory
    for filename in os.listdir(folder_path):
        # Match "graph_" followed by digits and ".pt"
        match = re.search(r'graph_(\+?\d+)\.pt', filename)
        if match:
            # Convert the matched group (the digits) to an integer
            indices.append(int(match.group(1)))
            
    return max(indices) if indices else -1

cat = (1,3,500)
# total_cat = [(1,3,50),(1,4,50),(1,5,50),(2,2,50),(2,3,50),(2,4,50)]
# total_cat = [(2,5,10),(3,2,10),(3,3,10),(3,4,10),(3,5,10)]
# total_cat = [(4,2,10),(4,3,10),(4,4,10),(4,5,10)]
# total_cat = [(16,5,1)]
total_cat = [(8,2,5),(8,3,5),(8,4,5),(8,5,5)]
if __name__ == '__main__':
    folder_path = 'data_storage_test_8days_50s'
    # generate_cat_samples(*cat)
    generate_complex_samples(total_cat, folder_path, add=False)
