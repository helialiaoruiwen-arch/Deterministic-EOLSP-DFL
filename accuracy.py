import torch
import os
import numpy as np
from torch_geometric.loader import DataLoader
from model_def import SchedulerForward, SchedulingDataset
from sklearn.metrics import confusion_matrix, classification_report
from CplexModel import solve_with_fixed_setups, solve_with_penalization_in_obj, evaluate_solution
from types import SimpleNamespace
import matplotlib.pyplot as plt

def native_scatter_max(src, index, num_segments):
    """
    src: [Num_Edges] - The probabilities (y_hat)
    index: [Num_Edges] - The slot IDs (target_slots)
    num_segments: The total number of slots in the batch
    """
    # 1. Get the max values per slot
    # fill_value should be very low so actual probs (0-1) always win
    out = src.new_full((num_segments,), -1e9) 
    max_values = out.scatter_reduce(0, index, src, reduce='amax', include_self=False)

    # 2. To get the ARGMAX (the winning indices in the original src)
    # We compare the original values to the broadcasted max values
    max_per_edge = max_values[index]
    
    # This creates a mask of where the original value matches the segment max
    is_max = (src == max_per_edge)
    
    # To handle ties (if two products have the same prob), we pick the first one
    # Note: This is a simplified argmax; torch_scatter is more optimized
    # We create a tensor of the original indices [0, 1, 2, ..., Num_Edges-1]
    # We set non-max indices to a very large number
    device = src.device
    indices = torch.arange(src.size(0), device=device)
    # If it's not the max, make the index infinity so it loses the 'min' check
    indices = indices.masked_fill(~is_max, src.size(0) + 1)
    
    # Use scatter_reduce with 'amin' to find the FIRST (lowest) index for each slot
    argmax_out = indices.new_full((num_segments,), src.size(0) + 1)
    argmax_indices = argmax_out.scatter_reduce(0, index, indices, reduce='amin', include_self=False)
    
    # Filter out slots that had no edges (those will still be src.size(0) + 1)
    valid_mask = argmax_indices <= src.size(0)
    final_indices = argmax_indices[valid_mask]
    return max_values, final_indices


def evaluate(test_data, model_path, feat_dims, edge_dims): # Add feat_dims as an argument
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = DataLoader(test_data, batch_size=1)

    # Initialize the CORRECT class with the correct dimensions
    model = SchedulerForward(hidden_channels=64, feat_dims=feat_dims, edge_dims_dict=edge_dims).to(device)
    # model.load_state_dict(torch.load(model_path, map_location=device))

    
    checkpoint = torch.load("checkpoints/var_150data_hybrid/checkpoint_epoch_49.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    
    all_targets = []

    accuracy = []

    # threshold = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
    threshold = [0.5]
    
    cplex_time = []
    gaps_fixe = [[] for i in range(len(threshold))]
    times_fixe = [[] for i in range(len(threshold))]
    gaps_penalize = []
    times_penalize = []
    relat_times_penalize = []
    obj_penalize = []
    obj_ref = []
    time_pen_eval = []
    time_ref = []

    loss = []
    loss_spo = []

    all_preds = [[] for i in range(len(threshold))]
    precision = [[] for i in range(len(threshold))]
    recall = [[] for i in range(len(threshold))]

    percentage_fixed = [[] for i in range(len(threshold))]

    print(f"Evaluating Setups on {len(test_data)} test scenarios...")

    
    with torch.no_grad():
        m=0
        for batch in loader:
            batch = batch.to(device)
            
            # Unpack the tuple (model returns y_hat)
            y_hat = model(batch)

            scenario = batch[0].scenario
            num_slots = batch['var_setup'].raw_r.max()+1
            slot_indices = batch['var_setup'].raw_r # Target nodes (microperiods)

            max_vals, argmax_indices = native_scatter_max(y_hat, slot_indices, num_slots)
            
            prod_indices = argmax_indices // scenario.R

            cplex_obj = scenario.obj

            cplex_time.append(scenario.resolution_time)

            y_true_binary = torch.round(batch['var_setup'].y)
            all_targets.append(y_true_binary.cpu())

            # evaluate the quality of the gnn precision by penalizing in the objective function
            # obj_pen, time_pen = solve_with_penalization_in_obj(scenario, prod_indices, confident_mask)

            y_hat_reshape = y_hat.reshape(scenario.J+1, scenario.R).detach().numpy()
            true_penaliz_obj, y_new, time_pen = solve_with_penalization_in_obj(scenario, y_hat_reshape)
            obj_pen, ti = evaluate_solution(scenario, y_new)

            gap_pen = (obj_pen - cplex_obj) / cplex_obj
            relative_time_pen = time_pen / scenario.resolution_time
            time_ref.append(scenario.resolution_time)

            gaps_penalize.append(gap_pen)
            times_penalize.append(time_pen)
            relat_times_penalize.append(relative_time_pen)
            time_pen_eval.append(ti)
            obj_penalize.append(obj_pen)
            obj_ref.append(cplex_obj)

            loss.append(obj_pen-cplex_obj)

            y_true_cpu = y_true_binary.reshape(scenario.J+1, scenario.R).detach().cpu().numpy()
            # Calculate SPO costs: 2*w - w_true
            spo_theta = 2 * y_hat_reshape - y_true_cpu
            
            # CLAMPING: We clip the theta be between 0 and 1
            spo_theta_safe = np.clip(spo_theta, 0.0, 1.0)
            # spo_theta_safe = (spo_theta + 1)/3.0
            
            print(m)
            # y*(2w - w_true)
            # spo_obj, spo_y, _ = solve_with_penalization_in_obj(scenario, spo_theta_safe)
            # eval_obj, _ = evaluate_solution(scenario, spo_y)
            # loss_spo.append(spo_obj-cplex_obj)

            # np.set_printoptions(precision=2, suppress=True)
            # print('theta', np.array(y_hat_reshape))
            # # print('spo theta', np.array(spo_theta))
            # print('cplex obj, true penalization obj, obj pen', cplex_obj, true_penaliz_obj, obj_pen)
            # print('penalization y', y_new.reshape(scenario.J+1, scenario.R+1))
            # # print('spo y', spo_y)
            # print('true y', y_true_cpu)

            
            for k in range(len(threshold)):
                thres = threshold[k]
                # Only the product with the highest prob in its slot and with probability above the threshold gets a '1'
                gnn_decisions = torch.zeros_like(y_hat)
                
                for i in argmax_indices:
                    if y_hat[i] >= thres:
                        gnn_decisions[i] = 1.0

                count = (gnn_decisions == 1).sum().item()

                
                fixed_perc = count / scenario.R
                percentage_fixed[k].append(fixed_perc)

                # # With those confident enough Y values fixed, resolve the problem and get the objective value
                # # evaluate the performance of the gnn prediction
                # confident_mask = max_vals >= thres
                
            
                # obj_fixe, time_fixe = solve_with_fixed_setups(scenario, prod_indices, confident_mask)

                # gap_fixe = (obj_fixe - cplex_obj) / cplex_obj
                # relative_time_fixe = time_fixe / scenario.resolution_time
                
                # gaps_fixe[k].append(gap_fixe)
                # times_fixe[k].append(relative_time_fixe)

                
                
                # print('objective value', obj_fixe, obj_pen, cplex_obj)

                # calculate the precision and recall of the gnn decision for a given threshold
                all_preds[k].append(gnn_decisions.cpu())

                prec, rec = calculate_metrics(gnn_decisions, y_true_binary)
                precision[k].append(prec)
                recall[k].append(rec)

            # if m == 12:
            #     print(obj, cplex_obj)
            #     print(gnn_decisions)
            #     print(y_true_binary)
            #     print(batch.D)
            #     print(batch.initial_stock)
            #     print(batch.lost_cost)

            m += 1
    
    print('true ave loss', sum(loss)/len(loss))
    # print('spo ave loss', sum(loss_spo)/len(loss_spo))
    
    # Convert gap data to percentages
    # gap_data_pct = [[g * 100 for g in threshold_list] for threshold_list in gaps_fixe]

    # plot_distribution(gap_data_pct, times_fixe, percentage_fixed, threshold, 'threshold_distribution_evolution.png')

    gap_data_pct_pen = [g * 100 for g in gaps_penalize]

    # relat_times_penalize is the relative time with respect to the reference time
    plot_distribution(gap_data_pct_pen, relat_times_penalize, 'distribution_evolution_with_penalization.png')

    plotfigure(gap_data_pct_pen, times_penalize, 'combined_performance_pen.png')
    plotfigure(gap_data_pct_pen, time_ref, 'combined_performance.png')

    # gaps = [[f'{t:.2f}' for t in gap_list] for gap_list in gaps]
    # print(gaps)
    plot_precision_recall(precision, recall, threshold)

    # print('Time needed to process the initial problem is', cplex_time)
    # print('cplex objectif', obj_ref)
    # print('penalization objective', obj_penalize)
    
    # Convert to numpy for metrics
    y_pred = [torch.cat(all_preds[k]).cpu().numpy() for k in range(len(threshold))]
    y_true = torch.cat(all_targets).cpu().numpy()

    y_true = y_true.astype(int).flatten()

    for k in range(len(threshold)):
        y_pred[k] = y_pred[k].astype(int).flatten()

        # --- Detailed Results ---
        print("\n" + "="*30)
        print(f"   GNN PERFORMANCE REPORT for threshold {threshold[k]}")
        print("="*30)

        # 1. Accuracy
        acc = (y_pred[k] == y_true).mean() * 100
        print(f"Overall Accuracy: {acc:.2f}%")
    
        # 2. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred[k])
        print("\nConfusion Matrix:")
        print(cm)

        # 3. Precision, Recall, F1
        # We focus on 'Setup' (Class 1)
        print("\nDetailed Metrics:")
        print(classification_report(y_true, y_pred[k], target_names=["No Setup", "Setup"], zero_division=0))

    # # # 4. Confidence Analysis
    # # print("\n--- Confidence Check ---")
    # # print(f"Avg probability assigned to actual Setups: {y_probs[y_true==1].mean():.4f}")
    # # print(f"Avg probability assigned to No-Setups:    {y_probs[y_true==0].mean():.4f}")
    
        # Logic Check
        print(f"\nTotal Setup decisions: CPLEX={int(y_true.sum())} | GNN={int(y_pred[k].sum())}")

    



def calculate_metrics(gnn_decisions, optimal_sol_tensor):
    gnn_decisions = gnn_decisions.float()
    # Ensure it's the same shape as y_hat_probs
    ground_truth = optimal_sol_tensor.float()

    # 3. Calculate TP, FP, FN
    # TP: Both are 1
    tp = torch.sum((gnn_decisions == 1.0) & (ground_truth == 1.0)).item()
    
    # FP: GNN says 1, Truth says 0
    fp = torch.sum((gnn_decisions == 1.0) & (ground_truth == 0.0)).item()
    
    # FN: GNN says 0 (or unsure), Truth says 1
    fn = torch.sum((gnn_decisions == 0.0) & (ground_truth == 1.0)).item()

    # 4. Final Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return precision, recall


def plotfigure(gaps, times, fig_name):
    indices = np.arange(len(gaps))
    labels = [f"{i+1}" for i in range(len(gaps))]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    bars = ax1.bar(indices, gaps, color='skyblue', alpha=0.7, label='Cost Gap (%)', edgecolor='black')
    ax1.set_xlabel('Test Instance')
    ax1.set_ylabel('Cost Gap (%)', color='blue', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticks(indices)
    ax1.set_xticklabels(labels)

    ax2 = ax1.twinx()
    line, = ax2.plot(indices, times, color='red', marker='o', linewidth=2, markersize=8, label='Time (s)')
    ax2.set_ylabel('Processing Time (s)', color='red', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('Performance Summary: Cost Gap vs. Solve Time', fontsize=14)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    ax1.legend([bars, line], ['Cost Gap (%)', 'Processing Time (s)'], loc='upper left')

    plt.tight_layout()
    plt.savefig(fig_name)



def plot_distribution(gap_data, time_data, fig_name):
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    
    # Bundle the unique parts of each plot
    plot_configs = [
        (gap_data, 'Cost Gap (%)', 'Cost Gap Distribution', 'skyblue', 'blue', (-10,20)),
        (time_data, 'Relative Proc. Time', 'Processing Time Distribution', 'salmon', 'red', (0,1.5))
        # (percent_data, 'Percentage fixed', 'Number of setups fixed', 'honeydew', 'forestgreen')
    ]
    
    # x_labels = [f"{t:.2f}" for t in thresholds]

    for ax, (data, ylabel, title, f_col, b_col, ylims) in zip(axes, plot_configs):
        ax.boxplot(data, showfliers=True, patch_artist=True,
                   boxprops=dict(facecolor=f_col, color=b_col),
                   medianprops=dict(color='black', linewidth=2))
        
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        # Check if any data exceeds the limit
        num_hidden = sum(1 for val in data if val > ylims[1])
        if num_hidden > 0:
            # For a single box, the x-position is always 1
            ax.text(1, ylims[1] * 0.96, f'↑{num_hidden} outside range, max is {max(data):.2f}', 
                    color='red', fontweight='bold', ha='center',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        ax.set_ylim(ylims)

    # axes[-1].set_xlabel('GNN Probability Threshold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(fig_name)

def plot_precision_recall(precision, recall, threshold):
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_with_shade(ax, threshold, precision, 'Precision', 'forestgreen')
    plot_with_shade(ax, threshold, recall, 'Recall', 'darkorange')

    ax.set_title('Metric Evolution: Mean and Std Dev across Instances', fontsize=14)
    ax.set_xlabel('GNN Probability Threshold', fontsize=12)
    ax.set_ylabel('Score (0.0 - 1.0)', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig('precision_recall.png')

def plot_with_shade(ax, x, data, label, color):
    means = np.array([np.mean(d) for d in data])
    stds = np.array([np.std(d) for d in data])
    
    ax.plot(x, means, label=label, color=color, marker='o', linewidth=2)
    ax.fill_between(x, means - stds, means + stds, color=color, alpha=0.2)


# from run_all import get_split_data
# DATA_PATH = "data_storage"
# MODEL_PATH = "scheduling_gnn.pth"
if __name__ == "__main__":
    # full_dataset = SchedulingDataset(DATA_PATH)
    # train_set, test_set = get_split_data(full_dataset)
    # evaluate(test_set, MODEL_PATH)
    max_val, indices = native_scatter_max(torch.tensor([0,0,0,1,2,0,3,2,1]),torch.tensor([0,0,0,1,1,1,2,2,2]),3)
    confident_mask = max_val >= 2
    # print(max_val, indices, confident_mask)
    a = torch.tensor([1,2,3])
    b= torch.tensor([0,0,1])
    gnn = torch.tensor([1,0,0,1,1])
    tar = torch.tensor([0,1,0,0,1])
    print(torch.cat([a,b]).numpy())