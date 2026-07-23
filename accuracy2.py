import torch
import os, pickle
import numpy as np
from torch_geometric.loader import DataLoader
from model_def import SchedulerForward, SchedulingDataset
from sklearn.metrics import confusion_matrix, classification_report
from CplexModel import solve_with_fixed_setups, solve_with_penalization_in_obj, solve_with_penalization_in_obj_hard, evaluate_solution
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


def evaluate(test_data, model_path, result_store_path, x_labels, feat_dims, edge_dims): # Add feat_dims as an argument
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = DataLoader(test_data, batch_size=1)

    # Initialize the CORRECT class with the correct dimensions
    model = SchedulerForward(hidden_channels=64, feat_dims=feat_dims, edge_dims_dict=edge_dims).to(device)
    # model.load_state_dict(torch.load(model_path, map_location=device))

    GAPS_PEN = []
    RELATIVE_TIME_PEN = []
    TIMES_PEN = []

    for path in model_path:
        checkpoint = torch.load(f"checkpoints/{path}")
        model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()

        
        all_targets = []

        accuracy = []

        threshold = [0.5, 0.60, 0.70, 0.80, 0.90, 0.95]
        # threshold = [0.6]
        
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

                y_true_binary = torch.round(batch['var_setup'].y)
                all_targets.append(y_true_binary.cpu())

                # # plot the figure for some instances for visualization
                # theta_flat = np.array(y_hat.detach().numpy()).flatten()
                # y_true_flat = np.array(y_true_binary.detach().numpy()).flatten()
                # # y_new_flat = np.array(y_new.reshape(scenario.J+1, scenario.R+1)[:,1:]).flatten()
                # plt.figure(figsize=(10, 5))
                # plt.plot(theta_flat, label='Original Theta', color='blue', alpha=0.7)
                # plt.plot(y_true_flat, label='True y', color='red', alpha=0.5, linestyle='--')
                # # plt.plot(y_new_flat, label='Pred y', color='blue', alpha=0.5, linestyle='dotted')
                # plt.show()

                # evaluate the quality of the gnn precision by penalizing in the objective function
                # obj_pen, time_pen = solve_with_penalization_in_obj(scenario, prod_indices, confident_mask)


                # ---------------------------------
                # y_hat_reshape = y_hat.reshape(scenario.J+1, scenario.R).detach().numpy()
                # true_penaliz_obj, y_new, time_pen = solve_with_penalization_in_obj(scenario, y_hat_reshape)
                # obj_pen, ti, _ = evaluate_solution(scenario, y_new)

                # np.set_printoptions(precision=2, suppress=True)
                # print(f'cplex obj {cplex_obj}')
                # print(f'true y {y_true_binary.reshape(scenario.J+1, scenario.R)}')
                # print(f'prediction y values {y_hat.reshape(scenario.J+1, scenario.R)}')
                # print(f'obj soft penalization {obj_pen}')
                # print('soft penalization', y_new.reshape(scenario.J+1, scenario.R+1))


                # gap_pen = (obj_pen - cplex_obj) / cplex_obj
                # relative_time_pen = time_pen / scenario.resolution_time
                # time_ref.append(scenario.resolution_time)

                # gaps_penalize.append(gap_pen)
                # times_penalize.append(time_pen)
                # relat_times_penalize.append(relative_time_pen)
                # time_pen_eval.append(ti)
                # obj_penalize.append(obj_pen)
                # obj_ref.append(cplex_obj)

                # loss.append(obj_pen-cplex_obj)
                #--------------------------------------------

                # print('loss:',obj_pen-cplex_obj)
                print(m)

                # y_true_cpu = y_true_binary.reshape(scenario.J+1, scenario.R).detach().cpu().numpy()

                # # Calculate SPO costs: 2*w - w_true
                # spo_theta = 2 * y_hat_reshape - y_true_cpu
                
                # # CLAMPING: We clip the theta be between 0 and 1
                # spo_theta_safe = np.clip(spo_theta, 0.0, 1.0)
                
                
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

                    # With those confident enough Y values fixed, resolve the problem and get the objective value
                    # evaluate the performance of the gnn prediction
                    confident_mask = max_vals >= thres
                    # print(confident_mask)
                    
                    # # Hard fixing strategy
                    # obj_fixe, time_fixe = solve_with_fixed_setups(scenario, prod_indices, confident_mask)

                    # Soft fixing strategy
                    obj_hard, y_new, time_fixe = solve_with_penalization_in_obj_hard(scenario, prod_indices, confident_mask)
                    obj_fixe, ti, _ = evaluate_solution(scenario, y_new)
                    # print(f'obj soft fixing {obj_hard} {obj_fixe}')
                    # print(f'prod indices {prod_indices}')
                    # print(f'soft fixing threshold {threshold[k]}', y_new.reshape(scenario.J+1, scenario.R+1))
                    

                    gap_fixe = (obj_fixe - cplex_obj) / cplex_obj
                    relative_time_fixe = time_fixe / scenario.resolution_time
                    
                    gaps_fixe[k].append(gap_fixe)
                    times_fixe[k].append(relative_time_fixe)

                    
                    
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

        gap_data_pct_pen = [g * 100 for g in gaps_penalize]

        GAPS_PEN.append(gap_data_pct_pen)
        RELATIVE_TIME_PEN.append(relat_times_penalize)
        TIMES_PEN.append(times_penalize)

        
        # print('true ave loss', sum(loss)/len(loss))
    # print('spo ave loss', sum(loss_spo)/len(loss_spo))

    y_pred = [torch.cat(all_preds[k]).cpu().numpy() for k in range(len(threshold))]
    y_true = torch.cat(all_targets).cpu().numpy()
    y_true = y_true.astype(int).flatten()

    results = {
        'gap_data': GAPS_PEN,
        'time_data': RELATIVE_TIME_PEN,
        'penalized_time_data': TIMES_PEN,
        'labels': x_labels,
        'ref_objectif': obj_ref,
        'ref_time': time_ref,
        'precision': precision,
        'recall': recall,
        'all_preds': y_pred,
        'all_targets': y_true,
        'threshold':threshold,
        'gap_fixing': gaps_fixe,
        'time_fixing': times_fixe,
        'percentage_fixed': percentage_fixed
    }
    # print(results)

    with open(f'results/{result_store_path}', 'wb') as f:
        pickle.dump(results, f)
    



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



def plot_distribution(gap_data, time_data, x_labels, fig_name):
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    
    # Bundle the unique parts of each plot
    plot_configs = [
        (gap_data, 'Cost Gap (%)', 'Cost Gap Distribution', 'skyblue', 'blue', (-1,15)),
        (time_data, 'Relative Proc. Time', 'Processing Time Distribution', 'salmon', 'red', (0,1.5))
        # (percent_data, 'Percentage fixed', 'Number of setups fixed', 'honeydew', 'forestgreen')
    ]
    

    for ax, (data, ylabel, title, f_col, b_col, ylims) in zip(axes, plot_configs):
        ax.boxplot(data, showfliers=True, patch_artist=True,
                   boxprops=dict(facecolor=f_col, color=b_col),
                   medianprops=dict(color='black', linewidth=2))
        
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        for i, sub_list in enumerate(data, 1):
            # Check if any data exceeds the limit
            num_hidden_s = sum(1 for val in sub_list if val > ylims[1])
            if num_hidden_s > 0:
                # For a single box, the x-position is always 1
                ax.text(i, ylims[1] * 0.96, r'$\uparrow$'f'{num_hidden_s} outside\nmax {max(sub_list):.2f}', 
                        color='red', fontweight='bold', ha='center',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            num_hidden_bot = sum(1 for val in sub_list if val < ylims[0])
            if num_hidden_bot > 0:
                # We place this at the bottom (ylims[0]) and shift it up slightly (* 1.05 or + offset)
                # Using va='bottom' ensures the text sits above the bottom line
                ax.text(i, ylims[0] + (ylims[1]-ylims[0])*0.02, r'$\downarrow$' + f'{num_hidden_bot} hidden\nmin: {min(sub_list):.2f}', 
                        color='blue', fontweight='bold', ha='center', va='bottom', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        ax.set_ylim(ylims)
        # print(len(data))
        ax.set_xticks(range(1, len(data) + 1))
        # ax.set_xticklabels(x_labels)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    
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


import matplotlib.pyplot as plt
import numpy as np

def plot_gap_vs_time(gap_data, time_data, x_labels, fig_name="gap_vs_time.png"):
    plt.figure(figsize=(11, 7)) # Slightly wider for legend
    
    # 1. Color map
    colors = plt.cm.plasma(np.linspace(0, 1, len(x_labels)))
    # colors = plt.cm.tab10(np.linspace(0, 1, len(x_labels)))
    
    # Storage for trend line coordinates
    means_x, means_y = [], []
    meds_x, meds_y = [], []

    # 2. Plotting loop
    for i, (g_group, t_group) in enumerate(zip(gap_data, time_data)):
        # Calculate stats for this specific cluster
        # m_x, m_y = np.mean(t_group), np.mean(g_group)
        # s_x, s_y = np.std(t_group), np.std(g_group)
        # means_x.append(m_x)
        # means_y.append(m_y)
         
        m_x, m_y = np.median(t_group), np.median(g_group)
        meds_x.append(m_x)
        meds_y.append(m_y)

        # A. Plot raw data points (Small, slightly transparent)
        # plt.scatter(t_group, g_group, color=colors[i], alpha=0.7, s=30, 
        #             edgecolors='none', zorder=1)

        # B. Plot Error Bars (Same color, but lighter/thinner)
        # x_err = [[m_x - min(t_group)], [max(t_group) - m_x]]
        # y_err = [[m_y - min(g_group)], [max(g_group) - m_y]]
        x_iqr = [[m_x - np.percentile(t_group, 25)], [np.percentile(t_group, 75) - m_x]]
        y_iqr = [[m_y - np.percentile(g_group, 25)], [np.percentile(g_group, 75) - m_y]]
        x_range = [[m_x - min(t_group)], [max(t_group) - m_x]]
        y_range = [[m_y - min(g_group)], [max(g_group) - m_y]]
        plt.errorbar(m_x, m_y, xerr=x_range, yerr=y_range, 
                     fmt='none', ecolor=colors[i], elinewidth=1, 
                     linestyle='--', alpha=0.3, zorder=1)
        plt.errorbar(m_x, m_y, xerr=x_iqr, yerr=y_iqr, 
                     fmt='none', ecolor=colors[i], elinewidth=3, # Thicker
                     capsize=0, alpha=0.7, zorder=2)

        # C. Plot Mean Point (Same color, larger, dense/opaque)
        plt.scatter(m_x, m_y, color=colors[i], s=120, edgecolors='black', 
                    linewidths=1.5, label=f"{x_labels[i]}", zorder=4)

        # D. Annotate Threshold
        # plt.annotate(f"{x_labels[i]}", (m_x, m_y), textcoords="offset points", 
        #              xytext=(0, 15), ha='center', fontsize=9, fontweight='bold',
        #              bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colors[i], alpha=0.7),
        #              zorder=5)

    # 3. Plot the connecting Trend Line in the background
    # plt.plot(means_x, means_y, color='black', linestyle='--', alpha=0.3, linewidth=1, zorder=3)

    # 4. Final Touches
    plt.xlabel('Relative Processing Time', fontsize=12)
    plt.ylabel('Cost Gap (%)', fontsize=12)
    plt.title('Performance of Different Models', fontsize=14)
    
    plt.legend(title="Models", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    plt.show()


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