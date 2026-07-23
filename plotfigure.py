import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report



def plot_results(data):
    GAPS_PEN = data['gap_data']
    RELATIVE_TIME_PEN = data['time_data']
    # PERCENT = data['percent_data']
    # TIMES_PEN = data['penalized_time_data']
    x_labels = data['labels']
    # obj_ref = data['ref_objectif']
    # time_ref = data['ref_time']
    # precision = data['precision']
    # recall = data['recall']
    # y_pred = data['all_preds']
    # y_true = data['all_targets']
    # threshold = data['threshold']

    # relat_times_penalize is the relative time with respect to the reference time
    plot_distribution(GAPS_PEN, RELATIVE_TIME_PEN, x_labels, 'distribution_evolution_with_penalization.png')
    # plot_distribution(GAPS_PEN, RELATIVE_TIME_PEN, PERCENT, x_labels, 'distribution_evolution_with_penalization.png')

    plot_gap_vs_time(GAPS_PEN,RELATIVE_TIME_PEN,x_labels)
    # plotfigure(gap_data_pct_pen, times_penalize, 'combined_performance_pen.png')
    # plotfigure(gap_data_pct_pen, time_ref, 'combined_performance.png')

    # gaps = [[f'{t:.2f}' for t in gap_list] for gap_list in gaps]
    # print(gaps)
    # plot_precision_recall(precision, recall, threshold)

    # print('Time needed to process the initial problem is', time_ref)
    # print('cplex objectif', obj_ref)
    # print('penalization objective', obj_penalize)
    

    # for k in range(len(threshold)):
    #     y_pred[k] = y_pred[k].astype(int).flatten()

    #     # --- Detailed Results ---
    #     print("\n" + "="*30)
    #     print(f"   GNN PERFORMANCE REPORT for threshold {threshold[k]}")
    #     print("="*30)

    #     # 1. Accuracy
    #     acc = (y_pred[k] == y_true).mean() * 100
    #     print(f"Overall Accuracy: {acc:.2f}%")
    
    #     # 2. Confusion Matrix
    #     cm = confusion_matrix(y_true, y_pred[k])
    #     print("\nConfusion Matrix:")
    #     print(cm)

    #     # 3. Precision, Recall, F1
    #     # We focus on 'Setup' (Class 1)
    #     print("\nDetailed Metrics:")
    #     print(classification_report(y_true, y_pred[k], target_names=["No Setup", "Setup"], zero_division=0))
    
    #     # Logic Check
    #     print(f"\nTotal Setup decisions: CPLEX={int(y_true.sum())} | GNN={int(y_pred[k].sum())}")

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



# def plot_distribution(gap_data, time_data, x_labels, fig_name):
#     fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    
#     # Bundle the unique parts of each plot
#     plot_configs = [
#         (gap_data, 'Cost Gap (%)', 'Cost Gap Distribution', 'skyblue', 'blue', (-1,15)),
#         (time_data, 'Relative Proc. Time', 'Processing Time Distribution', 'salmon', 'red', (0,1.5))
#         # (percent_data, 'Percentage fixed', 'Number of setups fixed', 'honeydew', 'forestgreen')
#     ]
    

#     for ax, (data, ylabel, title, f_col, b_col, ylims) in zip(axes, plot_configs):
#         ax.boxplot(data, showfliers=True, patch_artist=True,
#                    boxprops=dict(facecolor=f_col, color=b_col),
#                    medianprops=dict(color='black', linewidth=2))
        
#         ax.set_ylabel(ylabel, fontsize=12)
#         ax.set_title(title, fontsize=14)
#         ax.grid(axis='y', linestyle='--', alpha=0.5)

#         for i, sub_list in enumerate(data, 1):
#             # Check if any data exceeds the limit
#             num_hidden_s = sum(1 for val in sub_list if val > ylims[1])
#             if num_hidden_s > 0:
#                 # For a single box, the x-position is always 1
#                 ax.text(i, ylims[1] * 0.96, r'$\uparrow$'f'{num_hidden_s} outside\nmax {max(sub_list):.2f}', 
#                         color='red', fontweight='bold', ha='center',
#                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

#             num_hidden_bot = sum(1 for val in sub_list if val < ylims[0])
#             if num_hidden_bot > 0:
#                 # We place this at the bottom (ylims[0]) and shift it up slightly (* 1.05 or + offset)
#                 # Using va='bottom' ensures the text sits above the bottom line
#                 ax.text(i, ylims[0] + (ylims[1]-ylims[0])*0.02, r'$\downarrow$' + f'{num_hidden_bot} hidden\nmin: {min(sub_list):.2f}', 
#                         color='blue', fontweight='bold', ha='center', va='bottom', fontsize=9,
#                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

#         ax.set_ylim(ylims)
#         # print(len(data))
#         ax.set_xticks(range(1, len(data) + 1))
#         # ax.set_xticklabels(x_labels)
#         ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    
#     plt.tight_layout()
#     plt.savefig(fig_name, dpi=300, bbox_inches='tight')



def plot_distribution(gap_data, time_data, x_labels, fig_name):
# def plot_distribution(gap_data, time_data, percent_data, x_labels, fig_name):
    """
    Renders boxplots comparing distribution performance profiles across configurations.
    Automatically catches shape orientations to guarantee label alignment accuracy.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    
    gap_arr = np.asarray(gap_data)
    time_arr = np.asarray(time_data)
    # perc_arr = np.asarray(percent_data)
    num_methods = len(x_labels)
    
    # DYNAMIC ORIENTATION DETECTION:
    # If the first dimension length matches your label count, it needs to be transposed 
    # from (Methods, Samples) -> (Samples, Methods) so boxplot constructs exactly N categories.
    if gap_arr.shape[0] == num_methods:
        gap_to_plot = gap_arr.T
        time_to_plot = time_arr.T
        # percent_to_plot = perc_arr.T
    else:
        gap_to_plot = gap_arr
        time_to_plot = time_arr
        # percent_to_plot = perc_arr

    plot_configs = [
        (gap_to_plot, 'Cost Gap (%)', 'Cost Gap Distribution', 'skyblue', 'blue', (-2, 25)),
        (time_to_plot, 'Relative Run Time', 'Run Time Distribution', 'salmon', 'red', (0, 1.2)),
        # (percent_to_plot, 'Percentage fixed', 'Number of setups fixed', 'honeydew', 'forestgreen', (0,1))
    ]
    
    for ax, (data_matrix, ylabel, title, f_col, b_col, ylims) in zip(axes, plot_configs):
        # Generate the N box plots cleanly across the categorical axis
        ax.boxplot(data_matrix, showfliers=True, patch_artist=True,
                   boxprops=dict(facecolor=f_col, color=b_col),
                   medianprops=dict(color='black', linewidth=2))
        
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        # Iterate strictly over active column categories for outlier annotations
        for i in range(data_matrix.shape[1]):
            sub_list = data_matrix[:, i]
            x_pos = i + 1  # Matplotlib box plots start horizontal positions at index 1
            
            # Label elements clipped above maximum bounds
            num_hidden_s = sum(1 for val in sub_list if val > ylims[1])
            if num_hidden_s > 0:
                ax.text(x_pos, ylims[1] * 0.92, r'$\uparrow$'f'{num_hidden_s} outside\nmax {max(sub_list):.2f}', 
                        color='red', fontweight='bold', ha='center', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            # Label elements clipped beneath minimum bounds
            num_hidden_bot = sum(1 for val in sub_list if val < ylims[0])
            if num_hidden_bot > 0:
                ax.text(x_pos, ylims[0] + (ylims[1]-ylims[0])*0.02, r'$\downarrow$' + f'{num_hidden_bot} hidden\nmin: {min(sub_list):.2f}', 
                        color='blue', fontweight='bold', ha='center', va='bottom', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        ax.set_ylim(ylims)
        
        # Configure X-Axis ticks cleanly based on category count
        ax.set_xticks(range(1, num_methods + 1))
        ax.set_xticklabels(x_labels, rotation=30, ha='right')
    
    plt.tight_layout()
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    # plt.close()

# def plot_precision_recall(precision, recall, threshold):
#     fig, ax = plt.subplots(figsize=(10, 6))

#     plot_with_shade(ax, threshold, precision, 'Precision', 'forestgreen')
#     plot_with_shade(ax, threshold, recall, 'Recall', 'darkorange')

#     ax.set_title('Metric Evolution: Mean and Std Dev across Instances', fontsize=14)
#     ax.set_xlabel('GNN Probability Threshold', fontsize=12)
#     ax.set_ylabel('Score (0.0 - 1.0)', fontsize=12)
#     ax.set_ylim(0, 1.05)
#     ax.grid(True, linestyle='--', alpha=0.6)
#     ax.legend(loc='best')

#     plt.tight_layout()
#     plt.savefig('precision_recall.png')

# def plot_with_shade(ax, x, data, label, color):
#     means = np.array([np.mean(d) for d in data])
#     stds = np.array([np.std(d) for d in data])
    
#     ax.plot(x, means, label=label, color=color, marker='o', linewidth=2)
#     ax.fill_between(x, means - stds, means + stds, color=color, alpha=0.2)




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
        x_iqr = [[m_x - np.percentile(t_group, 25)], [np.percentile(t_group, 75) - m_x]]
        y_iqr = [[m_y - np.percentile(g_group, 25)], [np.percentile(g_group, 75) - m_y]]
        x_range = [[m_x - min(t_group)], [max(t_group) - m_x]]
        y_range = [[m_y - min(g_group)], [max(g_group) - m_y]]
        # plt.errorbar(m_x, m_y, xerr=x_range, yerr=y_range, 
        #              fmt='none', ecolor=colors[i], elinewidth=1, 
        #              linestyle='--', alpha=0.3, zorder=1)
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
    plt.xlabel('Relative Run Time', fontsize=12)
    plt.ylabel('Cost Gap (%)', fontsize=12)
    # plt.title('Performance of Different Models', fontsize=14)
    plt.title('Performance with respect to different threshold', fontsize=14)
    
    plt.legend(title="Models", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    plt.show()


def plot_gap_vs_time_two_groups(
    group1_gap, group1_time, group1_labels,
    group2_gap, group2_time, group2_labels,
    fig_name="gap_vs_time_two_groups.png"
):
    plt.figure(figsize=(12, 7)) # Adjusted size for dual-group legend
    
    # 1. Color map (Using plasma for group 1, and another distinctive map like viridis or cividis for group 2)
    colors1 = plt.cm.plasma(np.linspace(0.1, 0.9, len(group1_labels)))
    colors2 = plt.cm.cividis(np.linspace(0, 0.9, len(group2_labels)))
    # colors2 = plt.cm.viridis(np.linspace(0, 0.9, len(group2_labels)))

    # Helper function to plot a single group to avoid code repetition
    def plot_single_group(gap_data, time_data, x_labels, colors, marker_style, group_name_suffix):
        for i, (g_group, t_group) in enumerate(zip(gap_data, time_data)):
            m_x, m_y = np.median(t_group), np.median(g_group)

            # Calculate IQRs
            x_iqr = [[m_x - np.percentile(t_group, 25)], [np.percentile(t_group, 75) - m_x]]
            y_iqr = [[m_y - np.percentile(g_group, 25)], [np.percentile(g_group, 75) - m_y]]
            
            # Plot Error Bars
            plt.errorbar(m_x, m_y, xerr=x_iqr, yerr=y_iqr, 
                         fmt='none', ecolor=colors[i], elinewidth=3, 
                         capsize=0, alpha=0.6, zorder=2)

            # Plot Median Point with unique Marker Style
            plt.scatter(m_x, m_y, color=colors[i], s=140, marker=marker_style, 
                        edgecolors='black', linewidths=1.5, 
                        label=f"{x_labels[i]}", zorder=4)

    # 2. Plot Group 1: Using Large Circles ('o')
    plot_single_group(group1_gap, group1_time, group1_labels, colors1, marker_style='o', group_name_suffix="")

    # 3. Plot Group 2: Using another style like Filled Squares ('s'), Triangles ('^'), or Diamonds ('D')
    plot_single_group(group2_gap, group2_time, group2_labels, colors2, marker_style='^', group_name_suffix="")

    # 4. Final Touches
    plt.xlabel('Relative Run Time', fontsize=12)
    plt.ylabel('Cost Gap (%)', fontsize=12)
    plt.title('Performance comparison across two threshold groups', fontsize=14)
    
    # Places the combined customized legend smoothly to the right side
    plt.legend(title="Threshold Groups", bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    plt.grid(True, linestyle=':', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    plt.show()



def load_and_filter_simulation_data(file_path: str, target_labels: list) -> dict:
    """
    Filters complex simulation data by target labels while maintaining the exact
    original multidimensional and flat array structural footprints.
    
    Uses deep copies to decouple data memory vectors from original structures.
    """
    with open(file_path, 'rb') as f:
        full_results = pickle.load(f)
        
    all_labels = np.asarray(full_results['labels'])
    
    # 1. Map labels to target indices
    matching_indices = np.where(np.isin(all_labels, target_labels))[0]
    
    if len(matching_indices) == 0:
        print(f"Warning: None of the target labels {target_labels} were found.")
        return {}
        
    filtered_results = {}
    
    # 2. Preserve exactly 1D/2D shapes for core matrices with clean allocation copies
    filtered_results['labels'] = all_labels[matching_indices].copy()
    filtered_results['gap_data'] = np.asarray(full_results['gap_data'])[matching_indices].copy()
    filtered_results['time_data'] = np.asarray(full_results['time_data'])[matching_indices].copy()
    filtered_results['penalized_time_data'] = np.asarray(full_results['penalized_time_data'])[matching_indices].copy()
    
    # 3. Dynamic adjustment for parallel metric shapes (precision/recall)
    num_runs = len(all_labels)             
    for key in ['precision', 'recall']:
        if key in full_results:
            metric_arr = np.asarray(full_results[key])
            if metric_arr.shape[0] == num_runs:
                filtered_results[key] = metric_arr[matching_indices].copy()
            else:
                filtered_results[key] = metric_arr.copy()
                
    # 4. Handle Flat Array Slicing safely based on total simulation constants
    num_data_points = np.asarray(full_results['gap_data']).shape[1] 
    total_elements = len(np.asarray(full_results['all_targets']))    
    hidden_dim = total_elements // (num_runs * num_data_points)
    
    # Structure components temporarily to isolate configuration rows cleanly
    targets_temp = np.asarray(full_results['all_targets']).reshape(num_runs, num_data_points, hidden_dim)
    preds_temp = np.asarray(full_results['all_preds']).squeeze().reshape(num_runs, num_data_points, hidden_dim)
    
    # Slice matched runs and return back to standard flat layout footprints
    filtered_results['all_targets'] = targets_temp[matching_indices].flatten()
    filtered_results['all_preds'] = preds_temp[matching_indices].flatten().reshape(1, -1)
    
    # 5. Bring over independent background/static baseline configuration keys
    global_keys = ['ref_objectif', 'ref_time', 'threshold']
    for key in global_keys:
        if key in full_results:
            filtered_results[key] = full_results[key]
            
    return filtered_results

def load_and_filter_simulation_data(file_label_mapping: dict) -> dict:
    """
    Loads multiple simulation files with completely different numbers of labels,
    runs, and data points, extracting ONLY the specified target labels.
    
    Bypasses 3D reshaping entirely by slicing 1D flat structures using 
    calculated per-run memory stride offsets.
    """
    combined_results = {}
    
    # Structural accumulators
    all_found_labels = []
    gap_list = []
    time_list = []
    penalized_time_list = []
    precision_list = []
    recall_list = []
    targets_list = []
    preds_list = []
    
    # Static metadata copies
    global_metadata = {}
    metadata_keys = ['ref_objectif', 'ref_time', 'threshold']

    for path, target_labels in file_label_mapping.items():
        try:
            with open(path, 'rb') as f:
                full_results = pickle.load(f)
        except FileNotFoundError:
            print(f"Warning: File not found at '{path}'. Skipping.")
            continue
            
        all_labels = np.asarray(full_results['labels'])
        matching_indices = np.where(np.isin(all_labels, target_labels))[0]
        
        if len(matching_indices) == 0:
            print(f"Warning: None of the specified targets {target_labels} found in '{path}'.")
            continue
            
        # Capture global metadata once
        for key in metadata_keys:
            if key in full_results and key not in global_metadata:
                global_metadata[key] = full_results[key]
                
        # 1. Collect core 2D matrices (Rows match the specific file's total labels)
        all_found_labels.append(all_labels[matching_indices])
        gap_list.append(np.asarray(full_results['gap_data'])[matching_indices])
        time_list.append(np.asarray(full_results['time_data'])[matching_indices])
        penalized_time_list.append(np.asarray(full_results['penalized_time_data'])[matching_indices])
        
        # 2. Collect precision/recall parallel metrics safely
        num_runs = len(all_labels) # Dynamically scales to this specific file's labels
        if 'precision' in full_results:
            prec_arr = np.asarray(full_results['precision'])
            precision_list.append(prec_arr[matching_indices] if prec_arr.shape[0] == num_runs else prec_arr)
        if 'recall' in full_results:
            rec_arr = np.asarray(full_results['recall'])
            recall_list.append(rec_arr[matching_indices] if rec_arr.shape[0] == num_runs else rec_arr)

        # 3. NO-RESHAPE CHUNK SLICING FOR FLAT ARRAYS:
        # Calculate exactly how many sequential elements belong to one single run label
        flat_targets = np.asarray(full_results['all_targets'])
        flat_preds = np.asarray(full_results['all_preds']).flatten()
        
        stride_targets = len(flat_targets) // num_runs
        stride_preds = len(flat_preds) // num_runs
        
        # Extract the precise flat chunks for the matching runs
        for idx in matching_indices:
            # Targets chunk extraction
            start_t = idx * stride_targets
            end_t = start_t + stride_targets
            targets_list.append(flat_targets[start_t:end_t])
            
            # Predictions chunk extraction
            start_p = idx * stride_preds
            end_p = start_p + stride_preds
            preds_list.append(flat_preds[start_p:end_p])

    if not all_found_labels:
        print("Warning: No matching data discovered across any specified configurations.")
        return {}

    # --- Merge everything cleanly ---
    combined_results['labels'] = np.concatenate(all_found_labels, axis=0)
    
    # Using vstack safely merges matrices even if they have different row counts originally
    combined_results['gap_data'] = np.vstack(gap_list)
    combined_results['time_data'] = np.vstack(time_list)
    combined_results['penalized_time_data'] = np.vstack(penalized_time_list)
    
    if precision_list:
        combined_results['precision'] = np.vstack(precision_list)
    if recall_list:
        combined_results['recall'] = np.vstack(recall_list)
        
    # Re-flatten the collected chunks back into their original format wrappers
    combined_results['all_targets'] = np.concatenate(targets_list, axis=0)
    combined_results['all_preds'] = np.concatenate(preds_list, axis=0).reshape(1, -1)
    
    for key, val in global_metadata.items():
        combined_results[key] = val
        
    return combined_results


if __name__ == "__main__":
    # with open('results/simulation_results_4days_40data.pkl', 'rb') as f:
    with open('results/simulation_results_1-2days_spo_aveY.pkl', 'rb') as f:
        data = pickle.load(f)
        print(data['labels'])
    # plot_results(data)

    targets_config = {
        'results/simulation_results_4days_1200s.pkl': [
            # '4800data_bce_epoch_20','2400data_bce_epoch_20', 
            '150data_bce_epoch_20',
            '150data_spo_epoch_50', '150data_hybrid_epoch_50', '150data_fy_epoch_50',
            '150data_spo_moving_ave_3', '150data_spo_moving_ave_5'
        ],
        # 'results/simulation_results_1-2days.pkl': [
        #     # '4800data_bce_epoch_20','2400data_bce_epoch_20',
        #     '150data_bce_epoch_20', 
        #     '150data_spo_epoch_50', '150data_hybrid_epoch_50','150data_fy_epoch_50'
        # ],
        # 'results/simulation_results_1-2days_moving_ave.pkl': [
        #     '150data_spo_moving_ave_3', '150data_spo_moving_ave_5'
        # ],
        # 'results/simulation_results_1-2days_soft_penalization.pkl': [
        #     '150data_bce_epoch_20_pen'
        # ],
        # 'results/simulation_results_1-2days_crossentropy_pen.pkl': [
        #     '150data_bce_epoch_20_CE_pen'
        # ],
        # 'results/simulation_results_1-2days_Train_SPO_CE_pen_test_CE.pkl': [
        #     '150data_SPO_epoch_50_CE_pen'
        # ],
        'results/simulation_results_4days_spo_aveY.pkl': [
            '150data_spo_aveY_epoch_50'
        ],
    }
    filtered_data_1 = load_and_filter_simulation_data(targets_config)
    # print(filtered_data)
    all_found_labels_1 = filtered_data_1['labels']
    gap_list_1 = filtered_data_1['gap_data']
    time_list_1 = filtered_data_1['time_data']

    # results for the hard fixing strategy
    thres = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    # thres = [0.5]
    # file_path = 'results/simulation_results_1-2days_vardataset_soft_fixing.pkl'
    file_path = 'results/simulation_results_150data_4days_soft_fixing_evaluate.pkl'
    # Open the file in read-binary ('rb') mode
    with open(file_path, 'rb') as file:
        full_results_2 = pickle.load(file)
    
    all_thresholds = full_results_2['threshold']
    indexes = np.where(np.isin(all_thresholds, thres))[0]

    all_found_labels_2 = [f'threshold_{x}' for x in thres]
    gap_list_2 = []
    time_list_2 = []
    perct_fixe = []

    gap_list_2.append(np.asarray(full_results_2['gap_fixing'])[indexes])
    time_list_2.append(np.asarray(full_results_2['time_fixing'])[indexes])
    perct_fixe.append(np.asarray(full_results_2['percentage_fixed'])[indexes])
    # print(gap_list_2)

    gap_list_2 = np.vstack(gap_list_2)*100
    time_list_2 = np.vstack(time_list_2)
    percent_list_2 = np.vstack(perct_fixe)
    

    row_idx_1, col_idx_1 = np.unravel_index(np.argmax(gap_list_1), gap_list_1.shape)
    print(f"Largest value: {gap_list_1[row_idx_1, col_idx_1]}")
    print(f"Position -> Row: {row_idx_1}, Column: {col_idx_1}")
    row_idx_2, col_idx_2 = np.unravel_index(np.argmax(gap_list_2), gap_list_2.shape)
    print(f"Largest value: {gap_list_2[row_idx_2, col_idx_2]}")
    print(f"Position -> Row: {row_idx_2}, Column: {col_idx_2}")

    combined_results = {}
    combined_results['labels'] = np.concatenate((all_found_labels_1, all_found_labels_2), axis=0)
    combined_results['gap_data'] = np.vstack((gap_list_1,gap_list_2))
    combined_results['time_data'] = np.vstack((time_list_1,time_list_2))

    # combined_results = {}
    # combined_results['labels'] = all_found_labels_2
    # combined_results['gap_data'] = gap_list_2
    # combined_results['time_data'] = time_list_2
    # combined_results['percent_data'] = percent_list_2

    # print(combined_results['gap_data'])
    
    # plot_results(filtered_data_1)
    plot_results(combined_results)
    plot_gap_vs_time_two_groups(gap_list_1, time_list_1, all_found_labels_1,
    gap_list_2, time_list_2, all_found_labels_2)
    