import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report



def plot_results(data):
    GAPS_PEN = data['gap_data']
    RELATIVE_TIME_PEN = data['time_data']
    TIMES_PEN = data['penalized_time_data']
    x_labels = data['labels']
    obj_ref = data['ref_objectif']
    time_ref = data['ref_time']
    precision = data['precision']
    recall = data['recall']
    y_pred = data['all_preds']
    y_true = data['all_targets']
    threshold = data['threshold']
    # relat_times_penalize is the relative time with respect to the reference time
    plot_distribution(GAPS_PEN, RELATIVE_TIME_PEN, x_labels, 'distribution_evolution_with_penalization.png')

    plot_gap_vs_time(GAPS_PEN,RELATIVE_TIME_PEN,x_labels)
    # plotfigure(gap_data_pct_pen, times_penalize, 'combined_performance_pen.png')
    # plotfigure(gap_data_pct_pen, time_ref, 'combined_performance.png')

    # gaps = [[f'{t:.2f}' for t in gap_list] for gap_list in gaps]
    # print(gaps)
    plot_precision_recall(precision, recall, threshold)

    # print('Time needed to process the initial problem is', time_ref)
    # print('cplex objectif', obj_ref)
    # print('penalization objective', obj_penalize)
    

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
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')

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
    plt.xlabel('Relative Processing Time', fontsize=12)
    plt.ylabel('Cost Gap (%)', fontsize=12)
    plt.title('Performance of Different Models', fontsize=14)
    
    plt.legend(title="Models", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    with open('results/simulation_results_4days_40data.pkl', 'rb') as f:
    # with open('results/simulation_results_1-2days.pkl', 'rb') as f:
        data = pickle.load(f)
    plot_results(data)