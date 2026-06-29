import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def plot_kde(raw_data_file):
    """
    Plots a Kernel Density Estimate of finishing positions to show the bimodal distribution.
    Requires the raw ranks JSON output.
    """
    if not os.path.exists(raw_data_file):
        print(f"File {raw_data_file} not found.")
        return

    with open(raw_data_file, 'r') as f:
        data = json.load(f)
        
    drivers = data['drivers']
    ranks = np.array(data['ranks'])
    active_mask = np.array(data['active_mask'])
    
    plt.figure(figsize=(12, 6))
    
    # We will plot the top 5 drivers to avoid clutter
    # Find top 5 by mean rank
    mean_ranks = np.mean(ranks, axis=0)
    top_5_indices = np.argsort(mean_ranks)[:5]
    
    for idx in top_5_indices:
        driver = drivers[idx]
        driver_ranks = ranks[:, idx]
        driver_active = active_mask[:, idx]
        
        # We need to map DNF ranks to 20 for the histogram/KDE to show the bimodal spike
        # In our simulation, DNFs were given np.inf which became a rank of 20 (or similar)
        # But let's explicitly push DNFs to position 20 to group them perfectly
        plot_ranks = np.where(driver_active, driver_ranks, 20)
        
        # Use a smooth density plot
        try:
            import seaborn as sns
            sns.kdeplot(plot_ranks, label=driver, bw_adjust=1.5, fill=True, alpha=0.1)
        except ImportError:
            # Fallback to matplotlib histogram if seaborn is not installed
            plt.hist(plot_ranks, bins=np.arange(0.5, 21.5, 1), density=True, histtype='step', label=driver, linewidth=2)

    plt.title('KDE of Finishing Positions (Showing Bimodal DNF Peak at P20)')
    plt.xlabel('Finishing Position (20 = DNF)')
    plt.ylabel('Probability Density')
    plt.xticks(range(1, 21))
    plt.xlim(1, 20)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('kde_plot.png')
    print("Saved KDE plot to kde_plot.png")

def plot_stacked_bar(results_file):
    """
    Plots a horizontal stacked bar chart of mutually exclusive probabilities:
    Win, Podium (P2-3), Points (P4-10), Other Finish (P11-20), DNF.
    """
    if not os.path.exists(results_file):
        print(f"File {results_file} not found.")
        return
        
    with open(results_file, 'r') as f:
        data = json.load(f)
        
    results = data['results']
    drivers = list(results.keys())
    
    # Sort drivers by Win probability, then Podium
    drivers.sort(key=lambda d: (results[d]['Race']['Win'], results[d]['Race']['Podium']), reverse=False)
    
    wins = []
    podiums_only = []
    points_only = []
    others = []
    dnfs = []
    
    for d in drivers:
        r = results[d]['Race']
        w = r['Win']
        p = r['Podium'] - w
        t10 = r['Top10'] - r['Podium']
        f = r['Finish'] - r['Top10']
        dnf = r['DNF']
        
        wins.append(w)
        podiums_only.append(p)
        points_only.append(t10)
        others.append(f)
        dnfs.append(dnf)
        
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y = np.arange(len(drivers))
    
    p1 = ax.barh(y, wins, color='gold', label='Win')
    p2 = ax.barh(y, podiums_only, left=wins, color='silver', label='Podium (P2-P3)')
    p3 = ax.barh(y, points_only, left=np.array(wins)+np.array(podiums_only), color='darkorange', label='Points (P4-P10)')
    p4 = ax.barh(y, others, left=np.array(wins)+np.array(podiums_only)+np.array(points_only), color='lightgray', label='Finish (>P10)')
    p5 = ax.barh(y, dnfs, left=np.array(wins)+np.array(podiums_only)+np.array(points_only)+np.array(others), color='red', label='DNF')
    
    ax.set_yticks(y)
    ax.set_yticklabels(drivers)
    ax.set_xlabel('Probability')
    ax.set_title('Race Outcome Probabilities per Driver')
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    
    plt.tight_layout()
    plt.savefig('stacked_bar.png')
    print("Saved Stacked Bar plot to stacked_bar.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 Monte Carlo Visualizer")
    parser.add_argument('--year', type=int, default=2023)
    parser.add_argument('--race', type=str, default='bahrain')
    
    args = parser.parse_args()
    
    race_str = args.race.lower()
    
    plot_kde(f'raw_ranks_{race_str}_{args.year}.json')
    plot_stacked_bar(f'results_{race_str}_{args.year}.json')
