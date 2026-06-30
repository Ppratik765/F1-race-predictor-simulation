import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse
import textwrap
import matplotlib.image as mpimg
from matplotlib.patches import Circle
# ── Aesthetic Configuration ──────────────────────────────────────────────────
# F1 Light & Clean Storytelling Theme
plt.rcParams.update({
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#ffffff',
    'axes.edgecolor': '#e0e0e0',
    'axes.labelcolor': '#333333',
    'text.color': '#333333',
    'xtick.color': '#555555',
    'ytick.color': '#555555',
    'grid.color': '#f0f0f0',
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ── Dynamic Team Color Loader ────────────────────────────────────────────────
TEAM_COLORS = {}  # Populated at runtime from JSON

def load_team_colors(results_file):
    """Load team colors from the results JSON. Falls back to empty."""
    global TEAM_COLORS
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        mapping = data.get('team_mapping', {})
        for drv, info in mapping.items():
            TEAM_COLORS[drv] = info.get('color', '#888888')
    except Exception:
        pass

def get_color(driver):
    return TEAM_COLORS.get(driver, '#888888')

def setup_output_dir(year, race):
    out_dir = 'visualizations'
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"{out_dir}/f1_sim_{year}_{race}"
    return prefix

def add_watermark(fig, year, race):
    # Bottom right corner watermark text
    fig.text(0.95, 0.02, 'Priyanshu Pratik', 
             fontsize=9, color='#aaaaaa', ha='right', va='center', fontweight='bold', alpha=0.8)
    
    # Add circular image
    try:
        img_path = r"C:\Users\ppmak\.gemini\antigravity-ide\brain\562f5954-531d-4f53-94aa-b5c965efc8d3\media__1782760359196.png"
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            
            # Create a small axes for the image in the bottom right (slightly smaller)
            ax_img = fig.add_axes([0.96, 0.005, 0.03, 0.03], anchor='C', zorder=10)
            ax_img.axis('off')
            
            im = ax_img.imshow(img)
            
            # Create circular clip path
            center = (img.shape[1]/2, img.shape[0]/2)
            radius = min(center[0], center[1])
            patch = Circle(center, radius, transform=ax_img.transData)
            im.set_clip_path(patch)
    except Exception as e:
        print(f"Watermark image error: {e}")

# ── Chart 1: The Championship Contenders (Win, Podium & Points) ──────────────
def plot_win_probabilities(results_file, prefix, year, race):
    print("Generating Win, Podium & Points Probabilities...")
    with open(results_file, 'r') as f:
        data = json.load(f)
        
    results = data['results']
    drivers = list(results.keys())
    
    # Calculate probabilities for the 3 tiers
    wins = {d: results[d]['Race']['Win'] * 100 for d in drivers}
    podiums = {d: (results[d]['Race']['Podium'] - results[d]['Race']['Win']) * 100 for d in drivers}
    points = {d: (results[d]['Race']['Top10'] - results[d]['Race']['Podium']) * 100 for d in drivers}
    
    # FILTER: Show drivers with > 5% chance of scoring points to keep it relevant but not too sparse
    contenders = [d for d in drivers if (wins[d] + podiums[d] + points[d]) > 5.0]
    
    if not contenders:
        contenders = sorted(drivers, key=lambda d: wins[d] + podiums[d] + points[d], reverse=True)[:10]
        
    # Sort by win chance, then podium chance, then points chance
    sorted_contenders = sorted(contenders, key=lambda d: (wins[d], podiums[d], points[d]))
    
    w_vals = [wins[d] for d in sorted_contenders]
    p_vals = [podiums[d] for d in sorted_contenders]
    pts_vals = [points[d] for d in sorted_contenders]
    
    fig, ax = plt.subplots(figsize=(10, max(5, len(sorted_contenders) * 0.6)))
    y = np.arange(len(sorted_contenders))
    
    # Use gold for wins, a soft grey for podiums, and a lighter blue/grey for points
    bars_w = ax.barh(y, w_vals, color='#FFC107', height=0.6, label='Win Probability')
    bars_p = ax.barh(y, p_vals, left=w_vals, color='#E0E0E0', height=0.6, label='Podium (P2-P3)')
    
    left_pts = [w + p for w, p in zip(w_vals, p_vals)]
    bars_pts = ax.barh(y, pts_vals, left=left_pts, color='#BBDEFB', height=0.6, label='Points (P4-P10)')
    
    ax.set_yticks(y)
    ax.set_yticklabels(sorted_contenders, fontweight='bold', fontsize=11)
    
    # Dynamic tick colors based on team
    for tick_label in ax.get_yticklabels():
        tick_label.set_color(get_color(tick_label.get_text()))
        
    # Clean data labels
    for i, (w, p, pts) in enumerate(zip(w_vals, p_vals, pts_vals)):
        # Win label
        if w > 2.0:
            ax.text(w/2, i, f"{w:.0f}%", va='center', ha='center', color='white', fontweight='bold', fontsize=9)
        # Podium label
        if p > 3.0:
            ax.text(w + p/2, i, f"{p:.0f}%", va='center', ha='center', color='#555555', fontsize=9)
        # Points label
        if pts > 4.0:
            ax.text(w + p + pts/2, i, f"{pts:.0f}%", va='center', ha='center', color='#1565C0', fontsize=9)
            
    ax.set_xlabel('Probability (%)', fontsize=10, color='#666')
    ax.set_title(f'The Battle for Glory\nWin, Podium & Points Probabilities', fontsize=16, fontweight='bold', pad=20, loc='left')
    ax.legend(loc='lower right', frameon=False)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.spines['left'].set_visible(False)
    
    add_watermark(fig, year, race)
    plt.tight_layout()
    plt.savefig(f"{prefix}_win_probabilities.png", dpi=300, bbox_inches='tight')
    plt.close()

# ── Chart 2: The Race for Points (Expected Points Yield) ─────────────────────
def plot_expected_points(results_file, raw_data_file, prefix, year, race):
    print("Generating Expected Points...")
    if not os.path.exists(raw_data_file):
        return
        
    with open(raw_data_file, 'r') as f:
        data = json.load(f)
        
    drivers = data['drivers']
    ranks = np.array(data['ranks'])
    active_mask = np.array(data['active_mask'])
    
    points_system = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    
    expected_points = {}
    for idx, driver in enumerate(drivers):
        driver_ranks = np.where(active_mask[:, idx], ranks[:, idx], 20)
        pts = np.array([points_system.get(r, 0) for r in driver_ranks])
        expected_points[driver] = np.mean(pts)
        
    # Sort descending and take TOP 10 to focus the story
    sorted_drivers = sorted(expected_points.keys(), key=lambda d: expected_points[d], reverse=True)[:10]
    
    # Reverse again so P1 is at the top of the chart (since it draws bottom-up)
    sorted_drivers.reverse()
    sorted_pts = [expected_points[d] for d in sorted_drivers]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(sorted_drivers))
    colors = [get_color(d) for d in sorted_drivers]
    
    ax.barh(y, sorted_pts, color=colors, height=0.6, alpha=0.85)
    
    ax.set_yticks(y)
    ax.set_yticklabels(sorted_drivers, fontweight='bold', fontsize=11)
    
    for i, pt in enumerate(sorted_pts):
        ax.text(pt + 0.3, i, f"{pt:.1f} pts", va='center', color='#333333', fontweight='bold')
            
    ax.set_xlabel('Expected Championship Points', fontsize=10, color='#666')
    ax.set_title(f'The Points Battle\nTop 10 Expected Scorers', fontsize=16, fontweight='bold', pad=20, loc='left')
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    add_watermark(fig, year, race)
    plt.tight_layout()
    plt.savefig(f"{prefix}_expected_points.png", dpi=300, bbox_inches='tight')
    plt.close()

# ── Chart 3: The Biggest Movers (Net Position Change) ────────────────────────
def plot_grid_vs_finish(results_file, raw_data_file, prefix, year, race):
    print("Generating Net Position Change (Movers)...")
    with open(results_file, 'r') as f:
        res_data = json.load(f)
    with open(raw_data_file, 'r') as f:
        raw_data = json.load(f)
        
    drivers = raw_data['drivers']
    ranks = np.array(raw_data['ranks'])
    active_mask = np.array(raw_data['active_mask'])
    
    net_change = {}
    
    for idx, driver in enumerate(drivers):
        # Only take ranks where the driver actually finished the race (active_mask is True)
        finishing_ranks = ranks[active_mask[:, idx], idx]
        
        if len(finishing_ranks) > 0:
            exp_finish = np.mean(finishing_ranks)
        else:
            exp_finish = 20 # Fallback
            
        grid_pos = res_data['results'][driver].get('Grid_Position', 20)
        net_change[driver] = grid_pos - exp_finish  # Positive means gaining places
        
    # Sort by net change (biggest losers at top, biggest gainers at bottom)
    sorted_drivers = sorted(drivers, key=lambda d: net_change[d], reverse=False)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(drivers) * 0.4)))
    y = np.arange(len(sorted_drivers))
    
    changes = [net_change[d] for d in sorted_drivers]
    
    # Colors: Green for gaining spots, Red for losing spots, grey for minimal change
    bar_colors = ['#E53935' if c < -0.5 else '#43A047' if c > 0.5 else '#B0BEC5' for c in changes]
    
    ax.barh(y, changes, color=bar_colors, height=0.5, alpha=0.9)
    
    ax.set_yticks(y)
    ax.set_yticklabels(sorted_drivers, fontweight='bold')
    
    # Add a vertical line at 0
    ax.axvline(0, color='#333333', linewidth=1.5)
    
    # Labels
    for i, c in enumerate(changes):
        if c > 0.5:
            ax.text(c + 0.1, i, f"+{c:.1f}", va='center', color='#2E7D32', fontweight='bold', fontsize=9)
        elif c < -0.5:
            ax.text(c - 0.1, i, f"{c:.1f}", va='center', ha='right', color='#C62828', fontweight='bold', fontsize=9)
            
    ax.set_xlabel('Expected Positions Gained / Lost vs. Grid', fontsize=10, color='#666')
    ax.set_title(f'The Chargers & The Fallers\nExpected Position Changes', fontsize=16, fontweight='bold', pad=20, loc='left')
    
    # Hide all spines except bottom
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    add_watermark(fig, year, race)
    plt.tight_layout()
    plt.savefig(f"{prefix}_net_position_change.png", dpi=300, bbox_inches='tight')
    plt.close()

# ── Chart 4: Finishing Position Densities (Tiers) ────────────────────────────
def plot_joyplot(raw_data_file, prefix, year, race):
    print("Generating Tiered KDE Plots...")
    if not os.path.exists(raw_data_file):
        return
        
    with open(raw_data_file, 'r') as f:
        data = json.load(f)
        
    drivers = data['drivers']
    ranks = np.array(data['ranks'])
    active_mask = np.array(data['active_mask'])
    
    # Calculate expected finish to assign tiers
    exp_finish = {}
    valid_results = {}
    for idx, driver in enumerate(drivers):
        driver_ranks = np.where(active_mask[:, idx], ranks[:, idx], 20)
        exp_finish[driver] = np.mean(driver_ranks)
        valid_results[driver] = driver_ranks
        
    sorted_drivers = sorted(drivers, key=lambda d: exp_finish[d])
    
    # Divide into 3 tiers
    tier1 = sorted_drivers[:6]
    tier2 = sorted_drivers[6:14]
    tier3 = sorted_drivers[14:]
    
    tiers = [("The Front-Runners", tier1), ("The Midfield Battle", tier2), ("The Backmarkers", tier3)]
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    for ax, (tier_name, tier_drivers) in zip(axes, tiers):
        for driver in reversed(tier_drivers):  # Reverse so best is painted last/on top
            sns.kdeplot(valid_results[driver], 
                        color=get_color(driver), 
                        fill=True, 
                        alpha=0.25, 
                        linewidth=2.5,
                        label=driver,
                        ax=ax,
                        warn_singular=False)
            
            # Add driver label at the peak of their KDE curve
            try:
                line = ax.lines[-1]
                x_data = line.get_xdata()
                y_data = line.get_ydata()
                max_idx = np.argmax(y_data)
                peak_x = x_data[max_idx]
                peak_y = y_data[max_idx]
                
                # Add an arrow pointing to the peak
                y_offset = np.max(y_data) * 0.15  # Label 15% above the peak
                
                ax.annotate(f" {driver} ",
                            xy=(peak_x, peak_y),
                            xytext=(peak_x, peak_y + y_offset),
                            color=get_color(driver),
                            fontsize=9,
                            fontweight='bold',
                            ha='center', va='bottom',
                            arrowprops=dict(arrowstyle="-|>", color=get_color(driver), linewidth=1.2, alpha=0.8),
                            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))
            except Exception:
                pass
            
        ax.set_title(tier_name, loc='left', fontweight='bold', fontsize=12)
        ax.set_xlim(0.5, 20.5)
        ax.set_xticks(range(1, 21))
        ax.set_ylabel('')
        ax.set_yticks([])  # Hide y axis completely
        
        # Add cutoff lines
        ax.axvline(3.5, color='#FFC107', linestyle=':', alpha=0.8, linewidth=2, zorder=0)
        ax.axvline(10.5, color='#43A047', linestyle=':', alpha=0.8, linewidth=2, zorder=0)
        
        # Legend outside
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
        
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        
    axes[2].set_xlabel('Finishing Position', fontsize=12)
    fig.suptitle(f'Finishing Position Scenarios\n{year} {race.title()}', fontsize=18, fontweight='bold', y=0.98, x=0.1, ha='left')
    
    # Custom annotations for the lines
    axes[0].text(3.5, axes[0].get_ylim()[1]*0.9, ' Podium Cutoff', color='#F57F17', fontweight='bold', fontsize=9)
    axes[0].text(10.5, axes[0].get_ylim()[1]*0.9, ' Points Cutoff', color='#2E7D32', fontweight='bold', fontsize=9)
    
    add_watermark(fig, year, race)
    plt.tight_layout()
    plt.savefig(f"{prefix}_tiered_finishes.png", dpi=300, bbox_inches='tight')
    plt.close()

# ── Chart 5: DNF Risk Profile (Lollipop) ─────────────────────────────────────
def plot_dnf_risk(results_file, prefix, year, race):
    print("Generating DNF Risk Profile (Lollipop)...")
    with open(results_file, 'r') as f:
        data = json.load(f)
        
    results = data['results']
    drivers = list(results.keys())
    
    dnfs = {d: results[d]['Race']['DNF'] * 100 for d in drivers}
    # Sort highest risk to lowest
    sorted_drivers = sorted(dnfs.keys(), key=lambda d: dnfs[d], reverse=False)
    sorted_dnfs = [dnfs[d] for d in sorted_drivers]
    
    avg_dnf = np.mean(sorted_dnfs)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(drivers) * 0.4)))
    y = np.arange(len(sorted_drivers))
    colors = [get_color(d) for d in sorted_drivers]
    
    # Draw sticks
    ax.hlines(y, xmin=0, xmax=sorted_dnfs, color=colors, alpha=0.5, linewidth=2)
    # Draw dots
    ax.scatter(sorted_dnfs, y, color=colors, s=80, zorder=3)
    
    ax.set_yticks(y)
    ax.set_yticklabels(sorted_drivers, fontweight='bold')
    
    # Add average line
    ax.axvline(avg_dnf, color='#E53935', linestyle='--', alpha=0.7, zorder=1)
    ax.text(avg_dnf + 0.2, len(drivers) - 1, f"Grid Average ({avg_dnf:.1f}%)", color='#C62828', fontsize=9, fontweight='bold', va='center')
    
    for i, p in enumerate(sorted_dnfs):
        ax.text(p + 0.5, i, f"{p:.1f}%", va='center', color='#555555', fontsize=9)
        
    ax.set_xlabel('Probability of Retirement (%)', fontsize=10, color='#666')
    ax.set_title(f'Danger Zone\nDNF Risk Profile', fontsize=16, fontweight='bold', pad=20, loc='left')
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    add_watermark(fig, year, race)
    plt.tight_layout()
    plt.savefig(f"{prefix}_dnf_risk.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Professional F1 Monte Carlo Visualizer")
    parser.add_argument('--year', type=int, default=2026)
    parser.add_argument('--race', type=str, default='austria')
    
    args = parser.parse_args()
    
    race_str = args.race.lower()
    raw_file = f'raw_ranks_{race_str}_{args.year}.json'
    res_file = f'results_{race_str}_{args.year}.json'
    
    prefix = setup_output_dir(args.year, race_str)
    load_team_colors(res_file)
    
    try:
        plot_joyplot(raw_file, prefix, args.year, args.race)
        plot_win_probabilities(res_file, prefix, args.year, args.race)
        plot_expected_points(res_file, raw_file, prefix, args.year, args.race)
        plot_grid_vs_finish(res_file, raw_file, prefix, args.year, args.race)
        plot_dnf_risk(res_file, prefix, args.year, args.race)
        print(f"\n[SUCCESS] Generated 5 charts in {prefix}*.png")
    except Exception as e:
        print(f"Error generating visualizations: {e}")
