"""
backtester.py — Orchestrator & Human-Readable Reporter
=======================================================
Loads FastF1 data → runs qualifying sim → runs race sim → prints a clean
Markdown-style ASCII table to the terminal and saves structured JSON.
"""

import json
import time
import numpy as np
from data_pipeline import get_session, extract_quali_stats, extract_race_pace_and_deg
from quali_engine import run_quali_sim
from race_engine import run_race_sim


# ── Pretty-print helpers ──────────────────────────────────────────────────

def _pct(value):
    """Format a 0-1 float as a clean percentage string like '82.4%'."""
    return f"{value * 100:.1f}%"


def _print_table(title, headers, rows):
    """
    Renders a Markdown-style ASCII grid table to stdout.

    Args:
        title:   str — table caption
        headers: list[str]
        rows:    list[list[str]]
    """
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    hdr = "|" + "|".join(f" {h:<{col_widths[i]}} " for i, h in enumerate(headers)) + "|"

    print(f"\n{'=' * len(sep)}")
    print(f"  {title}")
    print(f"{'=' * len(sep)}")
    print(sep)
    print(hdr)
    print(sep.replace("-", "="))
    for row in rows:
        line = "|" + "|".join(
            f" {cell:<{col_widths[i]}} " for i, cell in enumerate(row)
        ) + "|"
        print(line)
    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────

def main(year=2023, race='Bahrain', num_iterations=100_000):
    print(f"\n{'─' * 60}")
    print(f"  F1 Monte Carlo Simulation  ·  {year} {race}")
    print(f"  Iterations: {num_iterations:,}")
    print(f"{'─' * 60}")

    t0 = time.time()

    # 1. Load Data ──────────────────────────────────────────────────────
    print("\n⏳ Loading telemetry via FastF1 …")
    try:
        session_fp3 = get_session(year, race, 'FP3')
        session_fp2 = get_session(year, race, 'FP2')
    except Exception as e:
        print(f"✗ Failed to load sessions: {e}")
        print("  Hint: Sprint weekends lack FP3 — use FP1 or Qualifying.")
        return

    print("⏳ Extracting driver statistics …")
    quali_stats = extract_quali_stats(session_fp3)
    race_stats  = extract_race_pace_and_deg(session_fp2)

    if not quali_stats or not race_stats:
        print("✗ Could not extract stats. Check session data.")
        return

    t1 = time.time()
    print(f"✓ Data ready in {t1 - t0:.1f} s")

    # 2. Qualifying ─────────────────────────────────────────────────────
    print(f"\n⏳ Qualifying sim ({num_iterations:,} iter) …")
    expected_grid, pole_probs, _, quali_drivers = run_quali_sim(
        quali_stats, num_iterations
    )
    t2 = time.time()
    print(f"✓ Qualifying done in {t2 - t1:.1f} s")

    # Print qualifying table
    grid_rows = sorted(expected_grid.items(), key=lambda x: x[1])
    _print_table(
        "QUALIFYING PREDICTIONS",
        ["Pos", "Driver", "Exp. Grid", "Pole %"],
        [
            [str(idx + 1), drv, f"{pos:.1f}", _pct(pole_probs.get(drv, 0))]
            for idx, (drv, pos) in enumerate(grid_rows)
        ],
    )

    # 3. Race ───────────────────────────────────────────────────────────
    num_laps = 57  # typical full race; configurable per track
    print(f"\n⏳ Race sim ({num_iterations:,} iter, {num_laps} laps) …")
    finishing_probs, final_ranks, race_drivers, active_mask = run_race_sim(
        race_stats=race_stats,
        grid_positions=expected_grid,
        num_iterations=num_iterations,
        num_laps=num_laps,
        num_pitstops=2,
        pitstop_time_loss=22.0,
    )
    t3 = time.time()
    print(f"✓ Race done in {t3 - t2:.1f} s")

    # Print race table (sorted by Win probability descending)
    race_rows = sorted(
        finishing_probs.items(),
        key=lambda x: x[1]['Win'],
        reverse=True,
    )
    _print_table(
        "RACE PREDICTIONS",
        ["#", "Driver", "Win", "Podium", "Top 10", "Finish", "DNF"],
        [
            [
                str(idx + 1),
                drv,
                _pct(probs['Win']),
                _pct(probs['Podium']),
                _pct(probs['Top10']),
                _pct(probs['Finish']),
                _pct(probs['DNF']),
            ]
            for idx, (drv, probs) in enumerate(race_rows)
        ],
    )

    # 4. Save JSON ──────────────────────────────────────────────────────
    output_data = {
        'metadata': {
            'year': year,
            'race': race,
            'iterations': num_iterations,
            'laps': num_laps,
        },
        'results': {},
    }

    for driver in race_drivers:
        output_data['results'][driver] = {
            'Qualifying': {
                'Expected_Grid': round(expected_grid.get(driver, 20), 2),
                'Pole_Probability': round(pole_probs.get(driver, 0.0), 4),
            },
            'Race': {
                'Win':    round(finishing_probs[driver]['Win'],    4),
                'Podium': round(finishing_probs[driver]['Podium'], 4),
                'Top10':  round(finishing_probs[driver]['Top10'],  4),
                'Finish': round(finishing_probs[driver]['Finish'], 4),
                'DNF':    round(finishing_probs[driver]['DNF'],    4),
            },
        }

    output_file = f'results_{race.lower()}_{year}.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

    # Save raw ranks subset for visualizer KDE
    raw_results = {
        'drivers': race_drivers,
        'ranks': final_ranks[:5000].tolist(),
        'active_mask': active_mask[:5000].tolist(),
    }
    with open(f'raw_ranks_{race.lower()}_{year}.json', 'w') as f:
        json.dump(raw_results, f)

    t4 = time.time()
    print(f"\n💾 JSON saved → {output_file}")
    print(f"⏱  Total wall time: {t4 - t0:.1f} s")


if __name__ == "__main__":
    main(year=2023, race='Bahrain', num_iterations=100_000)
