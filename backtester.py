"""
backtester.py — Smart Orchestrator & Human-Readable Reporter
==============================================================
Detects which sessions are available for a given race weekend and
adapts its behaviour accordingly:

  • Future race  → prints countdown to weekend / data availability
  • FP2+FP3 exist, but Qualifying hasn't happened → runs quali prediction
  • Qualifying results exist → uses REAL grid for race simulation
  • All sessions exist → full backtest with real grid
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import time
import datetime
import argparse
import numpy as np
import fastf1
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
fastf1.set_log_level('ERROR')

from data_pipeline import (
    Spinner,
    get_session,
    extract_quali_stats,
    extract_race_pace_and_deg,
    extract_real_grid,
    extract_reliability_stats,
    extract_season_trends,
    extract_team_mapping,
    extract_weather_context,
    extract_speed_metrics,
    extract_pitstop_stats,
    apply_team_sandbag_correction,
    detect_tow_assisted_laps,
    _get_track_type,
)
from quali_engine import run_quali_sim
from race_engine import run_race_sim


# ── Pretty-print helpers ──────────────────────────────────────────────────

def _pct(value):
    """Format a 0-1 float as a clean percentage string like '82.4%'."""
    return f"{value * 100:.1f}%"


def _print_table(title, headers, rows):
    """Renders a Markdown-style ASCII grid table to stdout."""
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
    print(sep.replace('-', '='))
    for row in rows:
        line = "|" + "|".join(
            f" {cell:<{col_widths[i]}} " for i, cell in enumerate(row)
        ) + "|"
        print(line)
    print(sep)


def _fmt_delta(td):
    """Format a timedelta into a human-readable string like '4 days, 3 hours'."""
    total_seconds = int(td.total_seconds())
    if total_seconds < 0:
        return "already passed"
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, _ = divmod(remainder, 60)
    parts = []
    if days > 0:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0 and days == 0:
        parts.append(f"{minutes} min")
    return ", ".join(parts) if parts else "< 1 min"


# ── Session availability checker ─────────────────────────────────────────

def _check_event_schedule(year, race):
    """
    Returns the event schedule row and session dates.
    Uses fastf1.get_event() to look up the race weekend.
    Returns (event, is_future, delta_to_weekend, delta_to_race) or None.
    """
    try:
        event = fastf1.get_event(year, race)
    except Exception:
        return None

    now = datetime.datetime.now(tz=datetime.timezone.utc)

    # Find the earliest session date (FP1 / Session1)
    weekend_start = None
    race_time = None
    for col in ['Session1DateUtc', 'Session2DateUtc', 'Session3DateUtc',
                'Session4DateUtc', 'Session5DateUtc']:
        val = event.get(col)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            try:
                dt = pd.Timestamp(val)
                if dt.tzinfo is None:
                    dt = dt.tz_localize('UTC')
                if weekend_start is None or dt < weekend_start:
                    weekend_start = dt
            except Exception:
                pass

    # Race is typically Session5
    for col in ['Session5DateUtc', 'Session4DateUtc']:
        val = event.get(col)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            try:
                dt = pd.Timestamp(val)
                if dt.tzinfo is None:
                    dt = dt.tz_localize('UTC')
                race_time = dt
                break
            except Exception:
                pass

    if weekend_start is None:
        return None

    now_ts = pd.Timestamp(now)
    is_future = now_ts < weekend_start
    delta_weekend = weekend_start - now_ts if is_future else None
    delta_race = (race_time - now_ts) if race_time and now_ts < race_time else None

    return event, is_future, delta_weekend, delta_race


def _try_load_session(year, race, session_type):
    """Attempts to load a session. Returns the session (tagged with which
    session_type string actually succeeded, via ._loaded_as) or None."""
    try:
        s = get_session(year, race, session_type)
        # Check if the session actually has lap data
        if s.laps is not None and len(s.laps) > 0:
            s._loaded_as = session_type
            return s
    except Exception:
        pass
    return None


# -- Main ------------------------------------------------------------------

def apply_manual_penalties(raw_grid, penalties_str):
    if not penalties_str or not raw_grid:
        return raw_grid, []
    
    penalties = {}
    pitlane_starters = []
    for p in penalties_str.split(','):
        if ':' in p:
            drv, pen = p.split(':')
            drv = drv.strip()
            pen = pen.strip().upper()
            if pen == 'PL':
                pitlane_starters.append(drv)
            else:
                try:
                    penalties[drv] = int(pen)
                except ValueError:
                    pass
    
    N = len(raw_grid)
    grid = [None] * N
    
    active_grid = {d: p for d, p in raw_grid.items() if d not in pitlane_starters}
    sorted_drivers = sorted(active_grid.items(), key=lambda x: x[1])
    
    # 1. Unpenalized drivers placed in qualifying positions
    for drv, pos in sorted_drivers:
        if drv not in penalties:
            grid[pos - 1] = drv
            
    # Remove gaps
    grid = [d for d in grid if d is not None]
    
    # 2. Penalized drivers
    penalized_drivers = []
    for drv, pos in sorted_drivers:
        if drv in penalties:
            penalized_drivers.append((drv, pos + penalties[drv], pos))
            
    # Sort penalized by temporary position ASC, then quali position DESC (slowest first)
    penalized_drivers.sort(key=lambda x: (x[1], -x[2]))
    
    for drv, temp_pos, _ in penalized_drivers:
        insert_idx = temp_pos - 1
        if insert_idx >= len(grid):
            grid.append(drv)
        else:
            grid.insert(insert_idx, drv)
            
    adjusted_grid = {}
    current_pos = 1
    for drv in grid:
        if drv is not None:
            adjusted_grid[drv] = current_pos
            current_pos += 1
            
    for drv in pitlane_starters:
        adjusted_grid[drv] = current_pos
        current_pos += 1
        
    return adjusted_grid, pitlane_starters

def main(year=2025, race='Monza', num_iterations=50_000, penalties_str="", min_laps=7):
    import pandas as pd  # local import for Timestamp usage

    print(f"\n{'-' * 60}")
    print(f"  F1 Monte Carlo Simulation  ·  {year} {race}")
    print(f"  Iterations: {num_iterations:,}")
    print(f"{'-' * 60}")

    t0 = time.time()

    # ── Step 0: Check if this is a future race ────────────────────────
    schedule_info = _check_event_schedule(year, race)
    if schedule_info is not None:
        event, is_future, delta_weekend, delta_race = schedule_info
        if is_future:
            event_name = event.get('EventName', race)
            print(f"\n🏁 {event_name} {year}")
            print(f"   Race weekend commences in: {_fmt_delta(delta_weekend)}")
            if delta_race:
                print(f"   Race prediction available in: {_fmt_delta(delta_race)}")
                print(f"   (Requires FP2/FP3 or Sprint telemetry to be available)")
            print(f"\n   ⏳ No telemetry data available yet. Check back after")
            print(f"      practice sessions have been completed.")
            return

    # ── Step 1: Load practice data (FP2 for race pace, FP3 for quali) ─
    t0 = time.time()
    with Spinner("Fetching data..."):
        session_fp2 = _try_load_session(year, race, 'FP2')
        session_fp3 = _try_load_session(year, race, 'FP3')

        # Fallback: if FP3 doesn't exist (sprint weekend), try Sprint Qualifying
        if session_fp3 is None:
            session_fp3 = _try_load_session(year, race, 'SQ')
        if session_fp3 is None:
            session_fp3 = _try_load_session(year, race, 'FP1')

        if session_fp2 is None:
            session_fp2 = _try_load_session(year, race, 'S')
        if session_fp2 is None:
            session_fp2 = _try_load_session(year, race, 'FP1')

        if session_fp2 is None:
            print("\n✗ No race-pace representative session found (FP2/S/FP1).")
            print("  Hint: Data may not yet be uploaded for this weekend.")
            return

        if session_fp3 is None:
            print("\n✗ No qualifying-representative session found (FP3/SQ/FP1).")
            print("  Hint: Data may not yet be uploaded for this weekend.")
            return

        # Get track specifics (df_type, turn_1_chaos, tow_factor, overtaking_diff, SC/VSC rates)
        try:
            event_info = fastf1.get_event(year, race)
            track_info = _get_track_type(event_info['EventName'])
        except Exception:
            track_info = None

        quali_stats, tow_flags = extract_quali_stats(session_fp3, track_info=track_info)
        race_stats  = extract_race_pace_and_deg(session_fp2, min_laps=min_laps)
        reliability_stats = extract_reliability_stats(year, race)
        season_trends, team_trends = extract_season_trends(year, race)
        team_mapping = extract_team_mapping(session_fp2)
        weather_context = extract_weather_context(session_fp2)

        # Straight-line speed / power-unit deployment index (speed-trap based,
        # blended across whichever practice sessions we actually loaded)
        power_index = extract_speed_metrics(session_fp2) or {}
        fp3_power_index = extract_speed_metrics(session_fp3) or {}
        for d, v in fp3_power_index.items():
            power_index[d] = (power_index.get(d, v) + v) / 2.0 if d in power_index else v


        # Team-specific pit stop loss + botch probability, anchored to this circuit's own pit loss
        track_pit_loss_base = (track_info.get('pit_loss_base', 22.0) if track_info else 22.0)
        pitstop_stats = extract_pitstop_stats(year, race, track_pit_loss_base=track_pit_loss_base)

        # ── Team-level sandbagging correction ──────────────────────────
        # Session label matters: FP1-derived stats are trusted less than FP2/FP3/Quali.
        fp3_label = getattr(session_fp3, '_loaded_as', None) or 'FP3'
        fp2_label = getattr(session_fp2, '_loaded_as', None) or 'FP2'
        quali_stats, _, quali_sandbag_flags = apply_team_sandbag_correction(
            quali_stats, {}, team_mapping, team_trends, session_label=fp3_label
        )
        _, race_stats, race_sandbag_flags = apply_team_sandbag_correction(
            {}, race_stats, team_mapping, team_trends, session_label=fp2_label
        )
        sandbag_flags = quali_sandbag_flags + race_sandbag_flags

    if not quali_stats or not race_stats:
        print("✗ Could not extract stats. Check session data.")
        return

    t1 = time.time()
    print(f"Data ready in {t1 - t0:.1f} s")

    if sandbag_flags:
        print("\n⚠️  SANDBAGGING / ANOMALY FLAGS")
        for flag in sandbag_flags:
            print(f"   • {flag}")

    if tow_flags:
        print("\n🌬️  TOW / DRAFT FLAGS (grid pace trusted less for these drivers)")
        for d, secs in sorted(tow_flags.items(), key=lambda x: -x[1]):
            print(f"   • {d}: fastest lap looked ~{secs:.2f}s better than their own session pace "
                  f"suggests — likely a big slipstream, not repeatable race pace.")

    low_confidence = []
    for d, stats in race_stats.items():
        soft = stats.get('SOFT', {})
        if soft.get('sample_laps', 0) > 0 and soft.get('longest_stint', 99) < 9:
            low_confidence.append((d, soft.get('longest_stint', 0), soft.get('num_stints', 0)))
        elif soft.get('sample_laps', 0) == 0:
            low_confidence.append((d, 0, 0))
    if low_confidence:
        print("\n📉  LOW-CONFIDENCE RACE PACE (short/no long-run sample — treat with caution)")
        for d, longest, n_stints in sorted(low_confidence):
            if longest == 0:
                print(f"   • {d}: no qualifying long-run stint found — using fallback estimate.")
            else:
                print(f"   • {d}: longest long-run stint was only {longest} laps ({n_stints} stint(s)) "
                      f"— could still reflect an unrepresentative fuel load.")

    if track_info:
        print(f"\n   Track history: SC {_pct(track_info.get('sc_probability', 0.35))} · "
              f"VSC {_pct(track_info.get('vsc_probability', 0.25))} · "
              f"Typical pit loss ~{track_info.get('pit_loss_base', 22.0):.1f}s")
    
    # Print weather conditions
    if weather_context:
        rain_str = "WET" if weather_context.get('rainfall', False) else "DRY"
        print(f"\n   Weather: Track {weather_context['track_temp']:.0f} C | Air {weather_context['air_temp']:.0f} C | {rain_str}")

    if season_trends:
        trend_rows = sorted(season_trends.items(), key=lambda x: x[1]['power_rank_delta'])
        _print_table(
            "SEASON POWER RANKINGS (Last 5 Races)",
            ["Rank", "Driver", "Avg Race Pace Deficit", "Sunday Conversion Factor"],
            [
                [str(idx + 1), drv, f"+{stats['power_rank_delta']:.3f}s", f"{stats['sunday_conversion']:+.3f}s"]
                for idx, (drv, stats) in enumerate(trend_rows)
            ],
        )

    # ── Step 2: Determine grid — real or simulated ────────────────────
    real_grid = None
    pole_probs = {}
    grid_source = "SIMULATED"
    pitlane_starters = []
    race_session = None

    # Try loading actual Race results first for official grid penalties
    with Spinner("Fetching real race grid..."):
        race_session = _try_load_session(year, race, 'R')
        if race_session is not None and race_session.results is not None and not race_session.results.empty:
            grid_res = extract_real_grid(race_session)
            if grid_res[0] is not None:
                real_grid = grid_res[0]
                pitlane_starters = grid_res[1]

    # ALWAYS load Q session to get tow flags (and grid if R wasn't available)
    with Spinner("Fetching real quali..."):
        quali_session = _try_load_session(year, race, 'Q')
        if quali_session is not None:
            if real_grid is None:
                grid_res = extract_real_grid(quali_session)
                if grid_res[0] is not None:
                    real_grid = grid_res[0]
                    pitlane_starters = grid_res[1]
            
            real_tow_flags = detect_tow_assisted_laps(quali_session, track_info=track_info)
            if real_tow_flags:
                tow_flags = {**tow_flags, **real_tow_flags}

    if real_grid is not None:
        grid_source = "ACTUAL (from Official Session)"
        grid_positions = {d: float(p) for d, p in real_grid.items()}
        print(f"Fetched official grid! ({len(real_grid)} drivers)")
        
        if race_session is not None:
            print("\n🚨  OFFICIAL RACE SESSION FOUND: Grid penalties and Pit Lane starts automatically applied from official FIA starting grid.")

        # Apply manual penalties only if we didn't get them from the Race session
        if race_session is None and penalties_str:
            grid_positions, pl_starters = apply_manual_penalties(grid_positions, penalties_str)
            pitlane_starters.extend(pl_starters)
            grid_source += " + MANUAL PENALTIES"

        if 'real_tow_flags' in locals() and real_tow_flags:
            print("\n🌬️  TOW / DRAFT FLAGS FROM ACTUAL QUALIFYING")
            for d, secs in sorted(real_tow_flags.items(), key=lambda x: -x[1]):
                print(f"   • {d}: grid slot looks ~{secs:.2f}s better than a repeatable lap — "
                      f"race pace will not be clamped as tightly to this grid position.")

        # Print actual grid
        grid_rows = sorted(grid_positions.items(), key=lambda x: x[1])
        
        formatted_rows = []
        for drv, pos in grid_rows:
            pos_str = "PIT LANE" if drv in pitlane_starters else str(int(pos))
            formatted_rows.append([pos_str, drv])
            
        _print_table(
            "STARTING GRID",
            ["Pos", "Driver"],
            formatted_rows,
        )
    else:
        # No real qualifying → run simulation
        with Spinner("Doing quali simulation..."):
            expected_grid, pole_probs, _, quali_drivers = run_quali_sim(
                quali_stats, track_info, season_trends, num_iterations, power_index=power_index
            )
        grid_positions = expected_grid
        pitlane_starters = []
        grid_source = "PREDICTED (Monte Carlo)"

        if penalties_str:
            discrete_grid = {drv: idx + 1 for idx, (drv, pos) in enumerate(sorted(expected_grid.items(), key=lambda x: x[1]))}
            grid_positions, pitlane_starters = apply_manual_penalties(discrete_grid, penalties_str)
            grid_source += " + MANUAL PENALTIES"

        t_q = time.time()
        print(f"✓ Qualifying sim done in {t_q - t1:.1f} s")

        grid_rows = sorted(grid_positions.items(), key=lambda x: x[1])
        formatted_rows = []
        for drv, pos in grid_rows:
            pos_str = "PIT LANE" if drv in pitlane_starters else f"{pos:.1f}" if not penalties_str else str(int(pos))
            formatted_rows.append([pos_str, drv, _pct(pole_probs.get(drv, 0))])

        _print_table(
            "QUALIFYING PREDICTIONS",
            ["Pos", "Driver", "Pole %"],
            formatted_rows,
        )

    print(f"\n   Grid source: {grid_source}")

    # ── Step 3: Race simulation ───────────────────────────────────────
    num_laps = 57
    t2 = time.time()
    
    # Ensure grid_positions only includes drivers we have race pace for
    num_drivers = len(race_stats.keys())
    sim_grid = {d: grid_positions.get(d, num_drivers) for d in race_stats.keys()}

    with Spinner("Doing race simulation..."):
        finishing_probs, final_ranks, race_drivers, active_mask = run_race_sim(
            race_stats=race_stats,
            reliability_stats=reliability_stats,
            grid_positions=sim_grid,
            num_iterations=num_iterations,
            num_laps=num_laps,
            num_pitstops=2,
            pitstop_time_loss=track_pit_loss_base,
            season_trends=season_trends,
            weather_context=weather_context,
            track_info=track_info,
            year=year,
            team_mapping=team_mapping,
            pitstop_stats=pitstop_stats,
            power_index=power_index,
            tow_flags=tow_flags,
            pitlane_starters=pitlane_starters,
        )
    t3 = time.time()
    print(f"✓ Race done in {t3 - t2:.1f} s")

    # Print race table
    race_rows = sorted(
        finishing_probs.items(),
        key=lambda x: (x[1]['Win'], x[1]['Podium'], x[1]['Top10'], x[1]['Finish']),
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

    # ── Step 4: Save JSON ─────────────────────────────────────────────
    output_data = {
        'metadata': {
            'year': year,
            'race': race,
            'iterations': num_iterations,
            'laps': num_laps,
            'grid_source': grid_source,
        },
        'results': {},
    }

    for driver in race_drivers:
        driver_data = {
            'Grid_Position': sim_grid.get(driver, len(race_drivers)),
            'Race': {
                'Win':    round(finishing_probs[driver]['Win'],    4),
                'Podium': round(finishing_probs[driver]['Podium'], 4),
                'Top10':  round(finishing_probs[driver]['Top10'],  4),
                'Finish': round(finishing_probs[driver]['Finish'], 4),
                'DNF':    round(finishing_probs[driver]['DNF'],    4),
            },
        }
        # Include pole probability only if we ran a quali simulation
        if pole_probs:
            driver_data['Pole_Probability'] = round(pole_probs.get(driver, 0.0), 4)

        output_data['results'][driver] = driver_data

    # Include team mapping for dynamic visualizer colors
    output_data['team_mapping'] = team_mapping
    output_data['weather'] = weather_context

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

    choice = input("\nRun visualizer to generate charts? (y/n): ").strip().lower()
    if choice == 'y':
        import subprocess
        print("Generating visualizations...")
        subprocess.run([sys.executable, "visualizer.py", "--year", str(year), "--race", race])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 Race Predictor (Monte Carlo)")
    parser.add_argument("--year", type=int, default=2025, help="Season year (e.g. 2024, 2025, 2026)")
    parser.add_argument("--race", type=str, default="Monza", help="Race location or name (e.g. Monza, Bahrain)")
    parser.add_argument("--iterations", type=int, default=50_000, help="Number of simulation iterations")
    parser.add_argument("--penalties", type=str, default="", help="Manual grid penalties (e.g. 'NOR:10,HAD:30,VER:PL')")
    parser.add_argument(
        '--min_laps', 
        type=int, 
        default=7, 
        help='Minimum continuous lap stint length required to include in race pace analysis (default: 7)'
    )
    
    args = parser.parse_args()
    main(year=args.year, race=args.race, num_iterations=args.iterations, penalties_str=args.penalties, min_laps=args.min_laps)