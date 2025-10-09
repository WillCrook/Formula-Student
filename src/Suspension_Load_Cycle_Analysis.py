import pandas as pd
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

#PARAMETERS

#Enter File below to analyse
file_name = "FSPT24_Endurance_IVAN_JESSIE_2024 Car_Generic testing_a_2797.csv" 

#visualise the acceleration 
VISUALISE = True

#amplitude bands for cycle counting
AMPLITUDE_BANDS = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0]  # in g

#toggle smoothing of IMU data (works by taking a rolling average)
APPLY_SMOOTHING = False

#optionally output results to a CSV file
OUPUT_CSV = True


#file handling
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data"

if not DATA_DIR.exists():
    print("Data folder not found. Creating one now at:", DATA_DIR)
    DATA_DIR.mkdir(exist_ok=True)
    print("Please add your data files to this folder and rerun the script.")
    exit()

data_file = DATA_DIR / file_name

# Ensure output directory exists
OUTPUT_DIR = BASE_DIR.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

#load IMU data from after the metadata
df = pd.read_csv(data_file, skiprows=14)


#change IMU values to floats
for c in ['InlineAcc', 'LateralAcc', 'VerticalAcc']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# clean noisy data by taking a rolling average
if APPLY_SMOOTHING:
    for c in ['InlineAcc', 'LateralAcc', 'VerticalAcc']:
        df[c] = df[c].rolling(5, center=True, min_periods=1).mean()


#find peaks and troughs of the acc
def find_peak_trough(acc):
    g = np.gradient(acc)
    # local max where g goes +ve to -ve
    is_max = (g[:-1] > 0) & (g[1:] < 0)
    # local min where g goes -ve to +ve
    is_min = (g[:-1] < 0) & (g[1:] > 0)
    max_idx = np.where(is_max)[0] + 1
    min_idx = np.where(is_min)[0] + 1
    return np.sort(np.concatenate([max_idx, min_idx])), max_idx, min_idx


#compute stats
def cycle_stats_from_signal(acc):
    acc = np.asarray(acc)
    if acc.size == 0: 
        return {
            'mean'       : np.nan,
            'total_cycles': 0, 
            'avg_amplitude': np.nan, 
            'rms'        : np.nan, 
            'pos_rms'    : 0.0,
            'neg_rms'    : 0.0,
            'band_counts': {band: 0 for band in AMPLITUDE_BANDS}
        }

    mean_acc = np.mean(acc)
    rms      = np.sqrt(np.mean(acc**2))
    positive_values = acc[acc > 0]
    negative_values = acc[acc < 0]

    if positive_values.size > 0:
        pos_rms = np.sqrt(np.mean(positive_values**2))
    else:
        pos_rms = 0.0

    if negative_values.size > 0:
        neg_rms = np.sqrt(np.mean(negative_values**2))
    else:
        neg_rms = 0.0
    
    peak_trough_idx, max_idx, min_idx = find_peak_trough(acc)

    # number of cycles roughly equal number of peak-trough pairs
    # produce alternating peaks and troughs, then amplitude = abs(diff between consecutive extrema)
    if peak_trough_idx.size < 2:
        return {
            'mean'        : mean_acc,
            'total_cycles': 0,
            'avg_amplitude': 0.0,
            'rms'         : rms,
            'pos_rms'     : pos_rms,
            'neg_rms'     : neg_rms,
            'band_counts' : {band: 0 for band in AMPLITUDE_BANDS}
        }

    ext_vals   = acc[peak_trough_idx]
    amplitudes = np.diff(ext_vals)  # keep sign of amplitude changes
    total_cycles = amplitudes.size
    avg_amp   = np.mean(np.abs(amplitudes)) if total_cycles > 0 else 0.0

    # Count cycles in each amplitude band
    band_counts = {}
    for band in AMPLITUDE_BANDS:
        band_counts[f'+>={band}g'] = np.sum(amplitudes >= band)
        band_counts[f'-<={-band}g'] = np.sum(amplitudes <= -band)

    return {
        'mean'        : mean_acc,
        'total_cycles': total_cycles,
        'avg_amplitude': avg_amp,
        'rms'         : rms,
        'pos_rms'     : pos_rms,
        'neg_rms'     : neg_rms,
        'band_counts' : band_counts
    }


#OUTPUT

#vector magnitude for the total acceleration
df['Mag'] = np.sqrt(df['InlineAcc']**2 + df['LateralAcc']**2 + df['VerticalAcc']**2)

#display stats 
results = []

for axis in ['InlineAcc', 'LateralAcc', 'VerticalAcc', 'Mag']:
    stats = cycle_stats_from_signal(df[axis].dropna().values)
    result = {
        'Signal'       : axis, 
        # 'Mean_g'     : f"{stats['mean']:.1f}", 
        'Total_cycles' : stats['total_cycles'], 
        # 'Avg_amp_g'  : f"{stats['avg_amplitude']:.1f}", 
        # 'RMS_g'      : f"{stats['rms']:.1f}",
        'RMS_Pos_g'    : f"{stats['pos_rms']:.1f}",
        'RMS_Neg_g'    : f"{stats['neg_rms']:.1f}"
    }
    for band in AMPLITUDE_BANDS:
        result[f'Cycles_>={band}g_pos'] = stats['band_counts'][f'+>={band}g']
        result[f'Cycles_<={-band}g_neg'] = stats['band_counts'][f'-<={-band}g']
    results.append(result)


#visualise acceleration
if VISUALISE:
    for axis in ['InlineAcc', 'LateralAcc', 'VerticalAcc']:
        stats = cycle_stats_from_signal(df[axis].dropna().values)
        plt.figure()
        plt.plot(df[axis], label=f'{axis} signal')
        plt.axhline(stats['pos_rms'], color='lightgreen', linestyle=':', label='+ RMS')
        plt.axhline(-stats['neg_rms'], color='red', linestyle=':', label='- RMS')
        plt.title(f'{axis}')
        plt.legend()
        plt.savefig(OUTPUT_DIR / f"{axis}.png", dpi=300)


#convert to dicts to dataframe for pandas functionallity 
df_results = pd.DataFrame(results)
print(df_results)

#optionally output to a csv file
if OUPUT_CSV:
    df_results.to_csv(OUTPUT_DIR / "results.csv", index=False)