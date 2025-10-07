import pandas as pd
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent  
file_name = "FSPT24_Endurance_IVAN_JESSIE_2024 Car_Generic testing_a_2797.csv" #enter file to analyse here
data_file = BASE_DIR.parent / "data" / file_name

#load IMU data from after the metadata
df = pd.read_csv(data_file, skiprows=14)

#change IMU values to floats
for c in ['InlineAcc', 'LateralAcc', 'VerticalAcc']:
    df[c] = pd.to_numeric(df[c], errors='coerce') #

#toggle smoothing of IMU data
APPLY_SMOOTHING = False

# clean noisy data by taking a rolling average
if APPLY_SMOOTHING:
    for c in ['InlineAcc','LateralAcc','VerticalAcc']:
        df[c] = df[c].rolling(5, center=True, min_periods=1).mean()
else:
    for c in ['InlineAcc','LateralAcc','VerticalAcc']:
        df[c] = df[c]

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
        return {'mean': np.nan, 'num_cycles':0, 'avg_amplitude':np.nan, 'rms': np.nan}
    mean_acc = np.mean(acc)
    rms = np.sqrt(np.mean(acc**2))
    peak_trough_idx, max_idx, min_idx = find_peak_trough(acc)
    # number of cycles roughly equal number of peak-trough pairs
    # produce alternating peaks and troughs, then amplitude = abs(diff between consecutive extrema)
    if peak_trough_idx.size < 2:
        return {'mean': mean_acc, 'num_cycles':0, 'avg_amplitude':0.0, 'rms': rms}
    ext_vals = acc[peak_trough_idx]
    amplitudes = np.abs(np.diff(ext_vals))  # peak-to-trough pairs
    num_cycles = amplitudes.size
    avg_amp = amplitudes.mean() if num_cycles > 0 else 0.0
    return {'mean': mean_acc, 'num_cycles': num_cycles, 'avg_amplitude': avg_amp, 'rms': rms}

#OUTPUT

#calculate per-axis stats 
results = []
for axis in ['InlineAcc','LateralAcc','VerticalAcc']:
    stats = cycle_stats_from_signal(df[axis].dropna().values)
    results.append({'Signal': axis, 
                    'Mean_g': f"{stats['mean']:.1f}", 
                    'Num_cycles': stats['num_cycles'], 
                    'Avg_amp_g': f"{stats['avg_amplitude']:.1f}", 
                    'RMS_g': f"{stats['rms']:.1f}"})

#vector magnitude for the total acceleration
df['Mag'] = np.sqrt(df['InlineAcc']**2 + df['LateralAcc']**2 + df['VerticalAcc']**2)
stats = cycle_stats_from_signal(df['Mag'].dropna().values)
results.append({'Signal':'Magnitude', 
                'Mean_g': f"{stats['mean']:.1f}",
                'Num_cycles': stats['num_cycles'], 
                'Avg_amp_g': f"{stats['avg_amplitude']:.1f}", 
                'RMS_g': f"{stats['rms']:.1f}"})


#visualise acceleration

VISUALISE = False

if VISUALISE:

    for axis in ['InlineAcc', 'LateralAcc', 'VerticalAcc']:
        stats = cycle_stats_from_signal(df[axis].dropna().values)
        plt.figure()
        plt.plot(df[axis], label=f'{axis} signal')
        plt.axhline(stats['mean'], color='black', linestyle='--', label='Mean')
        plt.axhline(stats['mean'] + stats['avg_amplitude'], color='red', linestyle=':', label='+ Avg Amplitude')
        plt.axhline(stats['mean'] - stats['avg_amplitude'], color='blue', linestyle=':', label='- Avg Amplitude')
        plt.title(f'{axis} Acceleration with Mean ± Avg Amplitude')
        plt.legend()
        plt.show()



print(pd.DataFrame(results))