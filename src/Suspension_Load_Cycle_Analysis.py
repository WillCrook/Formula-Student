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

#amplitude bands for cycle counting
AMPLITUDE_BANDS = [0.1, 0.2, 0.3, 0.4, 0.5]  # in g

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
        return {'mean': np.nan, 'total_cycles':0, 'avg_amplitude':np.nan, 'rms': np.nan, 'band_counts': {band: 0 for band in AMPLITUDE_BANDS}}
    mean_acc = np.mean(acc)
    rms = np.sqrt(np.mean(acc**2))
    peak_trough_idx, max_idx, min_idx = find_peak_trough(acc)
    # number of cycles roughly equal number of peak-trough pairs
    # produce alternating peaks and troughs, then amplitude = abs(diff between consecutive extrema)
    if peak_trough_idx.size < 2:
        return {'mean': mean_acc, 'total_cycles':0, 'avg_amplitude':0.0, 'rms': rms, 'band_counts': {band: 0 for band in AMPLITUDE_BANDS}}
    ext_vals = acc[peak_trough_idx]
    amplitudes = np.abs(np.diff(ext_vals))  # peak-to-trough pairs
    total_cycles = amplitudes.size
    avg_amp = amplitudes.mean() if total_cycles > 0 else 0.0
    # Count cycles in each amplitude band
    band_counts = {}
    for band in AMPLITUDE_BANDS:
        band_counts[band] = np.sum(amplitudes >= band)
    return {'mean': mean_acc, 'total_cycles': total_cycles, 'avg_amplitude': avg_amp, 'rms': rms, 'band_counts': band_counts}

#OUTPUT

#vector magnitude for the total acceleration
df['Mag'] = np.sqrt(df['InlineAcc']**2 + df['LateralAcc']**2 + df['VerticalAcc']**2)

#display stats 
results = []
for axis in ['InlineAcc','LateralAcc','VerticalAcc','Mag']:
    stats = cycle_stats_from_signal(df[axis].dropna().values)
    result = {'Signal': axis, 
              'Mean_g': f"{stats['mean']:.1f}", 
              'Total_cycles': stats['total_cycles'], 
              'Avg_amp_g': f"{stats['avg_amplitude']:.1f}", 
              'RMS_g': f"{stats['rms']:.1f}"}
    for band in AMPLITUDE_BANDS:
        result[f'Cycles_>={band}g'] = stats['band_counts'][band]
    results.append(result)


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

#convert to dicts to dataframe for pandas functionallity 
df_results = pd.DataFrame(results)
print(df_results)

#optionally output to a csv file
OUPUT_CSV = True
if OUPUT_CSV:
    df_results.to_csv("results.csv", index=False)







