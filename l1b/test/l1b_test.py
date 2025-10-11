# EODP-TS-L1B-0001. EQUALIZATION AND RESTORATION TEST

import numpy as np
from common.io.writeToa import readToa
from config.globalConfig import globalConfig
from l1b.src.l1b import l1b
from common.src.auxFunc import getIndexBand
import matplotlib.pyplot as plt
import os

# --- Directories ---
auxdir = r'C:\\Users\\HP\\Desktop\\EODP\\auxiliary'
indir = r"C:\\Users\\HP\\Desktop\\EODP_TER_2021-20250911T164647Z-1-001\\EODP_TER_2021\\EODP-TS-L1B\\input"
ref_dir = r"C:\\Users\\HP\\Desktop\\EODP_TER_2021-20250911T164647Z-1-001\\EODP_TER_2021\\EODP-TS-L1B\\output"
out_dir = r"C:\\Users\\HP\\Desktop\\EODP_TER_2021-20250911T164647Z-1-001\\EODP_TER_2021\\EODP-TS-L1B\\My_outputs"

myL1b = l1b(auxdir, indir, out_dir)

# --- Parameters ---
tol = 1e-4
three_sigma = 1 - 0.997

cfg = globalConfig()
bands = cfg.bands

# --- Main Loop ---
for idx, band in enumerate(bands):
    # --- File names ---
    fname_eq = f"l1b_toa_eq_{band}.nc"
    fname_ism = f"ism_toa_isrf_{band}.nc"

    # --- Read TOAs ---
    toa_eq_out = readToa(out_dir, fname_eq)
    toa_eq_ref = readToa(ref_dir, fname_eq)
    toa_ism = readToa(indir, fname_ism)

    # --- Check 3-sigma differences ---
    diff_eq = toa_eq_out - toa_eq_ref
    points_threshold = toa_eq_ref.size * three_sigma
    cnt_eq = np.sum(np.abs(diff_eq) > tol)
    print(f"\nBAND {band}")
    print(f"Test with eq {fname_eq}: {'OK' if cnt_eq < points_threshold else 'NOK'}")

    # --- Restore signal to radiance (mW/m²/sr) ---
    toa_restored = myL1b.restoration(toa_eq_out, myL1b.l1bConfig.gain[getIndexBand(band)])

    # --- Plot 1: TOA restored (l1b_toa) ---
    mid_row = toa_restored.shape[0] // 2
    plt.figure()
    plt.plot(toa_restored[mid_row, :], color='red')
    plt.title(f'Band {band} - TOA Restored (l1b_toa)')
    plt.xlabel('Pixel ACT [-]')
    plt.ylabel('Radiance [mW/m²/sr]')
    plt.grid(True)
    plt_path1 = os.path.join(out_dir, f'l1b_toa_restored_{band}.png')
    plt.savefig(plt_path1)
    plt.close()

    # --- Plot 2: ISRF signal (ism_toa_isrf) ---
    mid_row_ism = toa_ism.shape[0] // 2
    plt.figure()
    plt.plot(toa_ism[mid_row_ism, :], color='blue')
    plt.title(f'Band {band} - TOA after ISRF (ism_toa_isrf)')
    plt.xlabel('Pixel ACT [-]')
    plt.ylabel('Radiance [DN or mW/m²/sr]')
    plt.grid(True)
    plt_path2 = os.path.join(out_dir, f'ism_toa_isrf_{band}.png')
    plt.savefig(plt_path2)
    plt.close()








