from common.io.writeToa import readToa
import numpy as np
from config.globalConfig import globalConfig

# --- Global Configuration ---
myglobal = globalConfig()
bands = myglobal.bands

# --- Directories ---
reference = r"C:\\Users\\HP\\Desktop\\EODP_TER_2021-20250911T164647Z-1-001\\EODP_TER_2021\\EODP-TS-ISM\\output"
outdir = r"C:\\Users\\HP\\Desktop\\EODP_TER_2021-20250911T164647Z-1-001\\EODP_TER_2021\\EODP-TS-ISM\\My_outputs"

# --- Parameters ---
tol = 1e-7
three_sigma = 1 - 0.997

# --- Optical Stage ---
print('Optical Stage Module TEST')
print('---------------------')

def run_test(names, names_isrf):
    # Executing the tests for the diferent TOA files
    for inn in range(len(bands)):
        toa_ism = readToa(outdir, names[inn])
        toa_ism_input = readToa(reference, names[inn])
        toa_isrf_ism = readToa(outdir, names_isrf[inn])
        toa_isrf_ism_input = readToa(reference, names_isrf[inn])

        # Computting the difference
        result = toa_ism - toa_ism_input
        result_isrf = toa_isrf_ism - toa_isrf_ism_input

        # How many are outside the tolerance
        counter = np.sum(np.abs(result) > tol)
        counter_isrf = np.sum(np.abs(result_isrf) > tol)

        points_threshold = toa_ism.size * three_sigma

        # Results
        print(f"Test {names[inn]} {'OK' if counter < points_threshold else 'NOK'}")
        print(f"Test {names_isrf[inn]} {'OK' if counter_isrf < points_threshold else 'NOK'}")
        print('---------------------')

namee = [f'ism_toa_optical_{b}.nc' for b in bands]
namee_isrf = [f'ism_toa_isrf_{b}.nc' for b in bands]

# Running the test
run_test(namee, namee_isrf)


# --- Test Detection and Video Conversion ---
print('\n')
print('Test Detection and Video Conversion Module')
print('---------------------')

name_e = [f'ism_toa_e_{b}.nc' for b in bands]
name_detection = [f'ism_toa_detection_{b}.nc' for b in bands]
name_ds = [f'ism_toa_ds_{b}.nc' for b in bands]
name_prnu = [f'ism_toa_prnu_{b}.nc' for b in bands]

if not (len(name_e) == len(name_detection) == len(name_ds) == len(name_prnu)):
    print('Error with the size of the toa in the Detection Module')


def run_detection_test():
    for inn in range(len(name_e)):
        # Reading output files
        toa_ism_e = readToa(outdir, name_e[inn])
        toa_ism_ds = readToa(outdir, name_ds[inn])
        toa_ism_detection = readToa(outdir, name_detection[inn])
        toa_ism_prnu = readToa(outdir, name_prnu[inn])

        # Reading reference files
        toa_ism_e_input = readToa(reference, name_e[inn])
        toa_ism_ds_input = readToa(reference, name_ds[inn])
        toa_ism_detection_input = readToa(reference, name_detection[inn])
        toa_ism_prnu_input = readToa(reference, name_prnu[inn])

        # Difference calculation
        result_e = toa_ism_e - toa_ism_e_input
        result_ds = toa_ism_ds - toa_ism_ds_input
        result_detection = toa_ism_detection - toa_ism_detection_input
        result_prnu = toa_ism_prnu - toa_ism_prnu_input

        # Counters
        counter_e = np.sum(np.abs(result_e) > tol)
        counter_ds = np.sum(np.abs(result_ds) > tol)
        counter_detection = np.sum(np.abs(result_detection) > tol)
        counter_prnu = np.sum(np.abs(result_prnu) > tol)

        points_threshold = toa_ism_e.size * three_sigma

        # Results
        print('---------------------')
        print(f"Test {name_e[inn]} {'OK' if counter_e < points_threshold else 'NOK'}")
        print(f"Test {name_ds[inn]} {'OK' if counter_ds < points_threshold else 'NOK'}")
        print(f"Test {name_detection[inn]} {'OK' if counter_detection < points_threshold else 'NOK'}")
        print(f"Test {name_prnu[inn]} {'OK' if counter_prnu < points_threshold else 'NOK'}")
        print('---------------------')


run_detection_test()

