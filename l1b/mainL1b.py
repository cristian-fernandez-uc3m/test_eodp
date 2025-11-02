
# MAIN FUNCTION TO CALL THE L1B MODULE

from l1b.src.l1b import l1b

# Directory - this is the common directory for the execution of the E2E, all modules
auxdir = r'C:\\Users\\HP\\Desktop\\EODP\\auxiliary'
#indir = r"C:\\Users\\HP\\Desktop\\EODP_TER_2021-20250911T164647Z-1-001\\EODP_TER_2021\\EODP-TS-L1B\\input"
#outdir = r"C:\\Users\\HP\\Desktop\\EODP_TER_2021-20250911T164647Z-1-001\\EODP_TER_2021\\EODP-TS-L1B\\My_outputs"

# Unncoment fot the test EODP-TS-E2E-0001
indir = r"C:\\Users\\HP\\Desktop\\EODP_TER_2021-20250911T164647Z-1-001\\EODP_TER_2021\\EODP-TS-E2E\\My_outputs_E2E_0001_ISM" # Input the outputs of the ISM
outdir = r"C:\\Users\\HP\\Desktop\\EODP_TER_2021-20250911T164647Z-1-001\\EODP_TER_2021\\EODP-TS-E2E\\My_outputs_E2E_0001_L1B" # Input the outputs of the ISM



# Initialise the ISM
myL1b = l1b(auxdir, indir, outdir)
myL1b.processModule()
