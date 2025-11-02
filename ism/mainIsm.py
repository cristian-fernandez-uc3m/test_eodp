
# MAIN FUNCTION TO CALL THE ISM MODULE

from ism.src.ism import ism

# Directory - this is the common directory for the execution of the E2E, all modules
auxdir = r'C:\\Users\\HP\\Desktop\\EODP\\auxiliary'
#indir = r"C:\\Users\\HP\\Desktop\\EODP_TER_2021-20250911T164647Z-1-001\\EODP_TER_2021\\EODP-TS-ISM\\input\\gradient_alt100_act150"
#outdir = r"C:\\Users\\HP\\Desktop\\EODP_TER_2021-20250911T164647Z-1-001\\EODP_TER_2021\\EODP-TS-ISM\\My_outputs_ISM_0001_and_0002"

# Unncoment fot the test EODP-TS-E2E-0001
indir = r"C:\\Users\\HP\\Desktop\\EODP_TER_2021-20250911T164647Z-1-001\\EODP_TER_2021\\EODP-TS-E2E\\sgm_out"
outdir = r"C:\\Users\\HP\\Desktop\\EODP_TER_2021-20250911T164647Z-1-001\\EODP_TER_2021\\EODP-TS-E2E\\My_outputs_E2E_0001_ISM"


# Initialise the ISM
myIsm = ism(auxdir, indir, outdir)
myIsm.processModule()
