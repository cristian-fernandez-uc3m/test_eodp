from config.ismConfig import ismConfig
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
from common.io.readMat import writeMat

class mtf:
    """
    Class MTF. Collects the analytical modelling of the different contributions
    for the system MTF
    """
    def __init__(self, logger, outdir):
        self.ismConfig = ismConfig()
        self.logger = logger
        self.outdir = outdir

    def system_mtf(self, nlines, ncolumns, D, lambd, focal, pix_size,
                   kLF, wLF, kHF, wHF, defocus, ksmear, kmotion, directory, band):
        """
        System MTF
        :param nlines: Lines of the TOA
        :param ncolumns: Columns of the TOA
        :param D: Telescope diameter [m]
        :param lambd: central wavelength of the band [m]
        :param focal: focal length [m]
        :param pix_size: pixel size in meters [m]
        :param kLF: Empirical coefficient for the aberrations MTF for low-frequency wavefront errors [-]
        :param wLF: RMS of low-frequency wavefront errors [m]
        :param kHF: Empirical coefficient for the aberrations MTF for high-frequency wavefront errors [-]
        :param wHF: RMS of high-frequency wavefront errors [m]
        :param defocus: Defocus coefficient (defocus/(f/N)). 0-2 low defocusing
        :param ksmear: Amplitude of low-frequency component for the motion smear MTF in ALT [pixels]
        :param kmotion: Amplitude of high-frequency component for the motion smear MTF in ALT and ACT
        :param directory: output directory
        :return: mtf
        """

        self.logger.info("Calculation of the System MTF")

        # Calculate the 2D relative frequencies
        self.logger.debug("Calculation of 2D relative frequencies")
        fn2D, fr2D, fnAct, fnAlt = self.freq2d(nlines, ncolumns, D, lambd, focal, pix_size)

        # Diffraction MTF
        self.logger.debug("Calculation of the diffraction MTF")
        Hdiff = self.mtfDiffract(fr2D)

        # Defocus
        Hdefoc = self.mtfDefocus(fr2D, defocus, focal, D)

        # WFE Aberrations
        Hwfe = self.mtfWfeAberrations(fr2D, lambd, kLF, wLF, kHF, wHF)

        # Detector
        Hdet  = self. mtfDetector(fn2D)

        # Smearing MTF
        Hsmear = self.mtfSmearing(fnAlt, ncolumns, ksmear)

        # Motion blur MTF
        Hmotion = self.mtfMotion(fn2D, kmotion)

        # Calculate the System MTF
        self.logger.debug("Calculation of the Sysmtem MTF by multiplying the different contributors")
        Hsys = Hdiff*Hwfe*Hdefoc*Hdet*Hsmear*Hmotion

        # Plot cuts ACT/ALT of the MTF
        self.plotMtf(Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion, Hsys, nlines, ncolumns, fnAct, fnAlt, directory, band)
        return Hsys

    def freq2d(self,nlines, ncolumns, D, lambd, focal, w):
        """
        Calculate the relative frequencies 2D (for the diffraction MTF)
        :param nlines: Lines of the TOA
        :param ncolumns: Columns of the TOA
        :param D: Telescope diameter [m]
        :param lambd: central wavelength of the band [m]
        :param focal: focal length [m]
        :param w: pixel size in meters [m]
        :return fn2D: normalised frequencies 2D (f/(1/w))
        :return fr2D: relative frequencies 2D (f/(1/fc))
        :return fnAct: 1D normalised frequencies 2D ACT (f/(1/w))
        :return fnAlt: 1D normalised frequencies 2D ALT (f/(1/w))
        """
        # Tiny epsilon for numerical array generation
        eps = 1e-6

        # Frequency step size (Along and Across)
        f_step_alt = 1 / (nlines * w)
        f_step_act = 1 / (ncolumns * w)

        # 1D spatial frequency vectors (f_spatial)
        f_alt = np.arange(-1 / (2 * w), 1 / (2 * w) - eps, f_step_alt)
        f_act = np.arange(-1 / (2 * w), 1 / (2 * w) - eps, f_step_act)

        # Normalized 1D frequencies (fn = f / Fs)
        fnAlt = f_alt / (1 / w)
        fnAct = f_act / (1 / w)

        # Cutoff frequency (fc)
        f_cutoff = D / (lambd * focal)

        # Relative 1D frequencies (fr = f / fc)
        frAlt = f_alt / f_cutoff
        frAct = f_act / f_cutoff

        # Create 2D normalized grid (mesh)
        [fn_across_2d, fn_along_2d] = np.meshgrid(fnAct, fnAlt, indexing='xy')

        # 2D Normalized frequency magnitude
        fn2D = np.sqrt(fn_across_2d ** 2 + fn_along_2d ** 2)

        # 2D Relative frequency
        fr2D = fn2D * (1 / w) / f_cutoff

        # Required side effects: writing to disk
        writeMat(self.outdir, "fn2D", fn2D)
        writeMat(self.outdir, "fr2D", fr2D)

        return fn2D, fr2D, fnAct, fnAlt

    def mtfDiffract(self,fr2D):
        """
        Optics Diffraction MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :return: diffraction MTF
        """
        # Clip fr2D to [-1, 1] for arccos domain
        fr_clip = np.clip(fr2D, -1.0, 1.0)

        # Identify valid region (fr2D <= 1.0)
        inside = fr2D <= 1.0

        # Init MTF array to zero
        Hdiff = np.zeros_like(fr2D)

        # Calculate MTF for valid region only
        valid_freqs = fr_clip[inside]

        # MTF calculation using the physics formula
        Hdiff[inside] = 2 / np.pi * (
                np.arccos(valid_freqs) - valid_freqs * np.sqrt(np.maximum(0.0, 1 - valid_freqs ** 2))
        )
        Hdiff[~inside] = 0.0
        return Hdiff


    def mtfDefocus(self, fr2D, defocus, focal, D):
        """
        Defocus MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :param defocus: Defocus coefficient (defocus/(f/N)). 0-2 low defocusing
        :param focal: focal length [m]
        :param D: Telescope diameter [m]
        :return: Defocus MTF
        """
        # Calculate argument for Bessel function
        x_arg = np.pi * defocus * fr2D * (1 - fr2D)

        # Init MTF to 1.0
        Hdefoc = np.ones_like(x_arg)

        # Find small values to avoid division by zero
        non_small = ~np.isclose(x_arg, 0.0)

        # Apply 2*J1(x)/x formula for non-zero points
        Hdefoc[non_small] = (2.0 * j1(x_arg[non_small])) / x_arg[non_small]

        # Convert NaNs/Infs to 0.0 for safety
        Hdefoc = np.nan_to_num(Hdefoc, nan=0.0, posinf=0.0, neginf=0.0)

        return Hdefoc

    def mtfWfeAberrations(self, fr2D, lambd, kLF, wLF, kHF, wHF):
        """
        Wavefront Error Aberrations MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :param lambd: central wavelength of the band [m]
        :param kLF: Empirical coefficient for the aberrations MTF for low-frequency wavefront errors [-]
        :param wLF: RMS of low-frequency wavefront errors [m]
        :param kHF: Empirical coefficient for the aberrations MTF for high-frequency wavefront errors [-]
        :param wHF: RMS of high-frequency wavefront errors [m]
        :return: WFE Aberrations MTF
        """
        # Low-frequency WFE term (kLF * (wLF/lambda)^2)
        lf_term = kLF * (wLF / lambd) ** 2

        # High-frequency WFE term (kHF * (wHF/lambda)^2)
        hf_term = kHF * (wHF / lambd) ** 2

        # Total WFE coefficient for the exponent
        wfe_coefficient = lf_term + hf_term

        # Calculate the exponent term
        exponent_val = -fr2D * (1 - fr2D) * wfe_coefficient

        # Apply the exponential MTF formula
        Hwfe = np.exp(exponent_val)

        return Hwfe

    def mtfDetector(self,fn2D):
        """
        Detector MTF
        :param fnD: 2D normalised frequencies (f/(1/w))), where w is the pixel width
        :return: detector MTF
        """
        # Calculate the argument for the sinc function: pi * fn
        sinc_arg = np.pi * fn2D

        # Initialize MTF to 1.0 (this handles the sinc(0)=1 case)
        Hdet = np.ones_like(fn2D)

        # Find non-zero points for division
        non_zero = ~np.isclose(sinc_arg, 0.0)

        # Calculate MTF = |sin(x) / x| for non-zero points (Sinc function definition)
        Hdet[non_zero] = np.abs(np.sin(sinc_arg[non_zero]) / sinc_arg[non_zero])

        return Hdet

    def mtfSmearing(self, fnAlt, ncolumns, ksmear):
        """
        Smearing MTF
        :param ncolumns: Size of the image ACT
        :param fnAlt: 1D normalised frequencies 2D ALT (f/(1/w))
        :param ksmear: Amplitude of low-frequency component for the motion smear MTF in ALT [pixels]
        :return: Smearing MTF
        """
        # Calculating 1D Smear MTF: sinc(k_smear * fn_alt)
        smear_mtf_1d = np.sinc(ksmear * fnAlt)

        # Repeating the 1D result across all columns and transpose to 2D
        Hsmear = np.transpose(np.tile(smear_mtf_1d, (ncolumns, 1)))

        return Hsmear

    def mtfMotion(self, fn2D, kmotion):
        """
        Motion blur MTF
        :param fnD: 2D normalised frequencies (f/(1/w))), where w is the pixel width
        :param kmotion: Amplitude of high-frequency component for the motion smear MTF in ALT and ACT
        :return: detector MTF
        """
        # Calculating MTF using the sinc function for motion blur
        Hmotion = np.sinc(kmotion*fn2D)

        return Hmotion

    def plotMtf(self, Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion, Hsys, nlines, ncolumns, fnAct, fnAlt, directory,
                band):
        """"
        Plotting the system MTF and all of its contributors
        :param Hdiff: Diffraction MTF
        :param Hdefoc: Defocusing MTF
        :param Hwfe: Wavefront electronics MTF
        :param Hdet: Detector MTF
        :param Hsmear: Smearing MTF
        :param Hmotion: Motion blur MTF
        :param Hsys: System MTF
        :param nlines: Number of lines in the TOA
        :param ncolumns: Number of columns in the TOA
        :param fnAct: normalised frequencies in the ACT direction (f/(1/w))
        :param fnAlt: normalised frequencies in the ALT direction (f/(1/w))
        :param directory: output directory
        :param band: band
        :return: N/A
        """
        # Get midpoint indices for slicing
        halfAct = int(fnAct.shape[0] / 2)
        halfAlt = int(fnAlt.shape[0] / 2)

        # --- Plot 1: Across-Track (ACT) ---

        # Start ACT plot figure
        fig1, ax1 = plt.subplots(figsize=(8, 6))

        # Plot all components
        ax1.plot(fnAct[halfAct:], Hdiff[halfAlt, halfAct:], 'b', label='Hdiff')
        ax1.plot(fnAct[halfAct:], Hdefoc[halfAlt, halfAct:], 'c', label='Hdefoc')
        ax1.plot(fnAct[halfAct:], Hwfe[halfAlt, halfAct:], 'g', label='Hwfe')
        ax1.plot(fnAct[halfAct:], Hdet[halfAlt, halfAct:], 'r', label='Hdet')
        ax1.plot(fnAct[halfAct:], Hsmear[halfAlt, halfAct:], 'm', label='Hsmear')
        ax1.plot(fnAct[halfAct:], Hmotion[halfAlt, halfAct:], 'y', label='Hmotion')

        # Plot total system MTF
        ax1.plot(fnAct[halfAct:], Hsys[halfAlt, halfAct:], 'k', label='Hsys', linewidth=2)

        # Nyquist line
        ax1.axvline(fnAct[-1], color='k', linestyle='--', label='Nyquist frequency')

        # Get MTF value at Nyquist (for title)
        mtf_nyquist_act = Hsys[0, halfAct]

        # Set labels, limits, and grid
        ax1.set_title(f'MTF Components (ACT) for band: {band}', fontsize=13, pad=10)
        ax1.set_xlabel('Spatial frequencies (f/(1/w)) [-]')
        ax1.set_ylabel('MTF Value')
        ax1.set_ylim(0, 1.05)
        ax1.legend(loc='lower left')
        ax1.grid(True)

        # Final plot title with Nyquist value
        plt.suptitle(f'MTF @ Nyquist (ACT) = {mtf_nyquist_act:.8f}', fontsize=11, y=0.93)

        # Save ACT plot
        fig1.tight_layout(rect=[0, 0, 1, 0.94])
        fig1.savefig(f'{directory}/graph_mtf_act_{band}.png', dpi=300)
        plt.close(fig1)

        # --- Plot 2: Along-Track (ALT) ---

        # Start ALT plot figure
        fig2, ax2 = plt.subplots(figsize=(8, 6))

        # Plot all components
        ax2.plot(fnAlt[halfAlt:], Hdiff[halfAlt:, halfAct], 'b', label='Hdiff')
        ax2.plot(fnAlt[halfAlt:], Hdefoc[halfAlt:, halfAct], 'c', label='Hdefoc')
        ax2.plot(fnAlt[halfAlt:], Hwfe[halfAlt:, halfAct], 'g', label='Hwfe')
        ax2.plot(fnAlt[halfAlt:], Hdet[halfAlt:, halfAct], 'r', label='Hdet')
        ax2.plot(fnAlt[halfAlt:], Hsmear[halfAlt:, halfAct], 'm', label='Hsmear')
        ax2.plot(fnAlt[halfAlt:], Hmotion[halfAlt:, halfAct], 'y', label='Hmotion')

        # Plot total system MTF
        ax2.plot(fnAlt[halfAlt:], Hsys[halfAlt:, halfAct], 'k', label='Hsys', linewidth=2)

        # Nyquist line
        ax2.axvline(fnAlt[-1], color='k', linestyle='--', label='Nyquist frequency')

        # Get MTF value at Nyquist (for title)
        mtf_nyquist_alt = Hsys[halfAlt, 0]

        # Set labels, limits, and grid
        ax2.set_title(f'MTF Components (ALT) for band: {band}', fontsize=13, pad=10)
        ax2.set_xlabel('Spatial frequencies (f/(1/w)) [-]')
        ax2.set_ylabel('MTF Value')
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc='lower left')
        ax2.grid(True)

        # Final plot title with Nyquist value
        plt.suptitle(f'MTF @ Nyquist (ALT) = {mtf_nyquist_alt:.8f}', fontsize=11, y=0.93)

        # Save ALT plot
        fig2.tight_layout(rect=[0, 0, 1, 0.94])
        fig2.savefig(f'{directory}/graph_mtf_alt_{band}.png', dpi=300)
        plt.close(fig2)