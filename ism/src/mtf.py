from math import pi
from config.ismConfig import ismConfig
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import j1
from numpy.matlib import repmat
from common.io.readMat import writeMat
from common.plot.plotMat2D import plotMat2D
from scipy.interpolate import interp2d
from numpy.fft import fftshift, ifft2
import os

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
        Hsys = Hdiff * Hdefoc * Hwfe * Hdet * Hsmear * Hmotion

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
        eps = 1e-6
        fc = D / (lambd * focal)  # cutoff frequency

        f_alt_axis = np.linspace(-0.5 / w, 0.5 / w - eps, nlines)
        f_act_axis = np.linspace(-0.5 / w, 0.5 / w - eps, ncolumns)

        fnAlt = f_alt_axis * w
        fnAct = f_act_axis * w

        alt_grid, act_grid = np.meshgrid(fnAlt, fnAct, indexing="ij")
        fn2D = np.sqrt(alt_grid ** 2 + act_grid ** 2)

        fr2D = (fn2D / w) / fc

        writeMat(self.outdir, "fn2D", fn2D)

        return fn2D, fr2D, fnAct, fnAlt

    def mtfDiffract(self,fr2D):
        """
        Optics Diffraction MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :return: diffraction MTF
        """
        Hdiff = np.zeros_like(fr2D, dtype=float)
        mask = fr2D < 1.0
        f_valid = fr2D[mask]
        # Here I compute diffraction MTF only where fr < 1
        Hdiff[mask] = (2 / np.pi) * (np.arccos(f_valid) - f_valid * np.sqrt(1 - f_valid ** 2))

        return np.atleast_2d(Hdiff)


    def mtfDefocus(self, fr2D, defocus, focal, D):
        """
        Defocus MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :param defocus: Defocus coefficient (defocus/(f/N)). 0-2 low defocusing
        :param focal: focal length [m]
        :param D: Telescope diameter [m]
        :return: Defocus MTF
        """
        arg = np.pi * defocus * fr2D * (1 - fr2D)
        J1_approx = arg / 2 - (arg ** 3) / 16 + (arg ** 5) / 384 - (arg ** 7) / 18432
        Hdefoc = 2 * J1_approx / arg

        return np.atleast_2d(Hdefoc)

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
        factorLF = kLF * (wLF / lambd) ** 2
        factorHF = kHF * (wHF / lambd) ** 2
        total_factor = factorLF + factorHF
        Hwfe = np.exp(-fr2D * (1 - fr2D) * total_factor)
        return np.atleast_2d(Hwfe)

    def mtfDetector(self,fn2D):
        """
        Detector MTF
        :param fnD: 2D normalised frequencies (f/(1/w))), where w is the pixel width
        :return: detector MTF
        """
        Hdet = np.where(fn2D != 0,np.abs(np.sin(np.pi * fn2D) / (np.pi * fn2D)),1.0)
        return np.atleast_2d(Hdet)

    def mtfSmearing(self, fnAlt, ncolumns, ksmear):
        """
        Smearing MTF
        :param ncolumns: Size of the image ACT
        :param fnAlt: 1D normalised frequencies 2D ALT (f/(1/w))
        :param ksmear: Amplitude of low-frequency component for the motion smear MTF in ALT [pixels]
        :return: Smearing MTF
        """
        sinc_alt = np.where(fnAlt != 0,np.sin(np.pi * fnAlt * ksmear) / (np.pi * fnAlt * ksmear),1.0)
        Hsmear = np.tile(sinc_alt[:, np.newaxis], (1, ncolumns))
        return np.atleast_2d(Hsmear)

    def mtfMotion(self, fn2D, kmotion):
        """
        Motion blur MTF
        :param fnD: 2D normalised frequencies (f/(1/w))), where w is the pixel width
        :param kmotion: Amplitude of high-frequency component for the motion smear MTF in ALT and ACT
        :return: detector MTF
        """
        # Avoid division by zero
        Hmotion = np.where(fn2D != 0,
                           np.sin(np.pi * fn2D * kmotion) / (np.pi * fn2D * kmotion),
                           1.0)  # sin(0)/0 = 1
        return np.atleast_2d(Hmotion)


    def plotMtf(self,Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion, Hsys, nlines, ncolumns, fnAct, fnAlt, directory, band):
        """
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

        # Central pixel indices
        center_act = fnAct.shape[0] // 2
        center_alt = fnAlt.shape[0] // 2

        # ===== Plot along ACT for central ALT =====
        fig_alt, ax_alt = plt.subplots()
        plt.suptitle(f'Alt = {center_alt} for {band}')
        x_act = fnAct[center_act:]
        contributors_alt = [Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion, Hsys]
        colors = ['r', 'g', 'b', 'k', 'y', 'c', 'm']
        labels = ['Hdiff', 'Hdefoc', 'Hwfe', 'Hdet', 'Hsmear', 'Hmotion', 'Hsys']

        for H, color, label in zip(contributors_alt, colors, labels):
            ax_alt.plot(x_act, H[center_alt, center_act:], color, label=label)

        ax_alt.set_xlabel('Spatial Frequencies [-]')
        ax_alt.set_ylabel('MTF')
        ax_alt.grid(True)
        ax_alt.legend(loc='lower left')
        fig_alt.savefig(f'{self.outdir}/graph_mtf_alt_{band}_graph.png')

        # ===== Plot along ALT for central ACT =====
        fig_act, ax_act = plt.subplots()
        plt.suptitle(f'Act = {center_act} for {band}')
        x_alt = fnAlt[center_alt:]
        for H, color, label in zip(contributors_alt, colors, labels):
            ax_act.plot(x_alt, H[center_alt:, center_act], color, label=label)

        ax_act.set_xlabel('Spatial Frequencies [-]')
        ax_act.set_ylabel('MTF')
        ax_act.grid(True)
        ax_act.legend(loc='lower left')
        fig_act.savefig(f'{self.outdir}/graph_mtf_act_{band}_graph.png')

        # ===== Save matrices =====
        writeMat(self.outdir, "Hsys", Hsys)
        writeMat(self.outdir, "Hdet", Hdet)

        # ===== Nyquist MTF report =====
        nyquist_file = r"C:\\Users\\HP\\Desktop\\EODP_TER_2021-20250911T164647Z-1-001\\EODP_TER_2021\\EODP-TS-ISM\\myoutput\\mtf_nyquist.txt"

        if band == 'VNIR-0':
            open(nyquist_file, 'w').close()  # truncate

        with open(nyquist_file, 'a') as f:
            f.write(f'{band}\n')
            f.write(f'mtf_nyquist Act={Hsys[0, center_act]}\n')
            f.write(f'mtf_nyquist Alt={Hsys[center_alt, 0]}\n')



