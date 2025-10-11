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
        # Sampling of the frequencies
        fstepAlt = 1 / nlines / w
        fstepAct = 1 / ncolumns / w

        # Frequencies vector
        eps = 1e-6  # epsilon
        # centered halves of detector width - eps
        fAlt = np.arange(-1 / (2 * w), 1 / (2 * w) - eps, fstepAlt)
        fAct = np.arange(-1 / (2 * w), 1 / (2 * w) - eps, fstepAct)

        fnAlt = fAlt / (1 / w)
        fnAct = fAct / (1 / w)

        # nyquist_f = 1/(2/w)
        # self.logger.info(f"Nyquist Frequency = {nyquist_f}")

        [fnAltxx, fnActxx] = np.meshgrid(fnAlt, fnAct, indexing='ij')
        fn2D = np.sqrt(fnAltxx * fnAltxx + fnActxx * fnActxx)

        f_co = D / (lambd * focal)

        fr2D = fn2D * (1 / w) / f_co

        return fn2D, fr2D, fnAct, fnAlt

    def mtfDiffract(self,fr2D):
        """
        Optics Diffraction MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :return: diffraction MTF
        """
        Hdiff = np.zeros((fr2D.shape[0], fr2D.shape[1]))
        for i in range(fr2D.shape[0]):
            for j in range(fr2D.shape[1]):
                Hdiff[i, j] = 2 / pi * (np.arccos(fr2D[i, j]) - fr2D[i, j] * np.sqrt((1 - np.square(fr2D[i, j]))))

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
        x = pi * defocus * fr2D * (1 - fr2D)
        Hdefoc = 2 * j1(x) / x

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
        Hwfe = np.exp(-fr2D * (1 - fr2D) * (kLF * (wLF * wLF / lambd / lambd) + kHF * (wHF * wHF / lambd / lambd)))
        return Hwfe

    def mtfDetector(self,fn2D):
        """
        Detector MTF
        :param fnD: 2D normalised frequencies (f/(1/w))), where w is the pixel width
        :return: detector MTF
        """
        Hdet = np.abs(np.sinc(fn2D))
        return Hdet

    def mtfSmearing(self, fnAlt, ncolumns, ksmear):
        """
        Smearing MTF
        :param ncolumns: Size of the image ACT
        :param fnAlt: 1D normalised frequencies 2D ALT (f/(1/w))
        :param ksmear: Amplitude of low-frequency component for the motion smear MTF in ALT [pixels]
        :return: Smearing MTF
        """
        Hsmear = np.zeros((fnAlt.shape[0], ncolumns))
        row_smear = np.sinc(fnAlt * ksmear)
        for i in range(ncolumns):
            Hsmear[:, i] = row_smear
        return Hsmear

    def mtfMotion(self, fn2D, kmotion):
        """
        Motion blur MTF
        :param fnD: 2D normalised frequencies (f/(1/w))), where w is the pixel width
        :param kmotion: Amplitude of high-frequency component for the motion smear MTF in ALT and ACT
        :return: detector MTF
        """
        Hmotion = np.sinc(fn2D * kmotion)
        return Hmotion

    def plotMtf(self, Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion, Hsys,
                nlines, ncolumns, fnAct, fnAlt, directory, band):
        """
        Plotting the system MTF and all of its contributors
        """

        halfAct = int(fnAct.shape[0] / 2)
        halfAlt = int(fnAlt.shape[0] / 2)

        # =====================================================
        # === PLOT ACT direction ==============================
        # =====================================================
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.plot(fnAct[halfAct:], Hdiff[halfAlt, halfAct:], 'b', label='Hdiff')
        ax1.plot(fnAct[halfAct:], Hdefoc[halfAlt, halfAct:], 'c', label='Hdefoc')
        ax1.plot(fnAct[halfAct:], Hwfe[halfAlt, halfAct:], 'g', label='Hwfe')
        ax1.plot(fnAct[halfAct:], Hdet[halfAlt, halfAct:], 'r', label='Hdet')
        ax1.plot(fnAct[halfAct:], Hsmear[halfAlt, halfAct:], 'm', label='Hsmear')
        ax1.plot(fnAct[halfAct:], Hmotion[halfAlt, halfAct:], 'y', label='Hmotion')
        ax1.plot(fnAct[halfAct:], Hsys[halfAlt, halfAct:], 'k', label='Hsys', linewidth=2)
        ax1.axvline(fnAct[-1], color='k', linestyle='--', label='Nyquist frequency')

        # Calcular valor de MTF en Nyquist (ACT)
        mtf_nyquist_act = Hsys[0, halfAct]

        # Títulos
        ax1.set_title(f'MTF Components (ACT) for band: {band}', fontsize=13, pad=10)
        ax1.set_xlabel('Spatial frequencies (f/(1/w)) [-]')
        ax1.set_ylabel('MTF Value')
        ax1.set_ylim(0, 1.05)
        ax1.legend(loc='lower left')
        ax1.grid(True)

        # Subtítulo con el valor Nyquist (8 decimales)
        plt.suptitle(f'MTF @ Nyquist (ACT) = {mtf_nyquist_act:.8f}', fontsize=11, y=0.93)

        # Guardar gráfico
        fig1.tight_layout(rect=[0, 0, 1, 0.94])
        fig1.savefig(f'{directory}/graph_mtf_act_{band}.png', dpi=300)
        plt.close(fig1)

        # =====================================================
        # === PLOT ALT direction ==============================
        # =====================================================
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.plot(fnAlt[halfAlt:], Hdiff[halfAlt:, halfAct], 'b', label='Hdiff')
        ax2.plot(fnAlt[halfAlt:], Hdefoc[halfAlt:, halfAct], 'c', label='Hdefoc')
        ax2.plot(fnAlt[halfAlt:], Hwfe[halfAlt:, halfAct], 'g', label='Hwfe')
        ax2.plot(fnAlt[halfAlt:], Hdet[halfAlt:, halfAct], 'r', label='Hdet')
        ax2.plot(fnAlt[halfAlt:], Hsmear[halfAlt:, halfAct], 'm', label='Hsmear')
        ax2.plot(fnAlt[halfAlt:], Hmotion[halfAlt:, halfAct], 'y', label='Hmotion')
        ax2.plot(fnAlt[halfAlt:], Hsys[halfAlt:, halfAct], 'k', label='Hsys', linewidth=2)
        ax2.axvline(fnAlt[-1], color='k', linestyle='--', label='Nyquist frequency')

        # Calcular valor de MTF en Nyquist (ALT)
        mtf_nyquist_alt = Hsys[halfAlt, 0]

        # Títulos
        ax2.set_title(f'MTF Components (ALT) for band: {band}', fontsize=13, pad=10)
        ax2.set_xlabel('Spatial frequencies (f/(1/w)) [-]')
        ax2.set_ylabel('MTF Value')
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc='lower left')
        ax2.grid(True)

        # Subtítulo con el valor Nyquist (8 decimales)
        plt.suptitle(f'MTF @ Nyquist (ALT) = {mtf_nyquist_alt:.8f}', fontsize=11, y=0.93)

        # Guardar gráfico
        fig2.tight_layout(rect=[0, 0, 1, 0.94])
        fig2.savefig(f'{directory}/graph_mtf_alt_{band}.png', dpi=300)
        plt.close(fig2)
