from ism.src.initIsm import initIsm
from ism.src.mtf import mtf
from numpy.fft import fftshift, ifft2, fft2
import numpy as np
from common.io.writeToa import writeToa
from common.io.readIsrf import readIsrf
from scipy.interpolate import interp1d, interp2d
from common.plot.plotMat2D import plotMat2D
from common.plot.plotF import plotF
from common.src.auxFunc import getIndexBand




class opticalPhase(initIsm):

    def _init_(self, auxdir, indir, outdir):
        super()._init_(auxdir, indir, outdir)

    def compute(self, sgm_toa, sgm_wv, band):
        """
        The optical phase is in charge of simulating the radiance
        to irradiance conversion, the spatial filter (PSF)
        and the spectral filter (ISRF).
        :return: TOA image in irradiances [mW/m2/nm],
                    with spatial and spectral filter
        """
        self.logger.info("EODP-ALG-ISM-1000: Optical stage")

        # Calculation and application of the ISRF
        # -------------------------------------------------------------------------------
        self.logger.info("EODP-ALG-ISM-1010: Spectral modelling. ISRF")
        toa = self.spectralIntegration(sgm_toa, sgm_wv, band)

        self.logger.debug("TOA [0,0] " +str(toa[0,0]) + " [e-]")

        if self.ismConfig.save_after_isrf:
            saveas_str = self.globalConfig.ism_toa_isrf + band
            writeToa(self.outdir, saveas_str, toa)

        # Radiance to Irradiance conversion
        # -------------------------------------------------------------------------------
        self.logger.info("EODP-ALG-ISM-1020: Radiances to Irradiances")
        toa = self.rad2Irrad(toa,
                             self.ismConfig.D,
                             self.ismConfig.f,
                             self.ismConfig.Tr,band)

        self.logger.debug("TOA [0,0] " +str(toa[0,0]) + " [e-]")

        # Spatial filter
        # -------------------------------------------------------------------------------
        # Calculation and application of the system MTF
        self.logger.info("EODP-ALG-ISM-1030: Spatial modelling. PSF/MTF")
        myMtf = mtf(self.logger, self.outdir)
        Hsys = myMtf.system_mtf(toa.shape[0], toa.shape[1],
                                self.ismConfig.D, self.ismConfig.wv[getIndexBand(band)], self.ismConfig.f, self.ismConfig.pix_size,
                                self.ismConfig.kLF, self.ismConfig.wLF, self.ismConfig.kHF, self.ismConfig.wHF,
                                self.ismConfig.defocus, self.ismConfig.ksmear, self.ismConfig.kmotion,
                                self.outdir, band)

        # Apply system MTF
        toa = self.applySysMtf(toa, Hsys) # always calculated
        self.logger.debug("TOA [0,0] " +str(toa[0,0]) + " [e-]")



        # Write output TOA & plots
        # -------------------------------------------------------------------------------
        if self.ismConfig.save_optical_stage:
            saveas_str = self.globalConfig.ism_toa_optical + band

            writeToa(self.outdir, saveas_str, toa)

            title_str = 'TOA after the optical phase [mW/sr/m2]'
            xlabel_str='ACT'
            ylabel_str='ALT'
            plotMat2D(toa, title_str, xlabel_str, ylabel_str, self.outdir, saveas_str)

            idalt = int(toa.shape[0]/2)
            saveas_str = saveas_str + '_alt' + str(idalt)
            plotF([], toa[idalt,:], title_str, xlabel_str, ylabel_str, self.outdir, saveas_str)

        return toa

    def rad2Irrad(self, toa, D, f, Tr, band):
        """
        Radiance to Irradiance conversion
        :param toa: Input TOA image in radiances [mW/sr/m2]
        :param D: Pupil diameter [m]
        :param f: Focal length [m]
        :param Tr: Optical transmittance [-]
        :param band: Spectral band (not used here, but kept for compatibility)
        :return: TOA image in irradiances [mW/m2]
        """

        # Radiance to irradiance conversion factor
        rad2irra = Tr * np.pi * (D / f) * (D / f) / 4.0

        toa = toa * rad2irra

        return toa


    def applySysMtf(self, toa, Hsys):
        """
        Application of the system MTF to the TOA
        :param toa: Input TOA image in irradiances [mW/m2]
        :param Hsys: System MTF
        :return: TOA image in irradiances [mW/m2]
        """
        # Image to frequency domain
        toa_in_freq = fft2(toa)

        # Centering the MTF kernel
        mtf_centered = fftshift(Hsys)

        # Apply MTF (multiplication in frequency domain)
        filtered_freq_result = toa_in_freq * mtf_centered

        # Back to spatial domain (IFFT)
        toa_ft = ifft2(filtered_freq_result)

        # Removing imaginary part (just noise)
        toa_ft = np.real(toa_ft)

        return toa_ft

    def spectralIntegration(self, sgm_toa, sgm_wv, band):
        """
        Integration with the ISRF to retrieve one band
        :param sgm_toa: Spectrally oversampled TOA cube 3D in irradiances [mW/m2]
        :param sgm_wv: wavelengths of the input TOA cube
        :param band: band
        :return: TOA image 2D in radiances [mW/m2]
        """

        # Getting the ISRF and normalizing it
        isrf_raw, isrf_wvs = readIsrf(self.auxdir + self.ismConfig.isrffile, band)
        isrf_n = isrf_raw / np.sum(isrf_raw)

        # Init 2D output image
        final_toa_2d = np.zeros((sgm_toa.shape[0], sgm_toa.shape[1]))

        # Loop pixel by pixel to integrate
        for r in range(sgm_toa.shape[0]):
            for c in range(sgm_toa.shape[1]):
                # Interpolate the pixel's spectrum to the ISRF grid
                interp_func = interp1d(sgm_wv, sgm_toa[r, c, :],
                                       fill_value=(0, 0), bounds_error=False)

                # Resample signal
                resampled_signal = interp_func(isrf_wvs * 1000.0)

                # Integrate: multiply by normalized ISRF and sum
                final_toa_2d[r, c] = np.sum(resampled_signal * isrf_n)

        toa = final_toa_2d

        return toa