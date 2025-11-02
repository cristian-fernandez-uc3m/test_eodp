from ism.src.initIsm import initIsm
import numpy as np
from common.io.writeToa import writeToa
from common.plot.plotMat2D import plotMat2D
from common.plot.plotF import plotF

class detectionPhase(initIsm):

    def __init__(self, auxdir, indir, outdir):
        super().__init__(auxdir, indir, outdir)

        # Initialise the random see for the PRNU and DSNU
        np.random.seed(self.ismConfig.seed)


    def compute(self, toa, band):

        self.logger.info("EODP-ALG-ISM-2000: Detection stage")

        # Irradiance to photons conversion
        # -------------------------------------------------------------------------------
        self.logger.info("EODP-ALG-ISM-2010: Irradiances to Photons")
        area_pix = self.ismConfig.pix_size * self.ismConfig.pix_size # [m2]
        toa = self.irrad2Phot(toa, area_pix, self.ismConfig.t_int, self.ismConfig.wv[int(band[-1])])

        self.logger.debug("TOA [0,0] " +str(toa[0,0]) + " [ph]")

        # Photon to electrons conversion
        # -------------------------------------------------------------------------------
        self.logger.info("EODP-ALG-ISM-2030: Photons to Electrons")
        toa = self.phot2Electr(toa, self.ismConfig.QE)

        self.logger.debug("TOA [0,0] " +str(toa[0,0]) + " [e-]")

        if self.ismConfig.save_after_ph2e:
            saveas_str = self.globalConfig.ism_toa_e + band
            writeToa(self.outdir, saveas_str, toa)

        # PRNU
        # -------------------------------------------------------------------------------
        if self.ismConfig.apply_prnu:

            self.logger.info("EODP-ALG-ISM-2020: PRNU")
            toa = self.prnu(toa, self.ismConfig.kprnu)

            self.logger.debug("TOA [0,0] " +str(toa[0,0]) + " [e-]")

            if self.ismConfig.save_after_prnu:
                saveas_str = self.globalConfig.ism_toa_prnu + band
                writeToa(self.outdir, saveas_str, toa)

        # Dark-signal
        # -------------------------------------------------------------------------------
        if self.ismConfig.apply_dark_signal:

            self.logger.info("EODP-ALG-ISM-2020: Dark signal")
            toa = self.darkSignal(toa, self.ismConfig.kdsnu, self.ismConfig.T, self.ismConfig.Tref,
                                  self.ismConfig.ds_A_coeff, self.ismConfig.ds_B_coeff)

            self.logger.debug("TOA [0,0] " +str(toa[0,0]) + " [e-]")

            if self.ismConfig.save_after_ds:
                saveas_str = self.globalConfig.ism_toa_ds + band
                writeToa(self.outdir, saveas_str, toa)

        # Bad/dead pixels
        # -------------------------------------------------------------------------------
        if self.ismConfig.apply_bad_dead:

            self.logger.info("EODP-ALG-ISM-2050: Bad/dead pixels")
            toa = self.badDeadPixels(toa,
                               self.ismConfig.bad_pix,
                               self.ismConfig.dead_pix,
                               self.ismConfig.bad_pix_red,
                               self.ismConfig.dead_pix_red)


        # Write output TOA
        # -------------------------------------------------------------------------------
        if self.ismConfig.save_detection_stage:
            saveas_str = self.globalConfig.ism_toa_detection + band

            writeToa(self.outdir, saveas_str, toa)

            title_str = 'TOA after the detection phase [e-]'
            xlabel_str='ACT'
            ylabel_str='ALT'
            plotMat2D(toa, title_str, xlabel_str, ylabel_str, self.outdir, saveas_str)

            idalt = int(toa.shape[0]/2)
            saveas_str = saveas_str + '_alt' + str(idalt)
            plotF([], toa[idalt,:], title_str, xlabel_str, ylabel_str, self.outdir, saveas_str)

        return toa


    def irrad2Phot(self, toa, area_pix, tint, wv):
        """
        Conversion of the input Irradiances to Photons
        :param toa: input TOA in irradiances [mW/m2]
        :param area_pix: Pixel area [m2]
        :param tint: Integration time [s]
        :param wv: Central wavelength of the band [m]
        :return: Toa in photons
        """
        # Planck and Speed of light cte
        h_const = self.constants.h_planck
        c_speed = self.constants.speed_light

        # Calculate energy per photon (E=hc/lambda)
        photon_energy = (h_const * c_speed) / wv

        # Total incoming energy (J) over the pixel and time
        # toa is mW/m2, so I multiply by 1e-3 to get W/m2
        total_energy_in = (tint * area_pix * toa) * 1e-3

        # Total photons = Total Energy / Energy per photon
        toa_ph = total_energy_in / photon_energy
        return toa_ph

    def phot2Electr(self, toa, QE):
        """
        Conversion of photons to electrons
        :param toa: input TOA in photons [ph]
        :param QE: Quantum efficiency [e-/ph]
        :return: toa in electrons
        """
        # Convert photons to electrons using the QE
        toa = toa * QE

        # Get the Full Well Capacity (FWC) from the configuration
        capacity_limit = self.ismConfig.FWC

        # Clip the electron count ('toa') so it doesn't exceed the FWC
        # The new 'toa' value is the minimum of the calculated electrons and the limit
        toa = np.minimum(toa, capacity_limit)

        return toa

    def badDeadPixels(self, toa,bad_pix,dead_pix,bad_pix_red,dead_pix_red):
        """
        Bad and dead pixels simulation
        :param toa: input toa in [e-]
        :param bad_pix: Percentage of bad pixels in the CCD [%]
        :param dead_pix: Percentage of dead pixels in the CCD [%]
        :param bad_pix_red: Reduction in the quantum efficiency for the bad pixels [-, over 1]
        :param dead_pix_red: Reduction in the quantum efficiency for the dead pixels [-, over 1]
        :return: toa in e- including bad & dead pixels
        """
        # Applying the bad pixel QE reduction to this specific column
        reduction_factor = (1 - bad_pix_red)

        # Update the signal in column 5 with the reduced efficiency factor
        toa[:, 5] = toa[:, 5] * reduction_factor
        return toa

    def prnu(self, toa, kprnu):
        """
        Adding the PRNU effect
        :param toa: TOA pre-PRNU [e-]
        :param kprnu: multiplicative factor to the standard normal deviation for the PRNU
        :return: TOA after adding PRNU [e-]
        """
        # Generation of the noise factor for each column/band
        prnu_map = np.random.normal(0, 1, toa.shape[1])

        # Scale the map and shift by 1 to make it a multiplier
        prnu_multiplier = (prnu_map * kprnu) + 1

        # Apply the multiplicative PRNU noise to the signal
        toa = prnu_multiplier * toa
        return toa


    def darkSignal(self, toa, kdsnu, T, Tref, ds_A_coeff, ds_B_coeff):
        """
        Dark signal simulation
        :param toa: TOA in [e-]
        :param kdsnu: multiplicative factor to the standard normal deviation for the DSNU
        :param T: Temperature of the system
        :param Tref: Reference temperature of the system
        :param ds_A_coeff: Empirical parameter of the model 7.87 e-
        :param ds_B_coeff: Empirical parameter of the model 6040 K
        :return: TOA in [e-] with dark signal
        """
        # Generation of the DSNU factor (absolute value for non-uniformity)
        dsnu_factor = np.abs(np.random.normal(0, 1, toa.shape[1]))

        # Scaling of the DSNU
        dsnu_scaled = dsnu_factor * kdsnu

        # Calculating the mean Dark Signal (SD) using the temperature model
        temp_term = (T / Tref) ** 3
        exp_term = np.exp(-ds_B_coeff * ((1 / T) - (1 / Tref)))
        mean_ds = ds_A_coeff * temp_term * exp_term

        # Total DS = SD * (1 + DSNU)
        total_ds = mean_ds * (1 + dsnu_scaled)

        # Dark signal is additive noise
        toa = toa + total_ds
        return toa