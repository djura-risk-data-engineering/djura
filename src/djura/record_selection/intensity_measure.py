# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
from typing import List, Union
import warnings

import numpy as np
from scipy.signal import butter, lfilter, find_peaks
from scipy.integrate import cumulative_trapezoid, trapezoid


_trapezoid = np.trapezoid if np.lib.NumpyVersion(np.__version__) >= "2.0.0" \
    else np.trapz


class IntensityMeasure:
    # Acceleration of gravity in [m/s2]
    g = 9.81

    def __init__(self) -> None:
        pass

    def _get_signal(
        self, acc: List[float], dt: float
    ) -> tuple[List[float], List[float], List[float]]:
        # Create time series
        time = dt * np.arange(0, len(acc), 1)
        # transform into [m/s2]
        acc = np.array(acc) * self.g

        # get velocity time series [m/s]
        vel = cumulative_trapezoid(acc, time, initial=0)
        # get displacement time series in [m]
        disp = cumulative_trapezoid(vel, time, initial=0)

        return disp, vel, acc

    def _fft_signal(
        self, acc: List[float], dt: float,
        period: Union[float, np.ndarray], damping: float
    ) -> tuple[List[float], List[float]]:
        if isinstance(period, int):
            period = float(period)

        if dt == 0 and isinstance(period, float) and period == 0.0:
            dt = 1e-20

        if dt == 0 and isinstance(period, float) and period != 0.0:
            raise ValueError("Time step must not be zero!")

        if isinstance(period, float):
            if period == 0.0:
                period = 1e-20
        else:
            period[period == 0.0] = 1e-20

        power = 1
        while np.power(2, power) < len(acc):
            power = power + 1

        n_points = np.power(2, power)
        fas = np.fft.fft(acc, n_points)
        d_freq = 1 / (dt * (n_points - 1))
        freq = d_freq * np.array(range(n_points))
        sym_idx = int(1 + n_points / 2)

        nat_freq = 1 / period

        if isinstance(period, float):
            h = np.ones(len(fas), 'complex')
        else:
            h = np.ones((len(fas), len(period)), 'complex')

        h[np.int_(np.arange(1, sym_idx))] = \
            np.array([nat_freq ** 2 * 1 / (
                (nat_freq ** 2 - i ** 2) + 2 * 1j
                * damping * i * nat_freq
            ) for i in freq[1: sym_idx]])

        h[np.int_(np.arange(len(h) - sym_idx + 2, len(h)))] = \
            np.flipud(np.conj(h[np.int_(np.arange(1, sym_idx - 1))]))

        return h, fas

    def get_sat(
            self, period: Union[float, np.array], acc: List[float], dt: float,
            damping: float) -> Union[float, np.array]:
        """Get the pseudo spectral acceleration (Sa(period, damping)) of a
        ground motion

        Parameters
        ----------
        period : float
            Period of interest, where the Sa(T) is being calculated in [s]
        acc : List[float]
            Acceleration time series in [g]
        dt : float
            Time step in [s], if period==0.0, may be set to any value
        damping : float
            Damping ratio, if period==0.0, may be set to any value

        Returns
        -------
        Union[float, np.array]
            Sa(period, damping) in [g], if T=0, Sa = PGA
        """
        h, fas = self._fft_signal(acc, dt, period, damping)

        if isinstance(period, float) or isinstance(period, int):
            sa = np.max(abs(np.real(np.fft.ifft(np.multiply(h, fas)))))
        else:
            sa = np.max(
                abs(np.real(np.fft.ifft(np.multiply(h, fas[:, np.newaxis]),
                                        axis=0))), axis=0)

        return sa

    def get_sdt(self, acc: List[float], dt: float, period: float,
                damping: float) -> float:
        """Get the pseudo spectral displacement (Sd(period, damping)) of a
        ground motion

        Parameters
        ----------
        acc : List[float]
            Acceleration time series in [g]
        dt : float
            Time step in [s], if period==0.0, may be set to any value
        period : float
            Period of interest, where the Sd(T) is being calculated in [s]
        damping : float
            Damping ratio, if period==0.0, may be set to any value

        Returns
        -------
        float
            Sd(period, damping) in [m], if T=0, Sd = PGD
        """
        h, fas = self._fft_signal(acc, dt, period, damping)

        # circular frequency
        omega = 2 * np.pi / period

        sa = max(abs(np.real(np.fft.ifft(np.multiply(h, fas)))))
        return sa * self.g / omega ** 2

    def get_svt(self, acc: List[float], dt: float, period: float,
                damping: float) -> float:
        """Get the pseudo spectral velocity (Sv(period, damping)) of a
        ground motion

        Parameters
        ----------
        acc : List[float]
            Acceleration time series in [g]
        dt : float
            Time step in [s], if period==0.0, may be set to any value
        period : float
            Period of interest, where the Sv(T) is being calculated in [s]
        damping : float
            Damping ratio, if period==0.0, may be set to any value

        Returns
        -------
        float
            Sv(period, damping) in [m/s], if T=0, Sv = PGV
        """
        h, fas = self._fft_signal(acc, dt, period, damping)

        # circular frequency
        omega = 2 * np.pi / period

        sa = max(abs(np.real(np.fft.ifft(np.multiply(h, fas)))))
        return sa * self.g / omega

    def get_pga(self, acc: List[float]) -> float:
        """Get the peak ground acceleration (PGA)

        Parameters
        ----------
        acc : List[float]
            Acceleration time series in [g]

        Returns
        -------
        float
            PGA in [g]
        """
        return self.get_sat(0.0, acc, 0.01, 0.05)

    def get_pgv(self, acc: List[float], dt: float) -> float:
        """Get peak ground velocity (PGV) in [m/s]

        Parameters
        ----------
        acc : List[float]
            Acceleration time series in [g]
        dt : float
            Time step in [s]

        Returns
        -------
        float
            PGV in [m/s]
        """
        _, vel, _ = self._get_signal(acc, dt)

        return np.max(np.abs(vel))

    def get_pgd(self, acc: List[float], dt: float) -> float:
        """Get peak ground displacement (PGD) in [m]

        Parameters
        ----------
        acc : List[float]
            Acceleration time series in [g]
        dt : float
            Time step in [s]

        Returns
        -------
        float
            PGD in [m]
        """
        disp, _, _ = self._get_signal(acc, dt)

        return np.max(np.abs(disp))

    def get_sa_avg(
            self, acc: List[float], dt: float, period: float,
            damping: float, bounds: List[float], size: int = 10) -> float:
        """Get average pseudo spectral acceleration (Sa_avg) with the
        selected bounds

        Parameters
        ----------
        acc : List[float]
            Acceleration time series in [g]
        dt : float
            Time step in [s]
        period : float
            Period of interest, where the Sa_avg is being calculated in [s]
        damping : float
            Damping ratio
        bounds : List[float]
            Bounds for the period, e.g. [0.2, 1.5], where lower period will be
            0.2*period, upper period bound will be 1.5*period
        size : int
            Number of uniformly spaced periods considered within the range of
            periods (bounds), by default 10

        Returns
        -------
        float
            Average pseudo-spectral acceleration (Sa_avg)
        """
        # TODO add support for batch mode for multiple periods
        if isinstance(period, float) and period == 0.0:
            raise ValueError("Conditioning period must not be zero!")

        period_range = period * np.linspace(bounds[0], bounds[1], size)

        # Vectorize
        spectral_accelerations = self.get_sat(
            period=period_range, acc=acc, dt=dt, damping=damping)

        # geometric mean
        sa_avg = spectral_accelerations.prod() \
            ** (1 / len(spectral_accelerations))

        return sa_avg

    def sat2(self, acc, dt, period, damping, osc_type="psa",
             max_freq_ratio=5.0):
        # TODO based on pyrotd package
        # psa matches get_sat
        osc_freq = 1 / period
        fourier_amp = np.fft.rfft(acc)
        freq = np.linspace(0, 1.0 / (2 * dt), num=fourier_amp.size)
        ang_freq = 2 * np.pi * freq
        osc_ang_freq = 2 * np.pi * osc_freq

        # Single-degree of freedom transfer function
        h = 1 / (
            ang_freq**2.0
            - osc_ang_freq**2
            - 2.0j * damping * osc_ang_freq * ang_freq
        )
        if osc_type == "sd":
            pass
        elif osc_type == "sv":
            h *= 1.0j * ang_freq
        elif osc_type == "sa":
            h *= 1 + (1.0j * ang_freq) ** 2
        elif osc_type == "psa":
            h *= -(osc_ang_freq**2)
        elif osc_type == "psv":
            h *= -osc_ang_freq
        else:
            raise RuntimeError

        # Adjust the maximum frequency considered. The maximum frequency is 5
        # times the oscillator frequency. This provides that at the oscillator
        # frequency there are at least tenth samples per wavelength.
        n = len(fourier_amp)
        m = max(n, int(max_freq_ratio * osc_freq / freq[1]))
        scale = float(m) / float(n)

        # Scale factor is applied to correct the amplitude of the motion
        # for the change in number of points
        resp = scale * np.fft.irfft(fourier_amp * h, 2 * (m - 1))
        resp = np.abs(resp).max()

        return resp

    def get_fiv3(self, acc: List[float], dt: float, tn: float,
                 alpha: float = 0.7, beta: float = 0.85):
        """Get the filtered incremental velocity IM for a ground motion

        References
        ----------
        Dávalos H, Miranda E. Filtered incremental velocity: A novel approach
        in intensity measures for seismic collapse estimation.
        Earthquake Engineering & Structural Dynamics 2019; 48(12): 1384-1405.
        DOI: 10.1002/eqe.3205.

        Parameters
        ----------
        acc : List[float]
            Acceleration time history in [g]
        dt : float
            Time step in [s]
        tn : float
            Period in [s]
        alpha : float
            Period factor, by default 0.7
        beta : float
            Cut-off frequency factor, by default 0.85

        Returns
        ----------
        float
            Intensity measure FIV3 (as per Eq. (3) of
            Davalos and Miranda (2019)) in [m/s]
        ndarray
            1D array - Filtered incremental velocity (as per Eq. (2) of
            Davalos and Miranda (2019)) in [m/s]
        ndarray
            1D array - Time series of FIV in [m/s]
        ndarray
            1D array - Filtered acceleration time history in [g]
        ndarray
            1D array - Three peak values used to compute FIV3
        ndarray
            1D array - Three trough values used to compute FIV3
        """
        # convert to m/s2
        acc = np.array(acc) * self.g

        # Time series of the signal
        time = dt * np.arange(0, len(acc), 1)

        # apply a 2nd order Butterworth low pass filter to the ground motion
        wn = beta / tn / (0.5 / dt)
        b, a = butter(2, wn, 'low')
        ugf = lfilter(b, a, acc, axis=0)

        # filtered incremental velocity (FIV)
        ugf_pc = np.zeros(
            (np.sum(time < time[-1] - alpha * tn),
             int(np.floor(alpha * tn / dt)) + 1))
        for i in range(int(np.floor(alpha * tn / dt)) + 1):
            ugf_pc[:, i] = ugf[np.where(time < time[-1] - alpha * tn)[0] + i]

        fiv = dt * trapezoid(ugf_pc, axis=1)
        t = time[time < time[-1] - alpha * tn]

        # Convert
        # Find the peaks and troughs of the FIV array
        pks_ind, _ = find_peaks(fiv)
        trs_ind, _ = find_peaks(-fiv)

        # Sort the values
        pks_srt = np.sort(fiv[pks_ind])
        trs_srt = np.sort(fiv[trs_ind])

        # Get the peaks
        pks = np.abs(pks_srt[-3:])
        trs = np.abs(trs_srt[:3])

        # Compute the FIV3
        fiv3 = np.max([np.sum(pks), np.sum(trs)])

        return fiv3, fiv, t, ugf, pks, trs

    def get_sa_rot_d_xx(
            self, acc1: List[float], acc2: List[float], dt: float,
            period: float, damping: float, percentiles: List[float] = None,
            num_theta: int = 180) -> List[float]:
        """Get the RotDxx IM of a ground motion signal pair

        References
        ----------
        Boore DM. Orientation-independent, nongeometric-mean measures of
        seismic intensity from two horizontal components of motion.
        Bulletin of the Seismological Society of America 2010; 100(4):
        1830-1835. DOI: 10.1785/0120090400.

        Parameters
        ----------
        acc1 : List[float]
            Acceleration time series in [g] in direction 1
        acc2 : List[float]
            Acceleration time series in [g] in direction 2
        dt : float
            Time step in [s]
        period : float
            Period of interest in [s]
        damping : float
            Damping ratio
        percentiles : List[float], optional
            Percentile to calculate, by default [16., 50., 84.]
        num_theta : int, optional
            Number of rotations to consider between 0 and 180°, by default 180

        Returns
        -------
        List[float]
            RotDxx values for given percentiles in [g]
        """
        if num_theta is None:
            num_theta = 180
        if percentiles is None:
            percentiles = [16., 50., 84.]

        if not isinstance(percentiles, list):
            if isinstance(percentiles, np.ndarray):
                percentiles = percentiles.tolist()
            else:
                percentiles = [percentiles]

        # verify length of acceleration time series
        if len(acc1) > len(acc2):
            warnings.warn(
                'Acceleration time series are not of the same size', Warning)
            acc2 = np.append(acc2, np.zeros(len(acc1) - len(acc2)))
        elif len(acc2) > len(acc1):
            warnings.warn(
                'Acceleration time series are not of the same size', Warning)
            acc1 = np.append(acc1, np.zeros(len(acc2) - len(acc1)))

        # rotation [deg]
        theta = np.linspace(0, 180, num_theta)

        # get response
        h, fas = self._fft_signal(acc1, dt, period, damping)
        resp1 = np.real(np.fft.ifft(np.multiply(h, fas)))
        h, fas = self._fft_signal(acc2, dt, period, damping)
        resp2 = np.real(np.fft.ifft(np.multiply(h, fas)))
        resp1 = resp1.reshape(len(resp1), 1)
        resp2 = resp2.reshape(len(resp2), 1)

        sa_vals = np.max(
            np.abs(np.multiply(resp1, np.cos(np.deg2rad(theta)))
                   + np.multiply(resp2, np.sin(np.deg2rad(theta)))), axis=0)

        rot_d_xx = np.percentile(sa_vals, percentiles, axis=0)

        return rot_d_xx

    def get_arias_intensity(self, acc: List[float], dt: float) -> float:
        """Get Arias Intensity (Ia) in [m/s]

        Parameters
        ----------
        acc : List[float]
            Acceleration time series in [g]
        dt : float
            Time step in [s]

        Returns
        -------
        float
            Ia in [m/s]
        """
        acc = np.array(acc) * self.g

        arias = np.pi / (2 * self.g) * \
            cumulative_trapezoid(acc ** 2, dx=dt, initial=0)

        return arias[-1]

    def get_significant_duration(
        self, acc: List[float], dt: float,
        start: float = 0.05, end: float = 0.95
    ) -> tuple[float, float, float]:
        """Get the significant duration using cumulative acceleration
        according to Trifunac and Brady (1975).

        Parameters
        ----------
        acc : List[float]
            Acceleration time series in [g]
        dt : float
            Time step in [s]
        start : float, optional
            Threshold for duration start, by default 0.05
        end : float, optional
            Threshold for duration end, by default 0.95

        Returns
        -------
        tuple[float, float, float]
            (duration, start time, end time) in [s]
        """
        # Transform into m/s2
        acc = np.array(acc) * self.g

        cum_acc = np.cumsum(acc ** 2)
        ind2 = np.where(
            (cum_acc > start * cum_acc[-1]) & (cum_acc < end * cum_acc[-1]))
        start_time = ind2[0][0] * dt
        end_time = ind2[0][-1] * dt

        return (end_time - start_time, start_time, end_time)

    def get_cav(self, acc: List[float], dt: float) -> float:
        """Get cumulative absolute velocity (CAV) in [m/s]

        Parameters
        ----------
        acc : List[float]
            Acceleration time series in [g]
        dt : float
            Time step in [s]

        Returns
        -------
        float
            CAV in [m/s]
        """
        abs_acc = np.abs(acc) * self.g
        time = dt * np.arange(0, len(acc), 1)

        return _trapezoid(abs_acc, time)

    def sa_to_sd(self, sa: float, period: float) -> float:
        """Convert to spectral displacement (Sd) from pseudo spectral
        acceleration (Sa) at a specific period

        Parameters
        ----------
        sa : float
            SA in [g]
        period : float
            Period, T in [s]

        Returns
        -------
        float
            SD in [m]
        """
        return sa * self.g * np.power(period / 2 / np.pi, 2)

    def sd_to_sa(self, sd: float, period: float) -> float:
        """Convert to pseudo spectral acceleration (Sa) from spectral
        displacement (Sd) at a specific period

        Parameters
        ----------
        sd : float
            SD in [m]
        period : float
            Period, T in [s]

        Returns
        -------
        float
            SA in [g]
        """
        return sd / self.g * np.power(2 * np.pi / period, 2)

    def get_ei(
            self, acc: List[float], dt: float, period: float, damping: float
    ) -> float:
        """Get the input energy (EI(period, damping)) of a ground motion for a
        particular period

        Method
        ------
        Piecewise linear exact method

        References
        ----------
        Aydinoglu M.N. and Y.M. Fahjan, 2003. A unified formulation of the
        piecewise exact method for inelastic seismic demand analysis including
        the P-delta effect

        Parameters
        ----------
        acc : List[float]
            Acceleration time series in [g]
        dt : float
            Time step in [s], if period==0.0, may be set to any value
        period : float
            Period of interest, where the EI(T) is being calculated in [s]
        damping : float
            Damping ratio, if period==0.0, may be set to any value

        Returns
        -------
        float
            EI(period, damping) in [m2/s2]
        """

        # convert to m/s2
        acc = np.array(acc) * self.g

        # natural and damped natural frequencies
        w = 2 * np.pi / period
        wd = w * np.sqrt(1 - damping ** 2)

        # Recurrence coefficients
        e = np.cos(wd * dt) * np.exp(-damping * w * dt)
        f = np.sin(wd * dt) * np.exp(-damping * w * dt)
        a11 = e + damping * w / wd * f
        a12 = f / wd
        a21 = -(w ** 2) * a12
        a22 = e - damping * w / wd * f
        b11 = (a11 - 1) / w ** 2
        b12 = (a12 - 2 * damping * w * b11 - dt) / (w ** 2 * dt)
        b21 = -a12
        b22 = b11 / dt

        # Recurrence relation
        acc_diff = np.diff(acc)
        vel = np.zeros(len(acc))
        disp = np.zeros(len(acc))
        accel = np.zeros(len(acc))
        pei = np.zeros(len(acc))
        for i in range(len(acc) - 1):
            disp[i + 1] = a11 * disp[i] + a12 * \
                vel[i] + b11 * acc[i] + b12 * acc_diff[i]
            vel[i + 1] = a21 * disp[i] + a22 * \
                vel[i] + b21 * acc[i] + b22 * acc_diff[i]
            accel[i + 1] = -2 * damping * w * \
                vel[i + 1] - (w ** 2) * disp[i + 1]
            pei[i + 1] = -acc[i + 1] * vel[i + 1] * dt

        # Compute EI(m2/s2)
        ei = max(np.cumsum(pei))

        return ei
