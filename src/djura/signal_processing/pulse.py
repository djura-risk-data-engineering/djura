# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
"""Pulse classification algorithms."""
import numpy as np
import pywt
from scipy.interpolate import interp1d


def _cwt(signal, scales, wavelet):
    """Continuous wavelet transform with a discrete wavelet.

    :func:`pywt.cwt` refuses a discrete wavelet such as ``'db4'``
    (PyWavelets issue 365), although the transform itself is well defined -
    the integrated wavelet comes from ``wavefun`` either way. This is
    PyWavelets' own FFT algorithm with that check removed, restricted to a
    real one-dimensional signal.

    Parameters
    ----------
    signal : np.ndarray
        Input 1D signal.
    scales : np.ndarray
        Positive scales.
    wavelet : pywt.Wavelet
        The wavelet, discrete or continuous.

    Returns
    -------
    np.ndarray
        Coefficients of shape ``(len(scales), len(signal))``.
    """
    signal = np.asarray(signal, dtype=np.float64)
    scales = np.atleast_1d(scales)
    if np.any(scales <= 0):
        raise ValueError("scales must be positive")

    # Precision 10, as MATLAB's cwt and pywt.cwt use
    int_psi, x = pywt.integrate_wavelet(wavelet, precision=10)
    int_psi = np.asarray(int_psi, dtype=np.float64)
    step = x[1] - x[0]

    out = np.empty((scales.size, signal.size), dtype=np.float64)
    for i, scale in enumerate(scales):
        j = (np.arange(scale * (x[-1] - x[0]) + 1) / (scale * step))
        j = j.astype(int)
        if j[-1] >= int_psi.size:
            j = np.extract(j < int_psi.size, j)
        int_psi_scale = int_psi[j][::-1]

        size = int(2 ** np.ceil(np.log2(signal.size + int_psi_scale.size - 1)))
        conv = np.fft.irfft(np.fft.rfft(int_psi_scale, size)
                            * np.fft.rfft(signal, size), n=size)
        conv = conv[:signal.size + int_psi_scale.size - 1]

        coef = -np.sqrt(scale) * np.diff(conv)
        excess = (coef.size - signal.size) / 2.0
        if excess < 0:
            raise ValueError("scale %g is too small for this signal" % scale)
        if excess > 0:
            coef = coef[int(np.floor(excess)):-int(np.ceil(excess))]
        out[i] = coef
    return out


class PulseClassifier:
    """A class implementing the Shahi and Baker (2014)
    multi-component pulse classification algorithm.

    This class takes a pair of horizontal acceleration time-histories and
    applies the Baker (2007); Shahi and Baker (2011, 2013, 2014) pulse
    detection and classification procedure. It computes velocity
    time-histories, performs a wavelet-based search for strong velocity
    pulses, reconstructs  the pulse contribution, and classifies whether the
    record exhibits near-fault pulse-like behavior.

    References
    ----------
    - Shahi, S.K. and Baker, J.W. (2014). “An efficient algorithm to identify
      strong velocity pulses in multi-component ground motions.”
      Bulletin of the Seismological Society of America, 104(5), 2456-2466.
    - Shahi, S.K. (2013). “A probabilistic framework to include the effects of
      near-fault directivity in seismic hazard assessment.”
      Ph.D. Thesis, Stanford University, Stanford, CA.
    - Shahi, S.K., and Baker, J. W. (2011). “An empirically calibrated
      framework for including the effects of near-fault directivity in
      Probabilistic  Seismic Hazard Analysis.”  Bulletin of the Seismological
      Society of America, 101(2), 742-755.
    - Shahi, S.K., and Baker, J.W. (2011). “Regression models for predicting
      the probability of near-fault earthquake ground motion pulses, and their
      period.” 11th International Conference on Applications of Statistics and
      Probability in Civil Engineering, Zurich, Switzerland, 8.
    - Baker J.W. (2007). Quantitative classification of near-fault ground
      motions using wavelet analysis. Bulletin of the Seismological Society
      of America. 97 (5), 1486-1501.
    """
    signal1: np.ndarray
    signal2: np.ndarray
    dt: float
    wname = 'db4'

    def __init__(self, ag1, ag2, dt):
        """
        Initialize the Pulse object from two acceleration components.

        The two input acceleration traces are truncated to a common length
        (if necessary), numerically integrated once to obtain velocities,
        and converted from m/s to cm/s (multiplying by 981).

        Parameters
        ----------
        ag1 : np.ndarray
            First horizontal acceleration component (in g).
        ag2 : np.ndarray
            Second horizontal acceleration component (in g).
        dt : float
            Time step of the input recordings (in seconds).
        """
        # Compute pulse period
        if len(ag1) != len(ag2):
            min_len = min(len(ag1), len(ag2))
            ag1 = ag1[:min_len]
            ag2 = ag2[:min_len]
        self.signal1 = np.cumsum(ag1) * dt * 981  # velocity from H1 component
        self.signal2 = np.cumsum(ag2) * dt * 981  # velocity from H2 component
        self.dt = dt  # time step of the recording

    def classify(self):
        """
        Run the multi-component pulse classification algorithm.

        This method:

        1. Computes continuous wavelet transforms of the two velocity
           components over a set of scales.
        2. Searches iteratively for up to five dominant pulse candidates,
           rotating the components into the direction of maximum wavelet
           coefficient at each step.
        3. For each candidate direction, reconstructs the pulse time-history,
           computes residual motion, and evaluates several pulse indicators.
        4. Stores all intermediate and final results in instance attributes.

        After calling this method, the following attributes are available:

        - ``pulse_data`` : list of dict
            Per-candidate information (Tp, pulse scale, pulse and residual
            time-histories, energy ratios, classification flags, etc.).
        - ``rot_angles`` : list of float
            Rotation angles (in radians) corresponding to each candidate.
        - ``columns`` : list of int
            Column indices (time index) of the maximum coefficients.
        - ``row`` : list of int
            Row indices (scale index) of the maximum coefficients.
        """
        pulse_data = [{} for _ in range(5)]
        rot_angles = [0.0] * 5
        columns = [0] * 5
        rows = [0] * 5

        # Wavelet parameters
        tp_min, tp_max = 0.25, 15
        num_scales = 50

        # Scale calculations
        scale_min = int(np.floor(tp_min / 1.4 / self.dt))
        scale_step = int(np.ceil((tp_max / 1.4 / self.dt - scale_min)
                                 / num_scales))
        scale_max = scale_min + num_scales * scale_step
        scales = np.arange(scale_min, scale_max + 1, scale_step)

        # Perform CWT with custom implementation
        coefs1 = self._wavelet_trans(self.signal1, scales)
        coefs2 = self._wavelet_trans(self.signal2, scales)
        max_coeffs = coefs1**2 + coefs2**2

        for i in range(5):
            # Find maximum coefficient location
            col = np.argmax(np.max(max_coeffs, axis=0))
            row = np.argmax(max_coeffs[:, col])

            # Calculate orientation
            max_dir = np.arctan(coefs2[row, col] / coefs1[row, col])
            signal = (self.signal1 * np.cos(max_dir)
                      + self.signal2 * np.sin(max_dir))

            # Analyze pulse
            data = self._analyze_record(signal, col, row, scales)
            scale = scales[row]

            # Store results
            pulse_data[i] = data
            rot_angles[i] = max_dir
            columns[i] = col
            rows[i] = row

            # Block surrounding region
            block_min = col - 10 / 25 * scale
            block_max = col + 10 / 25 * scale
            idx = np.where(
                (np.arange(len(self.signal1)) > block_min)
                & (np.arange(len(self.signal1)) < block_max)
            )[0]
            max_coeffs[:, idx] = 0

        self.pulse_data = pulse_data
        self.rot_angles = rot_angles
        self.columns = columns
        self.row = rows

    def _wavelet_trans(self, signal, scales):
        """
        Compute continuous wavelet transform coefficients for a signal.

        Uses :func:`_cwt`, PyWavelets' own algorithm with the discrete
        wavelet check removed.

        Parameters
        ----------
        signal : np.ndarray
            Input 1D signal (e.g., velocity time-history).
        scales : np.ndarray
            Array of positive scales at which to compute the transform.

        Returns
        -------
        np.ndarray
            CWT coefficient matrix of shape
            ``(len(scales), len(signal))``.
        """
        return _cwt(signal, scales, pywt.Wavelet(self.wname))

    def _analyze_record(self, signal, col, row, scales):
        """
        Analyze a rotated velocity record and extract pulse information.

        This method refines the wavelet scale around the detected maximum,
        reconstructs the dominant pulse by iteratively extracting wavelet
        contributions, computes energy-based indicators, and classifies
        the record as pulse-like or not.

        Parameters
        ----------
        signal : np.ndarray
            Rotated velocity time-history in the candidate pulse direction.
        col : int
            Column index (time index) of the initial maximum wavelet
            coefficient.
        row : int
            Row index (scale index) of the initial maximum wavelet coefficient.
        scales : np.ndarray
            Original set of wavelet scales used in the coarse search.

        Returns
        -------
        dict
            Dictionary containing:
            - ``'dt'`` : float
            - ``'Tp'`` : float, identified pulse period
            - ``'pulse_scale'`` : float, dominant scale
            - ``'coefs'`` : np.ndarray, extracted wavelet coefficients
            - ``'PGV'`` : float, peak ground velocity of full signal
            - ``'PGV_resid'`` : float, peak ground velocity of residual
            - ``'late'`` : bool, whether the pulse arrives late
            - ``'pulse_indicator'`` : float, logistic classifier value
            - ``'is_pulse'`` : bool, final pulse classification
            - ``'signal'`` : np.ndarray, original velocity
            - ``'pulse_th'`` : np.ndarray, reconstructed pulse
            - ``'resid_th'`` : np.ndarray, residual velocity.
        """
        num_coefs = 10
        num_scales = len(scales)

        # Refine scales around detected row
        refined_scales = np.arange(
            scales[max(0, row - 1)], scales[min(num_scales - 1, row + 1)] + 1
        )

        # Custom CWT with refined scales
        cwt_coefs = self._wavelet_trans(signal, refined_scales)
        z = np.abs(cwt_coefs[:, col])
        new_row = np.argmax(z)
        pulse_scale = refined_scales[new_row]

        # Time array and wavelet function
        n_points = len(signal)
        time = np.linspace(0, (n_points - 1) * self.dt, n_points)
        wavelet = pywt.Wavelet(self.wname)
        phi, psi, xval = wavelet.wavefun(level=4)

        # Pulse reconstruction
        resid_th = signal.copy()
        pulse_th = np.zeros_like(signal)
        coefs = np.zeros(num_coefs)
        cols = np.zeros(num_coefs, dtype=int)

        for i in range(num_coefs):
            coef, pulse_scale, col_i, Tp = self._extract_wavelet(
                resid_th, pulse_scale, col if i == 0 else cols[0]
            )
            coefs[i] = coef
            cols[i] = col_i

            # Create basis function
            basis = xval * pulse_scale
            basis = basis + (col_i - np.median(basis))
            basis = basis * self.dt
            y_vals = psi * coef / np.sqrt(pulse_scale)

            # Interpolation setup
            delta = basis[1] - basis[0]
            num_pads = int(np.ceil((np.max(time) - np.max(basis)) / delta))

            left_side = np.arange(0, np.min(basis) - 0.00001, delta)
            # Adjust right_side to match the number of elements you’ll pad
            right_side = np.arange(
                np.max(basis) + delta,
                np.max(basis) + delta * (num_pads + 1),  # one extra step
                delta,
            )

            # This will now match the length of right_side
            right_zeros = np.zeros_like(right_side)

            # Concatenate with y_vals
            final_basis = np.concatenate([left_side, basis, right_side])
            final_yvals = np.concatenate(
                [np.zeros_like(left_side), y_vals, right_zeros]
            )
            interp_func = interp1d(
                final_basis, final_yvals, bounds_error=False, fill_value=0
            )
            pulse_segment = interp_func(time)
            pulse_th += np.nan_to_num(pulse_segment)
            resid_th = signal - pulse_th

        # Energy calculations
        signal_energy = np.cumsum(signal**2) / np.sum(signal**2) * 100
        pulse_energy = np.cumsum(pulse_th**2) / np.sum(pulse_th**2) * 100
        late = self._check_late_arrival(signal_energy, pulse_energy)

        # DWT calculations
        # Calculate energy from DWT coefficients
        # Approximation coefficients only, as MATLAB's single-output dwt
        approx, _ = pywt.dwt(signal, self.wname)
        approx_resid, _ = pywt.dwt(resid_th, self.wname)
        dwt_squared_orig = np.sum(approx ** 2)
        dwt_squared_resid = np.sum(approx_resid ** 2)

        # Get PGV from original signal
        pgv = np.max(np.abs(signal))

        # Pulse classification
        pgv_ratio, energy_ratio = self._calculate_ratios(
            signal, resid_th, dwt_squared_orig, dwt_squared_resid
        )
        pulse_indicator, is_pulse = self._classify_pulse(
            pgv_ratio, energy_ratio, late, pgv
        )

        return {
            "dt": self.dt,
            "Tp": Tp,
            "pulse_scale": pulse_scale,
            "coefs": coefs,
            "PGV": np.max(np.abs(signal)),
            "PGV_resid": np.max(np.abs(resid_th)),
            "late": late,
            "pulse_indicator": pulse_indicator,
            "is_pulse": is_pulse,
            "signal": signal,
            "pulse_th": pulse_th,
            "resid_th": resid_th,
        }

    def _extract_wavelet(self, signal, pulse_scale, pulse_row):
        """
        Extract the dominant wavelet coefficient at a given scale and time
        window.

        This helper performs a 1-scale CWT around the previously detected
        pulse scale and searches within a small time window around
        ``pulse_row`` for the maximum absolute coefficient.

        Parameters
        ----------
        signal : np.ndarray
            Input velocity time-history.
        pulse_scale : float
            Current candidate scale associated with the pulse.
        pulse_row : int
            Column index (time index) around which to search for the maximum.

        Returns
        -------
        coef : complex
            Wavelet coefficient at the dominant time location.
        pulse_scale : float
            (Same as input) currently used pulse scale.
        col : int
            Column index at which the maximum coefficient was found.
        Tp : float
            Approximate pulse period implied by the assumed center frequency.
        """
        row_range = 10 / 25
        scales = [pulse_scale]
        cwt_coefs = self._wavelet_trans(signal, scales)

        # Find maximum coefficient
        half = int(np.ceil(pulse_scale * row_range))
        search_start = max(0, pulse_row - half)
        search_end = min(len(signal) - 1, pulse_row + half)
        magnitude = np.abs(cwt_coefs[0])
        peak = magnitude[search_start:search_end + 1].max()
        col = int(np.flatnonzero(magnitude == peak)[0])
        coef = cwt_coefs[0, col]

        # Pulse period from the scale, as MATLAB's 1 / scal2frq
        Tp = 1 / (pywt.scale2frequency(self.wname, pulse_scale) / self.dt)

        return coef, pulse_scale, col, Tp

    @staticmethod
    def _check_late_arrival(signal_e, pulse_e):
        """Determine if pulse arrives late in the signal.

        The pulse is considered to arrive late if, by the time the pulse
        energy has reached 5% of its total, the cumulative signal energy
        has already reached at least 17% of its total.

        Parameters
        ----------
        signal_e : np.ndarray
            Cumulative percentage energy of the full signal (0–100).
        pulse_e : np.ndarray
            Cumulative percentage energy of the reconstructed pulse (0–100).

        Returns
        -------
        bool
            ``True`` if the pulse arrives late according to the criterion,
            ``False`` otherwise.
        """
        valid_indices = np.where(pulse_e <= 5)[0]
        if len(valid_indices) == 0:
            return False
        late_idx = valid_indices[-1]
        return signal_e[late_idx] >= 17

    @staticmethod
    def _calculate_ratios(signal, resid, dwt_squared_orig, dwt_squared_resid):
        """Calculate PGV and energy ratios.

        This function computes two key indicators used in the logistic
        pulse classification:

        - PGV ratio: residual PGV / full PGV.
        - DWT energy ratio: residual DWT energy / full DWT energy.

        Parameters
        ----------
        signal : np.ndarray
            Original velocity time-history.
        resid : np.ndarray
            Residual velocity time-history after pulse removal.
        dwt_squared_orig : float
            Total squared DWT coefficient energy of the original signal.
        dwt_squared_resid : float
            Total squared DWT coefficient energy of the residual signal.

        Returns
        -------
        pgv_ratio : float
            Ratio between residual and original PGV.
        energy_ratio : float
            Ratio between residual and original DWT energies.
        """
        pga = np.max(np.abs(signal))
        pgv_resid = np.max(np.abs(resid))
        pgv_ratio = pgv_resid / pga

        energy_ratio = dwt_squared_resid / dwt_squared_orig
        return pgv_ratio, energy_ratio

    @staticmethod
    def _classify_pulse(pgv_ratio, energy_ratio, late, pgv):
        """Classify pulse using logistic regression parameters.

        The classification follows Shahi and Baker's regression model:
        a combined pulse coefficient is computed from PGV and energy
        ratios, centered and scaled, and then used in a quadratic
        logistic function. A record is classified as pulse-like if
        the resulting pulse indicator is positive and the pulse does
        not arrive late.

        Parameters
        ----------
        pgv_ratio : float
            Ratio of residual to original PGV.
        energy_ratio : float
            Ratio of residual to original DWT energy.
        late : bool
            Flag indicating late arrival (from `_check_late_arrival`).
        pgv : float
            Peak ground velocity of the full signal.

        Returns
        -------
        pulse_indicator : float
            Continuous logistic indicator; positive values imply
            pulse-like behavior.
        is_pulse : bool
            Boolean classification flag; ``True`` if the record is
            classified as pulse-like.
        """
        pc = 0.63 * pgv_ratio + 0.777 * energy_ratio

        # Centering parameters (from MATLAB code)
        p = (pc - 1.208421) / 0.2462717
        v = (pgv - 11.58861) / 18.88015  # PGV comes from signal

        # Classification formula
        pulse_indicator = (
            -7.817
            - 0.5679 * p**2
            - 0.1516 * v**2
            - 3.0253 * p
            - 1.7396 * v
            - 2.7156 * p * v
        )
        is_pulse = (pulse_indicator > 0) and not late

        return pulse_indicator, is_pulse

    def get_tp(self):
        """
        Return the pulse period ``Tp`` of the first pulse-like record.

        The method scans the first five pulse candidates stored in
        ``self.pulse_data`` and returns the ``'Tp'`` value of the first
        candidate for which ``'is_pulse'`` is ``True``.

        Returns
        -------
        float
            Identified pulse period of the first pulse-like candidate, or
            ``-999`` if no pulse is detected among the first five candidates.
        """
        for data in self.pulse_data[:5]:
            if data['is_pulse']:
                return data['Tp']
        return -999

    def make_plot(self):
        """
        Plot the original, extracted pulse, and residual velocity records.

        This method finds the first record classified as pulse-like in
        ``self.pulse_data`` and creates a figure with three stacked subplots:

        1. Original rotated velocity time-history.
        2. Reconstructed pulse time-history.
        3. Residual velocity time-history.

        All panels share the same vertical limits to facilitate visual
        comparison of amplitudes.

        Notes
        -----
        Only the first pulse-like candidate is plotted. If no pulse-like
        record is found, nothing is plotted.
        """
        import matplotlib.pyplot as plt

        for data in self.pulse_data[:5]:
            if data.get('is_pulse', True):
                signal = np.array(data['signal'])
                pulse_th = np.array(data['pulse_th'])
                resid_th = np.array(data['resid_th'])
                dt = data['dt']

                np_points = len(signal)
                time = np.linspace(dt, dt * np_points, np_points)

                fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

                # First subplot: Original ground motion
                axs[0].plot(time, signal, '-k')
                axs[0].legend(['Original ground motion'])
                axs[0].set_ylim([-np.max(np.abs(axs[0].get_ylim())), np.max(
                    np.abs(axs[0].get_ylim()))])
                axs[0].tick_params(labelbottom=False)

                # Second subplot: Extracted pulse
                axs[1].plot(time, pulse_th, '-r')
                axs[1].legend(['Extracted pulse'])
                axs[1].set_ylabel('Velocity [cm/s]')
                axs[1].set_ylim(axs[0].get_ylim())
                axs[1].tick_params(labelbottom=False)

                # Third subplot: Residual ground motion
                axs[2].plot(time, resid_th, '-k')
                axs[2].legend(['Residual ground motion'])
                axs[2].set_xlabel('Time [s]')
                axs[2].set_ylim(axs[0].get_ylim())

                plt.tight_layout()
                plt.show()

                break
