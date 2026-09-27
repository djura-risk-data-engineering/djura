# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
"""Linear response history analysis solvers for SDOF oscillators."""
from numba import njit, float32, float64, prange
from typing import Union, Literal, Tuple
import numpy as np
import time
from scipy.signal import lfilter


def lrha(
    ag: Union[list, np.ndarray],
    dt: float,
    periods: Union[float, list, np.ndarray],
    xi: float,
    m: float = 1.0,
    mode: Literal['numpy', 'numba', 'numba-parallel', 'auto'] = 'auto',
    formulation: Literal['incremental', 'direct'] = 'incremental',
    precision: Literal[32, 64] = 32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Linear Response History Analysis (LRHA) of a Single Degree-of-Freedom
    (SDOF) system, using Newmark Beta method.

    Parameters
    ----------
    ag : list or numpy.ndarray
        Ground acceleration array.
    dt : float
        Time step [sec].
    periods : float, list, or numpy.ndarray
        Oscillator periods [sec]. Can be scalar or array-like.
    xi : float
        Damping ratio (e.g., 0.05 for 5%).
    m : float, optional
        Mass of the SDOF system (default is 1).
    mode : {'numpy', 'numba', 'numba-parallel', 'auto'}, optional
        Method used for calculation:
        - 'numpy': Reference NumPy-only implementation (slowest).
        - 'numba': Numba JIT-compiled single-loop version (fast).
        - 'numba-parallel': Parallelized version for multiple periods.
        (fastest for large n2)
        - 'auto': Automatically selects best mode based on problem size.
    formulation : Literal['incremental', 'direct']
        - 'incremental' : Newmark-beta method with incremental formulation.
        - 'direct' : Newmark-beta method with direct formulation.
    precision : Literal[32, 64]
        Precision of floating numbers.

    Returns
    -------
    u : numpy.ndarray
        Relative displacement response history [n_time, n_periods]
    v : numpy.ndarray
        Relative velocity response history [n_time, n_periods]
    ac : numpy.ndarray
        Relative acceleration response history [n_time, n_periods]

    References
    ----------
    Clough, R. W., & Penzien, J. (1995). Dynamics of Structures (3rd ed.).
    Computers & Structures, Inc., University Ave., Berkeley.
    Chopra, A.K. 2012. Dynamics of Structures: Theory and
    Applications to Earthquake Engineering, fourth edition, Prentice Hall.
    N. M. Newmark, “A Method of Computation for Structural Dynamics,”
    ASCE Journal of the Engineering Mechanics Division, Vol. 85, 1959,
    pp. 67-94.
    Rajasekaran, S. (2009). Structural dynamics of earthquake engineering:
    Theory and application using Mathematica and Matlab. Woodhead Publishing.

    Notes
    -----
    - Linear Acceleration Method: Gamma = 1/2, Beta = 1/6
    - Average Acceleration Method: Gamma = 1/2, Beta = 1/4
    - Average acceleration method is unconditionally stable,
    whereas linear acceleration method is stable only if dt/Tn <= 0.551
    Linear acceleration method is preferable if stable due to its accuracy.
    """

    if precision == 32:
        ag = np.asarray(ag, dtype=np.float32)  # ground excitation
        dt = np.float32(dt)  # time step
        xi = np.float32(xi)  # damping ratio
        m = np.float32(m)  # mass
        p = -m * ag  # external force array
        if isinstance(periods, (int, float)):
            periods = np.array([periods], dtype=np.float32)
        else:
            periods = np.asarray(periods, dtype=np.float32).flatten()

        # --- Check formulation ---
        if formulation == 'incremental':
            formulation = np.float32(0)
        elif formulation == 'direct':
            formulation = np.float32(1)
        else:
            raise ValueError("formulation must be 'incremental' or 'direct'")

        if mode == 'auto':
            mode = 'numba-parallel' if len(periods) > 16 else 'numba'

        if mode == 'numpy':
            return _lrha_numpy_vectorized(
                p, dt, periods, xi, m, formulation, 32
            )
        elif mode == 'numba':
            return _lrha_njit_vectorized_f32(
                p, dt, periods, xi, m, formulation
            )
        elif mode == 'numba-parallel':
            return _lrha_njit_parallel_f32(p, dt, periods, xi, m, formulation)
        else:
            raise ValueError(f"Invalid mode '{mode}'. Choose from 'numpy', "
                             "'numba', 'numba-parallel', or 'auto'.")
    else:
        ag = np.asarray(ag, dtype=np.float64)  # ground excitation
        dt = np.float64(dt)  # time step
        xi = np.float64(xi)  # damping ratio
        m = np.float64(m)  # mass
        p = -m * ag  # external force array
        if isinstance(periods, (int, float)):
            periods = np.array([periods], dtype=np.float64)
        else:
            periods = np.asarray(periods, dtype=np.float64).flatten()

        if formulation == 'incremental':
            formulation = np.float64(0)
        elif formulation == 'direct':
            formulation = np.float64(1)
        else:
            raise ValueError("formulation must be 'incremental' or 'direct'")

        if mode == 'auto':
            mode = 'numba-parallel' if len(periods) > 16 else 'numba'

        if mode == 'numpy':
            return _lrha_numpy_vectorized(
                p, dt, periods, xi, m, formulation, 64
            )
        elif mode == 'numba':
            return _lrha_njit_vectorized_f64(
                p, dt, periods, xi, m, formulation
            )
        elif mode == 'numba-parallel':
            return _lrha_njit_parallel_f64(
                p, dt, periods, xi, m, formulation
            )
        else:
            raise ValueError(f"Invalid mode '{mode}'. Choose from 'numpy', "
                             "'numba', 'numba-parallel', or 'auto'.")


def _lrha_numpy_vectorized(
    p: np.ndarray,         # External force array [n_time]
    dt: float,             # Time step [sec]
    periods: np.ndarray,   # Oscillator periods [sec]
    xi: float,             # Damping ratio (e.g., 0.05)
    m: float,              # Mass of the SDOF system
    formulation: int,      # 0 = 'incremental', 1 = 'direct'
    precision: int         # floating precision 32 or 64
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Damped vibration response to an external time-dependent loading.
    The solution is based on Newmark-beta method (incremental or direct form),
    computed using a vectorized NumPy routine.

    Parameters
    ----------
    p : numpy.ndarray
        External force history, shape (n_time,)
    dt : float
        Time step size [sec]
    periods : numpy.ndarray
        Oscillator periods [sec], shape (n_periods,)
    xi : float
        Damping ratio (e.g. 0.05 for 5%)
    m : float
        Mass of the SDOF system
    formulation : int
        0 for incremental formulation (Rajasekaran Table 7.11),
        1 for direct formulation (Chopra Table 5.4.2)
    precision : Literal[32, 64]
        Precision of floating numbers.

    Returns
    -------
    u : numpy.ndarray
        Relative displacement [n_time, n_periods]
    u_dot : numpy.ndarray
        Relative velocity [n_time, n_periods]
    u_ddot : numpy.ndarray
        Relative acceleration [n_time, n_periods]
    """
    # Constants
    if precision == 32:
        ZERO = np.float32(0.0)
        ONE = np.float32(1.0)
        TWO = np.float32(2.0)
        PI = np.float32(np.pi)
        GAMMA = np.float32(0.5)
        BETA_LA = np.float32(1.0 / 6.0)
        BETA_AA = np.float32(1.0 / 4.0)
    else:
        ZERO = np.float64(0.0)
        ONE = np.float64(1.0)
        TWO = np.float64(2.0)
        PI = np.float64(np.pi)
        GAMMA = np.float64(0.5)
        BETA_LA = np.float64(1.0 / 6.0)
        BETA_AA = np.float64(1.0 / 4.0)
    # Stability limit for linear acceleration eq. 7.49 (Rajasekaran)
    LA_THRESHOLD = ONE / ((PI * TWO**0.5) * (GAMMA - 2 * BETA_LA)**0.5)

    # Get array sizes
    n1 = p.shape[0]
    n2 = periods.shape[0]

    # Initialize the response history arrays
    if precision == 32:
        u = np.zeros((n1, n2), dtype=np.float32)
        u_dot = np.zeros((n1, n2), dtype=np.float32)
        u_ddot = np.zeros((n1, n2), dtype=np.float32)
    else:
        u = np.zeros((n1, n2), dtype=np.float64)
        u_dot = np.zeros((n1, n2), dtype=np.float64)
        u_ddot = np.zeros((n1, n2), dtype=np.float64)

    # Calculate system properties which depend on period
    fn = ONE / periods  # natural frequency
    wn = TWO * PI * fn  # circular natural frequency
    k = m * wn**2  # stiffness
    c = TWO * m * wn * xi  # damping coefficient

    # Newmark Beta Method coefficients
    if precision == 32:
        gamma = np.full(n2, GAMMA, dtype=np.float32)
        # Default to Average Acceleration Method
        beta = np.full(n2, BETA_AA, dtype=np.float32)
    else:
        gamma = np.full(n2, GAMMA, dtype=np.float64)
        # Default to Average Acceleration Method
        beta = np.full(n2, BETA_AA, dtype=np.float64)
    # Use Linear Acceleration Method based on eq. 7.49 (Rajasekaran)
    beta[dt / periods <= LA_THRESHOLD] = BETA_LA

    # Set the initial conditions (Step A.1 - Rajasekaran Table 7.11)
    u[0, :] = ZERO
    u_dot[0, :] = ZERO
    u_ddot[0, :] = (p[0] - c * u_dot[0, :] - k * u[0, :]) / m

    # Compute the constants used in Newmark's integration
    if formulation == ZERO:  # Incremental Formulation
        # Step A.3 (Rajasekaran Table 7.11)
        k_eff = k + gamma / (beta * dt) * c + ONE / (beta * dt**2) * m
        # Step A.4 (Rajasekaran Table 7.11)
        a = m / (beta * dt) + gamma * c / beta
        b = m / (TWO * beta) + dt * (gamma / (TWO * beta) - 1) * c
    else:  # Direct Formulation (Chopra Table 5.4.2)
        # Step 1.3
        a1 = (ONE / (beta * dt**2)) * m + (gamma / (beta * dt)) * c
        a2 = (ONE / (beta * dt)) * m + (gamma / beta - ONE) * c
        a3 = (ONE / (TWO * beta) - ONE) * m + dt * (
            gamma / (TWO * beta) - ONE
        ) * c
        # Step 1.4
        k_eff = k + a1

    for i in range(n1 - 1):
        if formulation == ZERO:  # Incremental Formulation
            # Step B.1 (Rajasekaran Table 7.11)
            dp_eff = (p[i + 1] - p[i]) + a * u_dot[i, :] + b * u_ddot[i, :]
            # Step B.2 (Rajasekaran Table 7.11)
            du = dp_eff / k_eff
            # Step B.3 (Rajasekaran Table 7.11)
            dv = (
                gamma / (beta * dt) * du
                - gamma / beta * u_dot[i, :]
                + dt * (ONE - gamma / (TWO * beta)) * u_ddot[i, :]
            )
            # Step B.4 (Rajasekaran Table 7.11)
            da = (
                ONE / (beta * dt**2) * du
                - ONE / (beta * dt) * u_dot[i, :]
                - ONE / (2 * beta) * u_ddot[i, :]
            )
            # Step B.5 (Rajasekaran Table 7.11)
            u[i + 1, :] = u[i, :] + du
            u_dot[i + 1, :] = u_dot[i, :] + dv
            u_ddot[i + 1, :] = u_ddot[i, :] + da
        else:  # Direct Formulation
            # Step 2.1 (Chopra Table 5.4.2)
            p_eff = (
                p[i + 1] + a1 * u[i, :] + a2 * u_dot[i, :] + a3 * u_ddot[i, :]
            )
            # Step 2.2 (Chopra Table 5.4.2)
            u[i + 1, :] = p_eff / k_eff
            # Step 2.3 (Chopra Table 5.4.2)
            u_dot[i + 1, :] = (
                gamma / (beta * dt) * (u[i + 1, :] - u[i, :])
                + (ONE - gamma / beta) * u_dot[i, :]
                + dt * (ONE - gamma / (TWO * beta)) * u_ddot[i, :]
            )
            # Step 2.4 (Chopra Table 5.4.2)
            u_ddot[i + 1, :] = (
                (ONE / (beta * dt**2)) * (u[i + 1, :] - u[i, :])
                - (ONE / (beta * dt)) * u_dot[i, :]
                - (ONE / (TWO * beta) - ONE) * u_ddot[i, :]
            )

    return u, u_dot, u_ddot


@njit((float32[:], float32, float32[:], float32, float32, float32), cache=True)
def _lrha_njit_vectorized_f32(
    p: np.ndarray,         # External force array [n_time]
    dt: float,             # Time step [sec]
    periods: np.ndarray,   # Oscillator periods [sec]
    xi: float,             # Damping ratio (e.g., 0.05)
    m: float,              # Mass of the SDOF system
    formulation: int       # 0 = 'incremental', 1 = 'direct'
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Damped vibration response to an external time-dependent loading.
    The solution is based on Newmark-beta method (incremental or direct form),
    computed using a vectorized and Numba-accelerated routine.

    Uses 32-bit floating-point numbers only (moderate precision).

    Parameters
    ----------
    p : numpy.ndarray
        External force history, shape (n_time,)
    dt : float
        Time step size [sec]
    periods : numpy.ndarray
        Oscillator periods [sec], shape (n_periods,)
    xi : float
        Damping ratio (e.g. 0.05 for 5%)
    m : float
        Mass of the SDOF system
    formulation : int
        0 for incremental formulation (Rajasekaran Table 7.11),
        1 for direct formulation (Chopra Table 5.4.2)

    Returns
    -------
    u : numpy.ndarray
        Relative displacement [n_time, n_periods]
    u_dot : numpy.ndarray
        Relative velocity [n_time, n_periods]
    u_ddot : numpy.ndarray
        Relative acceleration [n_time, n_periods]
    """
    # Constants
    ZERO = np.float32(0.0)
    ONE = np.float32(1.0)
    TWO = np.float32(2.0)
    PI = np.float32(np.pi)
    GAMMA = np.float32(0.5)
    BETA_LA = np.float32(1.0 / 6.0)
    BETA_AA = np.float32(1.0 / 4.0)
    # Stability limit for linear acceleration eq. 7.49 (Rajasekaran)
    LA_THRESHOLD = ONE / ((PI * TWO**0.5) * (GAMMA - 2 * BETA_LA)**0.5)

    # Get array sizes
    n1 = p.shape[0]
    n2 = periods.shape[0]

    # Initialize the response history arrays
    u = np.zeros((n1, n2), dtype=np.float32)
    u_dot = np.zeros((n1, n2), dtype=np.float32)
    u_ddot = np.zeros((n1, n2), dtype=np.float32)

    # Calculate system properties which depend on period
    fn = ONE / periods  # natural frequency
    wn = TWO * PI * fn  # circular natural frequency
    k = m * wn**2  # stiffness
    c = TWO * m * wn * xi  # damping coefficient

    # Newmark Beta Method coefficients
    gamma = np.full(n2, GAMMA, dtype=np.float32)
    # Default to Average Acceleration Method
    beta = np.full(n2, BETA_AA, dtype=np.float32)
    for j in range(n2):
        # Based on eq. 7.49 (Rajasekaran)
        if dt / periods[j] <= LA_THRESHOLD:
            # Use Linear Acceleration Method
            beta[j] = BETA_LA

    # Set the initial conditions (Step A.1 - Rajasekaran Table 7.11)
    u[0, :] = ZERO
    u_dot[0, :] = ZERO
    u_ddot[0, :] = (p[0] - c * u_dot[0, :] - k * u[0, :]) / m

    # Compute the constants used in Newmark's integration
    if formulation == ZERO:  # Incremental Formulation
        # Step A.3 (Rajasekaran Table 7.11)
        k_eff = k + gamma / (beta * dt) * c + ONE / (beta * dt**2) * m
        # Step A.4 (Rajasekaran Table 7.11)
        a = m / (beta * dt) + gamma * c / beta
        b = m / (TWO * beta) + dt * (gamma / (TWO * beta) - ONE) * c
    else:  # Direct Formulation
        # Step 1.3 (Chopra Table 5.4.2)
        a1 = (ONE / (beta * dt**2)) * m + (gamma / (beta * dt)) * c
        a2 = (ONE / (beta * dt)) * m + (gamma / beta - ONE) * c
        a3 = (ONE / (TWO * beta) - ONE) * m + dt * (
            gamma / (TWO * beta) - ONE
        ) * c
        # Step 1.4 (Chopra Table 5.4.2)
        k_eff = k + a1

    for i in range(n1 - 1):
        if formulation == ZERO:  # Incremental Formulation
            # Step B.1 (Rajasekaran Table 7.11)
            dp_eff = (p[i + 1] - p[i]) + a * u_dot[i, :] + b * u_ddot[i, :]
            # Step B.2 (Rajasekaran Table 7.11)
            du = dp_eff / k_eff
            # Step B.3 (Rajasekaran Table 7.11)
            dv = (
                gamma / (beta * dt) * du
                - gamma / beta * u_dot[i, :]
                + dt * (ONE - gamma / (TWO * beta)) * u_ddot[i, :]
            )
            # Step B.4 (Rajasekaran Table 7.11)
            da = (
                ONE / (beta * dt**2) * du
                - ONE / (beta * dt) * u_dot[i, :]
                - ONE / (TWO * beta) * u_ddot[i, :]
            )
            # Step B.5 (Rajasekaran Table 7.11)
            u[i + 1, :] = u[i, :] + du
            u_dot[i + 1, :] = u_dot[i, :] + dv
            u_ddot[i + 1, :] = u_ddot[i, :] + da
        else:  # Direct Formulation
            # Step 2.1 (Chopra Table 5.4.2)
            p_eff = (
                p[i + 1] + a1 * u[i, :] + a2 * u_dot[i, :] + a3 * u_ddot[i, :]
            )
            # Step 2.2 (Chopra Table 5.4.2)
            u[i + 1, :] = p_eff / k_eff
            # Step 2.3 (Chopra Table 5.4.2)
            u_dot[i + 1, :] = (
                gamma / (beta * dt) * (u[i + 1, :] - u[i, :])
                + (ONE - gamma / beta) * u_dot[i, :]
                + dt * (ONE - gamma / (TWO * beta)) * u_ddot[i, :]
            )
            # Step 2.4 (Chopra Table 5.4.2)
            u_ddot[i + 1, :] = (
                (ONE / (beta * dt**2)) * (u[i + 1, :] - u[i, :])
                - (ONE / (beta * dt)) * u_dot[i, :]
                - (ONE / (TWO * beta) - ONE) * u_ddot[i, :]
            )

    return u, u_dot, u_ddot


@njit((float64[:], float64, float64[:], float64, float64, float64), cache=True)
def _lrha_njit_vectorized_f64(
    p: np.ndarray,         # External force array [n_time]
    dt: float,             # Time step [sec]
    periods: np.ndarray,   # Oscillator periods [sec]
    xi: float,             # Damping ratio (e.g., 0.05)
    m: float,              # Mass of the SDOF system
    formulation: int       # 0 = 'incremental', 1 = 'direct'
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Damped vibration response to an external time-dependent loading.
    The solution is based on Newmark-beta method (incremental or direct form),
    computed using a vectorized and Numba-accelerated routine.

    Uses 64-bit floating-point numbers only (high precision).

    Parameters
    ----------
    p : numpy.ndarray
        External force history, shape (n_time,)
    dt : float
        Time step size [sec]
    periods : numpy.ndarray
        Oscillator periods [sec], shape (n_periods,)
    xi : float
        Damping ratio (e.g. 0.05 for 5%)
    m : float
        Mass of the SDOF system
    formulation : int
        0 for incremental formulation (Rajasekaran Table 7.11),
        1 for direct formulation (Chopra Table 5.4.2)

    Returns
    -------
    u : numpy.ndarray
        Relative displacement [n_time, n_periods]
    u_dot : numpy.ndarray
        Relative velocity [n_time, n_periods]
    u_ddot : numpy.ndarray
        Relative acceleration [n_time, n_periods]
    """
    # Constants
    ZERO = np.float64(0.0)
    ONE = np.float64(1.0)
    TWO = np.float64(2.0)
    PI = np.float64(np.pi)
    GAMMA = np.float64(0.5)
    BETA_LA = np.float64(1.0 / 6.0)
    BETA_AA = np.float64(1.0 / 4.0)
    # Stability limit for linear acceleration eq. 7.49 (Rajasekaran)
    LA_THRESHOLD = ONE / ((PI * TWO**0.5) * (GAMMA - 2 * BETA_LA)**0.5)

    # Get array sizes
    n1 = p.shape[0]
    n2 = periods.shape[0]

    # Initialize the response history arrays
    u = np.zeros((n1, n2), dtype=np.float64)
    u_dot = np.zeros((n1, n2), dtype=np.float64)
    u_ddot = np.zeros((n1, n2), dtype=np.float64)

    # Calculate system properties which depend on period
    fn = ONE / periods  # natural frequency
    wn = TWO * PI * fn  # circular natural frequency
    k = m * wn**2  # stiffness
    c = TWO * m * wn * xi  # damping coefficient

    # Newmark Beta Method coefficients
    gamma = np.full(n2, GAMMA, dtype=np.float64)
    # Default to Average Acceleration Method
    beta = np.full(n2, BETA_AA, dtype=np.float64)
    for j in range(n2):
        # Based on eq. 7.49 (Rajasekaran)
        if dt / periods[j] <= LA_THRESHOLD:
            # Use Linear Acceleration Method
            beta[j] = BETA_LA

    # Set the initial conditions (Step A.1 - Rajasekaran Table 7.11)
    u[0, :] = ZERO
    u_dot[0, :] = ZERO
    u_ddot[0, :] = (p[0] - c * u_dot[0, :] - k * u[0, :]) / m

    # Compute the constants used in Newmark's integration
    if formulation == ZERO:  # Incremental Formulation
        # Step A.3 (Rajasekaran Table 7.11)
        k_eff = k + gamma / (beta * dt) * c + ONE / (beta * dt**2) * m
        # Step A.4 (Rajasekaran Table 7.11)
        a = m / (beta * dt) + gamma * c / beta
        b = m / (TWO * beta) + dt * (gamma / (TWO * beta) - ONE) * c
    else:  # Direct Formulation
        # Step 1.3 (Chopra Table 5.4.2)
        a1 = (ONE / (beta * dt**2)) * m + (gamma / (beta * dt)) * c
        a2 = (ONE / (beta * dt)) * m + (gamma / beta - ONE) * c
        a3 = (ONE / (TWO * beta) - ONE) * m + dt * (
            gamma / (TWO * beta) - ONE
        ) * c
        # Step 1.4 (Chopra Table 5.4.2)
        k_eff = k + a1

    for i in range(n1 - 1):
        if formulation == ZERO:  # Incremental Formulation
            # Step B.1 (Rajasekaran Table 7.11)
            dp_eff = (p[i + 1] - p[i]) + a * u_dot[i, :] + b * u_ddot[i, :]
            # Step B.2 (Rajasekaran Table 7.11)
            du = dp_eff / k_eff
            # Step B.3 (Rajasekaran Table 7.11)
            dv = (
                gamma / (beta * dt) * du
                - gamma / beta * u_dot[i, :]
                + dt * (ONE - gamma / (TWO * beta)) * u_ddot[i, :]
            )
            # Step B.4 (Rajasekaran Table 7.11)
            da = (
                ONE / (beta * dt**2) * du
                - ONE / (beta * dt) * u_dot[i, :]
                - ONE / (TWO * beta) * u_ddot[i, :]
            )
            # Step B.5 (Rajasekaran Table 7.11)
            u[i + 1, :] = u[i, :] + du
            u_dot[i + 1, :] = u_dot[i, :] + dv
            u_ddot[i + 1, :] = u_ddot[i, :] + da
        else:  # Direct Formulation
            # Step 2.1 (Chopra Table 5.4.2)
            p_eff = (
                p[i + 1] + a1 * u[i, :] + a2 * u_dot[i, :] + a3 * u_ddot[i, :]
            )
            # Step 2.2 (Chopra Table 5.4.2)
            u[i + 1, :] = p_eff / k_eff
            # Step 2.3 (Chopra Table 5.4.2)
            u_dot[i + 1, :] = (
                gamma / (beta * dt) * (u[i + 1, :] - u[i, :])
                + (ONE - gamma / beta) * u_dot[i, :]
                + dt * (ONE - gamma / (TWO * beta)) * u_ddot[i, :]
            )
            # Step 2.4 (Chopra Table 5.4.2)
            u_ddot[i + 1, :] = (
                (ONE / (beta * dt**2)) * (u[i + 1, :] - u[i, :])
                - (ONE / (beta * dt)) * u_dot[i, :]
                - (ONE / (TWO * beta) - ONE) * u_ddot[i, :]
            )

    return u, u_dot, u_ddot


@njit((float32[:], float32, float32[:], float32, float32, float32),
      parallel=True, cache=True)
def _lrha_njit_parallel_f32(
    p: np.ndarray,         # External force array [n_time]
    dt: float,             # Time step [sec]
    periods: np.ndarray,   # Oscillator periods [sec]
    xi: float,             # Damping ratio (e.g., 0.05)
    m: float,              # Mass of the SDOF system
    formulation: int       # 0 = 'incremental', 1 = 'direct'
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Damped vibration response to an external time-dependent loading.
    The solution is based on Newmark-beta method (incremental or direct form),
    computed in parallel over periods using Numba.

    Uses 32-bit floating-point numbers only (moderate precision).

    Parameters
    ----------
    p : numpy.ndarray
        External force history, shape (n_time,)
    dt : float
        Time step size [sec]
    periods : numpy.ndarray
        Oscillator periods [sec], shape (n_periods,)
    xi : float
        Damping ratio (e.g. 0.05 for 5%)
    m : float
        Mass of the SDOF system
    formulation : int
        0 for incremental formulation (Rajasekaran Table 7.11),
        1 for direct formulation (Chopra Table 5.4.2)

    Returns
    -------
    u : numpy.ndarray
        Relative displacement [n_time, n_periods]
    u_dot : numpy.ndarray
        Relative velocity [n_time, n_periods]
    u_ddot : numpy.ndarray
        Relative acceleration [n_time, n_periods]
    """
    # Constants
    ZERO = np.float32(0.0)
    ONE = np.float32(1.0)
    TWO = np.float32(2.0)
    PI = np.float32(np.pi)
    GAMMA = np.float32(0.5)
    BETA_LA = np.float32(1.0 / 6.0)
    BETA_AA = np.float32(1.0 / 4.0)
    # Stability limit for linear acceleration eq. 7.49 (Rajasekaran)
    LA_THRESHOLD = ONE / ((PI * TWO**0.5) * (GAMMA - 2 * BETA_LA)**0.5)

    # Get array sizes
    n1 = p.shape[0]
    n2 = periods.shape[0]

    # Initialize the response history arrays
    u = np.zeros((n1, n2), dtype=np.float32)
    u_dot = np.zeros((n1, n2), dtype=np.float32)
    u_ddot = np.zeros((n1, n2), dtype=np.float32)

    for j in prange(n2):
        # System properties for each oscillator
        # natural frequency
        fn = ONE / periods[j]
        # natural circular frequency
        wn = TWO * PI * fn
        # stiffness
        k = m * wn**2
        # damping
        c = TWO * m * wn * xi

        # Newmark-beta parameters - Eq. 7.49 (Rajasekaran)
        if dt / periods[j] <= LA_THRESHOLD:
            # Linear Acceleration Method
            beta = BETA_LA
        else:
            # Average Acceleration Method
            beta = BETA_AA
        gamma = GAMMA

        # Initial conditions (Step A.1 - Rajasekaran Table 7.11)
        u[0, j] = ZERO
        u_dot[0, j] = ZERO
        u_ddot[0, j] = (p[0] - c * u_dot[0, j] - k * u[0, j]) / m

        # Precompute constants
        if formulation == ZERO:  # Incremental Formulation
            # Step A.3 (Rajasekaran Table 7.11)
            k_eff = k + gamma / (beta * dt) * c + ONE / (beta * dt**2) * m
            # Step A.4 (Rajasekaran Table 7.11)
            a = m / (beta * dt) + gamma * c / beta
            b = m / (TWO * beta) + dt * (gamma / (TWO * beta) - ONE) * c
        else:  # Direct Formulation
            # Step 1.3 (Chopra Table 5.4.2)
            a1 = (ONE / (beta * dt**2)) * m + (gamma / (beta * dt)) * c
            a2 = (ONE / (beta * dt)) * m + (gamma / beta - ONE) * c
            a3 = (ONE / (TWO * beta) - ONE) * m + dt * (
                gamma / (TWO * beta) - ONE
            ) * c
            # Step 1.4 (Chopra Table 5.4.2)
            k_eff = k + a1

        for i in range(n1 - 1):
            if formulation == ZERO:  # Incremental Formulation
                # Step B.1 - Effective load increment (Rajasekaran Table 7.11)
                dp_eff = (p[i + 1] - p[i]) + a * u_dot[i, j] + b * u_ddot[i, j]
                # Step B.2 - Displacement increment (Rajasekaran Table 7.11)
                du = dp_eff / k_eff
                # Step B.3 - Velocity increment (Rajasekaran Table 7.11)
                dv = (
                    gamma / (beta * dt) * du
                    - gamma / beta * u_dot[i, j]
                    + dt * (ONE - gamma / (TWO * beta)) * u_ddot[i, j]
                )
                # Step B.4 - Acceleration increment (Rajasekaran Table 7.11)
                da = (
                    ONE / (beta * dt**2) * du
                    - ONE / (beta * dt) * u_dot[i, j]
                    - ONE / (TWO * beta) * u_ddot[i, j]
                )
                # Step B.5 - Update state (Rajasekaran Table 7.11)
                u[i + 1, j] = u[i, j] + du
                u_dot[i + 1, j] = u_dot[i, j] + dv
                u_ddot[i + 1, j] = u_ddot[i, j] + da

            else:  # Direct Formulation
                # Step 2.1 - Effective load (Chopra Table 5.4.2)
                p_eff = (
                    p[i + 1]
                    + a1 * u[i, j]
                    + a2 * u_dot[i, j]
                    + a3 * u_ddot[i, j]
                )
                # Step 2.2 - Displacement update (Chopra Table 5.4.2)
                u[i + 1, j] = p_eff / k_eff
                # Step 2.3 - Velocity update (Chopra Table 5.4.2)
                u_dot[i + 1, j] = (
                    gamma / (beta * dt) * (u[i + 1, j] - u[i, j])
                    + (ONE - gamma / beta) * u_dot[i, j]
                    + dt * (ONE - gamma / (TWO * beta)) * u_ddot[i, j]
                )
                # Step 2.4 - Acceleration update (Chopra Table 5.4.2)
                u_ddot[i + 1, j] = (
                    (ONE / (beta * dt**2)) * (u[i + 1, j] - u[i, j])
                    - (ONE / (beta * dt)) * u_dot[i, j]
                    - (ONE / (TWO * beta) - ONE) * u_ddot[i, j]
                )

    return u, u_dot, u_ddot


@njit((float64[:], float64, float64[:], float64, float64, float64),
      parallel=True, cache=True)
def _lrha_njit_parallel_f64(
    p: np.ndarray,         # External force array [n_time]
    dt: float,             # Time step [sec]
    periods: np.ndarray,   # Oscillator periods [sec]
    xi: float,             # Damping ratio (e.g., 0.05)
    m: float,              # Mass of the SDOF system
    formulation: int       # 0 = 'incremental', 1 = 'direct'
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Damped vibration response to an external time-dependent loading.
    The solution is based on Newmark-beta method (incremental or direct form),
    computed in parallel over periods using Numba.

    Uses 64-bit floating-point numbers only (high precision).

    Parameters
    ----------
    p : numpy.ndarray
        External force history, shape (n_time,)
    dt : float
        Time step size [sec]
    periods : numpy.ndarray
        Oscillator periods [sec], shape (n_periods,)
    xi : float
        Damping ratio (e.g. 0.05 for 5%)
    m : float
        Mass of the SDOF system
    formulation : int
        0 for incremental formulation (Rajasekaran Table 7.11),
        1 for direct formulation (Chopra Table 5.4.2)

    Returns
    -------
    u : numpy.ndarray
        Relative displacement [n_time, n_periods]
    u_dot : numpy.ndarray
        Relative velocity [n_time, n_periods]
    u_ddot : numpy.ndarray
        Relative acceleration [n_time, n_periods]
    """
    # Constants
    ZERO = np.float64(0.0)
    ONE = np.float64(1.0)
    TWO = np.float64(2.0)
    PI = np.float64(np.pi)
    GAMMA = np.float64(0.5)
    BETA_LA = np.float64(1.0 / 6.0)
    BETA_AA = np.float64(1.0 / 4.0)
    # Stability limit for linear acceleration eq. 7.49 (Rajasekaran)
    LA_THRESHOLD = ONE / ((PI * TWO**0.5) * (GAMMA - 2 * BETA_LA)**0.5)

    # Get array sizes
    n1 = p.shape[0]
    n2 = periods.shape[0]

    # Initialize the response history arrays
    u = np.zeros((n1, n2), dtype=np.float64)
    u_dot = np.zeros((n1, n2), dtype=np.float64)
    u_ddot = np.zeros((n1, n2), dtype=np.float64)

    for j in prange(n2):
        # System properties for each oscillator
        # natural frequency
        fn = ONE / periods[j]
        # natural circular frequency
        wn = TWO * PI * fn
        # stiffness
        k = m * wn**2
        # damping
        c = TWO * m * wn * xi

        # Newmark-beta parameters - Eq. 7.49 (Rajasekaran)
        if dt / periods[j] <= LA_THRESHOLD:
            # Linear Acceleration Method
            beta = BETA_LA
        else:
            # Average Acceleration Method
            beta = BETA_AA
        gamma = GAMMA

        # Initial conditions (Step A.1 - Rajasekaran Table 7.11)
        u[0, j] = ZERO
        u_dot[0, j] = ZERO
        u_ddot[0, j] = (p[0] - c * u_dot[0, j] - k * u[0, j]) / m

        # Precompute constants
        if formulation == ZERO:  # Incremental Formulation
            # Step A.3 (Rajasekaran Table 7.11)
            k_eff = k + gamma / (beta * dt) * c + ONE / (beta * dt**2) * m
            # Step A.4 (Rajasekaran Table 7.11)
            a = m / (beta * dt) + gamma * c / beta
            b = m / (TWO * beta) + dt * (gamma / (TWO * beta) - ONE) * c
        else:  # Direct Formulation
            # Step 1.3 (Chopra Table 5.4.2)
            a1 = (ONE / (beta * dt**2)) * m + (gamma / (beta * dt)) * c
            a2 = (ONE / (beta * dt)) * m + (gamma / beta - ONE) * c
            a3 = (ONE / (TWO * beta) - ONE) * m + dt * (
                gamma / (TWO * beta) - ONE
            ) * c
            # Step 1.4 (Chopra Table 5.4.2)
            k_eff = k + a1

        for i in range(n1 - 1):
            if formulation == ZERO:  # Incremental Formulation
                # Step B.1 - Effective load increment (Rajasekaran Table 7.11)
                dp_eff = (p[i + 1] - p[i]) + a * u_dot[i, j] + b * u_ddot[i, j]
                # Step B.2 - Displacement increment (Rajasekaran Table 7.11)
                du = dp_eff / k_eff
                # Step B.3 - Velocity increment (Rajasekaran Table 7.11)
                dv = (
                    gamma / (beta * dt) * du
                    - gamma / beta * u_dot[i, j]
                    + dt * (ONE - gamma / (TWO * beta)) * u_ddot[i, j]
                )
                # Step B.4 - Acceleration increment (Rajasekaran Table 7.11)
                da = (
                    ONE / (beta * dt**2) * du
                    - ONE / (beta * dt) * u_dot[i, j]
                    - ONE / (TWO * beta) * u_ddot[i, j]
                )
                # Step B.5 - Update state (Rajasekaran Table 7.11)
                u[i + 1, j] = u[i, j] + du
                u_dot[i + 1, j] = u_dot[i, j] + dv
                u_ddot[i + 1, j] = u_ddot[i, j] + da

            else:  # Direct Formulation
                # Step 2.1 - Effective load (Chopra Table 5.4.2)
                p_eff = (
                    p[i + 1]
                    + a1 * u[i, j]
                    + a2 * u_dot[i, j]
                    + a3 * u_ddot[i, j]
                )
                # Step 2.2 - Displacement update (Chopra Table 5.4.2)
                u[i + 1, j] = p_eff / k_eff
                # Step 2.3 - Velocity update (Chopra Table 5.4.2)
                u_dot[i + 1, j] = (
                    gamma / (beta * dt) * (u[i + 1, j] - u[i, j])
                    + (ONE - gamma / beta) * u_dot[i, j]
                    + dt * (ONE - gamma / (TWO * beta)) * u_ddot[i, j]
                )
                # Step 2.4 - Acceleration update (Chopra Table 5.4.2)
                u_ddot[i + 1, j] = (
                    (ONE / (beta * dt**2)) * (u[i + 1, j] - u[i, j])
                    - (ONE / (beta * dt)) * u_dot[i, j]
                    - (ONE / (TWO * beta) - ONE) * u_ddot[i, j]
                )

    return u, u_dot, u_ddot


def benchmark(precision):
    # Generate benchmark input data
    n1 = 5000  # number of time steps
    n2 = 100   # number of oscillator periods
    if precision == 32:
        ag = np.random.randn(n1).astype(np.float32)
        periods = np.linspace(0.1, 4.0, n2).astype(np.float32)
        dt = np.float32(0.01)
        xi = np.float32(0.05)
        m = np.float32(1.0)
    elif precision == 64:
        ag = np.random.randn(n1).astype(np.float64)
        periods = np.linspace(0.1, 4.0, n2).astype(np.float64)
        dt = np.float64(0.01)
        xi = np.float64(0.05)
        m = np.float64(1.0)

    # Warm-up
    _ = lrha(ag, dt, periods, xi, m, mode='numba', precision=precision)
    _ = lrha(
        ag, dt, periods, xi, m, mode='numba-parallel', precision=precision
    )

    # Benchmark and compare outputs
    results = {}
    timings = {}

    for mode in ['numpy', 'numba', 'numba-parallel']:
        t0 = time.time()
        u, u_dot, u_ddot = lrha(
            ag, dt, periods, xi, m, mode=mode, precision=precision
        )
        timings[mode] = time.time() - t0
        results[mode] = (u, u_dot, u_ddot)
        print(f"{mode}:     {timings[mode]:.4f} sec")

    # Compare results using np.allclose
    closeness = {}
    base = results['numpy']
    for mode in ['numba', 'numba-parallel']:
        close_u = np.allclose(base[0], results[mode][0], rtol=1e-5, atol=1e-7)
        close_v = np.allclose(base[1], results[mode][1], rtol=1e-5, atol=1e-7)
        close_a = np.allclose(base[2], results[mode][2], rtol=1e-5, atol=1e-7)
        closeness[mode] = (close_u, close_v, close_a)
        print(f"{mode} --> u: {close_u}, v: {close_v}, a: {close_a}")

    return timings, closeness


if __name__ == '__main__':
    benchmark(32)
    benchmark(64)


def nigam_jennings(
    acc: np.ndarray,
    dt: float,
    periods: Union[float, list, np.ndarray],
    xi: float = 0.05,
    precision: Literal[32, 64] = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Relative displacement response, exact for piecewise-linear excitation.

    Unlike :func:`lrha`, which integrates numerically and so carries a
    time-step error, this evaluates the closed-form solution of the same
    equation assuming the ground acceleration varies linearly between samples.
    Accuracy therefore does not degrade as ``dt`` approaches the oscillator
    period. The step is written as a linear map on ``(u, u_dot)``, which in the
    z-domain is a second order IIR filter, so each period costs one
    :func:`scipy.signal.lfilter` call.

    References
    ----------
    Nigam, N. C., and Jennings, P. C. (1969). Calculation of response spectra
    from strong-motion earthquake records. Bulletin of the Seismological
    Society of America, 59(2), 909-922.

    Parameters
    ----------
    acc : numpy.ndarray
        Ground acceleration.
    dt : float
        Time step [s].
    periods : float, list or numpy.ndarray
        Oscillator periods [s].
    xi : float, optional
        Damping ratio, 0.05 by default. Must be below 1.
    precision : {32, 64}, optional
        dtype of the result. The recursion always runs in float64, since at
        long periods the poles sit close to the unit circle.

    Returns
    -------
    u : numpy.ndarray
        Relative displacement, shape ``(len(acc), len(periods))``, in the
        acceleration's unit times s2.
    u_dot : numpy.ndarray
        Relative velocity.
    u_ddot : numpy.ndarray
        Relative acceleration, from the equation of motion
        ``u_ddot = -ag - 2 xi w u_dot - w^2 u``, which is exact given ``u``
        and ``u_dot``.
    """
    acc = np.asarray(acc, dtype=np.float64).ravel()
    periods = np.atleast_1d(np.asarray(periods, dtype=np.float64)).ravel()
    if not 0.0 <= xi < 1.0:
        raise ValueError("damping must be in [0, 1); got %g" % xi)
    if np.any(periods <= 0):
        raise ValueError("periods must be positive")

    root = np.sqrt(1.0 - xi ** 2)
    dtype = np.float32 if precision == 32 else np.float64
    u = np.empty((acc.size, periods.size), dtype=dtype)
    u_dot = np.empty_like(u)
    u_ddot = np.empty_like(u)

    for index, period in enumerate(periods):
        wn = 2.0 * np.pi / period
        wd = wn * root
        decay = np.exp(-xi * wn * dt)
        sin, cos = np.sin(wd * dt), np.cos(wd * dt)
        ratio = xi / root

        # State-transition matrix A
        a11 = decay * (ratio * sin + cos)
        a12 = decay * sin / wd
        a21 = -decay * wn * sin / root
        a22 = decay * (cos - ratio * sin)

        # Load matrix B, columns B1 (a[i]) and B2 (a[i+1])
        w2, w3 = wn ** 2, wn ** 3
        c1 = (2.0 * xi ** 2 - 1.0) / (w2 * dt)
        c2 = 2.0 * xi / (w3 * dt)
        b11 = decay * ((c1 + xi / wn) * sin / wd
                       + (c2 + 1.0 / w2) * cos) - c2
        b12 = -decay * (c1 * sin / wd + c2 * cos) - 1.0 / w2 + c2
        b21 = (decay * ((c1 + xi / wn) * (cos - ratio * sin)
                        - (c2 + 1.0 / w2) * (wd * sin + xi * wn * cos))
               + 1.0 / (w2 * dt))
        b22 = (-decay * (c1 * (cos - ratio * sin)
                         - c2 * (wd * sin + xi * wn * cos))
               - 1.0 / (w2 * dt))

        # u(z)/a(z) = [1 0] adj(zI - A) (B1 + z B2) / det(zI - A)
        denominator = [1.0, -(a11 + a22), a11 * a22 - a12 * a21]
        numerator = [b12,
                     b11 - a22 * b12 + a12 * b22,
                     a12 * b21 - a22 * b11]
        u[:, index] = lfilter(numerator, denominator, acc)
        # the same with [0 1] in place of [1 0]
        numerator_dot = [b22,
                         a21 * b12 + b21 - a11 * b22,
                         a21 * b11 - a11 * b21]
        u_dot[:, index] = lfilter(numerator_dot, denominator, acc)
        u_ddot[:, index] = -(acc + 2.0 * xi * wn * u_dot[:, index]
                             + wn ** 2 * u[:, index])

    return u, u_dot, u_ddot
