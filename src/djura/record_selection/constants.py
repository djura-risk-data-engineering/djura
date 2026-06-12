# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
SUPPORTED_IM_DESCRIPTORS = {
    "SA": "Spectral acceleration [g]",
    "SA_vert": "Vertical component spectral acceleration [g]",
    "Sa_avg2": "Average spectral acceleration, n=10, bounds=[0.2T, 2.0T] [g]",
    "Sa_avg3": "Average spectral acceleration, n=10, bounds=[0.2T, 3.0T] [g]",
    "Sa_avg": "Average spectral acceleration, Indirect",
    "PGA": "Peak ground acceleration [g]",
    "PGV": "Peak ground velocity [cm/s]",
    # "PGD": "Peak ground displacement [cm]",
    "Ds575": "Significant duration for time intervals 5-75% [seconds]",
    "Ds595": "Significant duration for time intervals 5-95% [seconds]",
    "FIV3": "Filtered incremental velocity [cm/s]",
    "IA": "Arias Intensity [m/s]",
}

SUPPORTED_IM_COMPONENTS = {
    "SA": ["RotD50", "RotD100", "geomean"],
    "Sa_avg2": ["RotD50", "RotD100", "geomean"],
    "Sa_avg3": ["RotD50", "RotD100", "geomean"],
    "Sa_avg": ["RotD50", "RotD100", "geomean"],
    "FIV3": ["geomean"],
}

SUPPORTED_IMS = frozenset(SUPPORTED_IM_DESCRIPTORS.keys())

CORRELATION_MODELS = {
    "SA-SA": ["aso2024", "baker_jayaram", "akkar", "eshm20", ],
    "Ds595-SA": ["aso2024", "bradley2011_ds595_sa"],
    "Ds575-SA": ["aso2024", "bradley2011_ds575_sa"],
    "PGA-SA": ["bradley2011_pga"],
    "PGV-PGA": ["bradley2012_pgv"],
    "PGV-SA": ["bradley2012_pgv"],
    "Ds595-PGA": ["bradley2011_ds595_sa"],
    "Ds575-PGA": ["bradley2011_ds575_sa"],
    "Ds575-Ds595": ["bradley2011_ds"],
    "Ds575-PGV": ["bradley2011_ds575_pgv"],
    "Ds595-PGV": ["bradley2011_ds595_pgv"],
    "Sa_avg3-Sa_avg3": ["aso2024", "dm18"],
    "Sa_avg-Sa_avg": ["aso2024"],
    "SA-Sa_avg2": ["aso2024"],
    "Sa_avg2-PGA": ["aso2024"],
    "Sa_avg3-PGA": ["aso2024"],
    "Sa_avg2-PGV": ["aso2024"],
    "Sa_avg3-PGV": ["aso2024"],
    "Sa_avg2-Ds575": ["aso2024"],
    "Sa_avg2-Ds595": ["aso2024"],
    "Sa_avg2-FIV3": ["aso2024"],
    "Sa_avg2-Sa_avg3": ["aso2024"],
    "Sa_avg2-Sa_avg2": ["aso2024"],
    "SA-Sa_avg3": ["aso2024"],
    "SA-FIV3": ["aso2024"],
    "FIV3-FIV3": ["aso2024"],
    "FIV3-PGA": ["aso2024"],
    "FIV3-PGV": ["aso2024"],
    "FIV3-Ds595": ["aso2024"],
    "FIV3-Ds575": ["aso2024"],
    "FIV3-Sa_avg3": ["aso2024"],
    "Ds595-Sa_avg3": ["aso2024"],
    "Ds575-Sa_avg3": ["aso2024"],
    "IA-SA": ["bradley2015_ia_sa", "baker2007_ia_sa"],
    "IA-PGA": ["bradley2015_ia_pga"],
    "IA-PGV": ["bradley2015_ia_pgv"],
    "IA-Ds575": ["bradley2015_ia_ds575"],
    "IA-Ds595": ["bradley2015_ia_ds595"],
    "SA_vert-SA_vert": ["kohrangi2020_sav_sav", "gkas2017_v"],
    "SA-SA_vert": ["kohrangi2020_sav_sah"],
    "SA-PGV_vert": ["kohrangi2020_sah_pgvv"],
    "SA_vert-Ds575": ["kohrangi2020_sav_ds575"],
    "SA_vert-Ds595": ["kohrangi2020_sav_ds595"],
    "SA_vert-PGA_vert": ["kohrangi2020_sav_pgav"],
    "SA_vert-PGV_vert": ["kohrangi2020_sav_pgvv"]
}


DB_CAUSAL_PARS = {
    "mechanism": {
        "name": "Mechanism",
        "description": "Fault mechanism based on rake angle"
    },
    "magnitude": {
        "name": "Magnitude",
        "description": "Moment magnitude of earthquake"
    },
    "Rjb": {
        "name": "Joyner-Boore distance",
        "description": "Joyner-Boore distance [km]"
    },
    "Rrup": {
        "name": "Rupture distance",
        "description": "Rupture distance [km]"
    },
    # "rake": {
    #     "name": "Rake angle",
    #     "description": "Rake angle, from -180 to 360 deg [deg]"
    # },
    # "dip": {
    #     "name": "Dip angle",
    #     "description": "Dip angle of the fault plane, from 0 to 90 deg [deg]"
    # },
    "Z2pt5": {
        "name": "Depth to Vs = 2.5 km/sec",
        "description": "Depth to Vs = 2.5 km/sec [m]"
    },
    # "Ztor": {
    #     "name": "Depth to top of fault rupture",
    #     "description": "Depth to top of fault rupture [km]"
    # },
    "Z1": {
        "name": "Depth to Vs=1.0 km/sec",
        "description": "Depth to Vs=1.0 km/sec [m]"
    },
    # "Z1pt5": {
    #     "name": "Depth to Vs = 1.5 km/sec",
    #     "description": "Depth to Vs = 1.5 km/sec [m]"
    # },
    "Vs30": {
        "name": "Shear-wave velocity",
        "description": "Time-averaged shear-wave velocity to 30m depth"
        " selected for analysis [m/s]"
    },
    # "strike": {
    #     "name": "Strike angle",
    #     "description": "Strike angle of the fault plane used to approximate"
    #     " the causative fault surface. 0 <= Strike <= 360 [deg]"
    # },
    "D_hyp": {
        "name": "Hypocentral depth",
        "description": "Hypocentral depth [km]"
    },
    # "Rx": {
    #     "name": "Rx",
    #     "description": "Distance measured perpendicular to the fault strike"
    #     " from the surface projection of the up-dip edge of the fault plane "
    #     "[km]"
    # },
    # "rup_width": {
    #     "name": "Fault rupture width",
    #     "description": "Fault rupture width [km]"
    # },
    "Ds595": {
        "name": "Significant duration",
        "description": "The time interval between 5-95% of Arias Intensity"
    },
    "npts": {
        "name": "Number of data points",
        "description": "Number of data points"
    },
    "duration": {
        "name": "Record duration",
        "description": "Record duration"
    },
    "Tp": {
        "name": "Pulse period",
        "description": "Pulse period"
    }
}

MECHANISM_MAP = {
    0: 'strike-slip fault',
    1: 'normal fault',
    2: 'reverse fault',
    3: 'reverse/oblique fault',
    4: 'normal/oblique fault'
}

MECHANISM_MAP_REV = {
    'strike-slip fault': 0,
    'normal fault': 1,
    'reverse fault': 2,
    'reverse/oblique fault': 3,
    'normal/oblique fault': 4,
}

ESHM20_COEFFICIENTS = {
    "total": (0.18141134, 0.1555742, -0.10851875, 0.08, 0.2),
    "between-event": (0.15881576, 0.08439678, -0.13915732, 0.08, 0.2),
    "between-site": (0.15751022, 0.15934185, -0.17513388, 0.08, 0.2),
    "within-event": (0.26023904, 0.27590487, -0.0951078, 0.08, 0.2)
}
