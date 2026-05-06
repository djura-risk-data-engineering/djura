# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from scipy.interpolate import interp1d
from .scaler import MinMaxScaler


def _require_xgboost():
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            "XGBPredict requires xgboost. "
            "Install it with: pip install 'djura[xgboost]'"
        ) from exc
    xgb.set_config(verbosity=0)
    return xgb


path = Path(__file__).parent.resolve()


class PredictionSchema(BaseModel):
    strength_ratio: float
    dispersion: float


class XGBPredict:
    ductility_range = np.arange(0.01, 12, 0.1)
    FEATURE_LOW_BOUND = np.array([[
        0.01, 0.02, 0.02, 2., 0.25669332
    ]])
    FEATURE_UP_BOUND = np.array([[
        3., 0.2, 0.1, 8., 19.28312206
    ]])
    FEATURE_LOW_BOUND_COLLAPSE = np.array([[
        0.01, 0.02, 0.02, 2.
    ]])
    FEATURE_UP_BOUND_COLLAPSE = np.array([[
        3., 0.2, 0.1, 8.
    ]])
    FEATURE_ORDER = ['period', 'damping', 'hardening_ratio',
                     'ductility', 'actual_ductility_end']

    def __init__(self, im_type: str, collapse: bool) -> None:
        """
        Initialize XGB model

        Parameters
        ----------
        im_type : str
            "sa" for R, "sa_avg" for rho2 or rho3
        collapse : bool
            True for collapse scenarios
            False for non-collapse scenarios
            Note: currently ro_2 is always for non-collapse, while ro_3 is
            for collapse

        Raises
        ------
        ValueError
            When im_type is neither 'sa' nor 'sa_avg'

        """
        if im_type.lower() == "sa":
            self.parameter = "R"
        elif im_type.lower() == "sa_avg" or im_type.lower() == "saavg":
            if collapse:
                self.parameter = "ro_3"
            else:
                self.parameter = "ro_2"
        else:
            raise ValueError("Wrong im_type, must be 'sa' or 'sa_avg'")

        self.collapse = collapse
        self.WARNINGS = []

    def _verify_input(
        self,
        period,
        damping,
        hardening_ratio,
        ductility
    ) -> None:
        if not (0.01 <= period <= 3.0):
            self.WARNINGS.append(
                "Period is not within recommended limits [0.01, 3.0]")

        if not (0.02 <= damping <= 0.2):
            self.WARNINGS.append(
                "Damping is not within recommended limits [0.02, 0.2]")

        if not (0.02 <= hardening_ratio <= 0.07):
            self.WARNINGS.append(
                "Hardening ratio is not within"
                " recommended limits [0.02, 0.07]")

        if not (2.0 <= ductility <= 8.0):
            self.WARNINGS.append(
                "Ductility is not within recommended limits [2.0, 8.0]")

    def generate_sr_for_ductility(
        self, scaler, model, disp_model,
        period, damping, hardening_ratio, ductility
    ):

        # feature order
        # period, damping, hardening_ratio, ductility, actual_ductility_end
        # feature_names = scaler.get_feature_names_out()
        xgb = _require_xgboost()
        if not self.collapse:
            # Add dynamic ductility for non-collapse predictions
            xgb_input = np.array([[period, damping, hardening_ratio,
                                  ductility, None]])
        else:
            xgb_input = np.array([[
                period, damping, hardening_ratio, ductility]])

        medians = np.zeros(self.ductility_range.shape)
        dispersions = np.zeros(self.ductility_range.shape)
        for i, duct in enumerate(self.ductility_range):
            if xgb_input.shape[1] == 5:
                xgb_input[0][-1] = duct
            else:
                xgb_input = np.append(xgb_input, [[duct]], axis=1)

            if duct < 1.0 and self.parameter == "R":
                medians[i] = duct
                dispersions[i] = 0.0
                continue

            x = scaler.transform(xgb_input)

            matrix = xgb.DMatrix(x)
            medians[i] = np.expm1(model.predict(matrix))[0]
            dispersions[i] = self._get_dispersion(
                disp_model, period, damping,
                hardening_ratio, ductility, duct)
            if not self.collapse and self.parameter != "R" \
                    and duct < 0.625:
                medians[i] = duct

        return medians, dispersions

    def estimate_ductility(
            self, medians, dispersions, strength_ratio):

        if strength_ratio > max(medians):
            strength_ratio = max(medians)

        if strength_ratio < min(medians):
            strength_ratio = min(medians)

        int_median = interp1d(medians, self.ductility_range)
        int_disp = interp1d(medians, dispersions)

        median = int_median(strength_ratio)
        disp = int_disp(strength_ratio)

        return {
            "median": median,
            "dispersion": disp
        }

    @property
    def model(self):
        if self.collapse:
            method = "_collapse"
            # Get the scaler
            scaler = MinMaxScaler(
                self.FEATURE_LOW_BOUND_COLLAPSE,
                self.FEATURE_UP_BOUND_COLLAPSE
            )
        else:
            method = ""
            scaler = MinMaxScaler(
                self.FEATURE_LOW_BOUND,
                self.FEATURE_UP_BOUND
            )

        # Read the XGB model
        xgb = _require_xgboost()
        model = xgb.Booster()
        model.load_model(
            str(path / f"models/{self.parameter}_xgb{method}.ubj"))

        dispersions = json.load(open(
            path / f"models/{self.parameter}_xgb{method}_dispersions.json"))

        return scaler, model, dispersions

    def make_prediction(
        self,
        scaler,
        model,
        dispersions,
        period: float,
        damping: float,
        hardening_ratio: float,
        ductility: float,
        dynamic_ductility: np.ndarray = None,
    ) -> PredictionSchema:
        """
        Make predictions using the XGB model

        Parameters
        ----------
        period : float
            Period
        damping : float
            Damping ratio
        hardening_ratio : float
            Hardening ratio
        ductility : float
            Hardening ductility of system
        dynamic_ductility : float, optional
            Ductility where the strength ratio is being predicted, required
            for non-collapse predictions, by default None
        strength_ratio : float
            Strength ratio corresponding to which a ductility value is
            being estimated, by default, None

        Returns
        ----------
        PredictionSchema
            Predictions in dict type::

                {"median": float,  # R, ro_2, ro_3, or ductility
                 "dispersion": float}
        """
        self._verify_input(period, damping, hardening_ratio, ductility)

        # feature order
        # period, damping, hardening_ratio, ductility, actual_ductility_end
        # feature_names = scaler.get_feature_names_out()
        if not self.collapse:
            if isinstance(dynamic_ductility, float):
                dynamic_ductility = np.array([dynamic_ductility])

            # Add dynamic ductility for non-collapse predictions
            xgb_input = np.column_stack((
                np.full(len(dynamic_ductility), period),
                np.full(len(dynamic_ductility), damping),
                np.full(len(dynamic_ductility), hardening_ratio),
                np.full(len(dynamic_ductility), ductility),
                dynamic_ductility
            ))

        else:
            xgb_input = np.array([[
                period, damping, hardening_ratio, ductility]])

        if dynamic_ductility is None and not self.collapse:
            raise ValueError(
                "Dynamic ductility not provided for non-collapse predictions")

        xgb = _require_xgboost()
        x = scaler.transform(xgb_input)
        matrix = xgb.DMatrix(x)
        median = np.expm1(model.predict(matrix))

        # Retrieve dispersion
        dispersion = self._get_dispersion(
            dispersions, period, damping,
            hardening_ratio, ductility, dynamic_ductility)

        if not self.collapse and self.parameter != "R":
            median[dynamic_ductility
                   <= 0.625] = dynamic_ductility[dynamic_ductility <= 0.625]

        if not self.collapse and self.parameter == "R":
            median[dynamic_ductility
                   <= 1.0] = dynamic_ductility[dynamic_ductility <= 1.0]
            dispersion[dynamic_ductility <= 1.0] = 0.0

        prediction = {
            "median": median,
            "dispersion": dispersion,
        }

        return prediction

    def _get_dispersion(
        self,
        dispersions: dict,
        period: float,
        damping: float,
        hardening_ratio: float,
        ductility: float,
        dynamic_ductility: np.ndarray
    ) -> float:
        """Gets dispersion values

        Parameters
        ----------
        dispersions : dict
            Dispersions, {
                "period": {
                    "damping": {
                        "hardening_ratio": {
                            "ductility": float
                        }
                    }
                }
            }
        period : float
            Period in [s]
        damping : float
            Damping
        hardening_ratio : float
            Hardening ratio
        ductility : float
            Ductility
        dynamic_ductility : float
            Dynamic ductility

        Returns
        ----------
        float
            Dispersion value
        """

        def is_valid_float(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        period_key = str(min((
            key for key in dispersions if is_valid_float(key)),
            key=lambda x: abs(float(x) - period), default=None
        ))

        damp_key = str(min((
            key for key in dispersions[period_key] if is_valid_float(key)),
            key=lambda x: abs(float(x) - damping), default=None
        ))

        hard_key = str(min((
            key for key in dispersions[period_key][damp_key]
            if is_valid_float(key)),
            key=lambda x: abs(float(x) - hardening_ratio), default=None
        ))

        duct_key = str(min((
            key for key in dispersions[period_key][damp_key][hard_key]
            if is_valid_float(key)),
            key=lambda x: abs(float(x) - ductility), default=None
        ))

        dispersion = np.asarray(
            dispersions[period_key][damp_key][hard_key][duct_key])
        if self.collapse:
            return dispersion

        ductilities = np.asarray(dispersions["ductility"])

        interpolator = interp1d(ductilities, dispersion)
        clipped_dynamic_ductility = np.clip(
            dynamic_ductility, ductilities[0], ductilities[-1])

        val = interpolator(clipped_dynamic_ductility)
        val[dynamic_ductility < ductilities[0]] = dispersion[0]
        val[dynamic_ductility > ductilities[-1]] = dispersion[-1]

        if np.isnan(val).any() or np.all(val == 0):
            self.WARNINGS.append(
                "Dispersion is null, as dynamic ductility is unattainable for "
                "given input... Try with smaller dynamic ductility value")
            val = np.where(np.isnan(val), max(val), val)

        return val
