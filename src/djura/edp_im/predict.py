# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
import numpy as np
import importlib
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List
from scipy.stats import lognorm
import traceback

from ..vulnerability_modeller.utilities import (
    filter_args, to_json_serializable, DUCTILITY_F, set_ductilities)


class BackboneModel(BaseModel):
    period: float
    period_c: Optional[float] = None
    case: Optional[str] = None
    site: Optional[str] = None
    period_g: Optional[float] = None
    period_cc: Optional[float] = None
    ductility: Optional[float] = None
    ductility_f: Optional[float] = DUCTILITY_F
    hardening_ratio: Optional[float] = None
    ah: Optional[float] = None
    damping: Optional[float] = None

    # Plots
    r_plot: Optional[List[float]] = Field(default=None, alias="r-plot")
    mu_plot: Optional[List[float]] = Field(default=None, alias="mu-plot")

    @model_validator(mode='before')
    def str_to_lower(cls, values):
        for field, value in values.items():
            if isinstance(value, str):
                values[field] = value.lower()
        return values

    @model_validator(mode='after')
    def compute_backbone_plots(cls, model):
        if model.hardening_ratio is not None and model.ductility is not None \
                and model.ductility_f is not None:
            ah = model.hardening_ratio
            mu = model.ductility
            mu_f = model.ductility_f
            model.r_plot = [0, 1, 1 + ah * (mu - 1), 0]
            model.mu_plot = [0, 1, mu, mu_f]
        else:
            model.r_plot = None
            model.mu_plot = None
        return model


class EDPIMModel(BaseModel):
    hysteresis: str = "bilin"
    im_type: Optional[str] = "sa"
    method: Optional[str] = "shahnazaryan-oreilly"
    backbone: BackboneModel

    @model_validator(mode='before')
    def str_to_lower(cls, values):
        for field, value in values.items():
            if isinstance(value, str):
                values[field] = value.lower()
        return values


class EDPIMInfillModel(BaseModel):
    period: float
    c_y: float
    c_rp: float
    mu_h: float
    mu_s: float
    mu_rp: float
    mu_ult: float
    im_type: Optional[str] = "sa"

    @model_validator(mode='before')
    def str_to_lower(cls, values):
        for field, value in values.items():
            if isinstance(value, str):
                values[field] = value.lower()
        return values


class EDPIMIsolModel(BaseModel):
    R: float
    mu: float
    delta_max: float = 1.5


def _ductility_based(data, func):
    ductility_f = data.get('ductility_f', DUCTILITY_F) + 0.01

    DYNAMIC_DUCTILITY = set_ductilities(ductility_f)

    # Make prediction
    outs = np.zeros(DYNAMIC_DUCTILITY.shape)
    disps = np.zeros(DYNAMIC_DUCTILITY.shape)

    for i, dyn_duct in enumerate(DYNAMIC_DUCTILITY):
        if callable(func):
            data["dynamic_ductility"] = dyn_duct
            prediction = func(**data)
        else:
            return {
                "status": "error",
                "message": "Wrong method or something went wrong"
            }

        outs[i] = float(prediction["median"])
        if "dispersion" in prediction:
            disps[i] = prediction["dispersion"]

    return {
        "strength-ratios": outs,
        "ductilities": DYNAMIC_DUCTILITY,
        "dispersions": disps,
    }


def _xgb(data, func):
    ductility_f = data.get('dynamic_ductility', DUCTILITY_F) + 0.01

    DYNAMIC_DUCTILITY = set_ductilities(ductility_f)

    if callable(func):
        data["dynamic_ductility"] = DYNAMIC_DUCTILITY
        prediction = func(**data)

        return {
            "strength-ratios": prediction['median'],
            "ductilities": DYNAMIC_DUCTILITY,
            "dispersions": prediction['dispersion'],
        }

    return {
        "status": "error",
        "message": "Wrong method or something went wrong"
    }


def _strength_ratio_based(data, func):
    STRENGTH_RATIO = np.arange(0.01, 12.01, 0.01)

    # Make prediction
    outs = np.zeros(STRENGTH_RATIO.shape)
    disps = np.zeros(STRENGTH_RATIO.shape)

    for i, sr in enumerate(STRENGTH_RATIO):
        if callable(func):
            data["strength_ratio"] = sr
            prediction = func(**data)
        else:
            return {
                "status": "error",
                "message": "Wrong method or something went wrong"
            }

        outs[i] = float(prediction["median"])
        if "dispersion" in prediction:
            disps[i] = prediction["dispersion"]

    return {
        "strength-ratios": STRENGTH_RATIO,
        "ductilities": outs,
        "dispersions": disps,
    }


def edp_im(data: EDPIMModel):
    data = EDPIMModel.model_validate(data)

    hysteresis = data.hysteresis
    im_type = data.im_type
    backbone = data.backbone.model_dump()
    method = data.method

    if not backbone:
        return {"status": "error", "message": "Backbone missing"}

    module = None
    warnings = []
    if method == "shahnazaryan-oreilly" and hysteresis == "bilin":
        from .XGBPredict import XGBPredict
        module = XGBPredict(
            im_type=im_type, collapse=False)
        func = module.make_prediction
        backbone["scaler"], backbone["model"], backbone["dispersions"] = \
            module.model

    elif method != "shahnazaryan-oreilly" and im_type == "sa":
        module = importlib.import_module(
            f"djura.edp_im.r_mu_t.{method}")
        func = getattr(module, 'make_prediction', None)

    if not module:
        return {
            "status": "error",
            "message": "Hysteresis method incorrect or missing"
        }

    backbone["dynamic_ductility"] = backbone['ductility_f']
    backbone["strength_ratio"] = None
    filtered_data = filter_args(func, backbone)

    # Make prediction
    if method == "guerrini":
        outs = _strength_ratio_based(filtered_data, func)
    elif method == "shahnazaryan-oreilly":
        outs = _xgb(filtered_data, func)
        warnings = module.WARNINGS
    else:
        outs = _ductility_based(filtered_data, func)

    dispersions = outs.get("dispersions")
    sr = outs.get("strength-ratios")

    if isinstance(dispersions, float):
        # No case for this condition yet
        pass
    elif dispersions is not None:
        sr16 = lognorm.ppf(0.16, s=dispersions, scale=sr)
        sr84 = lognorm.ppf(0.84, s=dispersions, scale=sr)
        sr16[dispersions == 0] = sr[dispersions == 0]
        sr84[dispersions == 0] = sr[dispersions == 0]

        outs["strength-ratios-16%"] = sr16
        outs["strength-ratios-84%"] = sr84

    return to_json_serializable({
        "status": "success",
        "data": outs,
        "message": "Predictions success",
        "warnings": warnings,
    })


def edp_im_isol(data: EDPIMIsolModel):
    data = EDPIMIsolModel.model_validate(data)

    r = data.R
    mu = data.mu

    if r is None or mu is None:
        return {"status": "error", "message": "'R' or 'mu' missing"}

    a1, b1, c1 = 0.337475, 0.8083212, 1.02
    a2, c2 = 1.182557, -0.08815
    a3, b3, c3 = 0.857213, 0.4183158, 0.4027454

    gamma = a1 * mu**b1 * r**c1
    kappa = a2 * r**c2
    beta = a3 * mu**b3 * r**c3

    delta = np.arange(0.01, data.delta_max, 0.01)

    rho = (delta / gamma) ** (1 / kappa)

    sa_t_iso = rho * mu

    outs = {"rho": rho, "deltas": delta, "beta": beta, "sa_t_iso": sa_t_iso}

    return to_json_serializable({
        "status": "success",
        "data": outs,
        "message": "Predictions success"
    })


def edp_im_infill(data: EDPIMInfillModel):
    data = EDPIMInfillModel.model_validate(data)

    period = data.period
    c_y = data.c_y
    c_rp = data.c_rp
    mu_h = data.mu_h
    mu_s = data.mu_s
    mu_rp = data.mu_rp
    mu_ult = data.mu_ult
    im_type = data.im_type

    mu = np.arange(0.01, mu_ult, 0.01)

    if im_type == "sa_avg":
        rho = np.zeros(mu.shape)

        a = 0.704 * (period / c_y) ** 0.1595 - 0.239
        b = 1.813 * (c_rp * (mu_rp - mu_s)) ** 0.0473 - 1.98

        rho = np.exp(a * np.log(mu) + b)
        rho[mu <= 1.0] = np.exp(
            np.log(mu[mu <= 1.0]) + b)

        return to_json_serializable({
            "status": "success",
            "data": {
                "strength-ratios": rho,
                "ductilities": mu,
                "dispersion": 0.27,
            },
            "message": "Predictions success"
        })

    elif im_type == "sa":

        if mu_h == mu_s:
            mu_s = mu_h + 0.01

        def _adjust(data, last):
            if len(data) == 0:
                return np.array([])

            data = np.maximum.accumulate(data)
            first = data[0]
            data = data - (first - last)
            return data

        sr = np.zeros(mu.shape) + 1
        sr16 = np.zeros(mu.shape) + 1
        sr84 = np.zeros(mu.shape) + 1

        # Hardening branch
        h_idxs = (mu > 1) & (mu <= mu_h)
        a1, b1 = _hard_branch_infill(period)
        sr[h_idxs] = a1[1] * mu[h_idxs] ** b1[1]
        sr16[h_idxs] = a1[0] * mu[h_idxs] ** b1[0]
        sr84[h_idxs] = a1[2] * mu[h_idxs] ** b1[2]
        sr[h_idxs] = _adjust(sr[h_idxs], 1)
        sr16[h_idxs] = _adjust(sr16[h_idxs], 1)
        sr84[h_idxs] = _adjust(sr84[h_idxs], 1)

        # Softening branch
        s_idxs = (mu > mu_h) & (mu <= mu_s)
        a2, b2, g2 = _soft_branch_infill(period)
        sr16[s_idxs] = a2[0] * mu[s_idxs] ** 2 + b2[0] * mu[s_idxs] + g2[0]
        sr[s_idxs] = a2[1] * mu[s_idxs] ** 2 + b2[1] * sr16[s_idxs] + g2[1]
        sr84[s_idxs] = a2[2] * mu[s_idxs] ** 2 + b2[2] * sr[s_idxs] + g2[2]
        sr[s_idxs] = _adjust(sr[s_idxs], sr[h_idxs][-1])
        sr16[s_idxs] = _adjust(sr16[s_idxs], sr16[h_idxs][-1])
        sr84[s_idxs] = _adjust(sr84[s_idxs], sr84[h_idxs][-1])

        # Residual plateau branch
        rp_idxs = (mu > mu_s) & (mu <= mu_rp)
        a3, b3 = _res_branch_infill(period)
        sr[rp_idxs] = a3[1] * mu[rp_idxs] + b3[1]
        sr16[rp_idxs] = a3[0] * mu[rp_idxs] + b3[0]
        sr84[rp_idxs] = a3[2] * mu[rp_idxs] + b3[2]
        sr[rp_idxs] = _adjust(sr[rp_idxs], sr[s_idxs][-1])
        sr16[rp_idxs] = _adjust(sr16[rp_idxs], sr16[s_idxs][-1])
        sr84[rp_idxs] = _adjust(sr84[rp_idxs], sr84[s_idxs][-1])

        # Strength degradation branch
        sd_idxs = (mu > mu_rp)
        a4, b4 = _sd_branch_infill(period)
        sr[sd_idxs] = a4[1] * mu[sd_idxs] + b4[1]
        sr16[sd_idxs] = a4[0] * mu[sd_idxs] + b4[0]
        sr84[sd_idxs] = a4[2] * mu[sd_idxs] + b4[2]
        sr[sd_idxs] = _adjust(sr[sd_idxs], sr[rp_idxs][-1])
        sr16[sd_idxs] = _adjust(sr16[sd_idxs], sr16[rp_idxs][-1])
        sr84[sd_idxs] = _adjust(sr84[sd_idxs], sr84[rp_idxs][-1])

        sr[np.isnan(sr)] = 1
        sr16[np.isinf(sr16)] = 1e9
        sr84[np.isinf(sr84)] = 1e9

        # Elastic branch
        sr[mu <= 1.0] = mu[mu <= 1.0]
        sr16[mu <= 1.0] = mu[mu <= 1.0]
        sr84[mu <= 1.0] = mu[mu <= 1.0]

        return to_json_serializable({
            "status": "success",
            "data": {
                "strength-ratios": sr,
                "strength-ratios-16%": sr16,
                "strength-ratios-84%": sr84,
                "ductilities": mu,
                "dispersion": 0,
            },
            "message": "Predictions success"
        })

    return {"status": "error", "message": "IM type not provided or incorrect"}


def _hard_branch_infill(period):
    a = np.array([0.8628, 0.9235, 0.9195, 0.9632, 0.4745, 0.0654, 0.04461])
    b = np.array([0.7624, 0.5041, 0.1785, 1.0220, 0.3253, 0.4064, 0.4479])
    c = np.array([0.1643, 0.1701, 0.1147, 0.1694, 0.09403, 0.02054, 0.01584])

    exponent_terms = - ((period - b) / c) ** 2
    alpha50 = np.sum(a * np.exp(exponent_terms))

    # Constants for the first table
    a1 = np.array([0.1460, 0.5926, 0.07312, 0.2965, 0.02688, 1.0630, 0.3127])
    b1 = np.array([0.5335, 0.4161, 0.4495, 0.2215, 0.3699, 1.0030, 0.1462])
    c1 = np.array([0.03444, 0.3194, 0.01667, 0.1087, 0.0158, 0.6460, 0.07181])
    exponent_terms = - ((period - b1) / c1) ** 2
    alpha16 = np.sum(a1 * np.exp(exponent_terms))

    # Constants for the second table
    a2 = np.array([1.024, 0.6034, 0.2466, 0.06141, 0.2511, 0.0001, 0.07086])
    b2 = np.array([0.9018, 0.1928, 0.4758, 0.6903, 0.3254, 0.9390, 0.3948])
    c2 = np.array([0.6555, 0.1072, 0.1232, 0.05664, 0.07067, 0.00132, 0.02287])
    exponent_terms = - ((period - b2) / c2) ** 2
    alpha84 = np.sum(a2 * np.exp(exponent_terms))

    # Constants for the first table
    a_b1 = np.array([-0.1334, 0.3312, 0.7985, 0.0001, 0.1543, 0.9252, 0.2809])
    b_b1 = np.array([0.7771, 0.7647, 0.04284, 0.5721, 0.4788, 0.8165, 0.3003])
    c_b1 = np.array([0.04907, 0.000986, 0.09365,
                    0.0001, 0.1050, 0.5100, 0.1216])
    exponent_terms = - ((period - b_b1) / c_b1) ** 2
    beta50 = np.sum(a_b1 * np.exp(exponent_terms))

    # Constants for the second table
    a_b2 = np.array([0.2008, 0.1790, 0.1425, 0.1533,
                    3.623e12, 0.09451, 0.1964])
    b_b2 = np.array([1.0930, 0.7169, 0.4876, 0.5709, 97.610, 0.4424, 0.3345])
    c_b2 = np.array([0.5405, 0.08836, 0.04956,
                    0.07256, 17.940, 0.06262, 0.09522])
    exponent_terms = - ((period - b_b2) / c_b2) ** 2
    beta16 = np.sum(a_b2 * np.exp(exponent_terms))

    # Constants for the third table
    a_b3 = np.array([0.7182, 0.1320, 0.1233, 0.09805, 0.1429, 0.6547, 0.0001])
    b_b3 = np.array([0.04151, 0.6058, 0.4904, 0.5448, 0.3652, 0.8431, 0.7115])
    c_b3 = np.array([0.09018, 0.04845, 0.04392,
                    0.01778, 0.09815, 0.71260, 0.0001803])
    exponent_terms = - ((period - b_b3) / c_b3) ** 2
    beta84 = np.sum(a_b3 * np.exp(exponent_terms))

    return [alpha16, alpha50, alpha84], [beta16, beta50, beta84]


def _soft_branch_infill(period):
    aa2 = np.array([0.0395, 0.01833, 0.009508])
    ba2 = np.array([-0.03069, -0.01481, -0.007821])
    ab2 = np.array([1.049, 0.8237, 0.4175])
    bb2 = np.array([0.2494, 0.04082, 0.03164])
    ag2 = np.array([-0.7326, -0.7208, -0.0375])
    bg2 = np.array([1.116, 1.279, 1.079])

    a2 = aa2 * period + ba2
    b2 = ab2 * period + bb2
    g2 = ag2 * period + bg2

    return a2, b2, g2


def _res_branch_infill(period):
    aa = np.array([-5.075, -2.099, -0.382])
    ba = np.array([7.112, 3.182, 0.6334])
    ca = np.array([-1.572, -0.6989, -0.051])
    da = np.array([0.1049, 0.0481, 0.002])

    ab = np.array([16.16, 8.417, -0.027])
    bb = np.array([-26.5, -14.51, -1.8])
    cb = np.array([10.92, 6.75, 2.036])
    db = np.array([1.055, 0.9061, 1.067])

    a = aa * period ** 3 + ba * period ** 2 + ca * period + da
    b = ab * period ** 3 + bb * period ** 2 + cb * period + db

    return a, b


def _sd_branch_infill(period):
    aa = np.array([-1.564, -0.5954, -0.06693])
    ba = np.array([2.193, 0.817, 0.1418])
    ca = np.array([-0.352, -0.09191, 0.0124])
    da = np.array([0.0149, 0.001819, -0.002012])

    ab = np.array([1.756, 0.7315, -0.408])
    bb = np.array([-8.719, -3.703, -1.333])
    cb = np.array([8.285, 4.391, 2.521])
    db = np.array([1.198, 1.116, 1.058])

    a = aa * period ** 3 + ba * period ** 2 + ca * period + da
    b = ab * period ** 3 + bb * period ** 2 + cb * period + db

    return a, b


def edp_im_batch(data: List[EDPIMModel]):
    if not data:
        return {"status": "error", "message": "Empty batch"}

    results = [None] * len(data)

    # Validate all inputs and group by method/hysteresis
    from collections import defaultdict
    grouped = defaultdict(list)

    for idx, _data in enumerate(data):
        try:
            from ..vulnerability_modeller.backbone import (
                Backbone, get_infill_spo)
            backbone = _data["backbone"]

            hysteresis = _data.get('hysteresis', 'bilin')
            method = _data.get('method', "")
            im_type = _data.get("im_type", "sa")

            b = Backbone(
                backbone,
                _data.get('backboneMethod', 'backbone'),
                hysteresis,
            )

            _data['backbone'] = b.backbone

            if method == "shahnazaryan-oreilly" and \
                    hysteresis == "bilin":

                backbone = BackboneModel.model_validate(b.backbone)

            elif hysteresis == "infill":
                b.backbone['im_type'] = im_type
                b.backbone["r-plot"], b.backbone["mu-plot"] = get_infill_spo(
                    b.backbone)

            _data["backbone"] = b.backbone

            if hysteresis != "infill":
                validated = EDPIMModel.model_validate(_data)
                key = (validated.method, validated.hysteresis,
                       validated.im_type)
                grouped[key].append((idx, validated))
            else:
                key = (method, hysteresis, im_type)
                # Validation will be performed later
                grouped[key].append((idx, _data))

        except Exception as e:
            results[idx] = {
                "index": idx,
                "status": "error",
                "error": f"Validation failed: {str(e)}",
                "error_type": "validation"
            }

    # Process each group
    for (method, hysteresis, im_type), items in grouped.items():
        try:
            if hysteresis == "bilin":
                _process_group_bilin(
                    items, method, hysteresis, im_type, results)
            elif hysteresis == "infill":
                _process_group_infill(items, results)
            else:
                raise ValueError(
                    f"Unsupported hysteresis type: '{hysteresis}'. "
                    f"Expected 'bilin' or 'infill'."
                )

        except Exception as e:
            # Group processing failed - mark all items in group as errors
            for idx, _ in items:
                # Only if not already marked as error
                if results[idx] is None:
                    results[idx] = {
                        "index": idx,
                        "status": "error",
                        "error": f"Group processing failed: {str(e)}",
                        "error_type": "group_processing",
                        "traceback": traceback.format_exc()
                    }

    for idx, result in enumerate(results):
        # Catch all (unexpected error)
        if result is None:
            results[idx] = {
                "index": idx,
                "status": "error",
                "error": "Unexpected: result not set",
                "error_type": "unexpected"
            }

    # Calculate statistics
    successes = [r for r in results if r["status"] == "success"]
    failures = [r for r in results if r["status"] == "error"]

    success_ids = [r["index"] for r in successes]
    failure_ids = [r["index"] for r in failures]

    # Group errors by type
    error_types = {}
    for failure in failures:
        error_type = failure.get("error_type", "unknown")
        error_types[error_type] = error_types.get(error_type, 0) + 1

    return to_json_serializable({
        "status": "success",
        "total": len(data),
        "successful": len(successes),
        "failed": len(failures),
        "error_summary": error_types,
        "successes": successes,
        "failures": failures,
        "success_ids": success_ids,
        "failure_ids": failure_ids,
        "message": f"Batch processing complete: {len(successes)} succeeded,"
        f"{len(failures)} failed"
    })


def _process_group_bilin(items, method, hysteresis, im_type, results):
    """Process a group of items with the same method/hysteresis/im_type"""
    # XGBoost batch processing
    if method == "shahnazaryan-oreilly":
        try:
            from .XGBPredict import XGBPredict
            module = XGBPredict(im_type=im_type, collapse=False)
            func = module.make_prediction
            scaler, model, dispersions_info = module.model
        except Exception as e:
            # Model loading failed - mark all items as errors
            for idx, _ in items:
                results[idx] = {
                    "index": idx,
                    "status": "error",
                    "error": f"Model loading failed: {str(e)}",
                    "error_type": "model_loading"
                }
            return

        # Process each item using make_prediction
        # (model loaded once for entire group)
        for idx, data in items:
            try:
                backbone = data.backbone.model_dump()
                backbone["dynamic_ductility"] = backbone['ductility_f']
                backbone["strength_ratio"] = None

                # Add scaler, model, dispersions to backbone
                # (required by make_prediction)
                backbone["scaler"] = scaler
                backbone["model"] = model
                backbone["dispersions"] = dispersions_info

                # Used in output preparation
                # Processed input is returned with outputs as metadata
                backbone["method"] = method
                backbone["hysteresis"] = hysteresis
                backbone["im_type"] = im_type

                # Filter args for make_prediction
                filtered_data = filter_args(func, backbone)

                # Call the actual make_prediction method
                outs = _xgb(filtered_data, func)

                # Process dispersions
                outs = _get_sr_fractiles(outs)

                # Remove xgboost scaler
                if 'scaler' in backbone:
                    # Not JSON-serializable
                    del backbone['scaler']
                    del backbone['model']
                    # Entire XGB dispersion predictions, unnecessary
                    del backbone['dispersions']

                inputs = data.model_dump(by_alias=True)

                results[idx] = {
                    "index": idx,
                    "status": "success",
                    "out": outs.get('data', {}),
                    "input": inputs,
                }

            except Exception as e:
                results[idx] = {
                    "index": idx,
                    "status": "error",
                    "error": f"Prediction failed: {str(e)}",
                    "error_type": "prediction",
                    "traceback": traceback.format_exc()
                }

    # Other methods (r_mu_t based) - process individually within group
    elif method != "shahnazaryan-oreilly" and im_type == "sa":
        # Only SA (R)
        try:
            module = importlib.import_module(f"djura.edp_im.r_mu_t.{method}")
            func = getattr(module, 'make_prediction', None)

            if not func:
                raise ValueError(
                    f"make_prediction function not found for method {method}")

        except Exception as e:
            # Module loading failed - mark all items as errors
            for idx, _ in items:
                results[idx] = {
                    "index": idx,
                    "status": "error",
                    "error": f"Module loading failed: {str(e)}",
                    "error_type": "module_loading"
                }
            return

        # Process each item individually (these methods don't support batching)
        for idx, data in items:
            try:
                backbone = data.backbone.model_dump()
                backbone["dynamic_ductility"] = backbone['ductility_f']
                backbone["strength_ratio"] = None
                filtered_data = filter_args(func, backbone)

                if method == "guerrini":
                    outs = _strength_ratio_based(filtered_data, func)
                else:
                    outs = _ductility_based(filtered_data, func)

                processed = _get_sr_fractiles(outs)

                results[idx] = {
                    "index": idx,
                    "status": "success",
                    "out": processed.get('data', {}),
                    "input": backbone,
                }

            except Exception as e:
                results[idx] = {
                    "index": idx,
                    "status": "error",
                    "error": f"Individual prediction failed: {str(e)}",
                    "error_type": "prediction",
                    "traceback": traceback.format_exc()
                }

    else:
        # Unknown method/hysteresis combination
        for idx, _ in items:
            results[idx] = {
                "index": idx,
                "status": "error",
                "error": f"Unsupported method/hysteresis/IM type combination: "
                f"{method}/{hysteresis}/{im_type}",
                "error_type": "unsupported_method"
            }


def _process_group_infill(items, results):
    for idx, data in items:
        try:
            outs = edp_im_infill(data['backbone'])

            results[idx] = {
                "index": idx,
                "status": "success",
                "out": outs.get("data", {}),
                "input": data
            }
        except Exception as e:
            results[idx] = {
                "index": idx,
                "status": "error",
                "error": f"Prediction failed: {str(e)}",
                "error_type": "prediction",
                "traceback": traceback.format_exc()
            }


def _get_sr_fractiles(outs):
    try:
        dispersions = outs.get("dispersions")
        sr = outs.get("strength-ratios")

        if isinstance(dispersions, float):
            pass
        elif dispersions is not None:
            sr16 = lognorm.ppf(0.16, s=dispersions, scale=sr)
            sr84 = lognorm.ppf(0.84, s=dispersions, scale=sr)
            sr16[dispersions == 0] = sr[dispersions == 0]
            sr84[dispersions == 0] = sr[dispersions == 0]

            outs["strength-ratios-16%"] = sr16
            outs["strength-ratios-84%"] = sr84

        return to_json_serializable({
            "status": "success",
            "data": outs,
            "message": "Prediction success"
        })

    except Exception as e:
        return {
            "status": "error",
            "data": {},
            "message": f"Dispersion calculation failed: {str(e)}"
        }
