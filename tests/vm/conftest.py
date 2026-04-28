"""Shared helpers for tests/vm.

Tests call the djura library directly. The helpers below mirror the small
amount of preprocessing previously done in route handlers (Backbone fitting
for SPO/idealized inputs, infill SPO unpack, async drivers).
"""
import asyncio

from djura.edp_im.predict import edp_im as _edp_im
from djura.edp_im.predict import edp_im_infill as _edp_im_infill
from djura.edp_im.predict import edp_im_isol as _edp_im_isol
from djura.vulnerability_modeller.backbone import Backbone, get_infill_spo
from djura.vulnerability_modeller.demands import Demands
from djura.vulnerability_modeller.vm_mdof import VMMDOF


def run_edp_im(body):
    """Run Backbone preprocessing then ``edp_im``."""
    body = dict(body)
    backbone = body.get("backbone", {})
    backbone_method = body.get("backboneMethod", "backbone")
    hysteresis = body.get("hysteresis", "bilin")
    body["backbone"] = Backbone(backbone, backbone_method, hysteresis).backbone
    return _edp_im(body)


def run_edp_im_isol(body):
    payload = body["backbone"] if "backbone" in body else body
    return _edp_im_isol(payload)


def run_edp_im_infill(body):
    backbone = body["backbone"]
    backbone_method = body.get("backboneMethod", "spo")
    bb = dict(Backbone(backbone, backbone_method, "infill").backbone)
    bb["im_type"] = body.get("im_type", "sa")
    bb["r-plot"], bb["mu-plot"] = get_infill_spo(bb)
    return _edp_im_infill(bb)


def run_drift(models):
    return asyncio.run(Demands(models).estimate_drifts())


def run_acceleration(models):
    return asyncio.run(Demands(models).estimate_accelerations())


def run_vm(models):
    data = models if isinstance(models, list) else [models]
    return asyncio.run(VMMDOF(data).vulnerability())
