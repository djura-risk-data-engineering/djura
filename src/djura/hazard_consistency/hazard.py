import requests
import xml.etree.ElementTree as ET
import numpy as np


class Europe:
    SUPPORTED_MODELS = frozenset({"share"})

    def __init__(self, model: str) -> None:
        self.model = self._select_model(model.lower())

    def _select_model(self, model: str):
        if hasattr(self, model):
            method = getattr(self, model)
            if callable(method):
                return method
            else:
                raise AttributeError(
                    f"'{model}' is not an acceptable model\n"
                    f"Supported models: {self.SUPPORTED_MODELS}")
        else:
            raise AttributeError(
                f"'{model}' is not an acceptable model\n"
                f"Supported models: {self.SUPPORTED_MODELS}")

    def share(self, longitude, latitude, poe, investigation_time_span,
              soiltype, aggregationtype, aggregationlevel, model_id):

        url = "http://appsrvr.share-eu.org:8080/share/spectra"

        params = {
            "lat": latitude,
            "lon": longitude,
            "id": int(model_id),
            "imt": "SA",
            "poe": poe,
            "timespanpoe": int(investigation_time_span),
            "soiltype": soiltype,
            "aggregationtype": aggregationtype,
            "aggregationlevel": aggregationlevel,
        }

        response = requests.get(url, params=params)

        root = ET.fromstring(response.text)
        imls = root.find(
            ".//{http://openquake.org/xmlns/nrml/0.3}IML[@IMT='SA']")\
            .text.strip().split()
        periods = root.find(
            ".//{http://openquake.org/xmlns/nrml/0.3}spectraPeriodList")\
            .text.strip().split()

        target = {
            "periods": [np.round(float(_val), 2) for _val in periods],
            "SA": [float(_val) for _val in imls],
        }

        return target
