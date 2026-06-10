Djura-Tools companion scripts
=============================

`djura-tools <https://github.com/djura-risk-data-engineering/djura-tools>`_
is a companion repository containing helper functions and ready-to-run
scripts that exercise the ``djura`` package end-to-end. It bundles example
PSHA datastores, OQ Engine hazard and disaggregation outputs, and selected
record sets, together with scripts that turn them into inputs for the
`Djura web apps <https://apps.djura.it/>`_.

Clone the repository alongside an installed ``djura`` to run the scripts::

    git clone https://github.com/djura-risk-data-engineering/djura-tools
    cd djura-tools
    pip install djura

The scripts live under ``scripts/`` and read their sample inputs from
``scripts/data/``. The sections below reproduce each script and describe
what it does.

Processing disaggregation (approximate)
---------------------------------------

``scripts/process_disagg_approx.py`` parses OQ Engine disaggregation and
hazard curve outputs, then builds the conditional record-selection input
expected by the
`conditional record selector <https://apps.djura.it/hazard/record-selector/conditional>`_.

.. code-block:: python

    """
    Process Disaggregation outputs from Oq-engine

    This is an approximate approach
    """

    # flake8: noqa
    from pathlib import Path
    import sys
    import json

    path = Path(__file__).resolve().parent

    sys.path.insert(0, str(path.parent))

    from djura.hazard.psha import proc_oq_disaggregation, proc_oq_hazard_curve


    # POEs of interest
    # Those are probability of exceedances (POEs) associated with intensity levels
    # used during record selection
    poes = [0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.0025, 0.001]

    # Process the disaggregation excel file from OQ-Engine
    disagg_all = proc_oq_disaggregation(
        path / 'data/psha', poes,
        # out_file=path / 'data/psha/disaggregation.json'
    )

    hz = proc_oq_hazard_curve(poes, path / 'data/psha')

    # SA(0.5) to be used as conditional IM
    imt = "SA(0.5)"

    imls = hz['cond_imls'][imt]

    rs_input = json.load(open(path / "data/djura-conditional-all-input.json"))
    rs_input["imi"] = [
        'SA(0.05)', 'SA(0.075)', 'SA(0.1)', 'SA(0.15)', 'SA(0.2)',
        'SA(0.25)', 'SA(0.3)', 'SA(0.4)', 'SA(0.5)', 'SA(0.59)', 'SA(0.6)',
        'SA(0.7)', 'SA(0.8)', 'SA(0.9)', 'SA(1.0)', 'SA(1.25)',
        'SA(1.5)', 'SA(2.0)', 'SA(2.5)', 'SA(3.0)'
    ]
    rs_input["im_weights"] = [1 / len(rs_input["imi"])] * len(rs_input["imi"])
    rs_input["gmms"] = [
        {"ID": 0, "SA": {"names": ['KothaEtAl2020ESHM20'], "weights": [1]}}
    ]
    rs_input["site-parameters"] = {
        "z2pt5": "1.38",
        "z1pt0": "391.34",
        "vs30": "370",
        "xvf": "150",
        "region": 0
    }
    rs_input["poes"] = {}

    for idx, poe in enumerate(poes):
        disagg = disagg_all["imt_disagg"][imt][
            'mag_dist_hazard_contributions'][f'poe_{poe}']
        ruptures = []
        for i in range(len(disagg['mag'])):
            ruptures.append(
                {
                    "ID": i,
                    "gmms": [0],
                    "rjb": disagg['dist'][i],
                    "mag": disagg['mag'][i],
                    "rake": 0.0,
                    "weight": disagg['hz_cont_occ'][i],
                    # "weight": disagg['hz_cont_exc'][i],
                }
            )
        rs_input["poes"][poe] = {
            "ruptures": ruptures,
            "im-star": {"type": imt, "value": imls[idx]},
        }
        # You may save the rs_input variable to a file for each POE
        # This input then may be directly uploaded in
        # https://apps.djura.it/hazard/record-selector/conditional

        # NOTE: however, be careful as browsers might not easily display
        # large amount of data
        # If you encounter issues, feel free to contact us for questions
        # or to help run the API without a UI

Processing disaggregation with a datastore
------------------------------------------

``scripts/process_disagg_approx_dstore.py`` extends the approximate
workflow by reading rupture context parameters directly from the OQ
Engine datastore (``calc_<id>.hdf5``) via ``get_context_from_dstore``,
matching each disaggregation scenario to its closest magnitude/distance
context and keeping only the most contributing ruptures.

.. code-block:: python

    """
    Process Disaggregation outputs from Oq-engine

    This is an approximate approach
    """

    # flake8: noqa
    from pathlib import Path
    import sys
    import json
    import heapq
    import numpy as np

    path = Path(__file__).resolve().parent

    sys.path.insert(0, str(path.parent))

    from djura.hazard.dstore import get_context_from_dstore
    from djura.hazard.psha import proc_oq_disaggregation, proc_oq_hazard_curve

    # POEs of interest
    # Those are probability of exceedances (POEs) associated with intensity levels
    # used during record selection
    poes = [0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.0025, 0.001]

    # Process the disaggregation excel file from OQ-Engine
    disagg_all = proc_oq_disaggregation(
        path / 'data/psha', poes,
        # out_file=path / 'data/psha/disaggregation.json'
    )

    hz = proc_oq_hazard_curve(poes, path / 'data/psha')

    # SA(0.5) to be used as conditional IM
    imt = "SA(0.5)"

    imls = hz['cond_imls'][imt]

    rs_input = json.load(open(path / "data/djura-conditional-all-input.json"))

    # Some of the following input parameters are taken from the job.ini file
    rs_input["imi"] = [
        'SA(0.05)', 'SA(0.075)', 'SA(0.1)', 'SA(0.15)', 'SA(0.2)',
        'SA(0.25)', 'SA(0.3)', 'SA(0.4)', 'SA(0.5)', 'SA(0.6)',
        'SA(0.7)', 'SA(0.8)', 'SA(0.9)', 'SA(1.0)', 'SA(1.25)',
        'SA(1.5)', 'SA(2.0)', 'SA(2.5)', 'SA(3.0)'
    ]
    rs_input["im_weights"] = [1 / len(rs_input["imi"])] * len(rs_input["imi"])
    rs_input["gmms"] = [
        {"ID": 0, "SA": {"names": ['KothaEtAl2020ESHM20'], "weights": [1]}}
    ]
    # can be read directly from the datastore too
    rs_input["site-parameters"] = {
        "z2pt5": "1.38",
        "z1pt0": "391.34",
        "vs30": "370",
        "xvf": "150",
        "region": 0
    }
    rs_input["poes"] = {}

    for idx, poe in enumerate(poes):
        disagg = disagg_all["imt_disagg"][imt][
            'mag_dist_hazard_contributions'][f'poe_{poe}']
        ruptures = []
        for i in range(len(disagg['mag'])):
            ruptures.append(
                {
                    "ID": i,
                    "gmms": [0],
                    "rjb": disagg['dist'][i],
                    "mag": disagg['mag'][i],
                    "rake": 0.0,
                    "weight": disagg['hz_cont_occ'][i],
                    # "weight": disagg['hz_cont_exc'][i],
                }
            )
        rs_input["poes"][poe] = {
            "ruptures": ruptures,
            "im-star": {"type": imt, "value": imls[idx]},
        }
        # You may save the rs_input variable to a file for each POE
        # This input then may be directly uploaded in
        # https://apps.djura.it/hazard/record-selector/conditional

        # NOTE: however, be careful as browsers might not easily display
        # large amount of data
        # If you encounter issues, feel free to contact us for questions
        # or to help run the API without a UI

    # Most contributing scenarios to select
    # Leave None for all
    n = 10
    if n is not None:
        for poe in poes:
            ruptures = rs_input['poes'][poe]['ruptures']

            # select the most contributing scenarios (unsorted)
            ruptures = heapq.nlargest(n, ruptures, key=lambda x: x['weight'])
            rs_input['poes'][poe]['ruptures'] = ruptures

    # Match the mag and rjb to required parameters
    # Get datastore
    dstore = "2"
    hdf_path = path.parents[3] / f"oqdata/calc_{dstore}.hdf5"
    ctx, oq = get_context_from_dstore(hdf_path, im_ref=imt)

    site_params = {}

    # Based on OQ assignment
    params = np.rec.array(np.concatenate(list(ctx['ctx_by_grp'].values())))

    for param in ctx['site-parameters']:
        site_params[param] = params[param][0]
    rs_input['site-parameters'] = site_params

    for poe in poes:
        ruptures = rs_input["poes"][poe]['ruptures']
        for i, rup in enumerate(ruptures):
            mag = rup['mag']
            rjb = rup['rjb']
            print(f"Target Magnitude: {mag}, and Rjb: {rjb}")

            # Euclidean distance for each recarray row
            mag_normalized = (params['mag'] - params['mag'].min()) / \
                (params['mag'].max() - params['mag'].min())
            rjb_normalized = (params['rjb'] - params['rjb'].min()) / \
                (params['rjb'].max() - params['rjb'].min())

            target_mag_norm = (mag - params['mag'].min()) / \
                (params['mag'].max() - params['mag'].min())
            target_rjb_norm = (rjb - params['rjb'].min()) / \
                (params['rjb'].max() - params['rjb'].min())

            distances = np.sqrt((mag_normalized - target_mag_norm) **
                                2 + (rjb_normalized - target_rjb_norm)**2)

            closest_idx = np.argmin(distances)
            closest_row = params[closest_idx]

            print(f"Closest mag: {closest_row['mag']}")
            print(f"Closest rjb: {closest_row['rjb']}")

            # Required parameters
            req_params = ctx['required-parameters']
            req_params_values = {}
            for param in req_params:
                req_params_values[param] = closest_row[param]

            rs_input["poes"][poe]['ruptures'][i] = {
                **rs_input["poes"][poe]['ruptures'][i], **req_params_values}


    def to_json_serializable(data):
        from numpy import float32, int32, ndarray

        if isinstance(data, dict):
            for key, value in data.items():
                data[key] = to_json_serializable(value)
        elif isinstance(data, list):
            return [to_json_serializable(item) for item in data]
        elif isinstance(data, ndarray):
            return data.tolist()
        elif isinstance(data, float32):
            return float(data)
        elif isinstance(data, int32):
            return float(data)

        return data


    rs_input = to_json_serializable(rs_input)
    with open("filename.json", "w") as json_file:
        json.dump(rs_input['num-components'], json_file)

Processing disaggregation (exact)
---------------------------------

``scripts/process_disagg_exact.py`` shows the exact approach, reading the
full rupture context directly from a datastore. This method is only
available through the API, not the web UI.

.. code-block:: python

    """
    Process Disaggregation outputs from OQ-engine
    Datastore

    This method is not available within the UI of Djura
    But it can be used through the API.

    If you are willing to utilise the API directly without the
    UI, feel free to contact us!

    This is an exact approach
    """

    # flake8: noqa
    from pathlib import Path
    import sys
    import json

    path = Path(__file__).resolve().parent

    sys.path.insert(0, str(path.parent))

    from djura.hazard.dstore import get_context_from_dstore


    # OQ Datastores (.hdf5) are typically located in:
    # C:\Users\<your-username>\oqdata or
    # C:\Users\<your-username>\Documents\oqdata

    # Conditional IM, does not have to match the conditional
    # IM used during conditional selection of OQ engine
    im_ref = "SA(0.5)"

    dstore = "85"
    hdf_path = path.parents[5] / f"oqdata/calc_{dstore}.hdf5"
    ctx, oq = get_context_from_dstore(hdf_path, im_ref=im_ref)

    # You may choose to save the context into a pickle
    # from djura.utilities import export_results

    # export_results(
    #     path / f"ctx_{dstore}",
    #     ctx,
    #     "pickle"
    # )

Checking hazard consistency
---------------------------

``scripts/check_hazard_consistency.py`` parses hazard curves and selected
record response spectra, then assembles the payload expected by the
`hazard consistency app <https://apps.djura.it/hazard/hazard-consistency>`_.

.. code-block:: python

    # flake8: noqa
    from pathlib import Path
    import sys

    path = Path(__file__).resolve().parent

    sys.path.insert(0, str(path.parent))

    from djura.utilities import to_json_serializable
    from djura.record_selector.hzc import prepare_rs_for_hzc
    from djura.hazard.psha import proc_oq_hazard_curve


    # POEs of interest
    # Those are probability of exceedances (POEs) associated with intensity levels
    # used during record selection
    poes = [0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.0025, 0.001]

    # IMs of interest
    imts = ["SA(0.5)"]

    # Conditional IM (for visualisations)
    cond_im = "SA(0.5)"

    # Set the hazard-curves
    hz = proc_oq_hazard_curve(poes, path / 'data/psha',
                              out_file=path / 'data/psha/hazard.json')

    curves = hz['hazard_curves']

    rs = prepare_rs_for_hzc(path / f'data/records', hz["cond_poes"], imts)

    # Finally prepare input that can be used inside hazard-consistency application
    # at https://apps.djura.it/hazard/hazard-consistency

    data = {
        "imts": imts,
        "response_spectra": rs,
        "conditional_poes": hz["cond_poes"],
        "conditional_intensities": hz["cond_imls"][cond_im],
        "investigation_time": hz["investigation_time"],
        "num_im": 500,
        "hazard_curves": curves,
    }

    data = to_json_serializable(data)

Splitting record-selection input per PoE
----------------------------------------

``scripts/rs_input_poe.py`` is a small utility that splits a combined
conditional record-selection input into one file per probability of
exceedance.

.. code-block:: python

    import json
    from pathlib import Path

    path = Path(__file__).parent
    imt = "SA(0.59)"

    # Load JSON as a Python dict
    with open(path / f"data/{imt}.json", "r") as f:
        data = json.load(f)

    poes = data['poes']

    for poe in poes:
        _data = data.copy()

        ruptures = data['poes'][str(poe)]['ruptures']
        im_star = data['poes'][str(poe)]['im-star']  # adjust if key name differs

        _data['ruptures'] = ruptures
        _data['im-star'] = im_star

        output_file = path / f"{imt}_{poe}.json"
        # with open(output_file, 'w') as f_out:
        #     json.dump(_data, f_out, indent=4)

        # print(f"Saved {output_file}")
