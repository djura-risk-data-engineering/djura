"""
Storey-loss-function (SLF) Generator

The tool allows the automatic production of SLFs based on input fragility,
consequence and quantity data.

Considerations for double counting should be done at the input level and the
consequence function should mirror it.

SLF estimation procedure:       Ramirez and Miranda 2009, CH.3 Storey-based
building-specific loss estimation (p. 17)
FEMA P-58 for fragilities:      https://femap58.atcouncil.org/reports
For consequence functions:      https://femap58.atcouncil.org/reports

EDP:    Engineering Demand Parameter
DV:     Decision Variable
DS:     Damage State
"""

from typing import List, Union, Dict
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit
import warnings
from pydantic import TypeAdapter

from .models import (
    ComponentDataModel,
    CorrelationTreeModel,
    FragilityModel,
    DamageStateModel,
    CostModel,
    SimulationModel,
    LossModel,
    FittedLossModel,
    FittingParametersModel,
    SLFModel,
    SLFPGModel,
)
from .utilities import to_json_serializable
from .regression_methods import papadopoulos, weibull

warnings.filterwarnings("ignore")


class SLF:
    """
    Storey-loss-function (SLF) Generator for Storey-Based Loss Assessment

    The tool allows the automatic production of SLFs based on input fragility,
    consequence and quantity data.

    Considerations for double counting should be done at the input level and
    the consequence function should mirror it.

    SLF estimation procedure:       Ramirez and Miranda 2009,
    CH.3 Storey-based building-specific loss estimation (p. 17)
    FEMA P-58 for fragilities:      https://femap58.atcouncil.org/reports
    For consequence functions:      https://femap58.atcouncil.org/reports

    EDP:    Engineering Demand Parameter
    DV:     Decision Variable
    DS:     Damage State
    """

    NEGLIGIBLE = 1e-8

    def __init__(
        self,
        inventory: ComponentDataModel,
        edp: str,
        correlations: CorrelationTreeModel = None,
        edp_range: Union[List[float], np.ndarray] = None,
        edp_bin: float = None,
        do_grouping: bool = True,
        conversion: float = 1.0,
        realizations: int = 20,
        replacement_cost: float = 1.0,
        regression: str = "Weibull",
        storey: Union[int, List[int]] = None,
        directionality: int = None,
        seed: int = None,
        max_psd: float = 10,
        max_pfa: float = 5,
        n_prev: int = 0
    ):
        """initialize storey-loss function (SLF) generator
        TODO, add option for mutually exclusive damage states
        TODO, include possibility of including quantity uncertainties along
        with the mean values

        Parameters
        ----------
        inventory : ComponentDataModel
            Component data inventory
        edp : str
            EDP type:
                'PSD' = peak storey drift
                'PFA' = peak floor acceleration
        correlations: CorrelationTreeModel
            Correlation tree of component data, by default None
        edp_range : Union[List[float], np.ndarray], optional
            EDP range, by default None
        edp_bin : float, optional
            EDP bin size, by default None
        do_grouping : bool, optional
            Perform performance grouping of components or not, by default True
        conversion : float, optional
            Conversion factor from usd to euro, by default 1.0,
            Example: if provided in euro, use 1.0;
                if 1 usd = 0.88euro, use 0.88
                or use 1.0 if ratios are used directly
                However, euro is just a convention, it can be any currency
        realizations : int, optional
            Number of realizations for Monte Carlo method, by default 20
        replacement_cost : float, optional
            Replacement cost of the building (used when normalizing the SLFs),
            by default 1.0
        regression : str, optional
            Regression function to be used,
            currently supports: 'Weibull', 'Papadopoulos', by default 'Weibull'
        storey : Union[int, List[int]], optional
            Storey levels, by default None
        directionality : int, optional
            Directionality, by default None (None means non-directional)
        """

        self.matrix = None
        self.component_groups = None
        self.max_ds = 0
        self.item_ids = set()

        if isinstance(inventory, pd.DataFrame):
            from .utilities import convert_inv
            inventory = convert_inv(inventory)
        if isinstance(correlations, pd.DataFrame):
            from .utilities import convert_corr
            correlations = convert_corr(correlations)

        self.inventory = self._validate_inventory(inventory)
        self.correlations = correlations

        # Engineering demand parameters
        self.edp = edp.lower()
        self.edp_bin = edp_bin
        self.edp_range = edp_range

        self.do_grouping = do_grouping
        self.realizations = int(realizations)
        self.replacement_cost = replacement_cost
        self.regression = regression.lower()

        self.conversion = conversion

        self.storey = storey
        self.directionality = directionality
        self.n_prev = n_prev

        # Get EDP range
        self._define_edp_range(max_psd, max_pfa)

        # Component inventory
        self._get_component_data()

        # Component correlation tree
        if self.correlations is not None and len(self.correlations) > 1:
            self._get_correlation_tree()

        # Grouping components
        self._group_components()

        # Seed
        if seed:
            np.random.seed(int(seed))

    def _validate_inventory(self, inventory):
        inv_ta = TypeAdapter(List[ComponentDataModel])
        inventory = inv_ta.validate_python(inventory)
        return inventory

    def _rearange_component_data(self):
        keys = [item['id'] for item in self.inventory]
        component_data = dict(zip(keys, self.inventory))

        return component_data

    def _define_edp_range(self, max_psd, max_pfa):
        """Define range of engineering demand parameters (EDP)

        Returns
        ------
        ValueError
            If incorrect EDP type is provided, must be 'psd' or 'pfa'
        """
        if self.edp == "idr" or self.edp == "psd":
            # Peak storey drift ratio
            self.edp_bin = self.edp_bin if self.edp_bin is not None \
                else 0.1 / 100
            if self.edp_range is None:
                self.edp_range = np.arange(0, max_psd / 100 + self.edp_bin,
                                           self.edp_bin)
        elif self.edp == "pfa":
            # Peak floor acceleration in [g]
            self.edp_bin = self.edp_bin if self.edp_bin is not None else 0.05
            if self.edp_range is None:
                self.edp_range = np.arange(
                    0, max_pfa + self.edp_bin, self.edp_bin)
        else:
            # New Engineering demand parameters to be added
            raise ValueError("Wrong EDP type, must be 'PSD' or 'PFA'")

        self.edp_range = np.asarray(self.edp_range)
        self.edp_range[self.edp_range == 0] = self.NEGLIGIBLE

    def _get_component_data(self):
        """Gets component information from the user provided .csv file

        Direct manipulation within the .csv file, add new entries with empty
        IDs (the tool will assign the IDs automatically) or select ID manually.
        Newly created entries will not be saved within the database, and will
        be deleted if the .csv file is modified.
        """
        # Validate base fields
        for i, data in enumerate(self.inventory):
            if data.id in self.item_ids:
                raise ValueError("ITEM id must be unique")
            self.item_ids.add(data.id)

            n_ds = data.damage_states
            self.max_ds = max(self.max_ds, n_ds)
            if n_ds != len(data.total_dispesion) != \
                    len(data.repair_cost) != len(data.cost_dispersion):
                raise ValueError(
                    "There must be equal amount of columns: 'median-demand', "
                    "'total-dispersion, 'repair-cost', "
                    "'cost-dispersion', 'best-fit"
                )

            if not data.best_fit or len(data.best_fit) == 0:
                data.best_fit = [None] * n_ds

            if len(data.best_fit) < n_ds:
                data.best_fit += [None] * (n_ds - len(data.best_fit))

            data.best_fit = [
                'normal' if fit is None else fit for fit in data.best_fit]
            data.best_fit = data.best_fit[:n_ds]
            self.inventory[i] = data

    def _group_components(self):
        """Component performance grouping"""
        """
        Update to consider groups based on:
            - inventory.group
            - inventory.component
            - inventory.edp
            - storey    (TODO)
            - directionality    (TODO)
        """

        self.inventory = [model.model_dump(
            by_alias=True) for model in self.inventory]
        component_data = pd.DataFrame.from_dict(self.inventory)
        groups = np.array(component_data["Group"])
        components = np.array(component_data["Component"])

        if not self.do_grouping:
            if components.dtype != "O":
                # Populate with a placeholder
                component_data["Component"].fillna("-1", inplace=True)

            # Populate with a placeholder
            component_data["Group"].fillna(-1, inplace=True)

            # If no performance grouping is done, the EDP value is assigned
            # as the default group tag
            key = component_data["EDP"].iloc[0]
            self.component_groups = {key: component_data}

            return

        groups[groups == None] = 1    # noqa: E711
        component_data["Group"] = groups

        if components.dtype != "O":
            # Populate with a placeholder
            component_data["Component"].fillna("-1", inplace=True)

        unique_groups = np.unique(groups)
        self.component_groups = {}
        for group in unique_groups:
            self.component_groups[group] = component_data[
                (component_data["Group"] == group)
            ]

    def _find_inventory_item_by_name(self, name: str):
        for index, component in enumerate(self.inventory):
            if component.name.lower() == name.lower():
                return component.id

        return None

    def _get_correlation_tree(self) -> np.ndarray[int]:
        """Get correlation tree from .csv file

        Updates
        ----------
        matrix: np.ndarray [number of components x
                            (number of damage states + 2)]
            Correlation table, relationships between Item IDs

        Examples
        ----------
            +------------+-------------+-------------+-------------+
            | Item ID    |Dependant on | MIN DS|DS0  | 	MIN DS|DS1 |
            +============+=============+=============+=============+
            | Item 1     | Independent | Independent | Independent |
            +------------+-------------+-------------+-------------+
            | Item 2     | 1           | Undamaged   | Undamaged   |
            +------------+-------------+-------------+-------------+
            | Item 3     | 1           | Undamaged   | Undamaged   |
            +------------+-------------+-------------+-------------+

            continued...

            +-------------+-------------+-------------+-------------+
            |  MIN DS|DS2 |  MIN DS|DS3 | MIN DS|DS4  | MIN DS|DS5  |
            +=============+=============+=============+=============+
            | Independent | Independent | Independent | Independent |
            +-------------+-------------+-------------+-------------+
            | Undamaged   | DS1         | DS1         | DS1         |
            +-------------+-------------+-------------+-------------+
            | DS1         | DS2         | DS3         | DS3         |
            +-------------+-------------+-------------+-------------+
        """

        min_ds_len = None
        for data in self.correlations:
            CorrelationTreeModel.model_validate(data)

            if not min_ds_len:
                min_ds_len = len(data['MIN DS'])
            else:
                if min_ds_len != len(data['MIN DS']):
                    raise ValueError(
                        "Length of 'MIN DS' for all components must match"
                    )

        if len(self.inventory) != len(self.correlations):
            raise ValueError(
                "[EXCEPTION] Number of items in the correlation tree "
                "and component data should match"
            )

        # Create the correlation matrix
        self.matrix = np.zeros(
            (len(self.inventory), min_ds_len + 2), dtype=int
        )

        for i, data in enumerate(self.correlations):
            if data['id'] not in self.item_ids:
                raise ValueError(
                    f"ITEM: {data['id']}, missing from inventory"
                )

            dependence = data['DEPENDANT ON ITEM'].lower()
            min_ds = np.char.lower(data['MIN DS'])
            self.matrix[i][0] = int(data['id'])
            if dependence == 'independent':
                self.matrix[i][1] = int(data['id'])
                self.matrix[i][2:] = 0
            elif dependence == "" or dependence is None:
                self.matrix[i][1] = int(data['id'])
                self.matrix[i][2:] = 0

            else:
                item_i = self._find_inventory_item_by_name(dependence)
                if item_i is None:
                    raise ValueError(
                        f"Item {dependence} not found in inventory")
                self.matrix[i][1] = item_i
                self.matrix[i][2:][min_ds == "independent"] = 0
                self.matrix[i][2:][min_ds == "undamaged"] = 0
                condition = (min_ds != 'independent') & (min_ds != 'undamaged')
                self.matrix[i][2:][condition] = np.char.replace(
                    min_ds[condition], "ds", "")

        self.matrix[:, 3:] = np.maximum.accumulate(self.matrix[:, 3:], axis=1)

    def fragility_function(
        self,
    ) -> tuple[FragilityModel, np.ndarray, np.ndarray]:
        """Derives fragility functions

        Returns
        -------
        dict, FragilityModel
            Fragility functions associated with each damage state and component
        """
        # Deriving the ordinates of the fragility functions
        fragilities = {"EDP": self.edp_range, "ITEMs": {}}
        for item in self.inventory:
            _id = item['id']

            means = np.array(item['median-demand'])
            covs = np.array(item['total-dispersion'])

            # TODO, ensure that this is correct compared to PACT db
            means = np.exp(
                np.log(means) - 0.5 * np.log(covs ** 2 + 1)
            )
            std = np.log(covs ** 2 + 1) ** 0.5
            std[std <= 0] = 0.01

            frag = norm.cdf(
                np.log(self.edp_range.reshape(-1, 1) / means) / std,
                loc=0, scale=1
            )
            frag[np.isnan(frag)] = 0
            fragilities["ITEMs"][_id] = frag

        return fragilities

    def perform_monte_carlo(
            self, fragilities: FragilityModel) -> DamageStateModel:
        """Performs Monte Carlo simulations and simulates damage state(DS) for
        each engineering demand parameter (EDP) value

        Parameters
        ----------
        fragilities : FragilityModel
            Fragility functions of all components at all DSs

        Returns
        ----------
        DamageStateModel
            Sampled damage states of each component for each simulation
        """
        damage_state = dict()

        # Evaluate the DS on the i-th component for EDPs at the n-th simulation
        for item, frag in fragilities["ITEMs"].items():
            damage_state[item] = dict()
            ds_range = np.arange(0, frag.shape[1] + 1, 1)

            # Simulations
            for n in range(self.realizations):
                random = np.random.rand(len(self.edp_range))
                damage = np.zeros(len(self.edp_range), dtype=int)

                for ds in range(frag.shape[1], 0, -1):
                    upper = frag[:, ds - 1]

                    if ds == frag.shape[1]:
                        damage = np.where(
                            random <= upper, ds_range[ds], damage
                        )
                    else:
                        bottom = frag[:, ds]
                        damage = np.where(
                            (random >= bottom) & (random < upper),
                            ds_range[ds], damage
                        )

                damage_state[item][n] = damage

        return damage_state

    def enforce_ds_dependent(
            self, damage_state: DamageStateModel) -> DamageStateModel:
        """Enforces new DS for each dependent component

        Parameters
        ----------
        damage_state : DamageStateModel
            Sampled damage states of each component for each simulation

        Returns
        ----------
        DamageStateModel
            Sampled DS of each component for each simulation after enforcing
            DS for dependent components if a correlation matrix is provided
        """
        if self.correlations is None or len(self.correlations) < 2:
            return damage_state

        for i in range(self.matrix.shape[0]):
            # Check if component is dependent or independent
            if self.matrix[i][0] != self.matrix[i][1]:
                # -- Component is dependent
                # Causation component ID
                m = self.matrix[i][1]
                # Dependent component ID
                j = self.matrix[i][0]
                # Loop for each simulation
                for n in range(self.realizations):
                    causation_ds = damage_state[m][n]
                    correlated_ds = damage_state[j][n]

                    # Get dependent components DS conditioned
                    # on causation component
                    temp = np.zeros(causation_ds.shape)
                    # Loop over each DS
                    for ds in range(1, self.matrix.shape[1]):
                        temp[causation_ds == ds - 1] = self.matrix[
                            j - 1 - self.n_prev][ds]

                    # Modify DS if correlated component is conditioned on
                    # causation component's DS, otherwise skip
                    damage_state[j][n] = np.maximum(correlated_ds, temp)

        return damage_state

    def calculate_costs(
        self,
        damage_state: DamageStateModel,
    ) -> tuple[CostModel, CostModel, SimulationModel]:
        """Evaluates the damage cost on the individual i-th component at each
        EDP level for each n-th simulation

        Parameters
        ----------
        damage_state : DamageStateModel
            Sampled damage states

        Returns
        ----------
        CostModel
            Total replacement costs in absolute values
        CostModel
            Total replacement costs as a ratio of replacement cost
        SimulationModel
            Repair costs associated with each component and simulation
        """

        component_data = self._rearange_component_data()

        repair_cost = {}

        for item, damage in damage_state.items():
            repair_cost[item] = {}
            num_ds = component_data[item]["damage-states"]
            best_fit_funcs = component_data[item]["best-fit"]
            means = np.array(component_data[item]['repair-cost']) * \
                self.conversion
            covs = component_data[item]['cost-dispersion']

            for n in range(self.realizations):
                for ds in range(num_ds + 1):
                    if ds == 0:
                        repair_cost[item][n] = np.where(
                            damage[n] == ds, ds, -1
                        )

                        continue

                    # best-fit function
                    best_fit = best_fit_funcs[ds - 1].lower()

                    # EDP ID where ds is observed
                    idx_list = np.where(damage[n] == ds)[0]
                    for idx_repair in idx_list:
                        if best_fit == "lognormal":
                            cost = np.random.normal(
                                means[ds - 1],
                                covs[ds - 1]
                                * means[ds - 1],
                            )
                            while cost < 0:
                                std = (
                                    covs[ds - 1]
                                    * means[ds - 1]
                                )
                                m = np.log(
                                    means[ds - 1] ** 2
                                    / np.sqrt(means[ds - 1] ** 2 + std**2)
                                )
                                std_log = np.sqrt(
                                    np.log(
                                        (means[ds - 1] ** 2
                                            + std**2)
                                        / means[ds - 1] ** 2
                                    )
                                )
                                cost = np.random.lognormal(m, std_log)
                        else:
                            cost = np.random.normal(
                                means[ds - 1],
                                covs[ds - 1]
                                * means[ds - 1],
                            )
                            while cost < 0:
                                cost = np.random.normal(
                                    means[ds - 1],
                                    covs[ds - 1]
                                    * means[ds - 1],
                                )

                        repair_cost[item][n][idx_repair] = cost

        # Evaluate the total damage cost multiplying the individual
        # cost by each element quantity
        total_repair_cost = {}
        for item, repairs in repair_cost.items():
            total_repair_cost[item] = {}
            for n in range(self.realizations):
                total_repair_cost[item][n] = (
                    repairs[n] * component_data[item]["Quantity"]
                )

        # Evaluate total loss for the storey segment
        total_loss_storey = {}
        for n in range(self.realizations):
            total_loss_storey[n] = np.zeros(len(self.edp_range))
            for total_repair in total_repair_cost.values():
                total_loss_storey[n] += total_repair[n]

        # Calculate if replCost was set to 0, otherwise use the provided value
        if self.replacement_cost == 0.0 or self.replacement_cost is None:
            raise ValueError(
                "Replacement cost should be a non-negative non-zero value."
            )
        else:
            total_replacement_cost = self.replacement_cost

        total_loss_storey_ratio = {}
        for n in range(self.realizations):
            total_loss_storey_ratio[n] = total_loss_storey[n] / \
                total_replacement_cost

        return total_loss_storey, total_loss_storey_ratio, repair_cost

    def perform_regression(
        self,
        loss: CostModel,
        loss_ratio: CostModel,
        percentiles: List[float] = None,
    ) -> tuple[LossModel, FittedLossModel, FittingParametersModel]:
        """Performs regression and outputs final fitted results as
        storey-loss functions (SLFs)

        Parameters
        ----------
        loss : CostModel
            Total loss for the floor segment in absolute values
        loss_ratio : CostModel
            Total loss for the floor segment as a ratio of replacement cost
        percentiles : List[float], optional
            Percentiles to estimate, by default [0.16, 0.50, 0.84],
            'mean' is always included

        Returns
        ----------
        LossModel
            Loss quantiles in terms of both absolute values and ratio
            to replacement cost
        FittedLossModel
            Fitted loss functions
        FittingParametersModel
            Fitting parameters or each quantiles and mean
        """
        if percentiles is None:
            percentiles = [0.16, 0.50, 0.84]

        # Into a DataFrame for easy access for manipulation
        loss = pd.DataFrame.from_dict(loss)
        loss_ratio = pd.DataFrame.from_dict(loss_ratio)

        losses = {
            "loss": loss.quantile(percentiles, axis=1),
            "loss_ratio": loss_ratio.quantile(percentiles, axis=1),
        }

        mean_loss = np.mean(loss, axis=1)
        mean_loss_ratio = np.mean(loss_ratio, axis=1)
        losses["loss"].loc["mean"] = mean_loss
        losses["loss_ratio"].loc["mean"] = mean_loss_ratio

        # Setting the edp range
        if self.edp == "idr" or self.edp == "psd":
            edp_range = self.edp_range * 100
        else:
            edp_range = self.edp_range

        # Fitting the curve, SLFs
        if self.regression == "weibull":

            def fitting_function(x, a, b, c):
                return weibull(x, [a, b, c])

        elif self.regression == "papadopoulos":

            def fitting_function(x, a, b, c, d, e):
                return papadopoulos(x, [a, b, c, d, e])

        else:
            raise ValueError(f"Regression type {self.regression} "
                             "is not supported...")

        # Fitted loss functions at specified quantiles normalised
        # by the Replacement Cost
        losses_fitted = {}
        fitting_parameters = {}
        for q in percentiles:
            q_key = str(q)
            max_val = max(losses["loss_ratio"].loc[q])
            popt, pcov = curve_fit(
                fitting_function,
                edp_range,
                losses["loss_ratio"].loc[q] / max_val,
                maxfev=10**6,
            )

            losses_fitted[q_key] = fitting_function(edp_range, *popt) * max_val
            # Truncating at zero to prevent negative values
            losses_fitted[q_key][losses_fitted[q_key] <= 0] = 0.0
            fitting_parameters[q_key] = {
                "popt": popt,
                "pcov": pcov,
                "multiplier": max_val,
            }

        # Fitting the mean
        max_val = max(losses["loss_ratio"].loc["mean"])
        popt, pcov = curve_fit(
            fitting_function,
            edp_range,
            losses["loss_ratio"].loc["mean"] / max_val,
            maxfev=10**6,
        )

        losses_fitted["mean"] = fitting_function(edp_range, *popt) * max_val

        fitting_parameters["mean"] = {"popt": popt, "pcov": pcov,
                                      "multiplier": max_val}

        return losses, losses_fitted, fitting_parameters

    def estimate_accuracy(
            self, y: np.ndarray, yhat: np.ndarray) -> tuple[float, float]:
        """Estimate prediction accuracy

        Parameters
        ----------
        y : np.ndarray
            Observations
        yhat : np.ndarray
            Predictions

        Returns
        -------
        (float, float)
            Maximum error in %, and Cumulative error in %
        """
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
        if not isinstance(yhat, np.ndarray):
            yhat = np.asarray(yhat)

        error_max = max(abs(y - yhat) / max(y)) * 100
        error_cum = self.edp_bin * sum(abs(y - yhat) / max(y)) * 100
        return error_max, error_cum

    def _transform_output(
        self,
        losses_fitted: FittedLossModel,
        fitting_parameters: FittingParametersModel,
    ) -> SLFModel:
        """Transforms SLF output to primary attributes supported by
        Loss assessment module

        Parameters
        ----------
        losses_fitted : FittedLossModel
            Fitted loss functions

        Returns
        -------
        SLFModel
            SLF output
        """
        # Avoid negative values and any large values before last zero due to
        # inaccuracies in regression
        zero_indices = np.where(losses_fitted["mean"] <= 0)[0]

        if len(zero_indices) > 0:
            last_index = zero_indices[-1]
            losses_fitted["mean"][: last_index + 1] = 0

        out = {
            "Directionality": self.directionality,
            "Storey": self.storey,
            "edp": self.edp,
            "n_simulations": self.realizations,
            "edp_range": list(self.edp_range),
            "slf": list(losses_fitted["mean"]),
            "fitting_parameters": to_json_serializable(fitting_parameters),
        }

        return out

    def generate_slfs(self) -> Dict[str, SLFPGModel]:
        """Genearte SLFs

        Returns
        -------
        Dict[SLFModel]
            SLFs per each performance group
        """
        out = {}

        # Obtain component fragility and consequence functions
        fragilities = self.fragility_function()

        # Perform Monte Carlo simulations for damage state sampling
        damage_state = self.perform_monte_carlo(fragilities)

        # Populate the damage state matrix for correlated components
        damage_state = self.enforce_ds_dependent(damage_state)

        for group in self.component_groups:
            if self.component_groups[group].empty:
                continue

            # Select component inventory to analyze
            inventory = self.component_groups[group]
            item_ids = list(inventory['id'])
            ds_group = {key: damage_state[key] for key in item_ids}

            if isinstance(group, np.int64):
                group_id = f"{group} - "
            else:
                group_id = ""

            comp_keys = ", ".join(inventory['Component'].unique().tolist())
            edp_keys = inventory['EDP'].iloc[0]

            group_id = f"{group_id}{comp_keys}: {edp_keys}"

            # Calculate the costs
            total, ratio, _ = self.calculate_costs(
                ds_group)

            # Perform regression
            losses, losses_fitted, fitting_parameters = \
                self.perform_regression(total, ratio)

            # Compute accuracy
            error_max, error_cum = self.estimate_accuracy(
                losses["loss_ratio"].loc["mean"], losses_fitted["mean"]
            )

            # Transform output
            out[group_id] = self._transform_output(
                losses_fitted, fitting_parameters
            )
            out[group_id]["group"] = str(group)
            out[group_id]["Component-type"] = comp_keys
            out[group_id]["scatter"] = ratio
            out[group_id]["loss-mean-data"] = list(
                losses["loss_ratio"].loc["mean"])
            out[group_id]["error_max"] = error_max
            out[group_id]["error_cum"] = error_cum
            out[group_id]["regression"] = self.regression

        return out

    def export_to_json(self, out: SLFModel, export_path: Path) -> None:
        import json
        with open(export_path, "w") as json_file:
            json.dump(out, json_file)
