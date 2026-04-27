import numpy as np
import re
from .constants import SUPPORTED_IMS


class _Filter:
    """Must be inherited by CodeSelect or GCIMSelect.
    """

    metadata: dict
    """Dictionary containing metadata of ground motion records."""

    def _get_im_component(self, name, num_components, component_definition):
        """Gets IM component values

        Parameters
        ----------
        name : str
            Name of IM
        num_components : int
            Number of components
        component_definition : str
            Component definition, RotD50 or RotD100,
            applicable to SA type IMs only

        Returns
        -------
        np.ndarray
            IM values

        Raises
        ------
        ValueError
            If given IM name is not supported in the metadata
        """
        # TODO: For now SA_vert is not available for GCIM.
        if name == 'SA_vert' and type(self).__name__ == "CodeSelect":
            pass
        elif name not in SUPPORTED_IMS:
            raise ValueError(
                f"Intensity measure (IM) {name} is not supported. "
                f"Supported IMs include: {SUPPORTED_IMS}"
            )

        if name.startswith("SA") and component_definition:
            component_definition = component_definition.lower()
        elif component_definition:
            component_definition = "geomean"

        im1 = f"{name}_1"
        im2 = f"{name}_2"
        # im_vert = f"{name}_vert"

        if im1 not in self.metadata and num_components in [2, 3]:
            # ims for vertical components or ims with no component definition
            return self.metadata[name]

        if im1 not in self.metadata and num_components == 1:
            # ims with no component definition
            return np.append(self.metadata[name], self.metadata[name], axis=0)

        if num_components == 1:
            im_vals = np.append(
                self.metadata[im1], self.metadata[im2], axis=0)

            return im_vals

        # 2 components
        if component_definition == "geomean":
            im_vals = np.sqrt(self.metadata[im1] * self.metadata[im2])
        elif component_definition == 'srss':
            im_vals = np.sqrt(self.metadata[im1] ** 2
                              + self.metadata[im2] ** 2)
        elif component_definition == 'arithmeticmean':
            im_vals = (self.metadata[im1] + self.metadata[im2]) / 2
        elif component_definition == 'rotd50':
            im_vals = self.metadata[f'{name}_RotD50']
        elif component_definition == 'rotd100':
            im_vals = self.metadata[f'{name}_RotD100']
        else:
            raise ValueError(
                f"Wrong component definition: {component_definition}")

        return im_vals

    def _get_imi_database(
            self, imi, num_components, component_definition):
        """Loops for each IMi and gets the corresponding values from the
        metadata

        Parameters
        ----------
        imi : dict
            IMis
        num_components : int
            Number of components
        component_definition : str
            Component definition

        Returns
        -------
        dict
            IMis and corresponding values
        """
        im_known = {}
        for name in imi.keys():
            im_known[name] = self._get_im_component(
                name, num_components, component_definition)

        return im_known

    def _analyze_database(self, unique_key, num_components,
                          component_definition, context_limits, imi):
        """Analyze the metadata file

        Parameters
        ----------
        unique_key : str
            Unique key distinguishing the ground motion record
        num_components : int
            Number of components of ground motion to consider
        component_definition : str
            Component definition
        context_limits : dict
            Limits on causal context
        imi : dict
            IMi types and corresponding periods where applicable

        Returns
        -------
        dict
            IMi types and corresponding values queried from the metadata
        np.ndarray
            Unique identifiers of ground motion records
        dict
            Causal context from metadata
        np.ndarray
            Filename of record in 1st primary direction
        np.ndarray
            Filename of record in 2nd primary direction
        np.ndarray
            Earthquake identifier, an earthquake might have multiple
            ground motion recordings
        Raises
        ------
        ValueError
            If number of components is neither 0 or 1
        """
        # sa_known is from arbitrary ground motion component
        filename2 = None
        context = {}
        if num_components == 1:

            filename1 = np.append(
                self.metadata['Filename_1'], self.metadata['Filename_2'],
                axis=0)
            eq_id = np.append(
                self.metadata['EQID'], self.metadata['EQID'], axis=0)

            rsn = np.append(
                self.metadata[unique_key], self.metadata[unique_key], axis=0)

            for key, val in context_limits.items():
                if val is None or key not in self.metadata.keys() \
                        or all(_val is None for _val in val):
                    continue

                context[key] = np.append(
                    self.metadata[key], self.metadata[key], axis=0)

        elif num_components in [2, 3]:

            component_definition = component_definition.lower()
            filename1 = self.metadata['Filename_1']
            filename2 = self.metadata['Filename_2']
            eq_id = self.metadata['EQID']
            rsn = self.metadata[unique_key]

            for key, val in context_limits.items():
                if val is None or key not in self.metadata.keys() \
                        or all(_val is None for _val in val):
                    continue

                context[key] = self.metadata[key]

        im_known = self._get_imi_database(
            imi, num_components, component_definition)

        return im_known, rsn, context, filename1, filename2, eq_id

    @staticmethod
    def _create_mask_for_strings(arr, search_string):
        """Create a boolean mask where True indicates the element contains
        any of the search terms (case-insensitive).

        Parameters
        ----------
        arr : np.ndarray
            numpy array of strings and/or numbers
        search_string : string
            String with terms separated by semicolon, or period
        """
        search_terms = [
            term.strip().lower()
            for term in re.split('[;.]', search_string)
            if term.strip()
        ]

        # convert array to lowercase strings for comparison
        arr_lower = np.char.lower(arr.astype(str))

        # Create the mask by checking if any search
        # term is in each element
        mask = np.zeros(len(arr), dtype=bool)
        for term in search_terms:
            found = np.char.find(arr_lower, term) == 0
            mask = mask | found

        return mask

    def _limit_context(self, context, context_limits, im_known, rsn):
        """Create a list of RSNs of ground motion records to disregard
        during selection based on causal context limits imposed

        Parameters
        ----------
        context : dict
            Causal context
        context_limits : dict
            Causal context limits
        im_known : dict
            IMis and respective values for each ground motion
        rsn : np.ndarray
            Ground motion unqiue identifiers

        Returns
        -------
        List
            Ground motion with unqiue identifiers to be ignored
        """
        not_allowed = []
        mask = np.zeros(len(rsn), dtype=bool)
        for im_vals in im_known.values():
            not_allowed.extend(np.unique(np.where(im_vals <= 0)[0]).tolist())

        for name, val in context.items():
            _limit = context_limits[name]

            if len(_limit) == 0:
                continue

            if name == "mechanism":
                _mech_values = np.array(
                    [_mech['value'] for _mech in _limit], dtype=np.int64)
                mask = np.isin(val.astype(np.int64), _mech_values)
            elif name == "EQ_name":
                # Ignore Earthquakes that match the key value provided
                mask = self._create_mask_for_strings(
                    val, _limit
                )
                mask = ~mask
            else:
                if _limit[0] == _limit[1] == "":
                    continue

                if _limit[0] == "" or _limit[0] is None:
                    _limit[0] = 0

                if _limit[1] == "" or _limit[1] is None:
                    _limit[1] = 1e5

                _limit = np.array(_limit, dtype=float)
                mask = (val >= min(_limit)) * (val <= max(_limit))

            temp = np.where(~mask)[0]
            not_allowed.extend(temp)

        return not_allowed

    def _filter_database(self, imi, num_records, context_limits,
                         num_components, component_definition):
        """Searches the database and does the filtering

        Parameters
        -------
        imi : dict
            IMi types and corresponding periods where applicable
        num_records : int
            Number of records to be selected
        context_limits : dict
            Limits on causal context
        num_components : int
            Number of components of ground motion to consider
        component_definition : str
            Component definition

        Returns
        -------
        dict
            IMi types and corresponding values queried from the metadata
        dict
            Causal context from metadata
        np.ndarray
            Filename of record in 1st primary direction
        np.ndarray
            Filename of record in 2nd primary direction
        np.ndarray
            Unique identifiers of ground motion records (RSNs)
        np.ndarray
            Earthquake identifier, an earthquake might have multiple
            ground motion recordings
        np.ndarray
            Allowed record indices, unique indices of records allowed
            for the selection

        Raises
        ------
        ValueError
            Unexpected Sa definition, exiting... Wrong spectrum definition
        ValueError
            Wrong number of components. Selection can only be performed
            for one or two components at the moment, exiting...
        ValueError
            NaNs found in input response spectra
        ValueError
            There are not enough records which satisfy, the given record
            selection criteria...Please broaden your selection criteria...
        """

        unique_key = "RSN"

        im_known, rsn, context, filename1, filename2, eq_id = \
            self._analyze_database(
                unique_key, num_components, component_definition,
                context_limits, imi)

        # Limiting the records to be considered using the
        # `not_allowed' variable
        # IM values cannot be negative or zero, remove those
        not_allowed = self._limit_context(
            context, context_limits, im_known, rsn)

        # Initialize indices for all available records
        all_indexes = set(range(len(rsn)))

        # get the unique values
        not_allowed = set(not_allowed)

        # Allowed set of indices
        allowed = np.array(list(all_indexes - not_allowed))

        # Use only allowed records
        for key, val in im_known.items():
            if len(val.shape) > 1:
                im_known[key] = val[allowed, :]
            else:
                im_known[key] = val[allowed]

        for key, val in context.items():
            context[key] = val[allowed]

        eq_id = eq_id[allowed]
        filename1 = filename1[allowed]
        rsn = rsn[allowed]

        if filename2 is not None:
            filename2 = filename2[allowed]

        # Arrange the available spectra in a usable format and check
        # for invalid input
        # Match periods (known periods and periods for error computations)
        im_known = self._parse_imi_of_interest(im_known, imi)

        if num_records > len(eq_id):
            raise ValueError('There are not enough records which satisfy',
                             'the given record selection criteria...',
                             'Please use broaden your selection criteria...')

        return im_known, context, filename1, filename2, rsn, eq_id, allowed

    def _parse_imi_of_interest(self, im_known: dict, imi: dict):
        """Select the IMi matching at periods of interest only

        Parameters
        ----------
        im_known : dict
            IMi and respective IM values from metadata
        imi : dict
            IMi and respective periods of interest

        Returns
        -------
        dict
            Reduced IMi and respective IM values from metadata

        Raises
        ------
        ValueError
            NaNs found in input response spectra in metadata file
        """

        for im, vals in im_known.items():
            # Loop over each IM type
            # If vals has two dimensions, then it is period-dependent
            # otherwise, period-independent, so skip
            if len(vals.shape) == 1:
                continue

            if im.startswith("Sa_avg"):
                period_key = "Periods_Sa_avg"
            elif '_vert' in im:
                period_key = "Periods_" + im.split('_vert')[0]
            else:
                period_key = f"Periods_{im}"

            im_idx = []
            for period in imi[im]:
                if period == 0.0:
                    period = min(self.metadata[period_key])

                period = np.round(period, 5)
                im_idx.append(
                    np.where(self.metadata[period_key] == period)[0][0])

            im_known[im] = vals[:, im_idx]

            # Check for invalid input
            if np.any(np.isnan(im_known[im])):
                raise ValueError('NaNs found in input response spectra')

        return im_known
