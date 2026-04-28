from typing import List
import numpy as np

from .hazard_model import HazardModel


class HazardConsistency:
    # TODO, add support for other types of IMs
    def __init__(
        self,
        conditional_intensities: List,
        conditional_rps: List = None,
        conditional_mafes: List = None,
        conditional_poes: List = None,
        conditional_apoes: List = None,
        investigation_time: float = 50,
    ) -> None:
        """Hazard consistency checks

        Parameters
        ----------
        conditional_intensities : List
            Intensity of the reference ground motion set
        conditional_rps : List, optional
            Return periods, by default None
        conditional_mafes : List, optional
            Mean annual frequency of exceeding an IM value (MAFE),
            by default None
        conditional_poes : List, optional
            Probability of exceedance (POE), by default None
        conditional_apoes : List, optional
            Annual probability of exceedance (APOE), by default None
        investigation_time : float, optional
            Investigation time in years, by default 50
        """
        self.im_ref = conditional_intensities

        if conditional_mafes is None:
            self.h_ref = HazardModel().get_mafe(
                conditional_rps, conditional_poes, conditional_apoes,
                investigation_time
            )
        else:
            self.h_ref = conditional_mafes

    def check(
        self,
        rs_imi_intensities: List,
        num_im: int = 500
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Check hazard consistency

        Parameters
        ----------
        rs_imi_intensities : List
            Intensity measure values of the ground motions with the shape =
            (number of intensity levels, number of ground motions)
        num_im : int, optional
            Number of intensity measures, by default 500

        Returns
        -------
        Dict
            poe_envelope: Envelope of each POE
            im_range: Intensity range
        """
        rs_imi_intensities = np.asarray(rs_imi_intensities)
        im_start = 10 ** np.floor(np.log10(np.min(rs_imi_intensities)))
        im_end = 10 ** np.ceil(np.log10(np.max(rs_imi_intensities)))

        # Remove trailing zeros
        h_ref = np.trim_zeros(self.h_ref, 'b')
        im_ref = self.im_ref[:h_ref.shape[0]]

        # Initialise arrays
        # (number of intensities, number of groud motions)
        num_int, ngms = np.shape(rs_imi_intensities)
        s_range = np.linspace(start=im_start, stop=im_end, num=num_im)
        dh_ind = np.zeros((num_int, num_im))
        dh_env = np.zeros(num_im)

        dh_ref = np.zeros(len(h_ref))
        # Compute the dh_ref
        for p in range(num_int):
            if p == 0:
                k1_u = np.log(h_ref[p] / h_ref[p + 1]) / \
                    np.log(im_ref[p + 1] / im_ref[p])
                k0_u = h_ref[p] * np.power(im_ref[p], k1_u)

                smin_u = im_ref[p] + 0.5 * (im_ref[p + 1] - im_ref[p])
                hmid_u = k0_u * np.power(smin_u, -k1_u)

                dh_ref[p] = h_ref[p] - hmid_u
            elif p == num_int - 1:
                k1_l = np.log(h_ref[p - 1] / h_ref[p]) / \
                    np.log(im_ref[p] / im_ref[p - 1])
                k0_l = h_ref[p - 1] * np.power(im_ref[p - 1], k1_l)

                smin_l = im_ref[p - 1] + 0.5 * (im_ref[p] - im_ref[p - 1])
                hmid_l = k0_l * np.power(smin_l, -k1_l)

                dh_ref[p] = hmid_l - h_ref[p]
            else:
                k1_l = np.log(h_ref[p - 1] / h_ref[p]) / \
                    np.log(im_ref[p] / im_ref[p - 1])
                k0_l = h_ref[p - 1] * np.power(im_ref[p - 1], k1_l)

                smin_l = im_ref[p - 1] + 0.5 * (im_ref[p] - im_ref[p - 1])
                hmid_l = k0_l * np.power(smin_l, -k1_l)

                k1_u = np.log(h_ref[p] / h_ref[p + 1]) / \
                    np.log(im_ref[p + 1] / im_ref[p])
                k0_u = h_ref[p] * np.power(im_ref[p], k1_u)

                smin_u = im_ref[p] + 0.5 * (im_ref[p + 1] - im_ref[p])
                hmid_u = k0_u * np.power(smin_u, -k1_u)

                dh_ref[p] = hmid_l - hmid_u

        for i, s_val in enumerate(s_range):
            for ii in range(num_int):
                dh_ind[ii, i] = dh_ind[ii, i] + dh_ref[ii] * \
                    np.sum(
                        [x > s_val for x in rs_imi_intensities[ii]]
                ) / ngms
            dh_env[i] = np.sum(dh_ind[:, i])

        # TODO, optimise the loops
        # exceedances = np.sum([x > s_range for x in sa_prd], axis=0) / ngms
        # dh_ind = dh_ref[:, np.newaxis] * exceedances

        # # Take the envelope for each poe
        # dh_env = np.sum(dh_ind, axis=0)

        # # Compute SSE
        # dh_env = np.trim_zeros(dh_env, 'b')
        # s_range = s_range[:dh_env.shape[0]]

        # interpolation = interp1d(im_ref, h_ref, bounds_error=False,
        #                          fill_value=np.nan)
        # h_interp = interpolation(s_range)
        # valid_mask = ~np.isnan(h_interp)

        # # TODO, metrics
        # sse = np.sum(np.square(np.log(  # noqa
        #     h_interp[valid_mask]) - np.log(dh_env[valid_mask])))

        dh_env = HazardModel().get_poe(
            mafe=dh_env)

        return {
            "poe_envelope": dh_env,
            "im_range": s_range,
        }
