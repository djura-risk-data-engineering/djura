from djura.record_selection.gsim.models import AristeidouEtAl2024


class TestAristeidou:
    SUGGESTED_LIMITS = {
        'magnitude': [4.5, 7.9],
        'Rjb': [0.0, 299.44],
        'Rrup': [0.07, 299.59],
        'D_hyp': [2.3, 18.65],
        'Vs30': [106.83, 1269.78],
        'mechanism': [0, 4],
        'Z2pt5': [0.0, 7780],
        'Rx': [-297.13, 292.39],
        'Ztor': [0, 16.23]
    }

    def test_suggested_limits(self):
        limits = AristeidouEtAl2024().get_suggested_parameter_limits()

        assert limits == self.SUGGESTED_LIMITS

    def test_supported_ims(self):
        ims = AristeidouEtAl2024().get_supported_ims()

        assert len(ims) == 169
