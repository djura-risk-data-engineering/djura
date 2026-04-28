from pathlib import Path

from djura.record_selection.metrics import hellinger_distance

path = Path(__file__).resolve().parent


class TestMetrics:
    def test_hellinger(self):
        mu1, s1 = 0.4, 0.5
        mu2, s2 = 0.5, 0.7

        h_quad = hellinger_distance(mu1, s1, mu2, s2)
        h_samp = hellinger_distance(mu1, s1, mu2, s2, "sampling")

        print(f"Hellinger distance (quadrature): {h_quad[0]:.6f}")
        print(f"Hellinger distance (sampling): {h_samp[0]:.6f}")
