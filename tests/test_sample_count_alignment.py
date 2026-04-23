import unittest

from blip.src.makeLISAdata import LISAdata


class DummyLISAData(LISAdata):
    def __init__(self, fs, dur):
        super().__init__({"fs": fs, "dur": dur, "tstart": 0.0}, inj={})


class SampleCountAlignmentTests(unittest.TestCase):
    def test_long_paper_run_uses_enough_half_overlap_splices(self):
        """
        The 3-year absolute multipoles paper run that previously failed should
        produce at least the requested number of SGWB samples before trimming.
        """

        data = DummyLISAData(fs=0.25, dur=9.4608e7)

        requested_samples = data.get_requested_num_samples()
        Npersplice, halfN, nsplice = data.compute_half_overlap_splice_setup(1e4)
        produced_samples = nsplice * halfN - halfN

        self.assertEqual(requested_samples, 23652000)
        self.assertEqual(Npersplice, 2500)
        self.assertEqual(halfN, 1250)
        self.assertEqual(nsplice, 18923)
        self.assertGreaterEqual(produced_samples, requested_samples)

    def test_historical_floor_formula_would_underproduce_this_run(self):
        """
        Guard the exact regression: the old splice-count formula stopped 2000
        samples short for the long paper configuration.
        """

        data = DummyLISAData(fs=0.25, dur=9.4608e7)
        requested_samples = data.get_requested_num_samples()
        Npersplice = data.resolve_sample_count(1e4)
        halfN = Npersplice // 2

        old_nsplice = 2 * int(data.params["dur"] / 1e4) + 1
        old_produced_samples = old_nsplice * halfN - halfN

        self.assertEqual(old_nsplice, 18921)
        self.assertEqual(old_produced_samples, 23650000)
        self.assertEqual(requested_samples - old_produced_samples, 2000)


if __name__ == "__main__":
    unittest.main()
