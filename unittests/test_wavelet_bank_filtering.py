import logging
import tempfile
import unittest
from shutil import rmtree

import numpy as np
from scipy.sparse import coo_matrix
from torch_geometric.graphgym import cfg
from torch_geometric.datasets import WebKB
from torch_geometric.transforms import ToDense, ToUndirected
from gtaxogym.transform.perturbations.spectral import WaveletBankFiltering


class TestWaveletBankFilteringWebKB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg.device = "cpu"
        logging.detail = lambda msg: logging.log(15, msg)

        cls.root = tempfile.mkdtemp()
        print(f"Temporary data saved to {cls.root}")

        data = WebKB(root=cls.root, name="Cornell").data
        ToUndirected()(data)

        edges = data.edge_index.cpu().detach().numpy()
        M = coo_matrix((np.ones(edges.shape[1]), edges)).toarray()
        M -= np.diag(M.diagonal())

        d = np.maximum(M.sum(0), 1)
        cls.I = I = np.eye(M.shape[0])
        cls.P_rw = 0.5 * (I + M / d)
        cls.P_sym = 0.5 * (I + (M / np.sqrt(d)).T / np.sqrt(d))

        print(cls.root)
        print(data)

    @classmethod
    def tearDownClass(cls):
        print(f"Removing temporary files in {cls.root}")
        rmtree(cls.root)

    def setUp(self):
        data = WebKB(root=self.root, name="Cornell").data
        ToUndirected()(data)
        self.data = data

    def tearDown(self):
        del self.data

    def test_P_rw(self):
        wbf = WaveletBankFiltering(bands=[0,0,1], norm="rw")
        P_rw = wbf.get_P(self.data).to_dense().cpu().detach().numpy()
        self.assertLess((P_rw - self.P_rw).sum(), 1e-6)

    def test_P_sym(self):
        wbf = WaveletBankFiltering(bands=[0,0,1], norm="sym")
        P_sym = wbf.get_P(self.data).to_dense().cpu().detach().numpy()
        self.assertLess((P_sym - self.P_sym).sum(), 1e-6)

    def test_P_sym_lp(self):
        x_bench = self.P_sym @ (self.P_sym @ self.data.x.cpu().detach().numpy())

        WaveletBankFiltering(bands=[0,0,1], norm="sym")(self.data)
        x_test = self.data.x.cpu().detach().numpy()

        self.assertLess((x_bench - x_test).max(), 1e-5)

    def test_P_sym_hp(self):
        x_bench = ((self.I - self.P_sym) @ self.data.x.cpu().detach().numpy())

        WaveletBankFiltering(bands=[1,0,0], norm="sym")(self.data)
        x_test = self.data.x.cpu().detach().numpy()

        self.assertLess((x_bench - x_test).max(), 1e-5)


if __name__ == '__main__':
    unittest.main()
