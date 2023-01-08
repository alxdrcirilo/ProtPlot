from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.PDB import FastMMCIFParser, PDBList, Polypeptide, Residue


class ProtPlot:
    def __init__(self, pdb: str):
        self._pdb = pdb
        self.data = None
        self.model = None
        self.struct = None
        self.chain1 = None
        self.chain2 = None

        self.path = Path("data")
        if not self.path.exists():
            self.path.mkdir()

        self.fetch()

    def __str__(self):
        s = f"PDB id:\t{self.pdb.upper()}\n"
        s += (len(s) - 1) * "=" + "\n"
        s += f"Model(s):\t{self.models}\n"
        s += f"Chain(s):\t{self.chains}"
        return s

    @property
    def pdb(self):
        return self._pdb

    @pdb.setter
    def pdb(self, value):
        self._pdb = value

    @property
    def chains(self):
        return [chain.get_id() for chain in self.struct.get_chains()]

    @property
    def models(self):
        return [model.get_id() for model in self.struct.get_models()]

    @property
    def local(self):
        return [e for e in self.path.iterdir() if e.is_file()]

    def clean(self):
        if len(self.local) > 1:
            print(f"Cleaned up {len(self.local)} files")
        else:
            print(f"Cleaned up {len(self.local)} file")
        for file in self.local:
            file.unlink()

    def fetch(self):
        """
        Downloads the appropriate pdb structure file from the PDB server or its mirrors.
        """
        pdbl = PDBList(verbose=False)
        pdbl.retrieve_pdb_file(
            pdb_code=self._pdb, pdir=str(self.path), file_format="mmCif"
        )

        fn = str(self._pdb + ".cif").lower()
        fp = self.path / fn
        assert fp.exists()

        parser = FastMMCIFParser(QUIET=True)
        self.struct = parser.get_structure(structure_id=self._pdb.upper(), filename=fp)

    @staticmethod
    def get_dist(residue1, residue2) -> float:
        """
        Returns the C-α distance between two aminoacid residues (in Angstroms).

        :param residue1: residue 1
        :param residue2: residue 2
        :return: distance between two aminoacid residues (in Angstroms)
        """
        try:
            distance = np.linalg.norm(residue1["CA"].coord - residue2["CA"].coord)
        except KeyError:
            distance = pd.NA

        return distance

    def get_seq(self, chain: str, three: bool = False) -> str | list:
        """
        Returns the aminoacid sequence in str|list format (i.e. 1-/3-format).

        :param chain: chain id
        :param three: whether to return the aminoacid sequence in 1- or 3-format
        :return: aminoacid sequence
        """
        assert hasattr(self, "model"), "Data must be parsed first"

        c = self.model[chain]
        residues = [r.get_resname() for r in c.get_residues() if r.id[0] == " "]
        if three:
            return residues
        else:
            return "".join([Polypeptide.three_to_one(s=r) for r in residues])

    def parse(self, model: int, chain1: str, chain2: str):
        """
        Parse MMCIF file.

        :param model: model id
        :param chain1: chain 1 id
        :param chain2: chain 2 id
        """
        self.model = list(self.struct.get_models())[model]
        self.chain1, self.chain2 = chain1, chain2
        c1 = self.model[self.chain1]
        c2 = self.model[self.chain2]

        def is_het(x: Residue) -> bool:
            return True if x.id[0] != " " else False

        d = dict()
        for row, res1 in enumerate(c1):
            for col, res2 in enumerate(c2):
                d[row, col] = {
                    "res1": res1.resname,
                    "res2": res2.resname,
                    "dist": self.get_dist(residue1=res1, residue2=res2),
                    "res1.is_het": is_het(x=res1),
                    "res2.is_het": is_het(x=res2),
                }

        df = pd.DataFrame.from_dict(data=d).T
        self.data = df[(df["res1.is_het"] == 0) & (df["res2.is_het"] == 0)]

    def save(self, filename: str):
        """
        Saves parsed data into a .csv file with the given filename.

        :param filename: filename used for saving the parsed data
        """
        assert isinstance(self.data, pd.DataFrame), "Data must be parsed first"
        self.data.to_csv(path_or_buf=filename, index=False)

    def plot(
        self,
        colormap: str = "binary",
        threshold: (float, float) = None,
        tril: bool = True,
        ip: str = None,
        dpi: int = 100,
    ) -> plt.Figure:
        """
        Plots a protein contact map.

        :param colormap: matplotlib colormap
        :param threshold: minimum and maximum thresholds (Angstrom)
        :param tril: lower triangle of 2D array
        :param ip: interpolation method
        :param dpi: resolution in DPI
        :return: matplotlib.figure.Figure object
        """
        shape = tuple(i + 1 for i in self.data.index[-1])
        array = np.array(self.data["dist"].values.reshape(shape), dtype=float)

        if tril:
            mask = np.tril(m=array, k=-1)
            array = np.ma.array(data=array, mask=mask).T

        if threshold:
            min_thresh, max_thresh = threshold
            array = np.logical_and(array >= min_thresh, array <= max_thresh)

        fig, ax = plt.subplots(nrows=1, ncols=1, dpi=dpi)
        ax.imshow(X=array, cmap=colormap, aspect="equal", interpolation=ip)
        ax.set_xlabel(f"Chain {self.chain1}")
        ax.set_ylabel(f"Chain {self.chain2}")
        ax.set_title(self._pdb.upper())

        if tril:
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

        return fig


pdb_id = "1mbn"
pp = ProtPlot(pdb=pdb_id)
pp.parse(model=0, chain1="A", chain2="A")
print(pp.get_seq(chain="A"))
pp.save(filename=f"{pdb_id}.csv")
figure = pp.plot(colormap="viridis", tril=True, ip="antialiased", dpi=300)
figure.savefig(f"{pdb_id}.png", transparent=True)

print(str(pp))
# pp.clean()
