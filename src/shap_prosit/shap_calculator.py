import sys

import os
from typing import Union
import numpy as np
import tensorflow as tf
from dlomix.models import PrositIntensityPredictor
from numpy.typing import NDArray

import shap


class ShapCalculator:
    def __init__(
        self,
        ion: str,
        dset: NDArray,
        bgd: NDArray,
        model: PrositIntensityPredictor,
        max_sequence_length: int = 30,
        max_charge: int = 6,
    ):
        self.val = dset
        self.bgd = bgd
        self.max_len = max_sequence_length
        self.max_charge = max_charge
        self.model = model

        self.bgd_sz = bgd.shape[0]
        self.ion_ind = self.annotations()[ion]
        self.ext = int(ion[1])
        self.fnull = np.array(
            [
                self.model(self.hx(bgd), training=False)[:, self.ion_ind]
                .numpy()
                .squeeze()
                .mean()
            ]
        )

        self.savepep = []
        self.savecv = []

    def annotations(self):
        """Generate dictionary with ion ids.

        Returns:
            dict: Ion ids dictionary.
        """
        ions = {}
        count = 0
        for i in np.arange(1, self.max_len):
            for j in ["y", "b"]:
                for k in [1, 2, 3]:
                    ion = f"{j}{i}+{k}"
                    ions[ion] = count
                    count += 1

        return ions

    @tf.function
    def mask_pep(self, zs, pep, bgd_inds, mask=True):
        out = tf.zeros((tf.shape(zs)[0], tf.shape(pep)[1]), dtype=tf.string)
        if mask:
            ### TF
            ## Collect all peptide tokens that are 'on' and place them in the out tensor
            oneinds = tf.where(zs == 1)
            onetokens = tf.gather_nd(tf.tile(pep, [tf.shape(zs)[0], 1]), oneinds)
            out = tf.tensor_scatter_nd_update(out, oneinds, onetokens)
            ## Replace all peptide tokens that are 'off' with background dataset
            zeroinds = tf.where(zs == 0)
            # Random permutation of BGD peptides
            # randperm = tf.random.uniform_candidate_sampler(
            #    tf.ones((tf.shape(zs)[0], BGD_SZ), dtype=tf.int64),
            #    num_true=BGD_SZ, num_sampled=tf.shape(zs)[0], unique=True, range_max=BGD_SZ
            # )[0][:,None]
            # randperm = tf.random.uniform((tf.shape(zs)[0], 1), 0, BGD_SZ, dtype=tf.int32)
            bgd_ = tf.gather_nd(self.bgd, bgd_inds[:, None])
            # gather specific tokens of background dataset that belong to 'off' inds
            bgd_ = tf.gather_nd(bgd_, zeroinds)
            # Place the bgd tokens in the out tensor
            out = tf.tensor_scatter_nd_update(out, zeroinds, tf.reshape(bgd_, (-1,)))
            # pad c terminus with b''
            inds2 = tf.cast(tf.argmax(tf.equal(out, b""), 1), tf.int32)

            # nok = tf.where(inds2==0)
            # amt = tf.shape(nok)[0] #tf.reduce_sum(tf.cast(nok, tf.int32))
            # inds2 = tf.tensor_scatter_nd_update(
            #    inds2, nok, tf.shape(pep)[1]*tf.ones((amt,), dtype=tf.int32)
            # )
            inds2_ = tf.tile(
                tf.range(self.max_len, dtype=tf.int32)[None], [tf.shape(out)[0], 1]
            )
            inds1000 = tf.where(inds2_ > inds2[:, None])
            out = tf.tensor_scatter_nd_update(
                out, inds1000, tf.fill((tf.shape(inds1000)[0],), b"")
            )
        else:
            out = pep

        # self.savepep.append(out)
        # self.savecv.append(zs)
        return out

    @tf.function
    def hx(self, tokens):
        sequence = tokens[:, :-2]
        collision_energy = tf.strings.to_number(tokens[:, -2:-1])
        precursor_charge = tf.one_hot(
            tf.cast(tf.strings.to_number(tokens[:, -1]), tf.int32) - 1, self.max_charge
        )
        z = {
            "sequence": sequence,
            "collision_energy": collision_energy,
            "precursor_charge": precursor_charge,
        }

        return z

    def ens_pred(self, pep, batsz=100, mask=True):
        # pep: coalition vectors, 1s and 0s; excludes absent AAs
        P = pep.shape

        # Chunk into batches, each <= batsz
        batches = (
            np.split(pep, np.arange(batsz, batsz * (P[0] // batsz), batsz), 0)
            if P[0] % batsz == 0
            else np.split(pep, np.arange(batsz, batsz * (P[0] // batsz) + 1, batsz), 0)
        )

        # Use these indices to substitute values from background dataset
        # - bgd sample is run for each coalition vector
        rpts = P[0] // self.bgd_sz + 1  # number of repeats
        bgd_indices = tf.concat(rpts * [tf.range(self.bgd_sz, dtype=tf.int32)], 0)

        out_ = []
        for I, batch in enumerate(batches):
            # AAs (cut out CE, charge)
            # Absent AAs (all 1s)
            # [CE, CHARGE]
            batch = np.concatenate(
                [
                    batch[:, :-2],
                    np.ones((tf.shape(batch)[0], self.max_len - tf.shape(pep)[1] + 2)),
                    batch[:, -2:],
                ],
                axis=1,
            )
            batch = tf.constant(batch, tf.int32)

            # Indices of background dataset to use for subbing in 0s
            bgd_inds = bgd_indices[I * batsz : (I + 1) * batsz][: tf.shape(batch)[0]]

            # Create 1/0 mask and then turn into model ready input
            inp = self.hx(self.mask_pep(batch, self.inp_orig, bgd_inds, mask))

            # Run through model
            out = self.model(inp, training=False)
            out_.append(out)

        out_ = tf.concat(out_, axis=0)

        return out_

    def score(self, peptide, mask=True):
        shape = tf.shape(peptide)
        x_ = self.ens_pred(peptide, mask=mask)[:, self.ion_ind]
        score = tf.squeeze(x_).numpy()
        if shape[0] == 1:
            score = np.array([score])[None, :]

        return score

    def calc_shap_values(self, index, samp=1000):
        # String array
        inp_orig = self.val[index : index + 1]
        self.inp_orig = inp_orig

        # Peptide length for the current peptide
        pl = sum(inp_orig[0] != "") - 2
        if pl <= self.ext:
            return False

        # Input coalition vector: All aa's on (1) + charge + eV
        # - Padded amino acids are added in as all ones (always on) in ens_pred
        inpvec = np.ones((1, pl + 2))

        # Mask vector is peptide length all off
        # - By turning charge and eV on, I am ignoring there contribution
        maskvec = np.zeros((self.bgd_sz, pl + 2))
        maskvec[:, -2:] = 1

        orig_spec = self.ens_pred(inpvec, mask=False)[:, self.ion_ind]

        # SHAP Explainer
        ex = shap.KernelExplainer(self.Score, maskvec)
        ex.fnull = self.fnull
        ex.expected_value = ex.fnull

        # Calculate the SHAP values
        seq = list(inp_orig.squeeze())
        seqrep = " ".join(seq[:pl]) + " " + " ".join(seq[-2:])
        # print(seqrep)
        inten = float(orig_spec.numpy().squeeze())
        # print("Calculated intensity: %f"%inten)
        # print("fnull: %f"%ex.fnull)
        # print("Expectation value: %f"%ex.expected_value)
        shap_values = ex.shap_values(inpvec, nsamples=samp)

        # for i,j in zip(seq, shap_values.squeeze()[:pl]):
        # print('%c: %10f'%(i,j))
        # print(np.sum(shap_values))

        return {"int": inten, "sv": shap_values.squeeze()[:pl], "seq": seqrep}

    def write_shap_values(self, out_dict, path="output.txt"):
        inten = out_dict["int"]
        shap_values = out_dict["sv"]
        seqrep = out_dict["seq"]
        with open(path, "a", encoding="utf-8") as f:
            f.write(seqrep + " %f\n" % inten)
            f.write(" ".join(["%s" % m for m in shap_values.squeeze()]) + "\n")


def save_shap_values(
    val_data_path: Union[str, bytes, os.PathLike], ion: str, bgd_sz: int = 100
):

    # Shuffle validation dataset and split it in background and validation.
    val_data = np.genfromtxt(val_data_path, delimiter=",")
    perm = np.random.permutation(np.arange(len(val_data)))
    np.savetxt("perm.txt", perm, fmt="%d")
    perm = np.loadtxt("perm.txt").astype(int)
    bgd = val[perm[:bgd_sz]]
    val = val[perm[bgd_sz:]]

    sc = ShapCalculator(ion, val, bgd)

    for INDEX in range(
        int(sys.argv[1]), int(sys.argv[1]) + int(sys.argv[2]), 1
    ):  # val.shape[0], 1):
        print("\r%d/%d" % (INDEX, len(val)), end="\n")
        out_dict = sc.calc_shap_values(INDEX, samp=1000)
        if out_dict != False:
            sc.write_shap_values(out_dict)


if __name__ == "__main__":
    save_shap_values(val_data_path="val_inps.csv", ion="b7+1")