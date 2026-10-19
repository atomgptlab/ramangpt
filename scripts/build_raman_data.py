"""RamanGPT inverse dataset (DiffractGPT text format) from the precomputed Raman spectra.

Mirrors atomgpt's make_diffractgpt_prompt output exactly (instruction/input/id/
output, get_crystal_string_t), but:
  - peaks come from the stored 200-bin Raman spectrum (50-1000 cm^-1), not XRD,
  - peak label is "<cm^-1> cm-1(<rel intensity>)",
  - a space-group hint line is added to the input (user request).

Split = the forward Raman model's ids_train_val_test.json so train/val/test match
the leaderboard irdb_ir benchmark. Output records go to data/raman_split/.
"""

import json
import os
import numpy as np
from scipy.signal import find_peaks
from jarvis.core.atoms import Atoms
from jarvis.analysis.structure.spacegroup import Spacegroup3D

Raman = "/data/kchoudh2/work/alignn2026/raman/train_raman"
OUT = "/data/kchoudh2/RamanGPT/data/raman_split"
GRID = np.linspace(
    50.0, 1000.0, 200
)  # cm^-1, matches irdb build (D=200, 50-1000)
NUM_PEAKS = 20


def get_crystal_string_t(atoms):
    lengths = atoms.lattice.abc
    angles = atoms.lattice.angles
    atom_ids = atoms.elements
    frac_coords = atoms.frac_coords
    return (
        " ".join(["{0:.2f}".format(x) for x in lengths])
        + "\n"
        + " ".join([str(int(x)) for x in angles])
        + "\n"
        + "\n".join(
            [
                str(t) + " " + " ".join(["{0:.3f}".format(x) for x in c])
                for t, c in zip(atom_ids, frac_coords)
            ]
        )
    )


def ir_peak_text(ir):
    y = np.asarray(ir, dtype=float)
    m = y.max()
    if m > 0:
        y = y / m
    peaks, props = find_peaks(y, height=0.01, distance=1, prominence=0.05)
    if len(peaks) == 0:  # fall back to the single largest bin
        peaks = np.array([int(np.argmax(y))])
        props = {"peak_heights": y[peaks]}
    top = peaks[np.argsort(props["peak_heights"])[::-1][:NUM_PEAKS]]
    top = top[np.argsort(GRID[top])]  # sort by position
    return ", ".join(
        f"{GRID[p]:.0f} cm-1({round(float(y[p]), 2)})" for p in top
    )


def make_record(jid, atoms_dict, ir):
    atoms = Atoms.from_dict(atoms_dict)
    formula = atoms.composition.reduced_formula
    try:
        sg = Spacegroup3D(atoms).space_group_number
    except Exception:
        sg = 0
    peak_text = ir_peak_text(ir)
    inp = (
        f"The chemical formula is: {formula}.\n"
        f"The space group number is: {sg}.\n"
        f"The Raman spectrum shows main peaks at: {peak_text}.\n"
        f"Generate atomic structure description with lattice lengths, "
        f"angles, coordinates and atom types."
    )
    return {
        "instruction": "Below is a description of a material.",
        "input": inp,
        "id": jid,
        "output": get_crystal_string_t(atoms),
    }


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    idp = json.load(open(f"{Raman}/id_prop.json"))
    by = {d["jid"]: d for d in idp}
    split = json.load(open(f"{Raman}/results/ids_train_val_test.json"))
    key = {"train": "id_train", "val": "id_val", "test": "id_test"}
    for sp, k in key.items():
        ids = split[k]
        recs = []
        for jid in ids:
            d = by.get(jid)
            if d is None:
                continue
            recs.append(make_record(jid, d["atoms"], d["target"]))
        json.dump(recs, open(f"{OUT}/{sp}.json", "w"))
        print(sp, len(recs))
    # show one sample for inspection
    s = json.load(open(f"{OUT}/test.json"))[0]
    print("=== SAMPLE ===")
    print("INPUT:\n" + s["input"])
    print("OUTPUT:\n" + s["output"])
