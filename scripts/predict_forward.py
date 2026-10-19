"""RamanGPT forward model: structure -> 200-bin Raman spectrum (50-1000 cm^-1).

Loads a trained ALIGNN (alignn_atomwise_pure, output_features=200) from its
config.json + best_model.pt and predicts the spectrum for a POSCAR. Requires the
`alignn` package (the alignn_2026_conn line) and jarvis-tools.
"""

import argparse
import json
import numpy as np
import torch
from jarvis.core.atoms import Atoms
from alignn.graphs import Graph
from alignn.models.alignn_atomwise_pure import (
    ALIGNNAtomWisePure,
    ALIGNNAtomWisePureConfig,
)

GRID = np.linspace(50.0, 1000.0, 200)  # cm^-1


def load_model(config_path, ckpt_path):
    cfg = json.load(open(config_path))
    net = ALIGNNAtomWisePure(ALIGNNAtomWisePureConfig(**cfg["model"]))
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net.load_state_dict(
        sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    )
    net.eval()
    return net, cfg


def predict(net, cfg, atoms):
    g, lg = Graph.atom_dgl_multigraph(
        atoms,
        neighbor_strategy=cfg["neighbor_strategy"],
        cutoff=cfg["cutoff"],
        max_neighbors=cfg["max_neighbors"],
        atom_features=cfg["atom_features"],
        use_canonize=cfg.get("use_canonize", True),
        three_body_cutoff=cfg.get("three_body_cutoff", 4.0),
    )
    cell = torch.tensor(np.array(atoms.lattice_mat)).type(
        torch.get_default_dtype()
    )
    n_out = cfg["model"]["output_features"]
    with torch.no_grad():
        out = net((g, lg, cell))
    cand = out.values() if isinstance(out, dict) else [out]
    for v in cand:
        arr = np.array(v.detach().cpu()).squeeze()
        if arr.ndim == 1 and arr.shape[0] == n_out:
            return arr
    shapes = {
        k: tuple(np.array(v.detach().cpu()).shape)
        for k, v in (out.items() if isinstance(out, dict) else [])
    }
    raise RuntimeError(f"no {n_out}-dim output found; got {shapes}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", required=True, help="trained model config.json"
    )
    ap.add_argument("--model", required=True, help="best_model.pt")
    ap.add_argument("--poscar", required=True)
    a = ap.parse_args()
    net, cfg = load_model(a.config, a.model)
    atoms = Atoms.from_poscar(a.poscar)
    spec = predict(net, cfg, atoms)
    for w, y in zip(GRID, spec):
        print(f"{w:7.1f} {y:.6f}")


if __name__ == "__main__":
    main()
