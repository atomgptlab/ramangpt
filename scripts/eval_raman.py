"""Evaluate the RamanGPT inverse model (Qwen3-4B + LoRA) on the held-out Raman test set.

Greedy-generates a structure from each formula+SG+Raman-peaks prompt, parses it,
and reports RamanGPT-style inverse metrics: parse rate, reduced-formula
consistency, lattice MAE (a,b,c,alpha,beta,gamma), space-group match.
"""

import argparse
import json
import re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from jarvis.core.atoms import Atoms
from jarvis.core.lattice import Lattice
from jarvis.analysis.structure.spacegroup import Spacegroup3D

ALPACA = "### Instruction:\n{}\n### Input:\n{}\n### Output:\n"


def parse_crystal(text):
    """Parse get_crystal_string_t text -> Atoms (a b c / ang / El x y z)."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    abc = [float(x) for x in lines[0].split()[:3]]
    ang = [float(x) for x in lines[1].split()[:3]]
    els, coords = [], []
    for ln in lines[2:]:
        p = ln.split()
        if len(p) < 4:
            continue
        if not re.match(r"^[A-Z][a-z]?$", p[0]):
            continue
        try:
            xyz = [float(p[1]), float(p[2]), float(p[3])]
        except ValueError:
            continue
        els.append(p[0])
        coords.append(xyz)
    if len(els) == 0:
        raise ValueError("no atoms")
    lat = Lattice.from_parameters(
        abc[0], abc[1], abc[2], ang[0], ang[1], ang[2]
    )
    return (
        Atoms(
            lattice_mat=lat.matrix,
            elements=els,
            coords=coords,
            cartesian=False,
        ),
        abc,
        ang,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="unsloth/Qwen3-4B-Base")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", required=True)  # test.json
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.base, dtype=torch.bfloat16, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    recs = json.load(open(a.data))[: a.limit]
    rows = []
    n_parse = n_formula = n_sg = 0
    lat_err = {k: [] for k in ["a", "b", "c", "alpha", "beta", "gamma"]}
    for i, r in enumerate(recs):
        prompt = ALPACA.format(r["instruction"], r["input"])
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=a.max_new,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        gen = tok.decode(
            out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True
        )
        gen = gen.split("###")[0].strip()
        try:
            t_at, t_abc, t_ang = parse_crystal(r["output"])
            p_at, p_abc, p_ang = parse_crystal(gen)
            n_parse += 1
            if (
                p_at.composition.reduced_formula
                == t_at.composition.reduced_formula
            ):
                n_formula += 1
            for k, tv, pv in zip(["a", "b", "c"], t_abc, p_abc):
                lat_err[k].append(abs(tv - pv))
            for k, tv, pv in zip(["alpha", "beta", "gamma"], t_ang, p_ang):
                lat_err[k].append(abs(tv - pv))
            try:
                if (
                    Spacegroup3D(p_at).space_group_number
                    == Spacegroup3D(t_at).space_group_number
                ):
                    n_sg += 1
            except Exception:
                pass
            ok = True
        except Exception as e:
            ok = False
            gen = f"[PARSE_FAIL] {e} :: {gen[:200]}"
        rows.append({"id": r["id"], "ok": ok, "gen": gen})
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(recs)} parsed={n_parse}", flush=True)

    n = len(recs)
    summary = {
        "n": n,
        "parse_rate": round(n_parse / n, 4),
        "reduced_formula_consistency": round(n_formula / n, 4),
        "spacegroup_match": round(n_sg / n, 4),
        "lattice_MAE": {
            k: round(float(np.mean(v)), 4) if v else None
            for k, v in lat_err.items()
        },
    }
    json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=1)
    print("=== RamanGPT inverse eval (greedy) ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
