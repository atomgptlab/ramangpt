"""RamanGPT inference: formula + space group + Raman peaks -> crystal structure.

Works with any causal-LM base + a trained LoRA adapter (or a plain base).
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ALPACA = "### Instruction:\n{}\n### Input:\n{}\n### Output:\n"
INSTR = "Below is a description of a material."


def build_input(formula, sg, peaks):
    """peaks: e.g. '523 cm-1(1.0), 611 cm-1(0.34), 1080 cm-1(0.5)'."""
    sg_line = f"The space group number is: {sg}.\n" if sg else ""
    return (
        f"The chemical formula is: {formula}.\n"
        f"{sg_line}"
        f"The Raman spectrum shows main peaks at: {peaks}.\n"
        "Generate atomic structure description with lattice lengths, angles, "
        "coordinates and atom types."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        default="unsloth/mistral-7b-bnb-4bit",
        help="any HF causal-LM base",
    )
    ap.add_argument(
        "--adapter",
        default="knc6/ramangpt_dft_model",
        help="LoRA adapter repo/path; '' to use the raw base",
    )
    ap.add_argument("--formula", required=True)
    ap.add_argument("--sg", default="", help="space group number (optional)")
    ap.add_argument(
        "--peaks", required=True, help="'523 cm-1(1.0), 611 cm-1(0.34), ...'"
    )
    ap.add_argument("--max-new", type=int, default=512)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.base, dtype=torch.bfloat16, device_map="auto"
    )
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    prompt = ALPACA.format(INSTR, build_input(a.formula, a.sg, a.peaks))
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=a.max_new,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    gen = (
        tok.decode(
            out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True
        )
        .split("###")[0]
        .strip()
    )
    print(gen)


if __name__ == "__main__":
    main()
