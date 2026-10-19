"""DiffractGPT fine-tune on the ALIGNN-CSP split, plain transformers + peft.

Follows atomgpt/inverse_models/inverse_models.py (the published DiffractGPT
recipe): unsloth/mistral-7b-bnb-4bit, LoRA r=16 alpha=16 dropout 0 on
q,k,v,o,gate,up,down; Alpaca prompt + "</s>"; loss on every non-pad token;
adamw_8bit, lr 2e-4 linear, 5 warmup steps, batch 2 x grad-accum 4, bf16,
gradient checkpointing. Differences: no unsloth kernels (they do not run with
the transformers version on the GB10 env), and dynamic padding instead of
padding to 2048 (same loss, much less compute).

    python train_dgpt.py --epochs 2 --output runs/dgpt/model
"""

import argparse
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

DATA = Path("/data/kchoudh2/DifferentialXRD/data/diffractgpt_split")
ALPACA = "### Instruction:\n{}\n### Input:\n{}\n### Output:\n{}"
BASE = "unsloth/mistral-7b-bnb-4bit"
BF16_BASE = "unsloth/mistral-7b"  # same Mistral-7B-v0.1 weights, unquantised


def texts(split, limit=None, eos="</s>", data=DATA):
    """Alpaca text + EOS ("</s>" for Mistral, as in atomgpt)."""
    recs = json.loads((Path(data) / f"{split}.json").read_text())[:limit]
    return [
        ALPACA.format(r["instruction"], r["input"], r["output"]) + eos
        for r in recs
    ]


class Encoded(torch.utils.data.Dataset):
    def __init__(self, tok, items, max_len):
        self.ids = [
            tok(
                t, truncation=True, max_length=max_len, add_special_tokens=True
            )["input_ids"]
            for t in items
        ]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i]


def collate(pad_id):
    def fn(batch):
        n = max(len(b) for b in batch)
        ids = torch.full((len(batch), n), pad_id)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for k, b in enumerate(batch):
            ids[k, : len(b)] = torch.tensor(b)
            att[k, : len(b)] = 1
        labels = ids.masked_fill(att == 0, -100)
        return {"input_ids": ids, "attention_mask": att, "labels": labels}

    return fn


class Timer(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        if state.global_step and state.max_steps:
            el = time.time() - self.t0
            eta = (
                el / state.global_step * (state.max_steps - state.global_step)
            )
            print(
                f"step {state.global_step}/{state.max_steps} "
                f"elapsed {el / 3600:.2f} h eta {eta / 3600:.2f} h {logs}",
                flush=True,
            )

    def on_train_begin(self, *a, **k):
        self.t0 = time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--save-steps", type=int, default=2000)
    ap.add_argument(
        "--bf16-base",
        type=int,
        default=0,
        help="1: LoRA on the bf16 base (mistralai weights) instead of 4-bit",
    )
    ap.add_argument("--grad-ckpt", type=int, default=1)
    ap.add_argument(
        "--base",
        default=None,
        help="other base model (bf16), e.g. unsloth/Qwen3-4B-Base",
    )
    ap.add_argument("--backend", choices=["hf", "unsloth"], default="hf")
    ap.add_argument("--data", default=str(DATA), help="dir with train.json")
    args = ap.parse_args()

    lora = dict(
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    name = args.base or (BF16_BASE if args.bf16_base else BASE)
    if args.backend == "unsloth":
        from unsloth import FastLanguageModel

        model, tok = FastLanguageModel.from_pretrained(
            name,
            max_seq_length=args.max_len,
            dtype=torch.bfloat16,
            load_in_4bit=not (args.bf16_base or args.base),
        )
        model = FastLanguageModel.get_peft_model(
            model,
            **lora,
            random_state=3407,
            use_gradient_checkpointing="unsloth" if args.grad_ckpt else False,
        )
    else:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.bfloat16, device_map={"": 0}
        )
        if args.bf16_base or args.base:
            if args.grad_ckpt:
                model.gradient_checkpointing_enable()
                model.enable_input_require_grads()
        else:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=bool(args.grad_ckpt)
            )
        model = get_peft_model(
            model, LoraConfig(task_type="CAUSAL_LM", **lora)
        )
    # Multimodal bases (e.g. Ministral 3) come with an image+text processor;
    # this is text-only training, so use its text tokenizer.
    tok = getattr(tok, "tokenizer", tok)
    tok.pad_token = tok.pad_token or tok.unk_token or tok.eos_token
    train = Encoded(
        tok, texts("train", args.limit, tok.eos_token, args.data), args.max_len
    )
    lens = [len(x) for x in train.ids]
    print(
        f"{name} [{args.backend}]: train {len(train)} records, tokens mean "
        f"{sum(lens) / len(lens):.0f} max {max(lens)}",
        flush=True,
    )
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_steps=5,
        optim="adamw_8bit",
        bf16=True,
        logging_steps=50,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=3407,
        report_to="none",
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train,
        data_collator=collate(tok.pad_token_id),
        callbacks=[Timer()],
    )
    last = sorted(
        Path(args.output).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    trainer.train(resume_from_checkpoint=str(last[-1]) if last else None)
    model.save_pretrained(Path(args.output) / "final")
    tok.save_pretrained(Path(args.output) / "final")


if __name__ == "__main__":
    main()
