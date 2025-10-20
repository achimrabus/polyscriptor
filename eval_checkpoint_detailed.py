import argparse
from pathlib import Path
import csv
import torch
from PIL import Image
import pandas as pd
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, AutoTokenizer

def edit_distance(a, b):
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[lb]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", default="./data/efendiev_3")
    p.add_argument("--val_csv", default="val.csv")
    p.add_argument("--model_name", default=None, help="fallback model name for processor/tokenizer")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--out_csv", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VisionEncoderDecoderModel.from_pretrained(args.checkpoint).to(device)

    # robust processor/tokenizer loading: checkpoint -> parent -> provided model_name
    processor = None
    checkpoint_path = Path(args.checkpoint)
    try:
        processor = TrOCRProcessor.from_pretrained(str(checkpoint_path))
    except Exception:
        # try parent directory (maybe processor saved at run output root)
        parent = checkpoint_path.parent
        try:
            processor = TrOCRProcessor.from_pretrained(str(parent))
        except Exception:
            if args.model_name:
                processor = TrOCRProcessor.from_pretrained(args.model_name)
            else:
                raise RuntimeError("No processor found in checkpoint/parent and no --model_name provided")

    # tokenizer fallback
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
    except Exception:
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path.parent))
        except Exception:
            if args.model_name:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
                except Exception:
                    tokenizer = None
            else:
                tokenizer = None

    if tokenizer is not None:
        processor.tokenizer = tokenizer

    df = pd.read_csv(Path(args.data_root) / args.val_csv, names=["image_path","text"], encoding="utf-8")
    batch_size = args.batch_size
    rows_out = []
    model.eval()

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        images = [Image.open(Path(args.data_root) / p).convert("RGB") for p in batch["image_path"]]
        inputs = processor(images=images, return_tensors="pt", padding=True).pixel_values.to(device)
        with torch.no_grad():
            outs = model.generate(inputs, max_length=128, num_beams=1)
        batch_preds = processor.batch_decode(outs, skip_special_tokens=True)
        for img_path, ref, pred in zip(batch["image_path"].astype(str).tolist(), batch_preds, batch["text"].astype(str).tolist()):
            ed = edit_distance(pred, ref)
            cer = ed / max(1, len(ref))
            rows_out.append({"image_path": img_path, "ref": ref, "pred": pred, "edits": ed, "cer": cer})

    total_edits = sum(r["edits"] for r in rows_out)
    total_chars = sum(max(1, len(r["ref"])) for r in rows_out)
    overall_cer = total_edits / total_chars if total_chars > 0 else float("nan")
    print(f"Samples: {len(rows_out)}, Overall CER: {overall_cer:.6f}")

    out_path = Path(args.out_csv) if args.out_csv else Path(args.checkpoint) / "preds.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path","ref","pred","edits","cer"])
        writer.writeheader()
        for r in rows_out:
            writer.writerow({"image_path": r["image_path"], "ref": r["ref"], "pred": r["pred"], "edits": r["edits"], "cer": f"{r['cer']:.6f}"})

    print(f"Wrote per-sample predictions to {out_path.resolve()}")

if __name__ == "__main__":
    main()