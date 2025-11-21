#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch
from export_broadcast_static import (
    load_model_flexible,
    export_encoder,
    export_decoder_broadcast,
    write_configs,
)


def main():
    ap = argparse.ArgumentParser(description='Export broadcast encoder/decoder ONNX (no quant)')
    ap.add_argument('checkpoint', type=str)
    ap.add_argument('output_dir', type=str)
    ap.add_argument('--opset', type=int, default=17)
    ap.add_argument('--d_model', type=int)
    ap.add_argument('--nhead', type=int)
    ap.add_argument('--num_encoder_layers', type=int)
    ap.add_argument('--num_decoder_layers', type=int)
    ap.add_argument('--dim_feedforward', type=int)
    ap.add_argument('--max_seq_len', type=int)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model, config, _ = load_model_flexible(ckpt, args)
    enc_p = out / 'encoder.onnx'
    dec_p = out / 'decoder.onnx'
    export_encoder(model, enc_p, args.opset)
    export_decoder_broadcast(model, dec_p, args.opset)
    write_configs(out, config)
    print('Wrote', enc_p, dec_p)


if __name__ == '__main__':
    main()

