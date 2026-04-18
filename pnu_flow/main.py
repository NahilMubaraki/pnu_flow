from __future__ import annotations
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import argparse
import json
import os
from datetime import datetime

try:
    from pnu_flow.pipelines.inference_pipeline import query_route
    from pnu_flow.pipelines.training_pipeline import run_training_pipeline
except ImportError:
    from pipelines.inference_pipeline import query_route
    from pipelines.training_pipeline import run_training_pipeline

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PNU-Flow Phase 3 CLI")
    sub = p.add_subparsers(dest="mode", required=True)
    sub.add_parser("train", help="Run full simulation + LSTM training pipeline")
    
    demo = sub.add_parser("demo", help="Train then immediately infer one route")
    demo.add_argument("--from", dest="source", default="Main_Lobby")
    demo.add_argument("--to", dest="destination", default="Grand_Auditorium")
    
    infer = sub.add_parser("infer", help="Run inference (requires prior train run)")
    infer.add_argument("--from", dest="source", required=True)
    infer.add_argument("--to", dest="destination", required=True)
    infer.add_argument("--time", dest="query_time", default=None)
    return p

def _print(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))

def main():
    args = build_parser().parse_args()

    print("Step 1: Updating Graph & Mapping...")

    if args.mode == "train":
        print("Step 2: Running Training Pipeline...")
        _print(run_training_pipeline())

    elif args.mode == "demo":
        print("Step 2: Running Training Pipeline...")
        train_out = run_training_pipeline()
        print("Step 3: Running Demo Inference...")
        infer_out = query_route(source=args.source, destination=args.destination)
        _print({"train": train_out, "infer": infer_out})

    elif args.mode == "infer":
        print("Step 2: Running Inference...")
        qt = datetime.fromisoformat(args.query_time) if args.query_time else None
        _print(query_route(source=args.source, destination=args.destination, query_time=qt))

    print("Process Completed Successfully.")

if __name__ == "__main__":
    main()