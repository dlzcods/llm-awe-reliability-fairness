#!/usr/bin/env python3
"""Simple CLI entrypoint exposing common commands for preprocess and evaluate."""
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from preprocess import main as preprocess_main
from evaluate import main as evaluate_main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['preprocess', 'evaluate'])
    args, rest = parser.parse_known_args()
    if args.command == 'preprocess':
        preprocess_main()
    elif args.command == 'evaluate':
        evaluate_main()


if __name__ == '__main__':
    main()
