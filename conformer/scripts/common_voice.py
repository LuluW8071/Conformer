
"""
Convert a Common Voice corpus (TSV + MP3 clips) into resampled audio and split
the transcripts into train/test JSON manifests for ASR training.
"""

import argparse
import csv
import os
import logging
from multiprocessing.pool import ThreadPool

from sox.core import SoxiError
from tqdm import tqdm

from conformer.scripts.helper import (
    resample,
    entry,
    write,
    split_train_test,
    clean_text,
)

logger = logging.getLogger("CommonVoice Data Prep")

def _output_name(file_name: str, output_format: str) -> str:
    return file_name.rpartition(".")[0] + "." + output_format


def process_row(row, clips_dir, corpus_dir, output_format):
    """Convert one clip and return its manifest entry (or None if unusable)."""
    out_path = os.path.join(clips_dir, _output_name(row["path"], output_format))
    text = clean_text(row["sentence"])
    src = os.path.join(corpus_dir, "clips", row["path"])

    if not os.path.exists(src):
        return None
    if os.path.exists(out_path):                      # already converted
        return entry(out_path, text)

    try:
        resample(src, out_path, mono=False)
    except SoxiError:
        logger.warning(f"Skipping file due to SoxiError: {src}")
        return None

    return entry(out_path, text)


def load_rows(tsv_path):
    with open(tsv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main(args):
    corpus_dir = os.path.dirname(os.path.abspath(args.file_path))
    clips_dir = os.path.abspath(os.path.join(args.save_json_path, "clips"))
    os.makedirs(clips_dir, exist_ok=True)

    rows = load_rows(args.file_path)
    logger.info(f"{len(rows)} files found.")

    if args.convert:
        logger.info(f"Converting MP3 -> {args.output_format.upper()} using {args.num_workers} workers.")
        tasks = [(row, clips_dir, corpus_dir, args.output_format) for row in rows]
        with ThreadPool(args.num_workers) as pool:
            results = tqdm(pool.imap(lambda t: process_row(*t), tasks), total=len(tasks))
            entries = [r for r in results if r is not None]
    else:
        entries = [
            entry(
                os.path.join(clips_dir, _output_name(row["path"], args.output_format)),
                clean_text(row["sentence"]),
            )
            for row in rows
            if os.path.exists(os.path.join(corpus_dir, "clips", row["path"]))
        ]

    logger.info("Creating train and test JSON sets")
    train, test = split_train_test(entries, args.percent, seed=args.seed)
    write(train, os.path.join(args.save_json_path, "train.json"))
    write(test, os.path.join(args.save_json_path, "test.json"))
    logger.info(f"Done! {len(train)} train / {len(test)} test entries.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare a Common Voice corpus for ASR training.")
    p.add_argument("--file_path", required=True,
                   help="path to a .tsv file found in cv-corpus")
    p.add_argument("--save_json_path", required=True,
                   help="output dir for the json manifests")
    p.add_argument("--percent", type=int, default=10,
                   help="percent of clips put into test.json")
    p.add_argument("--convert", default=True, action="store_true",
                   help="convert mp3 to flac/wav")
    p.add_argument("--not-convert", dest="convert", action="store_false",
                   help="skip audio conversion")
    p.add_argument("-w", "--num_workers", type=int, default=2,
                   help="number of worker threads")
    p.add_argument("--output_format", default="flac",
                   help="output audio format (flac or wav)")
    p.add_argument("--seed", type=int, default=None,
                   help="random seed for the train/test split")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())