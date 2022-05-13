#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from constants import S3_BUCKET, TRAINING_DATA_DIR, TRAINING_DATA_PACKAGE_NAME
from quilt3 import Package

from speakerbox import preprocess
from speakerbox.datasets import seattle_2021_proto

###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
)
log = logging.getLogger(__name__)


###############################################################################
# Args


class Args(argparse.Namespace):
    def __init__(self) -> None:
        self.__parse()

    def __parse(self) -> None:
        # Setup parser
        p = argparse.ArgumentParser(
            prog="prepare-dataset-for-training",
            description="Pull and prepare the dataset for training a new model.",
        )
        p.add_argument(
            "--top-hash",
            # Generated package hash from upload-dataset-to-s3
            default=None,
            help=(
                "A specific version of the S3 stored data to retrieve. "
                "If none, will use latest."
            ),
        )
        p.add_argument(
            "-e",
            "--equalize",
            "--equalize-data-within-splits",
            dest="equalize",
            action="store_true",
            help="Should the prepared dataset be equalized by label counts.",
        )

        # Parse
        p.parse_args(namespace=self)


###############################################################################
# Build package


def prepare_dataset_for_training(top_hash: Optional[str], equalize: bool) -> Path:
    # Setup storage dir
    storage_dir = TRAINING_DATA_DIR.resolve()
    storage_dir.mkdir(exist_ok=True)

    # Pull / prep original Seattle data
    seattle_2021_proto_dir = storage_dir / "seattle-2021-proto"
    seattle_2021_proto_dir = seattle_2021_proto.unpack(
        dest=seattle_2021_proto_dir,
        clean=True,
    )
    seattle_2021_ds_items = seattle_2021_proto.pull_all_files(
        annotations_dir=seattle_2021_proto_dir / "annotations",
        transcript_output_dir=seattle_2021_proto_dir / "unlabeled_transcripts",
        audio_output_dir=seattle_2021_proto_dir / "audio",
    )

    # Expand annotated gecko data
    seattle_2021_ds = preprocess.expand_gecko_annotations_to_dataset(
        seattle_2021_ds_items,
        audio_output_dir=TRAINING_DATA_DIR / "chunked-audio-from-gecko",
        overwrite=True,
    )

    # Pull diarized data
    package = Package.browse(
        TRAINING_DATA_PACKAGE_NAME,
        S3_BUCKET,
        top_hash=top_hash,
    )

    # Download
    package.fetch(storage_dir)

    # Expand diarized data
    diarized_ds = preprocess.expand_labeled_diarized_audio_dir_to_dataset(
        [
            "training-data/diarized/01e7f8bb1c03/",
            "training-data/diarized/2cdf68ae3c2c/",
            "training-data/diarized/6d6702d7b820/",
            "training-data/diarized/9f55f22d8e61/",
            "training-data/diarized/9f581faa5ece/",
        ],
        audio_output_dir=TRAINING_DATA_DIR / "chunked-audio-from-diarized",
        overwrite=True,
    )

    # Combine into single
    combined_ds = pd.concat([seattle_2021_ds, diarized_ds], ignore_index=True)

    # Generate train test validate splits
    dataset, _ = preprocess.prepare_dataset(
        combined_ds,
        equalize_data_within_splits=equalize,
    )

    return storage_dir


###############################################################################
# Runner


def main() -> None:
    args = Args()
    prepare_dataset_for_training(top_hash=args.top_hash, equalize=args.equalize)


###############################################################################
# Allow caller to directly run this module (usually in development scenarios)

if __name__ == "__main__":
    main()
