#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from quilt3 import Package

from speakerbox import preprocess
from speakerbox.datasets import seattle_2021_proto

from .constants import S3_BUCKET, TRAINING_DATA_DIR, TRAINING_DATA_PACKAGE_NAME

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

        # Parse
        p.parse_args(namespace=self)


###############################################################################
# Build package


def prepare_dataset_for_training(top_hash: Optional[str]) -> Path:
    # Setup storage dir
    storage_dir = TRAINING_DATA_DIR.resolve()
    storage_dir.mkdir(exists_ok=True)

    # Pull / prep original Seattle data
    seattle_2021_ds_items = seattle_2021_proto.pull_all_files(
        annotations_dir="training-data/seattle-2021-proto/annotations/",
        transcript_output_dir="training-data/seattle-2021-proto/unlabeled_transcripts/",
        audio_output_dir="training-data/seattle-2021-proto/audio/",
    )

    # Expand annotated gecko data
    seattle_2021_ds = preprocess.expand_gecko_annotations_to_dataset(
        seattle_2021_ds_items,
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
        overwrite=True,
    )

    # Combine into single
    combined_ds = pd.concat([seattle_2021_ds, diarized_ds], ignore_index=True)

    # Generate train test validate splits
    dataset, value_counts = preprocess.prepare_dataset(combined_ds, equalize_data=False)
    log.info(f"Dataset subset value counts:\n{value_counts}")

    # dataset.save_to_disk(SOME_PATH)

    return storage_dir


###############################################################################
# Runner


def main() -> None:
    args = Args()
    prepare_dataset_for_training(top_hash=args.top_hash)


###############################################################################
# Allow caller to directly run this module (usually in development scenarios)

if __name__ == "__main__":
    main()
