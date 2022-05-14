#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Optional, Union

import fire

###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
)
log = logging.getLogger(__name__)


###############################################################################

TRAINING_DATA_PACKAGE_NAME = "speakerbox/training-data"
TRAINED_MODEL_PACKAGE_NAME = "speakerbox/model"
S3_BUCKET = "s3://evamaxfield-uw-equitensors-speakerbox"
TRAINING_DATA_DIR = Path(__file__).parent / "training-data"
TRAINING_DATA_DIRS_FOR_UPLOAD = [TRAINING_DATA_DIR / "diarized"]
PREPARED_DATASET_DIR = Path(__file__).parent / "prepared-speakerbox-dataset"
TRAINED_MODEL_NAME = "trained-speakerbox"

###############################################################################


class SpeakerboxManager:
    @staticmethod
    def upload_training_data(dry_run: bool = False, force: bool = False) -> str:
        """
        Upload data required for training a new model to S3.

        Parameters
        ----------
        dry_run: bool
            Conduct dry run of the package generation. Will create a JSON manifest file
            of that package instead of uploading.
            Default: False (commit push)
        force: bool
            Should the current repo status be ignored and allow a dirty git tree.
            Default: False (do not allow dirty git tree)

        Returns
        -------
        top_hash: str
            The generated package top hash.

        Raises
        ------
        ValueError
            Git tree is dirty and force was not specified.
        """
        import git
        from quilt3 import Package

        # Report with directory will be used for upload
        log.info(f"Using contents of directories: {TRAINING_DATA_DIRS_FOR_UPLOAD}")

        # Create quilt package
        package = Package()
        for training_data_dir in TRAINING_DATA_DIRS_FOR_UPLOAD:
            package.set_dir(training_data_dir.name, training_data_dir)

        # Report package contents
        log.info(f"Package contents: {package}")

        # Check for dry run
        if dry_run:
            # Attempt to build the package
            top_hash = package.build(TRAINING_DATA_PACKAGE_NAME)

            # Get resolved save path
            manifest_save_path = Path("upload-manifest.jsonl").resolve()
            with open(manifest_save_path, "w") as manifest_write:
                package.dump(manifest_write)

            # Report where manifest was saved
            log.info(f"Dry run generated manifest stored to: {manifest_save_path}")
            log.info(f"Completed package dry run. Result hash: {top_hash}")
            return top_hash

        # Check repo status
        repo = git.Repo(search_parent_directories=True)
        if repo.is_dirty():
            if not force:
                raise ValueError(
                    "Repo has uncommitted files and force was not specified. "
                    "Commit your files before continuing."
                )
            else:
                log.warning(
                    "Repo has uncommitted files but force was specified. "
                    "I hope you know what you're doing..."
                )

        # Get current git commit
        commit = repo.head.object.hexsha

        # Upload
        pushed = package.push(
            TRAINING_DATA_PACKAGE_NAME,
            S3_BUCKET,
            message=f"From commit: {commit}",
        )
        log.info(f"Completed package push. Result hash: {pushed.top_hash}")
        return pushed.top_hash

    @staticmethod
    def prepare_dataset(
        prepared_dataset_storage_dir: Union[str, Path] = PREPARED_DATASET_DIR,
        top_hash: Optional[str] = None,
        equalize: bool = False,
    ) -> Path:
        """
        Pull and prepare the dataset for training a new model.

        Parameters
        ----------
        prepared_dataset_storage_dir: Union[str, Path]
            Directory name for where the prepared dataset should be stored.
            Default: prepared-speakerbox-dataset/
        top_hash: Optional[str]
            A specific version of the S3 stored data to retrieve.
            Default: None (use latest)
        equalize: bool
            Should the prepared dataset be equalized by label counts.
            Default: False (do not equalize)

        Returns
        -------
        dataset_path: Path
            Path to the prepared and serialized dataset.
        """
        import pandas as pd
        from quilt3 import Package

        from speakerbox import preprocess
        from speakerbox.datasets import seattle_2021_proto

        # Setup storage dir
        training_data_storage_dir = TRAINING_DATA_DIR.resolve()
        training_data_storage_dir.mkdir(exist_ok=True)

        # Pull / prep original Seattle data
        seattle_2021_proto_dir = training_data_storage_dir / "seattle-2021-proto"
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
        package.fetch(training_data_storage_dir)

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

        # Store to disk
        dataset.save_to_disk(prepared_dataset_storage_dir)
        return Path(prepared_dataset_storage_dir)

    @staticmethod
    def train_and_eval(
        dataset_dir: Union[str, Path] = PREPARED_DATASET_DIR,
        model_name: str = TRAINED_MODEL_NAME,
    ) -> str:
        """
        Train and evaluate a new speakerbox model.

        Parameters
        ----------
        dataset_dir: Union[str, Path]
            Directory name for where the prepared dataset is stored.
            Default: prepared-speakerbox-dataset/
        model: str
            Name for the trained model.
            Default: trained-speakerbox

        Returns
        -------
        top_hash: str
            The generated package top hash. Includes both the model and eval results.
        """
        import shutil
        from datetime import datetime

        from datasets import DatasetDict
        from quilt3 import Package

        from speakerbox import eval_model, train

        # Record training start time
        training_start_dt = datetime.utcnow().replace(microsecond=0).isoformat()

        # Load dataset
        dataset = DatasetDict.load_from_disk(dataset_dir)

        # Train
        model_storage_path = train(dataset, model_name=model_name)

        # Create reusable model storage function
        def store_model_dir(message: str) -> str:
            package = Package()
            package.set_dir(model_name, model_name)

            # Log contents
            dir_contents = list(Path(model_name).glob("*"))
            log.info(f"Uploading directory contents: {dir_contents}")

            # Upload
            pushed = package.push(
                TRAINED_MODEL_PACKAGE_NAME,
                S3_BUCKET,
                message=message,
                force=True,
            )
            return pushed.top_hash

        # Remove checkpoints and runs subdirs
        shutil.rmtree(model_storage_path / "runs")
        for checkpoint_dir in model_storage_path.glob("checkpoint-*"):
            if checkpoint_dir.is_dir():
                shutil.rmtree(checkpoint_dir)

        # Store model to S3
        top_hash = store_model_dir(
            message=f"{training_start_dt} -- initial storage before eval",
        )
        log.info(f"Completed initial storage of model. Result hash: {top_hash}")

        # Eval
        accuracy, precision, recall = eval_model(
            dataset["valid"],
            model_name=model_name,
        )
        eval_results_str = (
            f"eval acc: {accuracy:.5f}, pre: {precision:.5f}, rec: {recall:.5f}"
        )
        log.info(eval_results_str)

        # Store eval results too
        top_hash = store_model_dir(message=f"{training_start_dt} -- {eval_results_str}")
        log.info(f"Completed storage of model eval results. Result hash: {top_hash}")
        return top_hash


if __name__ == "__main__":
    manager = SpeakerboxManager()
    fire.Fire(manager)
