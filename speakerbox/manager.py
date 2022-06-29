#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
from uuid import uuid4

import fire
from dataclasses_json import DataClassJsonMixin

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


@dataclass
class _TranscriptMeta(DataClassJsonMixin):
    event_id: str
    session_id: str
    session_datetime: datetime
    stored_annotated_transcript_uri: Optional[str] = None


@dataclass
class _TranscriptApplicationReturn(DataClassJsonMixin):
    annotated_transcript_path: str
    transcript_meta: _TranscriptMeta


@dataclass
class _TranscriptApplicationError(DataClassJsonMixin):
    transcript: str
    error: str


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
        accuracy, precision, recall, loss = eval_model(
            dataset["valid"],
            model_name=model_name,
        )
        eval_results_str = (
            f"eval acc: {accuracy:.5f}, pre: {precision:.5f}, "
            f"rec: {recall:.5f}, loss: {loss:.5f}"
        )
        log.info(eval_results_str)

        # Store eval results too
        top_hash = store_model_dir(message=f"{training_start_dt} -- {eval_results_str}")
        log.info(f"Completed storage of model eval results. Result hash: {top_hash}")
        return top_hash

    @staticmethod
    def prepare_train_and_eval(
        prepared_dataset_storage_dir: Union[str, Path] = PREPARED_DATASET_DIR,
        top_hash: Optional[str] = None,
        equalize: bool = False,
        model_name: str = TRAINED_MODEL_NAME,
    ) -> None:
        """
        Runs prepare_dataset and train_and_eval one after the other.

        Parameters are passed down to the appropriate functions.

        See Also
        --------
        SpeakerboxManager.prepare_dataset
            The function to prepare the dataset to be ready for training.
        SpeakerboxManager.train_and_eval
            The function to train and evaluate a model.
        """
        SpeakerboxManager.prepare_dataset(
            prepared_dataset_storage_dir=prepared_dataset_storage_dir,
            top_hash=top_hash,
            equalize=equalize,
        )

        SpeakerboxManager.train_and_eval(
            dataset_dir=prepared_dataset_storage_dir,
            model_name=model_name,
        )

    @staticmethod
    def pull_model(
        top_hash: Optional[str] = None,
        dest: Union[str, Path] = "./",
    ) -> None:
        """
        Pull down a single model.

        Parameters
        ----------
        top_hash: Optional[str]
            Specific model version to pull.
            Default: None (latest)
        dest: Union[str, Path]
            Location to store the model.
            Default: current directory
        """
        from quilt3 import Package

        package = Package.browse(
            TRAINED_MODEL_PACKAGE_NAME,
            S3_BUCKET,
            top_hash=top_hash,
        )
        package["trained-speakerbox"].fetch(dest)

    @staticmethod
    def list_models(n: int = 10) -> None:
        """
        List all stored models.

        Parameters
        ----------
        n: int
            Number of models to check
            Default: 10
        """
        from quilt3 import Package, list_package_versions

        # Get package versions
        lines = []
        versions = list(list_package_versions(TRAINED_MODEL_PACKAGE_NAME, S3_BUCKET))
        checked = 0
        for _, version in versions[::-1]:
            p = Package.browse(
                TRAINED_MODEL_PACKAGE_NAME,
                S3_BUCKET,
                top_hash=version,
            )
            for line in p.manifest:
                message = line["message"]
                lines.append(f"hash: {version} -- message: '{message}'")
                break

            checked += 1
            if checked == n:
                break

        single_print = "\n".join(lines)
        log.info(f"Models:\n{single_print}")

    @staticmethod
    def _pull_or_use_model(model_top_hash: str, model_storage_path: str) -> None:
        # Pull model
        if Path(model_storage_path).exists():
            log.info(f"Using existing model found in directory: '{model_storage_path}'")
        else:
            log.info(
                f"Pulling and using model from hash: '{model_top_hash}' "
                f"(storing to: '{model_storage_path}')"
            )
            SpeakerboxManager.pull_model(
                top_hash=model_top_hash,
                dest=model_storage_path,
            )

    @staticmethod
    def apply_single(
        transcript: Union[str, Path],
        audio: Union[str, Path],
        dest: Optional[Union[str, Path]] = None,
        model_top_hash: str = (
            "453d51cc7006d2ba26640ba91eed67a5f8a9315d7c25d95f81072edb20054054"
        ),
        model_storage_path: str = "trained-speakerbox",
        transcript_meta: Optional[_TranscriptMeta] = None,
        remote_storage_dir: Optional[str] = None,
        fs_kwargs: Dict[str, Any] = {},
    ) -> Union[Path, _TranscriptApplicationReturn, _TranscriptApplicationError]:
        """
        Apply a trained Speakerbox model to a single transcript.

        Parameters
        ----------
        transcript: Union[str, Path]
            The path to the transcript file to annotate.
        audio: Union[str, Path]
            The path to the audio file to use for classification.
        dest: Optional[Union[str, Path]]
            Optional local storage destination
        model_top_hash: str
            The model top hash to pull from remote store and use for annotation.
            Default: 453d51... (highest accuracy model for Seattle to date)
        transcript_meta: Optional[_TranscriptMeta]
            Optional metadata to hand back during return. Used in parallel application.
        remote_storage_dir: Optional[str]
            An optional remote storage dir to store the annotated transcripts to.
            Should be in the form of '{bucket}/{dir}'. The file will be stored with a
            random uuid.
        fs_kwargs: Dict[str, Any]
            Extra arguments to pass to the created file system connection.

        Returns
        -------
        Union[Path, _TranscriptApplicationReturn, _TranscriptApplicationError]
            If transcript_meta was not provided and no errors arose during application,
            only the Path is returned.

            If transcript_meta was provided and no errors arose during application,
            a _TranscriptApplicationReturn is returned that passed back the annotated
            path and the metadata.

            If any error occurs during application, a _TranscriptApplicationError is
            returned.
        """
        from cdp_backend.annotation.speaker_labels import annotate

        # Pull or use model
        SpeakerboxManager._pull_or_use_model(
            model_top_hash=model_top_hash,
            model_storage_path=model_storage_path,
        )

        # Configure destination file
        transcript = Path(transcript)
        if dest is None:
            transcript_name_no_suffix = transcript.with_suffix("").name
            dest_name = f"{transcript_name_no_suffix}-annotated.json"
            dest = transcript.parent / dest_name

        # Dest should always be a path
        dest = Path(dest)

        # Annotate and store
        try:
            annotated_transcript = annotate(
                transcript=transcript,
                audio=audio,
                model=model_storage_path,
            )
        except Exception as e:
            return _TranscriptApplicationError(
                transcript=str(transcript),
                error=str(e),
            )

        # Store and return
        with open(dest, "w") as open_f:
            open_f.write(annotated_transcript.to_json(indent=4))

        # Optionally store to S3
        if remote_storage_dir:
            import s3fs

            fs = s3fs.S3FileSystem(**fs_kwargs)

            # Make remote path
            remote_path = f"{remote_storage_dir}/{uuid4()}.json"
            fs.put_file(str(dest), remote_path)
            log.info(f"Stored '{dest}' to '{remote_path}'")

            # Attach remote path to meta
            if transcript_meta is not None:
                transcript_meta.stored_annotated_transcript_uri = f"s3://{remote_path}"

        # Return simple path (likely single application)
        if transcript_meta is None:
            return dest

        # Return application return (likely batch / parallel apply)
        return _TranscriptApplicationReturn(
            annotated_transcript_path=str(dest),
            transcript_meta=transcript_meta,
        )

    @staticmethod
    def apply_across_cdp_dataset(
        instance: str,
        start_datetime: str,
        end_datetime: str,
        model_top_hash: str = (
            "453d51cc7006d2ba26640ba91eed67a5f8a9315d7c25d95f81072edb20054054"
        ),
        model_storage_path: str = "trained-speakerbox",
        remote_storage_dir: Optional[str] = None,
        fs_kwargs: Dict[str, Any] = {},
    ) -> str:
        """
        Apply a trained Speakerbox model across a large CDP session dataset.

        Parameters
        ----------
        instance: str
            The CDP instance infrastructure slug (i.e. cdp_data.CDPInstances.Seattle).
        start_datetime: str
            The start datetime in ISO format (i.e. "2021-01-01")
        end_datetime: str
            The end datetime in ISO format (i.e. "2021-02-01")
        model_top_hash: str
            The model top hash to pull from remote store and use for annotation.
            Default: 453d51... (highest accuracy model for Seattle to date)
        remote_storage_dir: Optional[str]
            An optional remote storage dir to store the annotated transcripts to.
            Should be in the form of '{bucket}/{dir}'. A directory with the datetime
            of when this function was called will be appended to the path as well.
            For example, when provided the following: 'my-bucket/application-results/'
            the annotated files will ultimately be placed in the directory:
            'my-bucket/application-results/2022-06-29T11:14:42/'.
            Each file will be given a random uuid.
        fs_kwargs: Dict[str, Any]
            Extra arguments to pass to the created file system connection.

        Returns
        -------
        str
            The path to the results parquet file.

        Notes
        -----
        When attempting to use remote storage, be sure to set your `AWS_PROFILE`
        environment variable.
        """
        from datetime import datetime
        from itertools import repeat

        import pandas as pd
        from cdp_data import datasets, instances
        from tqdm.contrib.concurrent import process_map

        if remote_storage_dir:
            # Clean up storage dir tail
            if remote_storage_dir[-1] == "/":
                remote_storage_dir = remote_storage_dir[:-1]

            # Store in directory with datetime of run
            dt = datetime.utcnow().replace(microsecond=0).isoformat()
            remote_storage_dir = f"{remote_storage_dir}/{dt}"
            log.info(f"Will store annotated transcripts to: '{remote_storage_dir}'")

        # Get session dataset to apply against
        ds = datasets.get_session_dataset(
            infrastructure_slug=getattr(instances.CDPInstances, instance),
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            store_transcript=True,
            store_audio=True,
        )

        # Pull or use model
        # Do this now to avoid parallel download problems
        SpeakerboxManager._pull_or_use_model(
            model_top_hash=model_top_hash,
            model_storage_path=model_storage_path,
        )

        # Parallel annotate
        transcript_metas = [
            _TranscriptMeta(
                event_id=r.event.id,
                session_id=r.id,
                session_datetime=r.session_datetime,
            )
            for _, r in ds.iterrows()
        ]

        log.info("Annotating transcripts...")
        annotation_returns = process_map(
            SpeakerboxManager.apply_single,
            ds.transcript_path,
            ds.audio_path,
            repeat(None),
            repeat(model_top_hash),
            repeat(model_storage_path),
            transcript_metas,
            repeat(remote_storage_dir),
            repeat(fs_kwargs),
        )

        # Filter any errors
        errors = pd.DataFrame(
            [
                e.to_dict()
                for e in annotation_returns
                if isinstance(e, _TranscriptApplicationError)
            ]
        )
        results = pd.DataFrame(
            [
                {
                    "annotated_transcript_path": r.annotated_transcript_path,
                    **r.transcript_meta.to_dict(),
                }
                for r in annotation_returns
                if isinstance(r, _TranscriptApplicationReturn)
            ]
        )

        # Log info
        log.info(f"Annotated {len(results)} transcripts; {len(errors)} errored")

        # Store errors to CSV for easy viewing
        # Store results to parquet for fast load
        errors.to_csv("errors.csv", index=False)
        results_save_path = (
            f"results--start_{start_datetime}--end_{end_datetime}.parquet"
        )
        results.to_parquet(results_save_path)

        # Return path to parquet file of results
        return results_save_path


if __name__ == "__main__":
    manager = SpeakerboxManager()
    fire.Fire(manager)
