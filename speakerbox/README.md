# Speakerbox

While the [`speakerbox`](https://github.com/CouncilDataProject/speakerbox) repo
contains the code to annotate and train a `speakerbox` model. This repo
and module contains the code to generate the dataset used for larger scale
training and evaluation.

All code is routed through the `manager.py` file which runs Python fire.

### Get Best Model

```bash
python manager.py pull_model --top_hash 453d51cc7006d2ba26640ba91eed67a5f8a9315d7c25d95f81072edb20054054
```

### All Commands

```
python manager.py --help

NAME
    manager.py

SYNOPSIS
    manager.py COMMAND

COMMANDS
    COMMAND is one of the following:

     list_models
       List all stored models.

     prepare_dataset
       Pull and prepare the dataset for training a new model.

     pull_model
       Pull down a single model.

     train_and_eval
       Train and evaluate a new speakerbox model.

     upload_training_data
       Upload data required for training a new model to S3.
```
