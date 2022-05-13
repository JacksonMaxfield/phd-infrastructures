# Speakerbox

While the [`speakerbox`](https://github.com/CouncilDataProject/speakerbox) repo
contains the code to annotate and train a `speakerbox` model. This repo
and module contains the code to generate the dataset used for larger scale
training and evaluation.

All code is routed through the `speakerbox-manager.py` file which runs Python fire.

```
python speakerbox-manager.py --help

NAME
    manager.py

SYNOPSIS
    manager.py COMMAND

COMMANDS
    COMMAND is one of the following:

     prepare_dataset
       Pull and prepare the dataset for training a new model.

     train_and_eval
       Train and evaluate a new speakerbox model.

     upload_training_data
       Upload data required for training a new model to S3.
```
