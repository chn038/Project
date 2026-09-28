# This is NOT a finished project

The code here is just for referencing and should not be used in any way.
Feel free to read or maybe try to run the code.

This project is built with `uv` and `marimo`, check their documents for more information.

## Installation

Run installation with
```
uv sync
```

Please note that this project uses [fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) dataset, specifically, the fineweb-sample-10BT subset.

This repo does not include the dataset nor the script to download the dataset, please load it yourself.

## Update

I am migrating the code from marimo notebook into plain python file.

The file `main.py` is archived and not up-to-date.

## Run

All the configuration lies in `const.py` file, remember, if you run the tokenization, please update the file based on its description.

This project does not contain the fineweb dataset, please download the fineweb dataset yourself via huggingface.

To tokenize dataset:
```
python3 tokenize_dataset.py
```
please update the value in const.py after tokenization is finished.

To run model training:
1. make sure you have dataset download
2. make sure you have tokenized the dataset
3. create the checkpoint folder first:
```
mkdir checkpoints
```
4. run the training script, it will automatically restart from the last checkpoint.
```
python3 model_train.py
```

To check model result (WIP):
please look into the file `model_inspect.py` and write the function you need into main()

To check tokenized result (WIP):
please look into the file `inspect_dataset.py` and write the function you need into main()
