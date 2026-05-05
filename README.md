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

## Run

### Caution

Due to the design of marimo, once starting up the script, **THE CODE WILL AUTOMATICALLY RUN**.

DO NOT run without looking into the `main.py` file first, it's a bare python script with marimo module, but it can still be run with `python3 main.py`

To view the notebook, run:
```
marimo edit main.py
```
