import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from const import CONFIG, CONST
from CustomDataset import getProcessedDataPath


def count_total_token():
    total_size = 0
    for num in tqdm(range(CONST.TOKENIZED_FILE_COUNT)):
        fileName = getProcessedDataPath(num)
        dataFrame = pd.read_parquet(fileName)
        size = dataFrame.size / 2
        total_size += size
    print("Total token count:", total_size * CONFIG.context_size)


def get_samples():
    tokenizer = AutoTokenizer.from_pretrained(CONST.model_name)
    fileName = getProcessedDataPath(0)
    dataFrame = pd.read_parquet(fileName)
    sample_token, _ = dataFrame.iloc[0]
    print(tokenizer.decode(sample_token, skip_special_tokens=False))


def main():
    get_samples()


if __name__ == "__main__":
    main()
