import pandas as pd
from tqdm import tqdm
from const import CONST, CONFIG
from CustomDataset import getProcessedDataPath
total_size = 0
for num in tqdm(range(CONST.TOKENIZED_FILE_COUNT)):
    fileName = getProcessedDataPath(num)
    dataFrame = pd.read_parquet(fileName)
    size = dataFrame.size / 2
    total_size += size
print("Total token count:", total_size*CONFIG.context_size)
