import math
import torch
import pandas as pd

DATA_PATH = "fineweb/tokenized/"
DATA_FILE_COUNT = 257

def getDataPath(num):
    if num >= DATA_FILE_COUNT or num < 0:
        raise ValueError(
            f"Number should be between 0 and {DATA_FILE_COUNT - 1}, get {num} instead"
        )

    return f"{DATA_PATH}{num:03d}_00000.parquet"

class CustomDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_blocks, data_path=None, data_file_count=None):
        self.data_blocks = data_blocks
        self.block_length = len(data_blocks)
        self.data_path = data_path if data_path else DATA_PATH
        self.data_file_count = data_file_count if data_file_count else DATA_FILE_COUNT

    def __iter__(self):
        """
        Handles parallelism, each worker process different files.
        Excess worker will not work.
        """
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        active_workers = (
            num_workers if num_workers < self.block_length else self.block_length
        )
        # extra worker just leave
        if worker_id >= active_workers:
            return

        # every worker process different files
        block_seg = math.floor(self.block_length / active_workers)
        excess_block = self.block_length - block_seg * active_workers
        if worker_id < excess_block:
            block_start = (block_seg + 1) * worker_id
            block_width = block_seg + 1
            data_block = self.data_blocks[block_start : block_start + block_width]
        else:
            block_start = (block_seg) * worker_id + excess_block
            block_width = block_seg
            data_block = self.data_blocks[block_start : block_start + block_width]

        for num in data_block:
            fileName = getDataPath(num)
            dataFrame = pd.read_parquet(fileName)
            for data in dataFrame.itertuples(index=False):
                tokens, attn_mask = data
                yield torch.tensor(tokens), torch.tensor(attn_mask)
