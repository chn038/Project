class CONFIG:
    train_test_ratio = 0.9
    train_step = 1000
    test_steps = 10
    epoch = 30
    buffer_size = int(1e4)
    worker_count = 1
    batch_size = 1
    context_size = 32768
    beta = 0.5
    segment_length = 2048
    lr = 5e-5
    warmup_step = 10000
    gradient_accumulation_step = 4
    chunk_size = 65536

class CONST:
    model_name = "google/gemma-3-270m-it"
    DATA_FILE_COUNT = 15
    TOKENIZED_FILE_COUNT = 43
    DATA_PATH = "fineweb/sample/10BT/"
    PROCESSED_PATH = "fineweb/tokenized/"
    CHECKPOINT_PATH = "checkpoints/"
