@dataclass
class DataTrainingArguments:
    def __init__(self):
        self.task = "e2e_qg"
        self.valid_for_qg_only = True
        self.model_type = "t5"
        self.dataset_path = "./data/squad_multitask"  #put the dataset path here
        self.qg_format = "highlight_qg_format"
        self.max_source_length = 512
        self.max_target_length = 512
        self.train_file_name = "train_data_e2e.pt"
        self.valid_file_name = "valid_data_e2e.pt"

class DataProcessor:
    def __init__(self, tokenizer, model_type="t5", max_source_length=512, max_target_length=32):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.model_type = model_type
        self.hl_token = "<hl>"
        
        if model_type == "t5":
            self.sep_token = "<sep>"
  
    def process(self, dataset):
        if self.model_type == "t5":
            dataset = dataset.map(self._add_eos_examples)
        
        dataset = dataset.map(self._add_special_tokens)
        dataset = dataset.map(self._convert_to_features, batched=True)
        
        return dataset
  
    def _add_eos_examples(self, example):
        example['source_text'] = example['source_text'] + " </s>"
        example['target_text'] = example['target_text'] + " </s>"
        return example
  
    def _add_special_tokens(self, example):
        example['source_text'] = example['source_text'].replace("{hl_token}", self.hl_token)    
        example['target_text'] = example['target_text'].replace("{sep_token}", self.sep_token)
        return example
  
    # tokenize the examples
    def _convert_to_features(self, example_batch):
        source_encoding = self.tokenizer.batch_encode_plus(
            example_batch['source_text'],
            max_length=self.max_source_length,
            padding='max_length',
            pad_to_max_length=True,
            truncation=True, 
        )
        target_encoding = self.tokenizer.batch_encode_plus(
            example_batch['target_text'],
            max_length=self.max_target_length,
            padding='max_length',
            pad_to_max_length=True,
            truncation=True, 
        )

        encodings = {
            'source_ids': source_encoding['input_ids'], 
            'target_ids': target_encoding['input_ids'],
            'attention_mask': source_encoding['attention_mask'],
        }

        return encodings

def filter_e2e_qg(example):
    return example['task'] == 'e2e_qg'


TASK_TO_FILTER_FN = {
    'e2e_qg': filter_e2e_qg,
}


def main():
    data_args = DataTrainingArguments()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    if data_args.model_type == 't5':
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        print(tokenizer)

    
    tokenizer.add_tokens(['<sep>', '<hl>'])
    
    train_dataset = nlp.load_dataset(data_args.dataset_path, name=data_args.qg_format, split=nlp.Split.TRAIN, ignore_verifications=True)
    valid_dataset = nlp.load_dataset(data_args.dataset_path, name=data_args.qg_format, split=nlp.Split.VALIDATION)

    processor = DataProcessor(
        tokenizer,
        model_type=data_args.model_type,
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length
    )

    train_dataset = train_dataset.filter(TASK_TO_FILTER_FN[data_args.task])
    valid_dataset = valid_dataset.filter(TASK_TO_FILTER_FN[data_args.task])
    
    train_dataset = processor.process(train_dataset)
    valid_dataset = processor.process(valid_dataset)

    columns = ["source_ids", "target_ids", "attention_mask"]
    train_dataset.set_format(type='torch', columns=columns)
    valid_dataset.set_format(type='torch', columns=columns)

    if data_args.train_file_name is None:
        train_file_name = f"train_data_{data_args.task}_{data_args.qg_format}_{data_args.model_type}.pt"
        train_path = os.path.join("data", train_file_name)

        valid_file_name = f"valid_data_{data_args.task}_{data_args.qg_format}_{data_args.model_type}.pt"
        valid_path = os.path.join("data", valid_file_name)
    else:
        train_path = os.path.join("data", data_args.train_file_name)
        valid_path = os.path.join("data", data_args.valid_file_name)
    
    torch.save(train_dataset, train_path)
    logger.info(f"saved train dataset at {train_path}")
    
    torch.save(valid_dataset, valid_path)
    logger.info(f"saved validation dataset at {valid_path}")
    
    tokenizer_path = f"{data_args.model_type}_qg_tokenizer"
    if not os.path.exists(tokenizer_path):
        os.mkdir(tokenizer_path)
    tokenizer.save_pretrained(tokenizer_path)
    logger.info(f"saved tokenizer at {tokenizer_path}")


if __name__ == "__main__":
    main()
