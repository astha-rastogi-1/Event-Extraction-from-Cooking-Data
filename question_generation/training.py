from run_qg import run_qg

args_dict = {
    "model_name_or_path": "t5-small",
    "model_type": "t5",
    "tokenizer_name_or_path": "t5_qg_tokenizer",
    "output_dir": "t5-small-qg-hl-test",
    "train_file_path": "data/train_data_e2e.pt",
    "valid_file_path": "data/valid_data_e2e.pt",
    "per_device_train_batch_size": 1, #32
    "per_device_eval_batch_size": 1, #32
    "gradient_accumulation_steps": 8,
    "learning_rate": 1e-4, #1e-3 is giving decent results#1e-4
    "num_train_epochs": 15, # 10
    "seed": 42,
    "do_train": True,
    "do_eval": True,
    "evaluate_during_training": True,
    "logging_steps": 50, #100
    "overwrite_output_dir": True
}

# start training
run_qg(args_dict)
