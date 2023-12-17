The code for question generation follows the end-to-end question generation to generate questions just from the context.

## Data Preparation
The initial dataset was created in a SQuAD-like dataset, and has to be converted to be fed as input to the T5 transformer.
This is done using the **highlight format**, where the answer span is highlighted within the text with special highlight tokens.

`<hl> 42 <hl> is the answer to life, the universe and everything.`

To create the formatted input, feed the JSON file as the dataset arguement in [data_prep.py](https://github.com/astha-rastogi-1/Event-Extraction-from-Cooking-Data/blob/main/question_generation/data_prep.py)

## Fine-tuning
To fine-tune the T5 transformer on the given dataset, modify the arguments in [training.py](https://github.com/astha-rastogi-1/Event-Extraction-from-Cooking-Data/blob/main/question_generation/training.py) and run it.

To check if the code works, run [model_test.py](https://github.com/astha-rastogi-1/Event-Extraction-from-Cooking-Data/blob/main/question_generation/model_test.py)

