# Event-Extraction-from-Cooking-Data

## Scraping
- Change the {{BASE_URL}} variable in scraper.py to contain the url of the page that contains all the recipes you would like to scrape
- in get_url_data() put in the class names for the locations of each item of the recipe (i.e. name, ingredients, procedure)

The data was collected and put into seperate files based on cooking method. The files were then preprocessed to remove any recipes where the actual methodology didn't involve the releveant cooking method keywords

## Question Generation

The code for question generation follows the end-to-end question generation to generate questions just from the context.
This happens after the clustering of recipes using interactive machine learning and human-in-the-loop training.

## Data Preparation
The initial dataset was created in a SQuAD-like dataset, and has to be converted to be fed as input to the T5 transformer.
This is done using the **highlight format**, where the answer span is highlighted within the text with special highlight tokens.

`<hl> 42 <hl> is the answer to life, the universe and everything.`

To create the formatted input, feed the JSON file as the dataset arguement in [data_prep.py](https://github.com/astha-rastogi-1/Event-Extraction-from-Cooking-Data/blob/main/question_generation/data_prep.py)

## Fine-tuning
To fine-tune the T5 transformer on the given dataset, modify the arguments in [training.py](https://github.com/astha-rastogi-1/Event-Extraction-from-Cooking-Data/blob/main/question_generation/training.py) and run it.

To check if the code works, run [model_test.py](https://github.com/astha-rastogi-1/Event-Extraction-from-Cooking-Data/blob/main/question_generation/model_test.py)

