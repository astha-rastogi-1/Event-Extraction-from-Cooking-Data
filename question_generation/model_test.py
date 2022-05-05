from pipelines import pipeline

nlp = pipeline("e2e-qg", model="t5-small-qg-hl-test", tokenizer="t5_qg_tokenizer")
nlp("bake the food for 20 minutes.")
