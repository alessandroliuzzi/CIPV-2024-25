from transformers import BertTokenizerFast, BertForSequenceClassification

checkpoint = "bert-base-multilingual-cased"

# Scarica e salva il tokenizer
tokenizer = BertTokenizerFast.from_pretrained(checkpoint)
tokenizer.save_pretrained("./bert-cache")

# Scarica e salva il modello
model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
model.save_pretrained("./bert-cache")

