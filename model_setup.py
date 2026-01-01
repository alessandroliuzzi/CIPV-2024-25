from transformers import BertTokenizerFast, BertForSequenceClassification

checkpoint = "bert-base-multilingual-cased"

# Saving the tokenizer
tokenizer = BertTokenizerFast.from_pretrained(checkpoint)
tokenizer.save_pretrained("./bert-cache")

# Saving the model
model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
model.save_pretrained("./bert-cache")

