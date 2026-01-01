
# FULL CHAT + SINGLE MESSAGE CLASSIFIER (WITH CONTEXT)

import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from tabulate import tabulate



# 1. Load dataset
df = pd.read_csv("data/dataset.csv", sep=";", encoding='latin1')
df["toxic"] = df["toxic"].map({"Sì": 1, "no": 0})
df["conversation_id"] = df.index
assert df["toxic"].isna().sum() == 0, "Error in label mapping"

# 2. Train/val/test split
train_df, temp_df = train_test_split(
    df,
    stratify=df["toxic"],
    test_size=0.3,
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    stratify=temp_df["toxic"],
    test_size=0.5,
    random_state=42
)



# 3. HuggingFace datasets
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
    "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
    "test": Dataset.from_pandas(test_df.reset_index(drop=True))
})


# 4. Tokenizer (BERT)
checkpoint = "./bert-cache"  # local cache directory
tokenizer = BertTokenizerFast.from_pretrained(checkpoint, local_files_only=True)

def tokenize_fn(batch):
    return tokenizer(batch["conversation"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_fn, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("toxic", "labels")
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 5. Model
model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# 6. Metrics
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# 7. Training arguments
training_args = TrainingArguments(
    output_dir="./output",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none", 
)


# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# 9. Training
trainer.train()


# 10. Evaluation on test set
print("\n--- Evaluation on test set ---")
eval_results = trainer.evaluate(tokenized_datasets["test"])
for k, v in eval_results.items():
    print(f"{k}: {v:.4f}")

# 11. Confusion matrix and classification report
predictions = trainer.predict(tokenized_datasets["test"])
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["non-toxic", "toxic"]))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["non-toxic", "toxic"],
            yticklabels=["non-toxic", "toxic"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — Full Chat Classifier")
plt.show()



# 12. SINGLE MESSAGE CLASSIFIER WITH CONTEXT
def split_conversation(conv):
    if isinstance(conv, str):
        parts = [p.strip() for p in conv.split("  ") if len(p.strip()) > 0]
        return parts
    elif isinstance(conv, list):
        return conv
    else:
        return [str(conv)]

results = []

model.eval()
for idx, row in test_df.iterrows():
    conv = split_conversation(row["conversation"])
    msg_results = []
    context = ""  # context accumulator

    for msg in conv:
        context_input = context + " " + msg if context else msg
        inputs = tokenizer(context_input, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            p_tossica = probs[0][1].item()
        
        msg_results.append({"message": msg, "p_tossica": p_tossica})
        context += " " + msg  # update context

    if msg_results:
        avg_tossicita = np.mean([m["p_tossica"] for m in msg_results])
        msg_results_sorted = sorted(msg_results, key=lambda x: x["p_tossica"], reverse=True)
        top_msg = msg_results_sorted[0]["message"]

        results.append({
            "conversation_id": idx,
            "avg_tossicita": avg_tossicita,
            "top_msg": top_msg
        })

df_results = pd.DataFrame(results)


# 13. Compare full chat vs single message
pred_full_chat = y_pred
df_pred = pd.DataFrame({
    "conversation_id": test_df["conversation_id"].values,
    "pred_full_chat": pred_full_chat,
    "toxic": test_df["toxic"].values
})

comparison = df_pred.merge(df_results, on="conversation_id", how="inner")
comparison["pred_single_msg"] = (comparison["avg_tossicita"] >= 0.5).astype(int)
comparison["result"] = np.where(comparison["pred_full_chat"] == comparison["pred_single_msg"], "✅ same", "❌ different")

print("\n--- Full Chat vs Single Message Comparison ---")
print(comparison.head(20))


# 14. Confusion matrix — Single Message
y_true = comparison["toxic"]
y_pred_single = comparison["pred_single_msg"]
print("\nClassification report- Single Message Classifier:")
print(classification_report(y_true,y_pred_single,target_names=["non-toxic","toxic"]))
cm_single = confusion_matrix(y_true, y_pred_single)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_single, display_labels=["non-toxic (0)", "toxic (1)"])

plt.figure(figsize=(5,5))
disp.plot(cmap="Purples", values_format="d")
plt.title("Confusion Matrix — Single Message Classifier")
plt.show()


# 15. Analysis of intermediate predictions
intermediate = comparison[(comparison["avg_tossicita"] >= 0.2) & (comparison["avg_tossicita"] <= 0.8)]
count_intermediate = len(intermediate)
total = len(comparison)

print(f"\n🔹 Total conversations: {total}")
print(f"🔸 Intermediate predictions (0.2 ≤ p ≤ 0.8): {count_intermediate} ({100*count_intermediate/total:.2f}%)")

if count_intermediate > 0:
    print("\nSample intermediate predictions:")
    print(intermediate[["conversation_id", "avg_tossicita", "toxic"]].head(10))
else:
    print("\nNo intermediate predictions found.")


# 16. Qualitative analysis: per-message toxicity probabilities
sample_convs = test_df.sample(10, random_state=42)

for i, row in sample_convs.iterrows():
    print(f"\n🗨️ Conversation ID {row['conversation_id']} - Actual label: {row['toxic']}")
    print("-" * 100)

    conv = split_conversation(row["conversation"])
    table_data = []
    context = ""

    for msg in conv:
        context_input = context + " " + msg if context else msg
        inputs = tokenizer(context_input, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            p_tossica = probs[0][1].item()

        table_data.append([msg[:120] + ("..." if len(msg) > 120 else ""), round(p_tossica, 10)])
        context += " " + msg

    print(tabulate(table_data, headers=["Message", "Toxicity Probability"]))

