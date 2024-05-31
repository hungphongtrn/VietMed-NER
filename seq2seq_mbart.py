from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from loguru import logger
import fire
import json

from slue import get_slue_format, get_ner_scores
from modified_seqeval import classification_report

# Load dataset
dataset = load_dataset("yuufong/vietmed_ner_v5")
id2label_list = dataset['train'].features["tags"].feature._int2str
id2label = {int(k): v for k, v in enumerate(id2label_list)}
label2id = dataset['train'].features["tags"].feature._str2int

num_labels = len(label2id)
# Append a dummy label to the label2id and id2label
label2id["dum"] = num_labels
id2label[num_labels] = "dum"


def convert_to_seq(words, tags):
    """Convert words and tags to a sequence of tags and words
    Example: tags = [1, 2, 3], words = ["I", "am", "fine"]
    Output: "1* I 1* 2* am 2* 3* fine 3*"
    """
    seq = ""
    for word, tag in zip(words, tags):
        seq += (f"{tag}* {word} {tag}* ")
    seq = seq.strip()
    return seq


def convert_seq_to_list(words, seq):
    seq = seq.split()
    tags = []
    for word in words:
        tag = num_labels
        if word in seq:
            id = seq.index(word)
            if id > 0 and id < len(seq) - 1:
                prev = seq[id - 1]
                after = seq[id + 1]
                if prev == after:
                    try:
                        tag = int(prev.replace("*", ""))
                    except:
                        pass
        tags.append(tag)
    return tags


def tokenize(example, tokenizer):
    """Tokenize the input and target sequences"""
    words, tags = example['words'], example['tags']
    target_seq = [convert_to_seq(word, tag) for word, tag in zip(words, tags)]
    input_seq = [" ".join(word) for word in words]

    inputs = tokenizer(text=input_seq, text_target=target_seq, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    return inputs


def build_eval_compute_metrics(tokenizer):
    def eval_compute_metrics(p):
        """Compute metrics for the evaluation phase"""
        predicted_seq, labeled_seq, input_seq = p

        # print(predicted_seq, labeled_seq, input_seq)
        # print("------------------")
        # Decode the predicted, labeled, and input sequences
        predicted_text = tokenizer.batch_decode(predicted_seq, skip_special_tokens=True)
        labeled_text = tokenizer.batch_decode(labeled_seq, skip_special_tokens=True)
        input_text = tokenizer.batch_decode(input_seq, skip_special_tokens=True)
        # print(predicted_text, labeled_text, input_text)
        # print("------------------")
        # Convert the sequences to lists of tags and words
        original_words = [text.split(" ")[1:] for text in input_text]  # ignore the first token "ner:"
        predictions = [convert_seq_to_list(word, text) for word, text in zip(original_words, predicted_text)]
        labels = [convert_seq_to_list(word, text) for word, text in zip(original_words, labeled_text)]
        # print(original_words, predictions, labels)
        # print("------------------")
        # Pad or truncate the predictions to match the labels
        for i in range(len(labels)):
            # Pad with "O" tag if the prediction is shorter than the label
            if len(labels[i]) > len(predictions[i]):
                predictions[i] += [label2id["0"]] * (len(labels[i]) - len(predictions[i]))
            # Truncate the prediction if it is longer than the label
            elif len(labels[i]) < len(predictions[i]):
                predictions[i] = predictions[i][:len(labels[i])]  # truncate the prediction
            else:
                pass

        # Convert ids to labels
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (_, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        # print(true_predictions, true_labels)
        # print("------------------")

        # Convert the sequences to SLUE format
        all_gt = [get_slue_format(original_words[i], true_labels[i], False) for i in range(len(labels))]
        all_pred = [get_slue_format(original_words[i], true_predictions[i], False) for i in range(len(predictions))]
        all_gt_dummy = [get_slue_format(original_words[i], true_labels[i], True) for i in range(len(original_words))]
        all_pred_dummy = [get_slue_format(original_words[i], true_predictions[i], True) for i in range(len(predictions))]

        # Compute the SLUE scores
        slue_scores = get_ner_scores(all_gt, all_pred)
        dummy_slue_scores = get_ner_scores(all_gt_dummy, all_pred_dummy)

        # Compute the classification report
        results = classification_report(true_predictions, true_labels, digits=4, output_dict=True)

        return {
            "precision": results["macro avg"]["precision"],
            "recall": results["macro avg"]["recall"],
            "f1": results["macro avg"]["f1-score"],
            "slue_scores": slue_scores,
            "dummy_slue_scores": dummy_slue_scores,
            "results": results
        }
    return eval_compute_metrics


def config_tokenizer_mbart(tokenizer):
    tokenizer.add_special_tokens({"bos_token": "<NER>"}, replace_additional_special_tokens=True)
    tokenizer.src_lang = "vi_VN"
    tokenizer.tgt_lang = "<NER>"
    return tokenizer


def prepare_model(model_name, device="cuda", is_train=False):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    if not is_train:
        model_name = json.load(open(f"{model_name}/config.json"))["_name_or_path"]
        tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="vi_VN", tgt_lang="vi_VN")
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer = config_tokenizer_mbart(tokenizer)
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    return model, tokenizer, data_collator, model_name


def prepare_train_args(model_name):
    # Define the training arguments and the trainer
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"outputs/{model_name}",
        predict_with_generate=True,
        learning_rate=2e-5,
        # gradient_accumulation_steps=8,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=32,
        num_train_epochs=1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_total_limit=1,
        logging_steps=20,
        push_to_hub=False,
        report_to="tensorboard",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        include_inputs_for_metrics=True,
        generation_max_length=128
    )
    return training_args


def train_single_model(model_name, device="cuda"):
    logger.info(f"Loading model {model_name}")
    model, tokenizer, data_collator, model_name = prepare_model(model_name, device, is_train=True)

    # Tokenize the dataset
    tokenized_dataset = dataset.map(tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer})
    compute_metrics = build_eval_compute_metrics(tokenizer)

    training_args = prepare_train_args(model_name)

    trainer = Seq2SeqTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )

    logger.info("Start training")
    trainer.train()

    logger.info("Start evaluation")
    result = trainer.predict(tokenized_dataset["test"])

    logger.info("Finish training, saving the results")
    with open(f"outputs/{model_name}/test_results.json", 'w') as f:
        json.dump(result.metrics, f)


def predict_results(model_name, device="cuda"):
    logger.info(f"Loading model {model_name}")
    model, tokenizer, data_collator, model_name = prepare_model(model_name, device)

    # Tokenize the dataset
    tokenized_testset = dataset.map(tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer})["test"]
    compute_metrics = build_eval_compute_metrics(tokenizer)

    training_args = prepare_train_args(model_name)
    trainer = Seq2SeqTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
    )
    logger.info("Start evaluation")
    result = trainer.predict(tokenized_testset)
    with open(f"outputs/{model_name}/test_results.json", 'w') as f:
        json.dump(result.metrics, f)


if __name__ == "__main__":
    fire.Fire()
