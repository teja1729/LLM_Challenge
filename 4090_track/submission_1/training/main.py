# dataset preperation
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset


def prompt_formatting(example: dict):
    """example.keys -- instruction ,context(optional),response"""
    if example.get("context", "") != "":
        input_prompt = (
            "Below is an instruction that describes a task, paired with an input that provides"
            " further context. Write a response that appropriately completes the request.\n\n###"
            f" Instruction:\n{example['instruction']}\n\n### Input: \n{example['context']}\n\n###"
            " Response:"
        )
    else:
        input_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{example['instruction']}\n\n"
            "### Response:"
        )
    return {
        "input_prompt": input_prompt,
        "input_output_prompt": input_prompt + f"{example['response']}",
    }


# dolly
dolly_15k = load_dataset("databricks/databricks-dolly-15k")
instruct_dolly = dolly_15k.map(prompt_formatting)
instruct_dolly = instruct_dolly["train"].map(lambda x: {"dataset": "dolly"})
# lima
lima_dataset = load_dataset("GAIR/lima", use_auth_token="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")


def prepare_lima(dataset_dict: dict):
    formatted_dataset = []
    for dp in dataset_dict:
        convo = dp["conversations"]
        formatted_dataset.append({"instruction": convo[0], "context": "", "response": convo[1]})
    return formatted_dataset


train_data = prepare_lima(lima_dataset["train"])
instruct_lima = Dataset.from_pandas(pd.DataFrame(data=train_data))
instruct_lima = instruct_lima.map(prompt_formatting)
instruct_lima = instruct_lima.map(lambda x: {"dataset": "lima"})
# sciq
sciq_dataset = load_dataset("sciq")


def prepare_sciq(example: dict):
    formatted_dataset = []
    for dp in example:
        perm = list(
            np.random.permutation(
                [
                    dp["distractor3"],
                    dp["distractor1"],
                    dp["distractor2"],
                    dp["correct_answer"],
                ]
            )
        )
        idx = perm.index(dp["correct_answer"])
        formatted_dataset.append(
            {
                "instruction": dp["question"],
                "context": f"A.{perm[0]}\nB.{perm[1]}\nC.{perm[2]}\nD{perm[3]}",
                "response": chr(65 + idx),
            }
        )
    return formatted_dataset


sciq_train_data = prepare_sciq(sciq_dataset["train"])
instruct_sciq = Dataset.from_pandas(pd.DataFrame(data=sciq_train_data))
instruct_sciq = instruct_sciq.map(prompt_formatting)
instruct_sciq = instruct_sciq.map(lambda x: {"dataset": "sciq"})
# oasst1
oasst1_dataset = load_dataset("OpenAssistant/oasst1")
oasst1_dataset_df = oasst1_dataset["train"].to_pandas()


def prepare_oasst1(df):
    df_assistant = df[(df.role == "assistant") & (df["rank"] == 0.0)].copy()
    df_prompter = df[(df.role == "prompter")].copy()
    df_prompter = df_prompter.set_index("message_id")
    df_assistant["response"] = df_assistant["text"].values

    inputs = []
    parent_ids = []
    for _, row in df_assistant.iterrows():
        input = df_prompter.loc[row.parent_id]
        inputs.append(input.text)
        parent_ids.append(input.parent_id)

    df_assistant["instruction"] = inputs
    df_assistant["parent_id"] = parent_ids

    df_assistant = df_assistant[df_assistant.lang == "en"]

    df_assistant = df_assistant[["instruction", "response"]]

    return df_assistant


instruct_oasst1 = Dataset.from_pandas(prepare_oasst1(oasst1_dataset_df))
instruct_oasst1 = instruct_oasst1.map(prompt_formatting)
instruct_oasst1 = instruct_oasst1.map(lambda x: {"dataset": "oasst1"})
# sciencqa
scienceqa = load_dataset("derek-thomas/ScienceQA")


def prepare_scienceqa(example: dict):
    formatted_dataset = []
    for dp in example:
        options = "\n"
        for idx, v in enumerate(dp["choices"]):
            options = chr(65 + idx) + v + "\n"

        formatted_dataset.append(
            {
                "instruction": dp["question"] + options,
                "context": dp["lecture"],
                "response": chr(65 + int(dp["answer"])),
            }
        )
    return formatted_dataset


instruct_scienceqa = prepare_scienceqa(scienceqa["train"])
instruct_scienceqa = Dataset.from_pandas(pd.DataFrame(data=instruct_scienceqa))
instruct_scienceqa = instruct_scienceqa.map(prompt_formatting)
instruct_scienceqa = instruct_scienceqa.map(lambda x: {"dataset": "scienceqa"})
# merged
merged_dataset = pd.DataFrame()
for i in ["input_prompt", "input_output_prompt", "dataset"]:
    merged_dataset[i] = (
        instruct_dolly[i]
        + instruct_scienceqa[i]
        + instruct_lima[i]
        + instruct_sciq[i]
        + instruct_oasst1[i]
    )
# train and val
# from sklearn.model_selection import train_test_split
# train_df,val_df = train_test_split(merged_dataset,stratify=merged_dataset['dataset'],test_size=0.1)
# df to dataset
train_dataset = Dataset.from_pandas(merged_dataset)
# val_dataset = Dataset.from_pandas(val_df)
# training
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTTrainer

model_name = "mistralai/Mistral-7B-v0.1"
run_name = "models_output_last_4_layers_final_train"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=nf4_config,
    device_map="cuda:0",
    trust_remote_code=True,
)

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    lora_dropout=0.01,
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    layers_to_transform=[28, 29, 30, 31],
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config=lora_config)

###### added for evaluation help
if Path("/mnt/shared").exists():  # we're running on mzai infra
    output_dir = Path("/mnt/shared/neurips_eval/" + run_name)

    class PeftSavingCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            kwargs["model"].save_pretrained(checkpoint_path)

            if "pytorch_model.bin" in os.listdir(checkpoint_path):
                os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))

    callbacks = [PeftSavingCallback()]
else:
    output_dir = run_name
    callbacks = None
###### end added for evaluation help


training_args = TrainingArguments(
    per_device_train_batch_size=3,
    gradient_accumulation_steps=2,
    save_steps=5000,
    logging_steps=500,
    # evaluation_strategy='steps',
    # eval_steps = 300,
    optim="adamw_8bit",
    gradient_checkpointing=True,
    learning_rate=2e-4,
    fp16_full_eval=True,
    fp16=True,
    num_train_epochs=6,
    # warmup_ratio= 0.03,
    log_level="info",
    output_dir=output_dir,
    lr_scheduler_type="linear",
    # load_best_model_at_end=True
)

# from transformers import EarlyStoppingCallback
# early_stopping = EarlyStoppingCallback(early_stopping_patience=3)


trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    dataset_text_field="input_output_prompt",
    # eval_dataset=val_dataset,
    tokenizer=tokenizer,
    max_seq_length=2048,
    args=training_args,
    callbacks=callbacks,
    peft_config=lora_config,
)


trainer.train()


# path to save the trained weights
model_save_path = os.path.join("outputs", run_name)
trainer.save_model(output_dir=model_save_path)


model_hf_name = "mistrail_28_31_final_train"
model = AutoPeftModelForCausalLM.from_pretrained(
    model_save_path,
)
tokenizer = AutoTokenizer.from_pretrained(model_save_path)
model.push_to_hub(model_hf_name,token="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")
