#dataset preperation
import numpy as np
import pandas as pd
from datasets import load_dataset,Dataset

def prompt_formatting(example:dict):
    """
    example.keys -- instruction ,context(optional),response
    """
    if example.get("context","") !="":
        input_prompt = (f"Below is an instruction that describes a task, paired with an input that provides further context. "
      "Write a response that appropriately completes the request.\n\n"
      "### Instruction:\n"
      f"{example['instruction']}\n\n"
      f"### Input: \n"
      f"{example['context']}\n\n"
      f"### Response:")
    else:
        input_prompt = (f"Below is an instruction that describes a task. "
      "Write a response that appropriately completes the request.\n\n"
      "### Instruction:\n"
      f"{example['instruction']}\n\n"
      f"### Response:")   
    return {"input_prompt":input_prompt,"input_output_prompt":input_prompt+f"{example['response']}"}
#dolly
dolly_15k = load_dataset("databricks/databricks-dolly-15k")
instruct_dolly = dolly_15k.map(prompt_formatting)
instruct_dolly = instruct_dolly['train'].map(lambda x: {'dataset':'dolly'})
#lima
lima_dataset = load_dataset("GAIR/lima",use_auth_token = "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")
def prepare_lima(dataset_dict: dict):
    formatted_dataset =[]
    for dp in dataset_dict:
        convo = dp["conversations"]
        formatted_dataset.append({"instruction":convo[0],"context":"","response":convo[1]})
    return formatted_dataset
train_data = prepare_lima(lima_dataset['train'])
instruct_lima  = Dataset.from_pandas(pd.DataFrame(data = train_data))
instruct_lima = instruct_lima.map(prompt_formatting)
instruct_lima = instruct_lima.map(lambda x: {'dataset':'lima'})
#sciq
sciq_dataset = load_dataset("sciq")
def prepare_sciq(example:dict):
    formatted_dataset = []
    for dp in example:
        perm = list(np.random.permutation([dp['distractor3'],dp['distractor1'],dp['distractor2'],dp['correct_answer']]))
        idx = perm.index(dp['correct_answer'])
        formatted_dataset.append({'instruction':dp['question'],"context":f"A.{perm[0]}\nB.{perm[1]}\nC.{perm[2]}\nD{perm[3]}",'response':chr(65+idx)})
    return formatted_dataset
sciq_train_data = prepare_sciq(sciq_dataset['train'])
instruct_sciq  = Dataset.from_pandas(pd.DataFrame(data = sciq_train_data))
instruct_sciq = instruct_sciq.map(prompt_formatting)
instruct_sciq = instruct_sciq.map(lambda x: {'dataset':'sciq'})
#oasst1
oasst1_dataset = load_dataset('OpenAssistant/oasst1')
oasst1_dataset_df = oasst1_dataset['train'].to_pandas()
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

    df_assistant = df_assistant[
        ["instruction", "response"]]
    
    return df_assistant
instruct_oasst1 = Dataset.from_pandas(prepare_oasst1(oasst1_dataset_df))
instruct_oasst1 = instruct_oasst1.map(prompt_formatting)
instruct_oasst1 = instruct_oasst1.map(lambda x: {'dataset':'oasst1'})
#sciencqa
scienceqa = load_dataset('derek-thomas/ScienceQA')
def prepare_scienceqa(example:dict):
    formatted_dataset = []
    for dp in example:
        options = "\n"
        for idx,v in enumerate(dp['choices']):
            options = chr(65+idx) + v +'\n'
        
        formatted_dataset.append({"instruction":dp['question']+options,"context":dp['lecture'],"response":chr(65+int(dp['answer']))})
    return formatted_dataset
instruct_scienceqa = prepare_scienceqa(scienceqa['train'])
instruct_scienceqa  = Dataset.from_pandas(pd.DataFrame(data = instruct_scienceqa))
instruct_scienceqa = instruct_scienceqa.map(prompt_formatting)
instruct_scienceqa = instruct_scienceqa.map(lambda x: {'dataset':'scienceqa'})
#merged
merged_dataset = pd.DataFrame()
for i in ['input_prompt','input_output_prompt','dataset']:
    merged_dataset[i] = instruct_dolly[i]+instruct_scienceqa[i] + instruct_lima[i]+ instruct_sciq[i]+ instruct_oasst1[i]
#train and val
from sklearn.model_selection import train_test_split
train_df,val_df = train_test_split(merged_dataset,stratify=merged_dataset['dataset'],test_size=0.1)
#df to dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
#training
from transformers import (AutoModelForCausalLM,
                         AutoTokenizer,
                         BitsAndBytesConfig,
                         TrainingArguments)
from peft import (AutoPeftModelForCausalLM,
                  LoraConfig,
                  prepare_model_for_kbit_training,
                  get_peft_model)

from trl import SFTTrainer 
import torch

model_name = "mistralai/Mistral-7B-v0.1"
run_name = 'models_output_last_4_layers_6'

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=nf4_config,device_map = "cuda:0",trust_remote_code = True)

lora_config = LoraConfig(task_type="CAUSAL_LM",
                         lora_dropout= 0.01,
                         r = 8,
                         lora_alpha = 16,
                         target_modules=['q_proj','k_proj','v_proj','o_proj'],
                         layers_to_transform=[28,29,30,31])

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model,peft_config=lora_config)

training_args = TrainingArguments(
    per_device_train_batch_size= 8, 
    gradient_accumulation_steps=4,
    save_steps = 300,
    logging_steps = 300,
    evaluation_strategy='steps',
    eval_steps = 300,
    optim = 'sgd',
    gradient_checkpointing= True,
    learning_rate=2e-4,
    fp16_full_eval= True,
    fp16 = True,
    num_train_epochs= 5,
    #warmup_ratio= 0.03,
    log_level='info',
    output_dir=run_name,
    lr_scheduler_type= 'constant',
    load_best_model_at_end=True

)

#from transformers import EarlyStoppingCallback
#early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

trainer = SFTTrainer(model = model,
                     train_dataset=train_dataset,
                     dataset_text_field= "input_output_prompt",
                     eval_dataset=val_dataset,
                     tokenizer=tokenizer,
                     max_seq_length= 1000,
                     args = training_args,
                     #callbacks=[early_stopping],
                     peft_config= lora_config)

trainer.train()

import os
model_save_path = os.path.join('layers_determination',run_name)
trainer.save_model(output_dir=model_save_path)


model_hf_name = 'mistrail_28_31_5'
model = AutoPeftModelForCausalLM.from_pretrained(model_save_path,)
tokenizer = AutoTokenizer.from_pretrained(model_save_path)
model.push_to_hub(model_hf_name)
