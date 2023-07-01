import pandas as pd
dataframe = pd.read_csv("self_promotion_scores_transcripts.csv")

# !pip install sentencepiece
# !pip install transformers
# !pip install torch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Setting up the device for GPU usage
from torch import cuda
device = 'cuda:1' if cuda.is_available() else 'cpu'


class redditDatasetReader():
    
    def __init__(self, dataframe, tokenizer, max_source_length, max_target_length, source_text, target_text):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.maximum_source_length = max_source_length
        self.maximum_target_length = max_target_length
        self.source_text = self.data["interview"]
        self.target_text = self.data["overallComments"]
    
    def __len__(self):
        return len(self.target_text)
    
    def __getitem__(self, index):
        
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.maximum_source_length,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.maximum_target_length,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


def train(epoch, tokenizer, model, device, loader, optimizer):

    """
    Function to be called for training with the parameters passed from main function

    """

    model.train()
    for _, data in enumerate(loader, 0):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]

        optimizer.zero_grad()
        #loss.sum().backward()
        loss.backward()
        optimizer.step()


def validate(epoch, tokenizer, model, device, loader):

  """
  Function to evaluate model for predictions

  """
  model.eval()
  predictions = []
  actuals = []
  with torch.no_grad():
      for _, data in enumerate(loader, 0):
          y = data['target_ids'].to(device, dtype = torch.long)
          ids = data['source_ids'].to(device, dtype = torch.long)
          mask = data['source_mask'].to(device, dtype = torch.long)

          generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask, 
              max_length=150, 
              num_beams=2,
              repetition_penalty=2.5, 
              length_penalty=1.0, 
              early_stopping=True
              )
          preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
          target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
#           if _%10==0:
#               console.print(f'Completed {_}')

          predictions.extend(preds)
          actuals.extend(target)
  return predictions, actuals


def T5Trainer(
    dataframe, source_text, target_text, model_params, output_dir="./outputs/"
):

    """
    T5 trainer

    """

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # logging
#     console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    #model= nn.DataParallel(model, device_ids=[0, 1])
    model = model.to(device)

    # logging
#     console.log(f"[Data]: Reading data...\n")

    # Importing the raw dataset
    dataframe = dataframe[[source_text, target_text]]
#     display_df(dataframe.head(2))

    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest for validation.
    train_size = 0.9
    train_dataset = dataframe.sample(frac=train_size, random_state=model_params["SEED"])
    val_dataset = dataframe.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

#     console.print(f"FULL Dataset: {dataframe.shape}")
#     console.print(f"TRAIN Dataset: {train_dataset.shape}")
#     console.print(f"TEST Dataset: {val_dataset.shape}\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = redditDatasetReader(
        train_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    val_set = redditDatasetReader(
        val_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }

    val_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=model_params["LEARNING_RATE"]
    )

    # Training loop
#     console.log(f"[Initiating Fine Tuning]...\n")

    for epoch in tqdm(range(model_params["TRAIN_EPOCHS"])):
        train(epoch, tokenizer, model, device, training_loader, optimizer)

#     console.log(f"[Saving Model]...\n")
    # Saving the model after training
    # path = "/kaggle/working/"
    # model.save_pretrained()
    # tokenizer.save_pretrained()

    # evaluating test dataset
#     console.log(f"[Initiating Validation]...\n")
    for epoch in tqdm(range(model_params["VAL_EPOCHS"])):
        predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
        final_df.to_csv("predictions.csv")

#     console.save_text(os.path.join(output_dir, "logs.txt"))

#     console.log(f"[Validation Completed.]\n")
#     console.print(
#         f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n"""
#     )
#     console.print(
#         f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir,'predictions.csv')}\n"""
#     )
#     console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n""")



# let's define model parameters specific to T5
model_params = {
    "MODEL": "t5-base",  # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE": 4,  # training batch size
    "VALID_BATCH_SIZE": 4,  # validation batch size
    "TRAIN_EPOCHS": 50,  # number of training epochs
    "VAL_EPOCHS": 5,  # number of validation epochs
    "LEARNING_RATE": 1e-4,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 128,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
}


# T5 accepts prefix of the task to be performed:
# Since we are summarizing, let's add summarize to source text as a prefix
dataframe["interview"] = "feedback: " + dataframe["interview"]

T5Trainer(
    dataframe=dataframe,
    source_text="interview",
    target_text="overallComments",
    model_params=model_params,
    output_dir="outputs",
)

