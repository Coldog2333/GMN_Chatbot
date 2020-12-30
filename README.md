# GMN Chatbot
## Introduction
This is a repository for GMN-based Chatbot, which is our final project for U-Tokyo Course: New Frontier Artificial Intelligence II.
## Dataset
+ name
https://drive.google.com/drive/folders/1g29ssimdZ6JzTST6Y8g6h-ogUNReBtJD

## Dataloader
- Please prepare the data by download the text files in the google drive link above. There should be the following files in the `data` directory
```
"data/healthcaremagic_splitted_idname_1.csv",
"data/healthcaremagic_splitted_idname_2.csv",
"data/healthcaremagic_splitted_idname_3.csv",
"data/healthcaremagic_splitted_idname_4.csv",
"data/icliniq_splitted_idname.csv"
```
- The preprocessing from text file to csv and generate the tokenized dataset files, please run `preprocessing/reformat_text_data.py` 
```bash
python preprocessing/reformat_text_data.py
```
- To get the (pytorch)dataloader, import from the following module. Note that the tokenization was done by huggingface `bert-base-uncased` tokenizer and the maximum length default is 256
```python
from preprocessing.get_en_dataloader import get_training_dev_test_dataset

train_dataset, dev_dataset, test_dataset = get_training_dev_test_dataset(debugging=False, max_length=256)

print(train_loader[0])
```
which the dataset can be used in [Huggingface trainer](https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer). In case of manual training, please wrap it with pytorch dataloader
```python
from torch.utils.data import DataLoader

# For train loader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
```

## Default Data split
Note that the data is split by the following
```
Train:
    - data/healthcaremagic_splitted_idname_1.csv
    - data/healthcaremagic_splitted_idname_2.csv
    - data/healthcaremagic_splitted_idname_3.csv

Dev:
    - data/healthcaremagic_splitted_idname_4.csv

Test:
    - data/icliniq_splitted_idname.csv
```
- The dataloader will return the following dictionary for each different index
```
Dict{
    'input_ids': tensor(2, max_length),
    'token_type_ids': same,
    'attention_mask': same,
    'doctor_input_ids': tensor(#negative_sample + 1, max_length),
    'doctor_token_type_ids': same,
    'doctor_attention_mask': same,

}
```
There are 2 things to note here,
- In all samples there are description, patient response and doctor response, in total 3-turns dialogue. So here, we treat the description and patient as first 2-turns dialogue and ask the model to output the probability of the third turn
- the negative samples are sampled by randomly chosen from different response in other conversation. **The correct response is always in the first index(0)** and followed by `#negative_sample` number of wrong response.
