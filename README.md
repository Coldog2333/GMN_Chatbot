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
from preprocessing.get_en_dataloader import get_training_dev_test_dataloader

train_loader, dev_loader, test_loader = get_training_dev_test_dataloader(debugging=False, max_length=256)

print(train_loader[0])
```
- The dataloader will return the following dictionary for each different index
```
Dict{
    'input_ids': tensor(2, max_length),
    'token_type_ids': same,
    'attention_mask': same,
    'doctor_input_ids': tensor(#negative_sample + 1, max_length),
    'doctor_token_type_ids': same,
    'doctor_ dattention_mask': same,

}
```
Note that negative samples are sampled by randomly chosen from different response in other conversation.
