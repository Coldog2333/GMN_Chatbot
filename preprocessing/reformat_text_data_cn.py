import os
import pickle
import re
from itertools import cycle
import warnings

import pandas as pd

warnings.filterwarnings('ignore')


def split_by_id_and_speaker(data_filename, output_filename):
    """
    Split dialogues by id and speaker (i.e., patient and doctor) with some cleaning

    Output filename format: each row is (idx, speaker, "dialogue")
    E.g.
    0, patient0, "#description0 #patient0_dialogue"
    0, doctor0, "#doctor0_dialogue"
    1, patient1, "#description1 #patient1_dialogue"
    1, doctor1, "#doctor1_dialogue"
    ...
    """
    # Available speakers (cycle for convenient switching)
    speaker_tags = ['医生：\n', '疾病：\n']
    # Dict containing words for starting and ending check for each speaker
    starting_words = dict(
        doctor=['Q. '],
        patient=['Q. '],
    )
    ending_words = dict(
        doctor=[],
        patient=[],
        description=[],
    )

    df_dict = {
        'dialogue_id': [],
        'speaker': [],
        'context': []
    }

    # Main function content
    with open(output_filename, 'a+') as f_out, open(data_filename) as f_data:
        current_speaker = None  # track current speaker (patient or doctor)
        in_dialogue = False  # Flag that currently in the dialouge
        contexts = []
        for line in f_data:
            # Detect id and speaker (id -> patient, 'Doctor' -> doctor)

            if in_dialogue and (line in speaker_tags or line == "\n"):
                in_dialogue = False
                df_dict['dialogue_id'].append(current_idx)
                df_dict['speaker'].append(current_speaker)
                df_dict['context'].append(' '.join(contexts))
                contexts = []
                if len(df_dict['speaker']) != len(df_dict['context']):
                    print(f"{current_idx}, {len(df_dict['speaker'])} len(df_dict['context'])")
                    pass

            if 'id=' in line:
                in_dialogue = False
                current_idx = line.split('id=')[1][:-1]

            if line in speaker_tags:

                in_dialogue = True

                if line == '医生：\n':
                    current_speaker = "doctor"
                if line == '疾病：\n':
                    current_speaker = "patient"

                # current_speaker = next(speakers) # switch speaker
                # f_out.write(f'{current_idx},{current_speaker},"') # Opening " mark of the next dialogue
                start = True  # Flag for start of the dialogue
                continue

            if in_dialogue:
                # Get rid of unnecessary characters
                # Drop \n and ....,,,,,
                line = re.sub(r'\n|([\.\,]{2,})', '', line).strip()
                url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
                line = re.sub(url_regex, '', line).strip()

                # Dialogue beginning check
                for starting_word in starting_words[current_speaker]:
                    if line.startswith(starting_word):
                        line = line.split(starting_word)[1]

                # Dialogue ending check
                for ending_word in ending_words[current_speaker]:
                    if ending_word in line:
                        # Drop the right-handed side of the ending word
                        line = line.split(ending_word)[0]

                contexts.append(line)

        print(len(df_dict['dialogue_id']), len(df_dict['speaker']), len(df_dict['context']))

        pd.DataFrame(df_dict).to_csv(output_filename, index=False)


def pretokenize_dataset() -> None:
    '''
    After finishing generate the messy text into CSV,
    we tokenize the text and save them as objects
    '''
    files = [f"data/{year}_split_by_idname.csv" for year in range(2011, 2021)]

    max_lengths = [256]

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    for file in files:
        df = pd.read_csv(file)
        df.fillna('', inplace=True)
        df_description = df[df.speaker == "description"]
        df_patient = df[df.speaker == "patient"]
        df_doctor = df[df.speaker == "doctor"]
        for max_length in max_lengths:

            # Remove csv
            file_name = file.split("/")[1][:-4]

            tokenize_description = tokenizer(df_description['context'].tolist(
            ), max_length=max_length, truncation=True, padding=True)
            tokenize_patient = tokenizer(df_patient['context'].tolist(
            ), max_length=max_length, truncation=True, padding=True)
            tokenize_doctor = tokenizer(df_doctor['context'].tolist(
            ), max_length=max_length, truncation=True, padding=True)

            with open(f"data/{file_name}_{max_length}_description.pkl", 'wb') as f:
                pickle.dump(tokenize_description, f)

            with open(f"data/{file_name}_{max_length}_patient.pkl", 'wb') as f:
                pickle.dump(tokenize_patient, f)

            with open(f"data/{file_name}_{max_length}_doctor.pkl", 'wb') as f:
                pickle.dump(tokenize_doctor, f)


# Main loop
if __name__ == '__main__':
    # Chinese dataset
    for year in [2011,2012,2013,2015,2016,2017,2019,2020]:
        file = f"data/{year}.txt"
        if not os.path.isfile(f"data/{year}_split_by_idname.csv"):
            split_by_id_and_speaker(
                file, f"data/{year}_split_by_idname.csv")

    # From text to input index
    print('--------- Tokenizing the data ---------')
    pretokenize_dataset()
