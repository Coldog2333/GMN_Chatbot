import pandas as pd
import re
from itertools import cycle
import warnings
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
    speaker_tags = ['Doctor:\n', 'Patient:\n', 'Description\n']
    # Dict containing words for starting and ending check for each speaker
    starting_words = dict(
        doctor = ['Q. '],
        patient = ['Q. '],
        description = [],
    )
    ending_words = dict(
        doctor = ['Take care', 'Hope I', 'Regards', 'Thank you,' 'Thank you.'],
        patient = ['Thank you,', 'Thank you.', 'Thanks'],
        description = [],
    )

    df_dict = {
        'dialogue_id':[],
        'speaker':[],
        'context':[]
    }
    # Closing line function
    def closing_line():
        # f_out.write('"\n') # Closing " mark of the previous dialogue and begin a new line
        pass

    # Main function content
    with open(output_filename, 'a+') as f_out, open(data_filename) as f_data:
        current_speaker = None  # track current speaker (patient or doctor)
        in_dialogue = False # Flag that currently in the dialouge
        contexts = []
        for line in f_data:
            # Detect id and speaker (id -> patient, 'Doctor' -> doctor)
            
            if in_dialogue and (line in speaker_tags or line == "\n"):
                in_dialogue = False
                df_dict['context'].append(' '.join(contexts))
                contexts = []

            if 'id=' in line: 
                current_idx = line.split('id=')[1][:-1] 

            if line in speaker_tags: 

                in_dialogue = True

                if line == "Doctor:\n":
                    current_speaker = "doctor"
                if line == "Patient:\n":
                    current_speaker = "patient"
                if line == "Description\n":
                    current_speaker = "description"
                
                # current_speaker = next(speakers) # switch speaker
                # f_out.write(f'{current_idx},{current_speaker},"') # Opening " mark of the next dialogue
                df_dict['dialogue_id'].append(current_idx)
                df_dict['speaker'].append(current_speaker)
                start = True # Flag for start of the dialogue
                continue

            if in_dialogue:
                # Get rid of unnecessary characters
                line = re.sub(r'\n|([\.\,]{2,})', '', line).strip() # Drop \n and ....,,,,,
                url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
                line = re.sub(url_regex, '', line).strip()

                # Dialogue beginning check
                for starting_word in starting_words[current_speaker]:
                    if line.startswith(starting_word):
                        line = line.split(starting_word)[1]

                # Dialogue ending check
                for ending_word in ending_words[current_speaker]:
                    if ending_word in line:
                        line = line.split(ending_word)[0] # Drop the right-handed side of the ending word

                contexts.append(line)

        pd.DataFrame(df_dict).to_csv(output_filename, index=False)

# Main loop
if __name__ == '__main__':
    # Health care magic
    for i in range(1, 5):
        split_by_id_and_speaker(f'../data/healthcaremagic_dialogue_{i}.txt', f'../data/healthcaremagic_splitted_idname_{i}.csv')
    # Icliniq
    split_by_id_and_speaker('../data/icliniq_dialogue.txt', '../data/icliniq_splitted_idname.csv')
