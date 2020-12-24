import pandas as pd
import re
from itertools import cycle
import warnings
warnings.filterwarnings('ignore')

def split_by_id_and_speaker(data_filename, output_filename):
    # Available speakers (cycle for convenient switching)
    speakers = cycle(['patient', 'doctor'])
    # Dict containing words for ending check for each speaker
    ending_words = dict(
        doctor = ['Take care', 'Hope I', 'Regards', 'Thank you,' 'Thank you.'],
        patient = ['Thank you,', 'Thank you.', 'Thanks']
    )
    # Closing line function
    def closing_line():
        f_out.write('"\n') # Closing " mark of the previous dialogue and begin a new line

    # Main function content
    with open(output_filename, 'a+') as f_out, open(data_filename) as f_data:
        current_speaker = None  # track current speaker (patient or doctor)
        for line in f_data:
            # Detect id and speaker (id -> patient, 'Doctor' -> doctor)
            if line.startswith('id=') or line == 'Doctor:\n': 
                if line.startswith('id='): current_idx = line[3:-1] 
                if current_speaker: closing_line()
                current_speaker = next(speakers) # switch speaker
                f_out.write(f'{current_idx},{current_speaker},"') # Opening " mark of the next dialogue
                start = True # Flag for start of the dialogue
                continue

            # Write to output file if there is a speaker
            if current_speaker:
                # Skip rows for patient
                if current_speaker == 'patient':
                    if line in ['Dialogue\n', 'Patient:\n', 'Description\n'] or line.startswith('https:'):  # Skip these rows
                        continue

                # Get rid of unnecessary characters
                line = re.sub(r'\n|(\.{2,})', '', line).strip() 

                # Dialogue ending check
                for ending_word in ending_words[current_speaker]:
                    if ending_word in line:
                        line = line.split(ending_word)[0] # Drop the right-handed side of the ending word
                        current_speaker = None
                
                # Skip blank string
                if line == '': 
                    if not current_speaker: closing_line()
                    continue

                # Write the dialogue
                content_to_write = line if start else ' ' + line # If the speaker just starts speaking, don't add space
                f_out.write(content_to_write) # Add space between each line of the dialogue (in case one dialogue has many lines)
                start = False # Flip start to False after writing the speaker's first sentence

                # Write closing " and new line if the dialogue already finished (current_speaker == None)
                if not current_speaker: closing_line()
        # Last line check
        if current_speaker: closing_line()

# Main loop
if __name__ == '__main__':
    for i in range(1, 5):
        split_by_id_and_speaker(f'../data/healthcaremagic_dialogue_{i}.txt', f'splitted_by_id_and_speaker_{i}.csv')