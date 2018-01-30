import pickle
import os
from tqdm import tqdm

from preprocess import process_wav

class Dataset:
    def __init__(self, folder_dir):
          self._folder_dir = folder_dir


class LJDataset(Dataset):
    def __init__(self, folder_dir):
        super(LJDataset, self).__init__(folder_dir=folder_dir)

    def run(self):
        if not os.path.isdir(self._folder_dir):
            raise FileNotFoundError

        csv_file_path = os.path.join(self._folder_dir, "metadata.csv")

        file_results = []

        with open(csv_file_path) as f:
            csv_data = f.read()
            csv_data = csv_data.split('\n')

            for data in tqdm(csv_data):
                linear_output, mel_output, filename, text = self._extract_data(data)  # it was always outputting 1 in ever rows and columns, figured out the problem was with min_level_db which should be -100 but was 100.
                linear_file = 'training/'+filename+'-linear'
                mel_file = 'training/'+filename+'-mel'
                pickle.dump(linear_output, linear_file)
                pickle.dump(mel_output, mel_file)

                result = linear_file + '|' + mel_file + '|' + text
                file_results.append(result)


        with open('training/train.txt', 'r') as f:
            for result in file_results:
                f.write(result)


    def _extract_data(self, data):
        data = data.split('|')  # data[0] = filename, data[1] = text

        wav_file = data[0] + '.wav'

        wav_path = os.path.join(self._folder_dir, 'wavs', wav_file)
        linear_output, mel_output = process_wav(wav_path)

        return linear_output, mel_output, data[0], data[1]  # linear output, mel output, text


dataset = LJDataset("LJSpeech-1.0")
dataset.run()