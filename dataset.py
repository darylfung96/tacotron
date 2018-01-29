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

        with open(csv_file_path) as f:
            csv_data = f.read()
            csv_data = csv_data.split('\n')

            for data in tqdm(csv_data):
                linear_output, mel_output, text = self._extract_data(data)



    def _extract_data(self, data):
        data = data.split('|')  # data[0] = filename, data[1] = text

        wav_file = data[0] + '.wav'

        wav_path = os.path.join(self._folder_dir, 'wavs', wav_file)
        linear_output, mel_output = process_wav(wav_path)

        return linear_output, mel_output, data[1] # linear output, mel output, text


dataset = LJDataset("LJSpeech-1.0")
dataset.run()