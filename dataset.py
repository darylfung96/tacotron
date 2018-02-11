import pickle
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as np

from preprocess.preprocess_wav import process_wav
from Hyperparameters import hp

class Dataset:
    def __init__(self, folder_dir):
          self._folder_dir = folder_dir


class LJDataset(Dataset):
    def __init__(self, folder_dir):
        super(LJDataset, self).__init__(folder_dir=folder_dir)

    def run(self):
        if not os.path.isdir(self._folder_dir):
            raise FileNotFoundError

        csv_file_path = 'LJSpeech-1.0/metadata.csv'#os.path.join(self._folder_dir, "metadata.csv")

        job_results = []

        if not os.path.isdir('training'):
            os.mkdir('training')

        with open(csv_file_path) as f:
            csv_data = f.read()
            csv_data = csv_data.split('\n')

            processor = ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())
            file_results = []
            index = 0

            for data in csv_data:
                #linear_output, mel_output, text = self._extract_data(data)  # it was always outputting 1 in ever rows and columns, figured out the problem was with min_level_db which should be -100 but was 100.

                job_done = processor.submit(self._extract_data, data, index)
                job_results.append(job_done)
                index += 1

        training_data = [job_done.result() for job_done in tqdm(job_results)]


        with open('training/train.txt', 'w') as f:
            for data in tqdm(training_data):
                f.write(data+"\n")

    def _extract_data(self, data, index):
        data = data.split('|')  # data[0] = filename, data[1] = text

        wav_file = data[0] + '.wav'
        text = data[1]

        wav_path = os.path.join(self._folder_dir, 'wavs', wav_file)
        linear_output, mel_output = process_wav(wav_path)

        linear_output = np.array(linear_output).transpose()
        mel_output = np.array(mel_output).transpose()


        linear_file = 'training/{}-linear.pkl'.format(index)
        mel_file = 'training/{}-mel.pkl'.format(index)

        with open(linear_file, 'wb') as f:
            pickle.dump(linear_output, f)
        with open(mel_file, 'wb') as f:
            pickle.dump(mel_output, f)

        result = linear_file + '|' + mel_file + '|' + text

        return result  # linear output, mel output, text


dataset = LJDataset(hp.folder_dir)
dataset.run()