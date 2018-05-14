# tacotron

Implementation of tacotron, a text to speech deep learning model. 

Paper can be found here: [tacotron](https://arxiv.org/abs/1703.10135).

## Getting Started
---
1. Dataset can be retrieved from [here](https://keithito.com/LJ-Speech-Dataset/). When extracted, file will contain LJSpeech-1.1 as a folder. Move that folder into the root directory beside where train.py is.

2. Run preprocess.py to preprocess the audio files. Audio files will be generated in a directory called training for the default parameters.

3. After running the preprocessing steps, we can start training the model.

### To train:

```
python train.py
```


### To evaluate:
todo.

TODO:
- train the network to generate a pretrained model

Reimplementation of this for education purposes.

Got lots of reference from: https://github.com/keithito/tacotron
Really grateful and appreciate the work of Keith Ito.
