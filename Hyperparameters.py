class hp:
    sample_rate = 24000
    pre_emphasis = 0.97
    num_freq = 2048
    num_mels = 80
    frame_length = 50  # ms
    frame_shift = 12.5  # ms
    r_frames = 4

    amp_reference = 20

    min_level_db = -100
    griffin_iter = 60

    folder_dir = "../keith_tacotron/tacotron/LJSpeech-1.0"

    symbols = '_' + '~' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + 'abcdefghijklmnopqrstuvwxyz' + '`1234567890' + '!\'(),-.:;?" '
