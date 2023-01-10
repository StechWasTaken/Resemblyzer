from resemblyzer import preprocess_wav, VoiceEncoder
from demo_utils import *
from pathlib import Path

wav_fpath = Path("audio_data", "spaceballs.mp3")
wav = preprocess_wav(wav_fpath)

segments = [[5, 11], [91, 96], [98, 100]]
speaker_names = ["Akaash", "Sander", "Wesley"]

# extract wave segments from wave based on the specified segments above
speaker_wavs = [wav[int(s[0] * sampling_rate):int(s[1] * sampling_rate)] for s in segments]

encoder = VoiceEncoder("cpu")
print("Running the continuous embedding on cpu, this might take a whilep...")
_, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=32)


speaker_embeds = [encoder.embed_utterance(speaker_wav) for speaker_wav in speaker_wavs]
similarity_dict = {name: cont_embeds @ speaker_embed for name, speaker_embed in 
                   zip(speaker_names, speaker_embeds)}

interactive_diarization(similarity_dict, wav, wav_splits)