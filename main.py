from TTS.api import TTS
import torch
from TTS.tts.configs.xtts_config import XttsConfig, XttsArgs
from TTS.tts.models.xtts import XttsAudioConfig  # Import the missing class
from TTS.config.shared_configs import BaseDatasetConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
# Allow safe globals so that XttsConfig and XttsAudioConfig are permitted during checkpoint loading
torch.serialization.add_safe_globals([XttsConfig,XttsArgs, XttsAudioConfig, BaseDatasetConfig])

# Initialize the TTS model (using XTTS-v2)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# The text to synthesize
text = "Welcome to our channel! Today we’re diving into the future with “10 Emerging Technologies That Will Change the World.” In this video, we’ll explore breakthrough innovations—from AI that acts on your behalf to quantum computers that solve unsolvable problems. These technologies aren’t just trends; they’re poised to reshape industries, improve our daily lives, and help address global challenges."

# Generate speech using the local reference voice file
tts.tts_to_file(
    text=text,
    file_path="output/output1.wav",
    speaker_wav="reference_voices/referance_male.wav",  # Local reference voice file
    language="en",                # Change language code if needed
    split_sentences=True,
    speed=1.2
)
