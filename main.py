
# Made by Bryce Gernon, edited from https://pytorch.org/tutorials/intermediate/text_to_speech_with_torchaudio.html
# Runs pytorch example code to run voice generator


import sys
import torch
import torchaudio
import matplotlib.pyplot as plt
import time
import IPython


# Define hyperparams + initial vars

OUTPUT_DIR = "C:\\Users\\brger\\PycharmProjects\\VoiceReplicator\\audio"
torch.random.manual_seed(91817)
device = "cpu"
symbol_list = '_-!\'(),.:;? abcdefghijklmnopqrstuvwxyz'
look_up_tables = {s: i for i, s in enumerate(symbol_list)}  # Build lookup tables for TacoTron2
symbol_set = set(symbol_list)  # Build mathematical set out of symbols
print(look_up_tables)
print(symbol_set)
#sys.exit()
source = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
transforms = source.get_text_processor()  # Load symbol->tensor transforms for tacotron, phoneme-based encoding, waveRNN vocoder
spectrolizer = source.get_tacotron2().to(device)  # Load tensor->spectrogram tacotron model
waveformer = source.get_vocoder().to(device)  # Load spectrogram->waveform WaveRNN model

def text_to_sequence(text):
    text = text.lower()  # Convert to lowercase to match symbol set
    return [look_up_tables[s] for s in text if s in symbol_list]  # Converts text to symbols using lookup table

def text_to_spectrogram(text):
    proc, len = transforms(text)
    proc = proc.to(device)
    len = len.to(device)
    spc = spectrolizer.infer(proc, len) # returns spectrogram + unused data, seperate spectrogram
    return spc

def spectrogram_to_waveform(spectrogram):
    return waveformer(spectrogram[0], spectrogram[1])


def main():
    start = time.perf_counter()
    test_text = "Testing Testing Testing"
    code = hash(test_text)
    print(code)
    with torch.inference_mode():
        processed_text = transforms(test_text)
        processed, lengths = processed_text
        print(transforms(test_text))
        print("Phonemes:")
        print([transforms.tokens[i] for i in processed[0, :lengths[0]]])
        spc = text_to_spectrogram(test_text)
        waveform = spectrogram_to_waveform(spc)
    # plt.figure(figsize=(10, 8))
    # plt.plot(x_new, y_new, 'b')
    # plt.plot(x, y, 'ro')
    # plt.title('Bryce Gernon')
    # plt.xlabel('Time(s)')
    # plt.ylabel('Velocity (ft/s)')
    # plt.show()
    # plt.plot(spc[0].cpu().detach().numpy())
    torchaudio.save(OUTPUT_DIR + "\\" + str(code) + "_output_char_wavernn.wav", waveform[0][0:1].cpu(), sample_rate=waveformer.sample_rate)
    plt.imshow(spc[0][0].cpu().detach())
    plt.savefig(OUTPUT_DIR + "\\" + str(code) + "_spectrogram.png")
    plt.show()
    print(time.perf_counter() - start)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
