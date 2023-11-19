import os 
import re
import tgt
import librosa
import numpy as np
import copy
import yaml
from text import grapheme_to_phoneme
import audio as Audio
from utils.pitch_tools import get_pitch, get_cont_lf0, get_lf0_cwt
from tqdm import tqdm
from g2p_en import G2p
from pathlib import Path
import argparse
import torch
import dac
from audiotools import AudioSignal

#Class for preprocessing text and wav files into input and output sentence and respective mels
class Preprocessor:
    def __init__(self,dataset):

        #load config
        self.preprocess_config = yaml.load(open(
            os.path.join("config",f"{dataset}_preprocess.yaml"), "r"), Loader=yaml.FullLoader)

        #load variables
        self.sampling_rate = self.preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length =  self.preprocess_config["preprocessing"]["stft"]["hop_length"]
        self.trim_top_db = self.preprocess_config["preprocessing"]["audio"]["trim_top_db"]
        self.filter_length = self.preprocess_config["preprocessing"]["stft"]["filter_length"]
        self.STFT = Audio.stft.TacotronSTFT(
            self.preprocess_config["preprocessing"]["stft"]["filter_length"],
            self.preprocess_config["preprocessing"]["stft"]["hop_length"],
            self.preprocess_config["preprocessing"]["stft"]["win_length"],
            self.preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            self.preprocess_config["preprocessing"]["audio"]["sampling_rate"],
            self.preprocess_config["preprocessing"]["mel"]["mel_fmin"],
            self.preprocess_config["preprocessing"]["mel"]["mel_fmax"],)

        self.window_time = self.preprocess_config["token"]["window_time"]
        self.time_step = self.hop_length / self.sampling_rate * 1000
        self.time_step = self.time_step/1000
        self.window_step = round(self.window_time/self.time_step)
        self.window_length = round((3/0.5)* self.window_step)

        # print(self.window_length,self.window_step)

        self.pitch_token = self.preprocess_config["token"]["pitch"]
        self.duration_token = self.preprocess_config["token"]["duration"]
        self.in_dir  = self.preprocess_config["path"]["in_dir"]
        self.text_grid_dir = self.preprocess_config["path"]["text_grid_dir"]
        self.save_dir = self.preprocess_config["path"]["save_dir"]

        # self.save_dir = os.path.join(self.save_dir,dataset)
        self.save_mels = os.path.join(self.save_dir,"mels")
        self.save_wavs = os.path.join(self.save_dir,"wavs")
        self.save_codecs = os.path.join(self.save_dir,"codecs")
        Path(self.save_mels).mkdir(parents=True, exist_ok=True)
        # Path(self.save_wavs).mkdir(parents=True, exist_ok=True)
        Path(self.save_codecs).mkdir(parents=True, exist_ok=True)
        self.dac_model = dac.DAC.load(dac.utils.download(model_type="24khz"))
        self.dac_model.to('cuda')
        self.dac_model.eval()
    #get alignment of phones to duration in wav file
    def get_alignment(self,tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        mel2ph = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate /self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        # Get mel2ph
        for ph_idx in range(len(phones)):
            mel2ph += [ph_idx + 1] * durations[ph_idx]
        assert sum(durations) == len(mel2ph)

        return phones, durations, mel2ph, start_time, end_time

    #get alignement of phone to words
    def get_phone_word(self,tier,word):
        sil_phones = ["sil", "sp", "spn"]

        phones = []

        words_intervals = [_word for _word in word._objects if word != ""]
        word_phone = [[] for _word in words_intervals]
        word_counter = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue

            if p not in sil_phones:
                word_phone[word_counter].append(p)
                if t.end_time >= words_intervals[word_counter].end_time:
                    word_counter += 1

        return word_phone

    #load audio from wav into numpy
    def load_audio(self,wav_path):
        wav_raw, _ = librosa.load(wav_path, self.sampling_rate)
        _, index = librosa.effects.trim(wav_raw, top_db= self.trim_top_db, frame_length= self.filter_length, hop_length= self.hop_length)
        wav = wav_raw[index[0]:index[1]]
        duration = (index[1] - index[0]) / self.hop_length
        return wav_raw.astype(np.float32), wav.astype(np.float32), int(duration)

    def get_f0cwt(self,f0):
        uv, cont_lf0_lpf = get_cont_lf0(f0)
        logf0s_mean_org, logf0s_std_org = np.mean(cont_lf0_lpf), np.std(cont_lf0_lpf)
        logf0s_mean_std_org = np.array([logf0s_mean_org, logf0s_std_org])
        cont_lf0_lpf_norm = (cont_lf0_lpf - logf0s_mean_org) / logf0s_std_org
        Wavelet_lf0, scales = get_lf0_cwt(cont_lf0_lpf_norm)
        return Wavelet_lf0, scales, logf0s_mean_std_org

    #generate mels using rolling window
    def get_mels(self,mel,duration):
        time = self.window_length
        total_duration = sum(duration)
        mels = []
        mels.append(mel[:,:time])
        time += self.window_step
        while time < total_duration:
            mels.append(mel[:,time-self.window_length:time])
            time += self.window_step
        return mels
    
    def encode_dac(self,wav):
        with torch.no_grad():

            signal = AudioSignal(wav,self.sampling_rate)
            signal.to(self.dac_model.device)
            x = self.dac_model.preprocess(signal.audio_data, signal.sample_rate)
            z, codes, latents, _, _ = self.dac_model.encode(x)
            del signal
            result = z.cpu()
            del z
            return result
    
    def get_codecs(self,wav,duration):
        time = self.window_length
        total_duration = sum(duration)
        codecs = []
        wav_slice = wav[:time*self.hop_length]
        codecs.append(self.encode_dac(wav_slice))
        time += self.window_step
        while time < total_duration:
            wav_slice = wav[(time-self.window_length)*self.hop_length:time*self.hop_length]
            codecs.append(self.encode_dac(wav_slice))
            time += self.window_step
        return codecs

    #process wav file and textgrid to generate input and output sentence
    def process_utterance( self,tg_path, speaker, basename, in_dir):
        sup_out_exist, unsup_out_exist = True, True
        wav_path = os.path.join(in_dir, speaker, "{}_{}.wav".format(speaker,basename))
        text_path = os.path.join(in_dir, speaker, "{}_{}.lab".format(speaker,basename))

        wav_raw, wav, duration = self.load_audio(wav_path)


        if duration < self.window_length:
            return (duration,None)

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")
        # phone = grapheme_to_phoneme(raw_text, G2p())
        # phones = "{" + "}{".join(phone) + "}"
        # phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
        # text_unsup = phones.replace("}{", " ")
        # print(tg_path)
        # print(os.path.exists(tg_path))
        # Supervised duration features

        if os.path.exists(tg_path):
        
            # Get alignments
            textgrid = tgt.io.read_textgrid(tg_path)
            phone, duration, mel2ph, start, end = self.get_alignment(
                textgrid.get_tier_by_name("phones")
            )

            #if wav file shorter than 3 secs
            if sum(duration) < self.window_length:
                return (sum(duration),None)
            
            #get phone to word alignments
            word_phone = self.get_phone_word(
                textgrid.get_tier_by_name("phones"),textgrid.get_tier_by_name("words")
            )
            # print(word_phone)

            text_sup = "{" + " ".join(phone) + "}"
            if start >= end:
                sup_out_exist = False
            else:
                # Read and trim wav files
                wav, _ = librosa.load(wav_path, self.sampling_rate)
                wav = wav.astype(np.float32)
                wav = wav[
                    int(self.sampling_rate * start) : int(self.sampling_rate * end)
                ]

                # Compute mel-scale spectrogram and energy
                mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
                # wav, mel_spectrogram, energy = self.STFT.mel_spectrogram(wav)
                mel_spectrogram = mel_spectrogram[:, : sum(duration)]
                # energy = energy[: sum(duration)]
                with_f0 =  True
                with_f0cwt =  True

                # Compute pitch
                if with_f0:
                    f0_sup, pitch_sup = get_pitch(wav, mel_spectrogram.T,self.preprocess_config)
                    if f0_sup is None: 
                        sup_out_exist = False
                    else:
                        # spec_f0_to_figure(mel_spectrogram.T, {"f0_sup":f0_sup}, filename=os.path.join(self.out_dir, f"{basename}_sup.png"))
                        f0_sup = f0_sup[: sum(duration)]
                        pitch_sup = pitch_sup[: sum(duration)]

                        if with_f0cwt:
                            cwt_spec_sup, cwt_scales_sup, f0cwt_mean_std_sup = self.get_f0cwt(f0_sup)
                            if np.any(np.isnan(cwt_spec_sup)):
                                sup_out_exist = False
                        assert mel_spectrogram.shape[1] == len(f0_sup)

            #clean random delimiters 
            delimiters = [" ", "-"]
            result = [raw_text]
            for delimiter in delimiters:
                temp_result = []
                for item in result:
                    temp_result.extend(item.split(delimiter))
                result = temp_result
            raw_words = result
            # print(raw_words)
            # print(len(raw_words),len(word_phone),len(raw_words)==len(word_phone))

            skip = False
            input_text = ""
            output_text = ""
            sil_phones = ["sil", "sp", "spn"]
            pitch_time = 0
            phone_count = 0
            word_count = 0

            #if words and word aligned phones dont match up skip
            if len(raw_words)!=len(word_phone):
                # print("misaligned matchup")
                return (sum(duration),None)
            if not skip:
                for word_ in word_phone:
                    # if word_count > 0:
                    #     input_text += " "
                    #     output_text += " "

                    input_text += raw_words[word_count]
                    output_text += raw_words[word_count]
                    input_text += " "
                    output_text += " "
                    for phone_ in word_:
                        phone_pitch = pitch_sup[pitch_time:pitch_time+duration[phone_count]]
                        phone_filtered = [i for i in phone_pitch if i != 1]
                        if phone_filtered != []:
                            average_pitch = round(sum(phone_filtered)/len(phone_filtered))
                        else:
                            average_pitch = 1
                        # print(average_pitch)
                        pitch_time += duration[phone_count]

                        input_text += f"{phone[phone_count]} "
                        input_text += f"{self.pitch_token} "
                        input_text += f"{self.duration_token} "

                        output_text += f"{phone[phone_count]} "
                        output_text += f"{average_pitch} "
                        output_text += f"{duration[phone_count]} "
                        phone_count += 1
                        
                        try:
                            if phone[phone_count] in sil_phones:
                                phone_pitch = pitch_sup[pitch_time:pitch_time+duration[phone_count]]
                                phone_filtered = [i for i in phone_pitch if i != 1]
                                if phone_filtered != []:
                                    average_pitch = round(sum(phone_filtered)/len(phone_filtered))
                                else:
                                    average_pitch = 1
                                # print(average_pitch)
                                pitch_time += duration[phone_count]

                                input_text += f"{phone[phone_count]} "
                                input_text += f"{self.pitch_token} "
                                input_text += f"{self.duration_token} "

                                output_text += f"{phone[phone_count]} "
                                output_text += f"{average_pitch} "
                                output_text += f"{duration[phone_count]} "
                                phone_count += 1
                        except:
                            pass
                    word_count += 1


            mels = self.get_mels(mel_spectrogram, duration)
            codecs = self.get_codecs(wav, duration)
            # print()
            # print(speaker, basename)
            # print(sum(duration)*time_step)
            # print(raw_text)
            # print(input_text)
            # print(output_text)
            # print()

            return (sum(duration),(speaker, basename,raw_text,input_text, output_text, mels,codecs))
        else:
            # print("No TextGrid")
            return (duration,None)

    #process all data for particular dataset
    def preprocess(self):
        in_sub_dirs = [p for p in os.listdir(self.in_dir) if os.path.isdir(os.path.join(self.in_dir, p))]
        # print(in_sub_dirs)
        save_file = []
        save_count = 0
        total_duration = 0
        duration_filtered = 0
        for i, speaker in enumerate(tqdm(in_sub_dirs)):
            if os.path.isdir(os.path.join(self.in_dir, speaker)):
                for wav_name in tqdm(os.listdir(os.path.join(self.in_dir, speaker))):
                    if ".wav" not in wav_name:
                        continue
                    # print(wav_name)
                    basename_ = wav_name.split(".")[0]
                    basename = "_".join(basename_.split("_")[1:])
                    chapter_name = basename_.split("_")[1]
                    # print("Here:",speaker,chapter_name,basename)

                    tg_path = os.path.join(
                        self.text_grid_dir, speaker,chapter_name, "{}_{}.TextGrid".format(speaker,basename)
                    )
                    # print(tg_path)

                    duration,results = self.process_utterance(tg_path, speaker, basename, self.in_dir)
                    total_duration += duration

                    #if longer than 3 seconds
                    if duration > self.window_length and results != None:
                        duration_filtered += duration
                        (speaker, basename,raw_text,input_text, output_text, mels,codecs) = results
                    
                        #for each generated mel
                        for mel_num in range(len(mels)):
                            #save mel and sentence and speaker basename
                            basename_mel =speaker + "_" + basename + "_" + str(mel_num).zfill(2)
                            save_file.append(speaker + "|" + basename_mel + "|" + raw_text + "|" + input_text + "|" + output_text)
                            save_mel_path = os.path.join(self.save_mels,f"{basename_mel}.npy")
                            np.save(save_mel_path,mels[mel_num])

                            # save_wav_path = os.path.join(self.save_wavs,f"{basename_mel}.wav")
                            # mel_torch = torch.from_numpy(mels[mel_num])
                            # Audio.tools.inv_mel_spec(mel_torch,save_wav_path, self.STFT)

                            save_codec_path = os.path.join(self.save_codecs,f"{basename_mel}.npy")
                            np.save(save_codec_path,codecs[mel_num].numpy())

                            # save_codec_path = os.path.join(self.save_codecs,f"{basename_mel}.wav")
                            # back = self.dac_model.decode(codecs[mel_num])
                            # back = back.cpu()
                            # back = back.detach().numpy()
                            # y = AudioSignal(back,self.sampling_rate)
                            # y.write(save_codec_path)
                            # del back
                            
                        del codecs



                        # save_count += 1

                        # #Save a few for testing
                        # if save_count >= 2:
                        #     break

            # #Save a few for testing
            # if save_count >= 2:
            #             break

        with open(os.path.join(self.save_dir,"dataset.txt"), "w") as f:
            for word in save_file:
                f.write(word)
                f.write("\n")
        print(total_duration)
        print(duration_filtered)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    args = parser.parse_args()
    preprocess = Preprocessor(args.dataset)
    preprocess.preprocess()

