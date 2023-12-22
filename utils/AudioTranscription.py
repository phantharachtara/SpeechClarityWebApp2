# Imports
import speech_recognition as sr

from pydub import AudioSegment
from pydub.silence import split_on_silence

import nltk
from nltk.tokenize import sent_tokenize

import spacy
import pickle

class AudioTranscription():
    
    def __init__(self):
        # self.nlp = spacy.load("en_core_web_sm")
        self.nlp = pickle.load(open('models/en_core_web_sm.pkl','rb'))
    
    def recognize_speech(self,recognizer,audio_data):
        ''' Takes speech recognizer instance and sr.AudioData as inputs
            Returns:
                - text extracted from audio data 
                - empty string if no words were extracted
                - empty string if model cannot be reached'''
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        
        except sr.UnknownValueError:
            print("Speech Recognition could not understand audio")
            return ""
        
        except sr.RequestError as e:
            print(f"Could not request results from model {e}")
            return ""

    def silence_split(self,audio_data):
        ''' Takes sr.AudioData as inputs and returns list of sentences'''
        sentences = split_on_silence(audio_data,silence_thresh=-40)
        return sentences
    
    def convert_to_wav(self,data_path,wav_path):
        ''' Converts audio that is not in wavefile format to wavefile and saves it'''
        audio = AudioSegment.from_file(data_path)
        audio.export(wav_path,format="wav")
    
    def sentence_split(self,text):
        ''' Split long text into sentences using Spacy's sentence segmentation module'''
        tokenized_sentence = [s.text for s in self.nlp(text).sents]
        return tokenized_sentence
    
    def process_audio_file(self,data_path):
        ''' Takes in path to audio signal and returns sentence text'''
        
        recognizer = sr.Recognizer() # Instantiate recognizer
        
        if data_path[-3:] != 'wav': # Check for correct file format
            wav_path = data_path.replace(data_path[-3:],'wav')
            self.convert_to_wav(data_path,wav_path)
            data_path = wav_path 
        
        with sr.AudioFile(data_path) as source:
            audio = recognizer.record(source)
            extracted_text = self.recognize_speech(recognizer,audio)
            return extracted_text

    def process_long_audio(self,data_path):
        ''' Takes in a datapath and returns a list of transcribed sentence text'''
        recognizer = sr.Recognizer() # Instantiate recognizer

        if data_path[-3:] != 'wav': # Check for correct file format
            wav_path = data_path.replace(data_path[-3:],'wav')
            self.convert_to_wav(data_path,wav_path)
            data_path = wav_path 

        # Generate audio segment
        audio_segment = AudioSegment.from_file(data_path,format='wav')
        
        # Split into segment by silence
        sentences = self.silence_split(audio_segment)

        # Generate list of sentences
        sentence_list = []
        for i,sentence in enumerate(sentences,start=1):
            raw_audio = sentence.raw_data
            audio_instance = sr.AudioData(raw_audio,sentence.frame_rate,sentence.sample_width)
            extracted_text = self.recognize(recognizer,audio_instance)
            sentence_list.append(extracted_text)

        return sentence_list
    
    def live_transcription(self):
        ''' Perform live transcription by using the microphone as the input '''
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening.....")
            try:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source,timeout=0.5)
                text = recognizer.recognize_google(audio)
                print("Transcription:",text)
            except sr.UnknownValueError:
                print("Sphinx could not understand")
            except sr.RequestError as e:
                print(f'could not request result at the moment')
        

        


    
