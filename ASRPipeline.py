import os
import librosa
import torch
import pandas as pd
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Dataset, DatasetDict
import soundfile as sf
from jiwer import wer
import json
from tqdm import tqdm
import re

class ASRPipeline:
    def __init__(self, data_path="data"):
        self.data_path = data_path
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def load_and_segment_data(self, split="train", max_segments=2000):  # ADDED LIMIT
        """Load and segment audio files based on timestamps"""
        split_path = os.path.join(self.data_path, split)
        segments = []
        
        folders = [f for f in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, f))]
        total_folders = len(folders)
        
        print(f"Processing {total_folders} folders in {split} set...")
        print(f"MEMORY LIMIT: Will stop at {max_segments} segments to prevent crash")
        
        processed_files = 0
        total_segments = 0
        
        for folder_idx, folder in enumerate(folders, 1):
            # MEMORY PROTECTION: Stop if we have enough segments
            if total_segments >= max_segments:
                print(f"Reached segment limit ({max_segments}). Stopping to prevent memory crash.")
                break
                
            folder_path = os.path.join(split_path, folder)
            
            # Find txt and wav files (STRICT: ignore ALL generated files)
            files = os.listdir(folder_path)
            txt_files = [f for f in files if f.endswith('.txt') 
                        and not ('_untuned' in f or '_tuned' in f or '_baseline' in f)]
            wav_files = [f for f in files if f.endswith('.wav')]
            
            if len(txt_files) != 1 or len(wav_files) != 1:
                print(f"Skipping {folder}: Expected 1 txt and 1 wav file")
                continue
                
            txt_file = txt_files[0]
            wav_file = wav_files[0]
            
            # Load transcript
            transcript_path = os.path.join(folder_path, txt_file)
            audio_path = os.path.join(folder_path, wav_file)
            
            # Load audio with error handling
            try:
                audio, sr = librosa.load(audio_path, sr=16000)
            except Exception as e:
                print(f"Error loading audio {wav_file}: {e}")
                print(f"Skipping folder {folder}")
                continue
            
            # Parse transcript
            with open(transcript_path, 'r') as f:
                lines = f.readlines()
            
            folder_segments = 0
            for line in lines:
                # MEMORY PROTECTION: Check limit before processing each segment
                if total_segments >= max_segments:
                    break
                    
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split('\t')
                if len(parts) >= 3:
                    start_time = float(parts[0])
                    end_time = float(parts[1])
                    text = parts[2]
                    
                    # Extract audio segment
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    
                    if end_sample <= len(audio):
                        segment_audio = audio[start_sample:end_sample]
                        
                        if len(segment_audio) > 0:
                            segments.append({
                                'audio': segment_audio,
                                'text': text,
                                'folder': folder,
                                'start_time': start_time,
                                'end_time': end_time,
                                'file_path': folder_path
                            })
                            folder_segments += 1
                            total_segments += 1
            
            processed_files += 1
            
            # Clear audio from memory after processing
            del audio
            
            if folder_idx % 10 == 0 or folder_idx == total_folders:
                print(f"Processed {folder_idx}/{total_folders} folders, {total_segments} segments so far")
        
        print(f"Total processed: {processed_files} folders, {total_segments} segments")
        return segments
    
    def preprocess_audio(self, audio_segments, batch_size=100):  # ADDED BATCH PROCESSING
        """Preprocess audio segments in batches to save memory"""
        processed = []
        
        # Process in batches to save memory
        for i in tqdm(range(0, len(audio_segments), batch_size), desc="Preprocessing audio batches"):
            batch = audio_segments[i:i+batch_size]
            
            for segment in batch:
                audio = segment['audio']
                
                # Normalization
                if len(audio) > 0:
                    audio = audio / np.max(np.abs(audio))
                    
                    # Remove silence from beginning and end
                    # Simple energy-based trimming
                    energy = np.abs(audio)
                    energy_threshold = 0.01 * np.max(energy)
                    
                    # Find start and end of speech
                    start_idx = 0
                    end_idx = len(audio)
                    
                    for j in range(len(energy)):
                        if energy[j] > energy_threshold:
                            start_idx = max(0, j - int(0.1 * 16000))  # 0.1s padding
                            break
                    
                    for j in range(len(energy) - 1, -1, -1):
                        if energy[j] > energy_threshold:
                            end_idx = min(len(audio), j + int(0.1 * 16000))  # 0.1s padding
                            break
                    
                    if end_idx > start_idx:
                        audio = audio[start_idx:end_idx]
                
                # Minimum length filter (0.5 seconds)
                if len(audio) >= 0.5 * 16000:
                    segment_copy = segment.copy()
                    segment_copy['audio'] = audio
                    processed.append(segment_copy)
        
        print(f"After preprocessing: {len(processed)} segments (filtered from {len(audio_segments)})")
        return processed
    
    def create_dataset(self, segments, for_training=False):
        """Create dataset for training/evaluation"""
        if for_training:
            # For training: convert to proper format IN BATCHES to save memory
            input_features = []
            labels = []
            
            print("Converting audio segments to training format...")
            batch_size = 50  # Process in small batches to prevent memory crash
            
            for i in tqdm(range(0, len(segments), batch_size), desc="Converting batches"):
                batch = segments[i:i+batch_size]
                
                for segment in batch:
                    # Convert audio to input features
                    inputs = self.processor(segment['audio'], sampling_rate=16000, return_tensors="pt")
                    input_features.append(inputs.input_features.squeeze().numpy())
                    
                    # Convert text to labels
                    labels_encoding = self.processor.tokenizer(segment['text'], return_tensors="pt")
                    labels.append(labels_encoding.input_ids.squeeze().numpy())
                    
                    # Clear memory after each conversion
                    del inputs, labels_encoding
                
                # Clear batch memory
                if i % (batch_size * 4) == 0:  # Every 4 batches
                    import gc
                    gc.collect()
            
            dataset_dict = {
                'input_features': input_features,
                'labels': labels
            }
        else:
            # For evaluation: keep original format
            dataset_dict = {
                'audio': [s['audio'] for s in segments],
                'text': [s['text'] for s in segments],
                'folder': [s['folder'] for s in segments]
            }
        
        dataset = Dataset.from_dict(dataset_dict)
        return dataset
    
    def evaluate_model(self, dataset, model=None, save_transcripts=True, suffix="untuned"):
        """Evaluate model and calculate WER"""
        if model is None:
            model = self.model
        
        model.eval()
        predictions = []
        references = []
        transcript_data = []
        
        print(f"Evaluating model on {len(dataset)} samples...")
        
        # ADDED: English-only and children's voice optimizations
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="english", task="transcribe")
        
        with torch.no_grad():
            for i, sample in enumerate(tqdm(dataset)):
                # Process audio on-the-fly
                audio = sample['audio']
                inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
                input_features = inputs.input_features.to(self.device)
                
                # Generate prediction with English-only and children's voice optimizations
                predicted_ids = model.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids,  # English-only
                    temperature=0.1,                        # Lower confidence threshold for children
                    num_beams=5,                           # More beam paths for soft voices
                    length_penalty=1.0,
                    repetition_penalty=1.1
                )
                predicted_text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                
                reference_text = sample['text']
                
                predictions.append(predicted_text)
                references.append(reference_text)
                
                transcript_data.append({
                    'folder': sample['folder'],
                    'reference': reference_text,
                    'prediction': predicted_text
                })
                
                # MEMORY CLEANUP: Clear GPU cache periodically
                if (i + 1) % 50 == 0:
                    torch.cuda.empty_cache()
                    print(f"Processed {i + 1}/{len(dataset)} samples")
                
                # MEMORY CLEANUP: Delete tensors after use
                del inputs, input_features, predicted_ids
        
        # Calculate WER
        wer_score = wer(references, predictions)
        print(f"Word Error Rate: {wer_score:.4f}")
        
        if save_transcripts:
            self.save_transcripts(transcript_data, suffix)
        
        return wer_score, predictions, references
    
    def save_transcripts(self, transcript_data, suffix):
        """Save transcripts to respective folders"""
        folder_transcripts = {}
        
        # Group by folder
        for item in transcript_data:
            folder = item['folder']
            if folder not in folder_transcripts:
                folder_transcripts[folder] = []
            folder_transcripts[folder].append(item)
        
        saved_count = 0
        
        for folder, transcripts in folder_transcripts.items():
            # Find the folder path
            for split in ['train', 'test']:
                folder_path = os.path.join(self.data_path, split, folder)
                if os.path.exists(folder_path):
                    # Find original transcript name (STRICT: ignore ALL generated files)
                    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt') 
                                and not ('_untuned' in f or '_tuned' in f or '_baseline' in f)]
                    if txt_files:
                        original_name = txt_files[0]  # e.g., T2027215.txt
                        base_name = original_name.replace('.txt', '')  # e.g., T2027215
                        new_filename = f"{base_name}_{suffix}.txt"  # e.g., T2027215_untuned.txt
                        new_path = os.path.join(folder_path, new_filename)
                        
                        # Write predictions
                        with open(new_path, 'w') as f:
                            for t in transcripts:
                                f.write(f"{t['prediction']}\n")
                        
                        saved_count += 1
                    break
        
        print(f"Saved transcripts to {saved_count} folders with suffix '{suffix}'")
    
    def run_baseline_evaluation(self):
        """Run evaluation on original Whisper model"""
        print("="*50)
        print("BASELINE EVALUATION (Original Whisper)")
        print("="*50)
        
        # Load and preprocess test data with memory limit
        test_segments = self.load_and_segment_data("test", max_segments=1000)  # REDUCED FOR SAFETY
        test_segments = self.preprocess_audio(test_segments)
        test_dataset = self.create_dataset(test_segments)
        
        # Evaluate original model
        wer_score, _, _ = self.evaluate_model(test_dataset, suffix="untuned")
        
        print(f"Baseline WER: {wer_score:.4f}")
        return wer_score

def main():
    # Initialize pipeline
    pipeline = ASRPipeline()
    
    # Run baseline evaluation
    baseline_wer = pipeline.run_baseline_evaluation()
    
    print(f"\nBaseline evaluation complete!")
    print(f"Original Whisper WER: {baseline_wer:.4f}")
    print(f"Check your data folders for *_untuned.txt files with transcriptions")

if __name__ == "__main__":
    main()
