# Research-Project-Testing-2

The project presents the fine-tuning of OpenAI's Whisper-base model for improved automatic speech recognition (ASR) performance on children's speech data using WER (Word Error Rate as the key measuring metric).

#Steps to Run the Project:

Open Project on VSCode and create virtual environment on using this command on VSCode terminal:
python3 -m venv venv

Activate virtual environment
source venv/bin/activate

Install Requirements:
pip install -r requirements.txt

To check only baseline untuned performance:
python run_baseline_only.py

for baseline+ fine tuned performance:

python fine_tuning_script.py


#Data Pre-processing Strategy

Audio Segmentation: Extracted audio segments based on provided timestamps from transcript files for getting only child’s speaking section.
Normalization: Applied amplitude normalization to handle varying volume levels.
Silence Removal: Energy-based trimming to remove silence from segment beginnings and ends.
Quality Filtering: Removed segments shorter than 0.5 seconds to ensure minimum audio quality for transcription.
Memory Management: Limited dataset size to 1500 training and 500 test segments to prevent memory overflow.

#Children's Speech Optimization
When initially transcribing I noticed a lot transcripts being generated in other languages when all audio was in English, so added English as language requirement. 

Language Forcing: Implemented English-only decoding to prevent incorrect language detection
Most of the kids voices were very feeble, faint, words were not properly enunciated, and high pitched, so lowered confidence threshold, and made beams search 5 as optimal observation. 
Generation Parameters:

Temperature: 0.1 (lower confidence threshold for soft speech)
Beam Search: 5 beams for better transcription paths

Fine-tuning Approach

Method: Parameter-Efficient Fine-tuning using LoRA (Low-Rank Adaptation)
Base Model: OpenAI Whisper-base (72.8M parameters)
Trainable Parameters: 294,912 (0.40% of total parameters)

3.2 LoRA Configuration

Rank (r): 8
Alpha: 16
Target Modules: q_proj, v_proj (attention layers)
Dropout: 0.1

3.3 Training Parameters

Batch Size: 4 per device with gradient accumulation (steps=2)
Learning Rate: 5e-5 with 100 warmup steps
Max Steps: 1000
Optimization: FP16 precision with gradient checkpointing
Evaluation: Every 100 steps using Word Error Rate (WER)

A few times facing memory error so fixed memory segments to 800 to handle the loads on Google Collab. 

#Evaluation Observation

Improvement of WER from 1.5524 in baseline 

To WER 0.8676 in fine tuned.

