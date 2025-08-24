import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from ASRPipeline import ASRPipeline  # Import from previous script

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def compute_metrics(eval_pred):
    pred_ids, label_ids = eval_pred
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    from jiwer import wer
    wer_score = wer(label_str, pred_str)
    return {"wer": wer_score}

class FineTuner:
    def __init__(self, data_path="data"):
        self.pipeline = ASRPipeline(data_path)
        self.processor = self.pipeline.processor
        self.model = self.pipeline.model
        
    def setup_lora(self, r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"]):
        """Setup LoRA configuration"""
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
    def fine_tune(self, epochs=3, batch_size=4, learning_rate=5e-5):
        """Fine-tune the model"""
        print("="*50)
        print("FINE-TUNING WHISPER WITH LORA")
        print("="*50)
        
        # Load training data
        print("Loading training data...")
        train_segments = self.pipeline.load_and_segment_data("train", max_segments=800)  # ADDED LIMIT
        train_segments = self.pipeline.preprocess_audio(train_segments)
        train_dataset = self.pipeline.create_dataset(train_segments, for_training=True)  # FIXED: Added for_training=True
        
        # Load test data
        print("Loading test data...")
        test_segments = self.pipeline.load_and_segment_data("test", max_segments=500)   # ADDED LIMIT
        test_segments = self.pipeline.preprocess_audio(test_segments)
        test_dataset = self.pipeline.create_dataset(test_segments, for_training=True)   # FIXED: Added for_training=True
        
        # Setup LoRA
        self.setup_lora()
        
        # Data collator
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./whisper-finetuned",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=2,
            learning_rate=learning_rate,
            warmup_steps=100,
            max_steps=1000,
            gradient_checkpointing=True,
            fp16=True,
            eval_strategy="steps",  # Changed from evaluation_strategy
            eval_steps=100,
            save_steps=500,
            logging_steps=25,
            report_to=[],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        # Start training
        print("Starting training...")
        trainer.train()
        
        # Save the model
        trainer.save_model("./whisper-finetuned-final")
        
        print("Fine-tuning completed!")
        return trainer
    
    def evaluate_finetuned(self, model_path="./whisper-finetuned-final"):
        """Evaluate the fine-tuned model"""
        print("="*50)
        print("EVALUATING FINE-TUNED MODEL")
        print("="*50)
        
        # Load fine-tuned model
        from peft import PeftModel
        base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
        finetuned_model = PeftModel.from_pretrained(base_model, model_path)
        finetuned_model.to(self.pipeline.device)
        
        # Load test data
        test_segments = self.pipeline.load_and_segment_data("test", max_segments=500)    # ADDED LIMIT
        test_segments = self.pipeline.preprocess_audio(test_segments)
        test_dataset = self.pipeline.create_dataset(test_segments)  # CORRECT: No for_training parameter for evaluation
        
        # Evaluate
        wer_score, _, _ = self.pipeline.evaluate_model(test_dataset, finetuned_model, suffix="tuned")
        
        print(f"Fine-tuned model WER: {wer_score:.4f}")
        return wer_score

def main():
    # Initialize fine-tuner
    fine_tuner = FineTuner()
    
    # First run baseline evaluation if not done
    print("Running baseline evaluation...")
    baseline_wer = fine_tuner.pipeline.run_baseline_evaluation()
    
    # Fine-tune the model
    trainer = fine_tuner.fine_tune(epochs=3, batch_size=4)
    
    # Evaluate fine-tuned model
    finetuned_wer = fine_tuner.evaluate_finetuned()
    
    # Results summary
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    print(f"Original Whisper WER: {baseline_wer:.4f}")
    print(f"Fine-tuned Whisper WER: {finetuned_wer:.4f}")
    print(f"Improvement: {((baseline_wer - finetuned_wer) / baseline_wer * 100):.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
