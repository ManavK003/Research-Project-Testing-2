from ASRPipeline import ASRPipeline

def main():
    pipeline = ASRPipeline()
    baseline_wer = pipeline.run_baseline_evaluation()
    print(f"\nBaseline WER: {baseline_wer:.4f}")

if __name__ == "__main__":
    main()
