import argparse
from OPRTrainer import OnPolicyReplayTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="model/Qwen2.5-7B-Instruct", type=str)
    parser.add_argument("--output-dir", default="output/qwen_opr_ru", type=str)
    parser.add_argument("--rho", default=0.01, type=float, help='replay ratio')
    args = parser.parse_args()

    trainer = OnPolicyReplayTrainer(args.model_path, args.output_dir, args.rho)
    trainer.run()


if __name__ == "__main__":
    main()