from huggingface_hub import hf_hub_download
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ResearchQA dataset from Hugging Face.")
    parser.add_argument('--split', type=str, default='test.json', choices=['full.json', 'test.json', 'valid.json'], help='Which split to download')
    args = parser.parse_args()

    file_path = hf_hub_download(
        repo_id="realliyifei/ResearchQA",
        filename=args.split,
        repo_type="dataset"
    )
    print(f"Downloaded {args.split} to: {file_path}") 