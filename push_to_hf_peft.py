import argparse
import os
import torch
from models.modeling_xvla import XVLA
from huggingface_hub import login
from dotenv import load_dotenv

def main():
    parser = argparse.ArgumentParser(description="Upload custom XVLA LoRA adapter to Hugging Face Hub")
    parser.add_argument("--model_name", type=str, required=True, help="Target HF repo for the ADAPTER")
    parser.add_argument("--model_path", type=str, required=True, help="Local directory containing base model checkpoints")
    parser.add_argument("--LoRA_path", type=str, required=True, help="Path to your local LoRA adapter weights")
    args = parser.parse_args()

    load_dotenv()

    hf_token = os.environ.get("HF_WRITE_TOKEN")

    if hf_token:
        login(hf_token)
    else:
        print("HF_WRITE_TOKEN not found in environment variables.")
        return

    # --------------------------------------------------------------------------
    # 1. Load the Base Model and Wrap with LoRA
    # --------------------------------------------------------------------------
    try:
        print("\n📦 Loading base XVLA model...")
        base_model = XVLA.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
        
        print(f"🔸 Wrapping base model with LoRA weights from {args.LoRA_path} ...")
        from peft import PeftModel
        
        peft_model = PeftModel.from_pretrained(
            base_model,
            args.LoRA_path,
            torch_dtype=torch.float32,
        )
        print("LoRA layers initialized onto base architecture.")

        print(f"Pushing LoRA adapter to {args.model_name}...")
        
        peft_model.push_to_hub(repo_id=args.model_name)
        print("Adapter successfully pushed to Hugging Face!")
        
    except Exception as e:
        # If your local LoRA directory already contains everything properly formatted, 
        # you can fallback to the direct huggingface_hub utility API.
        print(f"Structural loading failed: {e}")
        # print("Attempting direct directory upload backup...")
        # from huggingface_hub import HfApi
        # api = HfApi()
        # try:
        #     api.upload_folder(
        #         folder_path=args.LoRA_path,
        #         repo_id=args.model_name,
        #         repo_type="model"
        #     )
        #     print("Backup folder upload successful.")
        # except Exception as backup_err:
        #     print(f"Backup upload failed: {backup_err}")

if __name__ == "__main__":
    main()
