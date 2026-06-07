import argparse
import os
import torch
from models.modeling_xvla import XVLA
from models.processing_xvla import XVLAProcessor
from huggingface_hub import login
from dotenv import load_dotenv

def main():
    parser = argparse.ArgumentParser(description="Upload custom XVLA checkpoint to Hugging Face Hub")
    parser.add_argument("--model_name", type=str, required=True, help="Target HF repo (e.g. 'your-username/X-VLA-Pt-Custom')")
    parser.add_argument("--model_path", type=str, required=True, help="Local directory containing your checkpoints/configs")
    args = parser.parse_args()

    load_dotenv()

    hf_token = os.environ.get("HF_WRITE_TOKEN")

    if hf_token:
        login(hf_token)
    else:
        print("HF_WRITE_TOKEN not found in environment variables.")
        return

    try:
        print("\n🧩 Loading XVLAProcessor...")
        processor = XVLAProcessor.from_pretrained(args.model_path)
        print("✅ XVLAProcessor loaded successfully.")
        
        # Link this custom class to AutoProcessor mapping
        XVLAProcessor._auto_class = "AutoProcessor"
        
        print(f"🚀 Pushing processor to {args.model_name}...")
        # This uploads preprocessor_config.json and maps it
        processor.push_to_hub(repo_id=args.model_name)
        print("✅ Processor successfully pushed.")
        
    except Exception as e:
        print(f"⚠️ Processor steps skipped or failed: {e}")

    try:
        print("\n📦 Loading XVLA model from pretrained checkpoint...")
        model = XVLA.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
        print("✅ Model successfully loaded.")
        
        XVLA._auto_class = "AutoModel"
        
        print(f"🚀 Pushing model weights to {args.model_name}...")
        model.push_to_hub(repo_id=args.model_name)
        print("✅ Model successfully pushed.")
        
    except Exception as e:
        print(f"❌ Failed to load or push model: {e}")
        return

    print("\npy 📄 Pushing custom Python architecture files...")
    from huggingface_hub import HfApi
    api = HfApi()
    
    try:
        api.upload_file(
            path_or_fileobj="models/modeling_xvla.py",
            path_in_repo="modeling_xvla.py",
            repo_id=args.model_name
        )
        api.upload_file(
            path_or_fileobj="models/processing_xvla.py",
            path_in_repo="processing_xvla.py",
            repo_id=args.model_name
        )
        print("Custom python code scripts uploaded successfully.")
    except Exception as e:
        print(f"Failed to automatically upload code files: {e}")
if __name__ == "__main__":
    main()
