from huggingface_hub import snapshot_download
import os

local_dir = "./libero_xvla"

snapshot_download(
    repo_id="2toINF/Libero-XVLA-format",
    repo_type="dataset",
    local_dir=local_dir,
    allow_patterns=[
        "libero_spatial/*",
    ]
)

print(f"Downloaded spatial data and metadata to: {os.path.abspath(local_dir)}")