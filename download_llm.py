from huggingface_hub import snapshot_download
from pathlib import Path

repo_id = "google/gemma-2-2b-it"  # specify the model repo ID from Hugging Face Hub
local_dir = "data/hf_models"  # hf_models storage dir
local_dir = Path(local_dir) / "--".join(repo_id.split("/"))  # download to a subfolder
local_dir.mkdir(parents=True, exist_ok=True)  # create the directory if it does not exist

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # forbid symlinks
    revision="main"                # the branch name or commit hash to download
)
