from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="tiiuae/Falcon-H1-1.5B-Instruct",
    local_dir="/home/alaktaou/OSAKA/models/Falcon-H1-1.5B-Instruct",
)

print(path)