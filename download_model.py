from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ByteDance-Seed/BFS-Prover-V1-7B",
    local_dir="/data/not_backed_up/yxu209/DSP-Plus/vllm/models/BFS-Prover-V1-7B",
)

print("Download complete.")
