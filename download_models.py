from huggingface_hub import snapshot_download

def download():
    """
    Downloads and caches the required models from Hugging Face Hub
    into the default cache location.
    """
    print("Downloading Hunyuan3D-2.1 model...")
    # This single download covers both shape generation and texture painting models.
    snapshot_download(repo_id="tencent/Hunyuan3D-2.1", repo_type="model")
    print("Hunyuan3D-2.1 model downloaded.")

    print("Downloading DINO model...")
    # This is required by the texture painting pipeline.
    snapshot_download(repo_id="facebook/dinov2-giant", repo_type="model")
    print("DINO model downloaded.")

if __name__ == "__main__":
    download() 