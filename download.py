import os

from runner import init_model

if __name__ == "__main__":
    init_model(
        hf_auth_token=os.environ["HF_AUTH_TOKEN"],
        torch_device="cpu",
    )
