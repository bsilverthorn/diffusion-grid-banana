```bash
# new dependency? edit Pipfile
# install dependencies (& update Pipfile.lock)
pipenv install --dev
# update requirements.txt
pipenv requirements > requirements.txt
```

local execution:

```bash
export HF_AUTH_TOKEN="..."
export DIFFUSION_TORCH_DEVICE="cpu"
python server.py
```
