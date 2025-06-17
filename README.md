## Training

You can train the model using the following command:

```
python scripts/train.py fit --config configs/classification.yaml
```

---

## Environment Setup

Follow the steps below to set up the environment.

### 1. Create and Activate Conda Environment

```
conda create -n senatra python=3.10 -y
conda activate senatra
```

### 2. Install PyTorch with CUDA 11.8 Support

```
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Required Dependencies

```
# NATTEN (Neighborhood Attention)
pip install natten==0.17.4+torch250cu118 -f https://shi-labs.com/natten/wheels

# PyTorch Lightning
pip install lightning

# General dependencies to fix pip
pip install sniffio certifi scipy itsdangerous
pip install "h11<0.15,>=0.13" "urllib3>=1.26.11"
pip install python-dateutil click fastapi protobuf pyjwt requests six websocket-client
pip install wcwidth pytz shellingham

# Argument parsing with signature support
pip install -U 'jsonargparse[signatures]>=4.27.7'

# Experiment tracking with Weights & Biases
pip install wandb
```

---

## Install `attn_gym` for Rotary Positional Embeddings (RoPE)

```
git clone https://github.com/pytorch-labs/attention-gym.git
cd attention-gym
pip install .
```

---

