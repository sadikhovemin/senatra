### Training

You can train the model using the following command:

```bash
python scripts/train.py fit --config configs/classification.yaml
```


### Create Env
conda create -n senatra python=3.10 -y
conda activate senatra
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install natten==0.17.4+torch250cu118 -f https://shi-labs.com/natten/wheels
pip install lightning
pip install sniffio certifi scipy itsdangerous
pip install "h11<0.15,>=0.13" "urllib3>=1.26.11"
pip install python-dateutil click fastapi protobuf pyjwt requests six websocket-client
pip install wcwidth pytz shellingham
pip install -U 'jsonargparse[signatures]>=4.27.7'
pip install wandb


# git clone attn_gym
cd attn_gym
pip install .