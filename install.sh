python -m venv scriptspeech --system-site-package
source venv/scriptspeech/bin/activate

pip install -U wheel pip

pip install -U transformers
pip install flash_attn-2.5.9.post1+cu122torch2.3cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
pip install torchaudio==2.3.0 deepspeed tensorboardX accelerate
pip install python-dotenv simplejson setproctitle attrdict pyarrow==15.0.0
pip install pyphen

