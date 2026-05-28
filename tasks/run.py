import collections
import collections.abc

for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))
import utils.commons.single_thread_env  # NOQA
import os
from utils.commons.hparams import hparams, set_hparams
import importlib
import subprocess
import setproctitle

def run_task():
    assert hparams['task_cls'] != ''
    pkg = ".".join(hparams["task_cls"].split(".")[:-1])
    cls_name = hparams["task_cls"].split(".")[-1]
    task_cls = getattr(importlib.import_module(pkg), cls_name)
    task_cls.start()


if __name__ == '__main__':
    if os.path.isfile('.env.local'):
        from dotenv import load_dotenv
        load_dotenv('.env.local')
        
    from utils.nn.fa import flash_attn_installed, flash_attn_backend
    print(f"| [Boot] FA installed={flash_attn_installed}, backend={flash_attn_backend}")

    if os.environ.get('CUDA_VISIBLE_DEVICES', '') == '':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    try:
        import libtmux

        tmux_session_name = subprocess.run(
            "echo $(tmux list-panes -t \"$TMUX_PANE\" -F '#S' | head -n1)",
            shell=True, check=True, stdout=subprocess.PIPE).stdout
        tmux_session_name = tmux_session_name.decode().strip()
        server = libtmux.Server()
        session = server.find_where({"session_name": tmux_session_name})
        window = session.attached_window
    except Exception as e:
        print('| libtmux load error.')

    set_hparams()
    setproctitle.setproctitle(f'MegaAvatar runner')
    # if len(hparams["exp_name"]) > 5: # 会干扰vscode debug
    #     try:
    #         subprocess.check_call(f'pkill -f "{hparams["exp_name"]}"', shell=True)
    #     except:
    #         pass
    try:
        if hparams['rename_tmux'] and not hparams['infer']:
            window.rename_window('_'.join(hparams['exp_name'].split("_")[:-1]))
    except:
        pass
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(",")
    for d in devices:
        os.system(f'pkill -f "voidgpu{d}"')
    run_task()