import os
import multiprocessing
import subprocess
import torch
from functools import partial
from multiprocessing import Pool, Process

from tasks.tts.dataset_utils.tts_datasets import load_ds_len
from utils.commons.dataset_utils import data_loader
from utils.commons.hparams import hparams


class TTSDatasetMixin:
    @data_loader
    def train_dataloader(self):
        node_id = None
        node_size = None
        if hparams['world_size'] > 0:
            node_id = hparams['start_rank'] // self.trainer.num_local_gpus
            node_size = self.trainer.num_total_gpus // self.trainer.num_local_gpus
        multiprocessing.set_start_method('spawn', force=True)
        shm_base = f'/dev/shm/data_shm_{hparams["exp_name"]}'
        if self.trainer.num_local_gpus == 8:
            subprocess.check_call(f'rm -rf /dev/shm/data_shm_* || exit 0', shell=True)
        if hparams['dataset_cls'].split('.')[-1] in ['DialogueSegmentShmDataset', 'PromptAudioShmDataset', 'DialogueSegmentEmbDataset']:
            train_dataset = self.dataset_cls(
                prefix='train', hparams=hparams, use_fast_dataloader=True, 
                rank_id=self.trainer.proc_rank_local, world_size=self.trainer.num_local_gpus, batch_size=1,
                node_id=node_id, node_size=node_size
            )
            return train_dataset.get_dataloader(
                seed=(self.global_step % 1231) + 1231,
                num_workers=4
            )
        else:
            l = hparams.get('max_updates', 10000000)
            if self.trainer.proc_rank_local == 0:
                Process(target=self.build_fast_dataloader, kwargs={
                    'shm_base': shm_base,
                    'seed': self.global_step * ((node_id if node_id else 0) + 1) % 1231,
                    'world_size': self.trainer.num_local_gpus,
                    'hparams': hparams,
                    'n_processer': self.trainer.num_local_gpus * hparams['ds_workers'],
                    'reader_chunk_size': hparams.get('reader_chunk_size', 64),
                    'processer_fn': self.processer_fn
                }).start()
            train_dataset = self.dataset_cls(
                l, shm_base, self.trainer.proc_rank_local, self.trainer.num_local_gpus, hparams)
            train_dataset.load_mel = True
            return torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=train_dataset.collater,
                num_workers=4,
                worker_init_fn=partial(train_dataset.init_worker, work_dir=hparams['work_dir']),
                persistent_workers=False,
                prefetch_factor=4
            )

    @data_loader
    def test_dataloader(self):
        data_dir = hparams["binary_data_dir"]
        ds_len = load_ds_len(data_dir, is_train=False)
        valid_dataset = self.val_dataset_cls(hparams['valid_set_name'], ds_len, shuffle=False, data_dir=data_dir)
        return torch.utils.data.DataLoader(
            dataset=valid_dataset, collate_fn=valid_dataset.collater,
            worker_init_fn=valid_dataset.init_worker,
            batch_size=1, num_workers=1, shuffle=False
        )

    @data_loader
    def val_dataloader(self):
        return self.test_dataloader()


class FastDatasetMixin:
    @data_loader
    def train_dataloader(self):
        rank = self.trainer.proc_rank_local
        node_id = None
        node_size = None
        if hparams['world_size'] > 0:
            node_id = hparams['start_rank'] // self.trainer.num_local_gpus
            node_size = self.trainer.num_total_gpus // self.trainer.num_local_gpus
        train_dataset = self.dataset_cls(
            'train', hparams, hparams['fast_ds'], rank,
            self.trainer.num_local_gpus, 1,
            node_id=node_id, node_size=node_size)
        seed = (self.global_step % 1231) + 1231
        print(f"Init train dataloader with seed {seed}, node_id {node_id} and node_size {node_size}.")
        train_dl = train_dataset.get_dataloader(
            seed=seed, num_workers=hparams.get('ds_workers', 4))
        return train_dl

    @data_loader
    def test_dataloader(self):
        test_dataset = self.dataset_cls(
            'val', hparams, False,
            batch_size=hparams['val_batch_size'], random_frame=False)
        test_dl = test_dataset.get_dataloader()
        return test_dl

    @data_loader
    def val_dataloader(self):
        return self.test_dataloader()