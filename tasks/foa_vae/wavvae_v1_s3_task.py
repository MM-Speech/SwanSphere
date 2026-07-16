from tasks.foa_vae.wavvae_v1_base_task import FOAWavVAEBaseTask


class FOAWavVAEStage3Task(FOAWavVAEBaseTask):
    stage_name = "d_pretrain_detached"
