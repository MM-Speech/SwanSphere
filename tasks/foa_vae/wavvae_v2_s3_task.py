from tasks.foa_vae.wavvae_v2_base_task import FOAWavVAEBaseTask


class FOAWavVAEStage3Task(FOAWavVAEBaseTask):
    stage_name = "d_pretrain_detached"
