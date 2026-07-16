from tasks.foa_vae.wavvae_v2_base_task import FOAWavVAEBaseTask


class FOAWavVAEStage2Task(FOAWavVAEBaseTask):
    stage_name = "recon_stabilization"
