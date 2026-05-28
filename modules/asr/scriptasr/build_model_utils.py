def build_asr_text_tokenizer():
    from transformers import AutoTokenizer, Qwen2Tokenizer
    text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    text_tokenizer.add_tokens([
        '<BOT>', '<EOT>', '<BOS>', '<EOS>', '<TAG>', '</TAG>', '<GPROMPT>', '</GPROMPT>', '<SPK>', '</SPK>', '<MASK>',
        '[breath]', '<breath>', '</breath>', '[laughter]', '<laughter>', '</laughter>', '[cough]', '<cough>', '</cough>',
        '[music]', '<music>', '</music>', '<strong>', '</strong>', '[noise]', '[hissing]', '[sigh]', '[vocalized-noise]', 
        '[lipsmack]', '[clucking]', '[quick_breath]'
    ], special_tokens=True)
    vocab_size = len(text_tokenizer)
    return text_tokenizer, vocab_size

def build_asr_model(hparams, text_tokenizer=None, init_pretrained=True, vocab_size=None, padding_idx=None):
    from modules.asr.scriptasr.causal_asr import CausalASRModel, ModelArgs
    if text_tokenizer is not None:
        vocab_size = len(text_tokenizer)
        padding_idx = text_tokenizer.encode('<|endoftext|>')[0]
    model_config = ModelArgs(
        vocab_size=vocab_size,
        padding_idx=padding_idx,
        audio_encoder_type=hparams.get('audio_encoder_type'),
        audio_encoder_ckpt=hparams.get('audio_encoder_ckpt'),
        init_pretrained=init_pretrained,
        model_spk_diarization=hparams.get('model_spk_diarization', False),
        spk_diarization_dim=hparams.get('spk_diarization_dim', 512),
        spk_diarization_after_lm=hparams.get('spk_diarization_after_lm', False),
        max_spk_num=hparams.get('max_spk_num', 10000000)
    )
    if hparams.get('backbone', 'llama') == 'llama':
        if hparams.get('model_size', 'base') == 'small':
            model_config.lm_config.n_layers = 12
            model_config.lm_config.n_heads = 12
            model_config.lm_config.dim = 768 
        elif hparams.get('model_size', 'base') == '1b':
            model_config.lm_config.n_layers = 28
            model_config.lm_config.n_heads = 16
            model_config.lm_config.dim = 1536 
    elif hparams.get('backbone', 'llama') == 'qwen3':
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        model_config.backbone = 'qwen3'
    elif hparams.get('backbone', 'llama') == 'llama_seq2seq':
        from modules.asr.llama.llama_seq2seq import Seq2SeqLLaMA, ModelArgs as LLaMaS2SModelArgs
        model_config.backbone = 'llama_seq2seq'
        model_config.lm_config = LLaMaS2SModelArgs()
        model_config.lm_config.enc_n_layers = 4
        model_config.lm_config.dec_n_layers = 20
        if hparams.get('model_size', 'base') == 'small':
            model_config.lm_config.enc_n_layers = 2
            model_config.lm_config.dec_n_layers = 10
            model_config.lm_config.n_heads = 12
            model_config.lm_config.dim = 768 
        elif hparams.get('model_size', 'base') == '1b':
            model_config.lm_config.enc_n_layers = 8
            model_config.lm_config.dec_n_layers = 24
            model_config.lm_config.n_heads = 16
            model_config.lm_config.dim = 1536 

    model = CausalASRModel(model_config)
    return model


