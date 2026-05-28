import os
import re
import torch
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess, sentence_postprocess
from utils.text.split_text import get_word_list

def build_asr_model(device='cuda'):
    asr_model_name = "iic/SenseVoiceSmall"
    modelscope_model_dir = os.path.join(os.environ['MODELSCOPE_CACHE'], 'models')
    asr_model_path = os.path.join(modelscope_model_dir, asr_model_name)
    vad_model_name = "fsmn-vad"
    vad_model_path = os.path.join(modelscope_model_dir, 'iic/speech_fsmn_vad_zh-cn-16k-common-pytorch')
    punc_model_name = "ct-punc-c"
    punc_model_path = os.path.join(modelscope_model_dir, 'iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch')
    if isinstance(device, torch.device):
        device_index = device.index
        device = device.type
        if device_index is not None:
            device = f"{device}:{device_index}"
    asr_model = AutoModel(
        model= asr_model_path if os.path.exists(asr_model_path) else asr_model_name,
        # model_path="/mnt/bn/sa-ag-data/liruiqi/code/modelscope/models/iic/SenseVoiceSmall"
        # trust_remote_code=True,
        # remote_code="./model.py",  
        vad_model= vad_model_path if os.path.exists(vad_model_path) else vad_model_name,
        vad_kwargs={"max_single_segment_time": 60000},
        punc_model= punc_model_path if os.path.exists(punc_model_path) else punc_model_name,
        punc_model_revision="v2.0.4",
        disable_pbar=True,
        device=device,
        disable_update=True
    )
    return asr_model

@torch.no_grad()
def run_asr_model(audio_file, asr_model, with_segments=True):
    res = asr_model.generate(
        input=audio_file,
        cache={},
        language="auto",
        use_itn=False,
        batch_size_s=300,
        merge_vad=True,
        merge_length_s=15,
    )

    result = []
    for i in range(len(res)):
        raw_text = res[i]["text"]
        text_pred = rich_transcription_postprocess(res[i]["text"])
        text_pred = re.sub(r'<\|.*?\|>', '', text_pred)
        text_normed = batch_remove_str(text_pred, ["😊", "😔", "😡", "😰", "🤢", "😮", "🎼", "👏", "😀", "😭", "🤧", "😷"])
        text_normed = text_pred.replace('🎼', ' ')
        
        if with_segments:
            # segment according to tags
            pattern = re.compile(r'(<\|[^|]+\|>)+([^<]+)')
            segments = []
            for match in re.finditer(pattern, raw_text):
                tags = re.findall(r'<\|([^|]+)\|>', match.group(0))
                sentence = match.group(2).strip()
                # Remove Spaces at the beginning and end of sentences
                # sentence = re.sub(r'\s+', ' ', sentence).strip()
                
                segments.append({
                    'lang': tags[0],
                    'emotion': tags[1],
                    'event': tags[2],
                    'use_itn': tags[3],
                    'text': sentence
                })

        ret = {
            'raw_text': raw_text,
            'text': text_pred,
            'text_normed': text_normed
        }
        if with_segments:
            ret['segments'] = segments
        result.append(ret)

    if (not isinstance(audio_file, list)) and len(result) == 1:
        return result[0]

    return result

def batch_remove_str(src: str, tgt_lst):
    for tgt in tgt_lst:
        src = src.replace(tgt, '')
    return src


def build_vad_model(device='cuda'):
    modelscope_model_dir = os.path.join(os.environ['MODELSCOPE_CACHE'], 'models')
    vad_model_name = "fsmn-vad"
    vad_model_path = os.path.join(modelscope_model_dir, 'iic/speech_fsmn_vad_zh-cn-16k-common-pytorch')
    if isinstance(device, torch.device):
        device_index = device.index
        device = device.type
        if device_index is not None:
            device = f"{device}:{device_index}"
    vad_model = AutoModel(
        model= vad_model_path if os.path.exists(vad_model_path) else vad_model_name,
        punc_model_revision="v2.0.4",
        disable_pbar=True,
        device=device,
        disable_update=True
    )
    return vad_model

def run_vad_model(audio_file, vad_model):
    res = vad_model.generate(
        input=audio_file,
        batch_size_s=300,
        merge_vad=False,
        merge_length_s=10,
    )
    print(res)


if __name__ == '__main__':
    if os.path.isfile('.env.local'):
        from dotenv import load_dotenv
        load_dotenv('.env.local')

    audio_file = 'user/prompts/mega_eval_prompt0731/0_vocal.wav'
    
    # vad_model = build_vad_model()
    # run_vad_model(audio_file, vad_model)

    asr_model = build_asr_model()
    asr_results = run_asr_model(audio_file, asr_model)

    print('asr_results', asr_results)

    