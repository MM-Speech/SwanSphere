from langdetect import detect as classify_language
from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnNormalizer

zh_normalizer = None
en_normalizer = None

def normalize_text(text):
    if classify_language(text) == 'en':
        global en_normalizer
        if en_normalizer is None:
            en_normalizer = EnNormalizer(overwrite_cache=False)
        text = en_normalizer.normalize(text)
    else:
        global zh_normalizer
        if zh_normalizer is None:
            zh_normalizer = ZhNormalizer(overwrite_cache=False)
        text = zh_normalizer.normalize(text)

    return text

def isChinese(ch: str):
    if "\u4e00" <= ch <= "\u9fff" or "\u0030" <= ch <= "\u0039":
        return True
    return False
