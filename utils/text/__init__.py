import re

SHENGMU = ["C0b", "C0c", "C0ch", "C0d", "C0f", "C0g", "C0h", "C0j", "C0k", "C0l", "C0m", "C0n", "C0p", "C0q", "C0r", "C0s", "C0sh", "C0t", "C0x", "C0z", "C0zh"]

# C0iii 是翘舌音，C0ii 是平舌音，C0i 用于发“yi”这种情况

YUNMU = ["C0a", "C0ai", "C0air", "C0an", "C0ang", "C0angr", "C0anr", "C0ao", "C0aor", "C0ar", "C0e", "C0ei", "C0eir", "C0en", "C0eng", "C0engr", "C0enr", "C0er", "C0i", "C0ia", "C0ian", "C0iang", "C0iangr", "C0ianr", "C0iao", "C0iaor", "C0iar", "C0ie", "C0ier", "C0ii", "C0iii", "C0iiir", "C0iir", "C0in", "C0ing", "C0ingr", "C0inr", "C0io", "C0iong", "C0iongr", "C0iou", "C0iour", "C0ir", "C0ng", "C0o", "C0ong", "C0ongr", "C0or", "C0ou", "C0our", "C0u", "C0ua", "C0uai", "C0uair", "C0uan", "C0uang", "C0uangr", "C0uanr", "C0uar", "C0uei", "C0ueir", "C0uen", "C0ueng", "C0uengr", "C0uenr", "C0uo", "C0uor", "C0ur", "C0v", "C0van", "C0vanr", "C0ve", "C0ver", "C0vn", "C0vnr", "C0vr"]

# YUNMU = YUNMU_WO_ERHUA + YUNMU_ERHUA

YUNMU_WO_ERHUA = ["C0a", "C0ai", "C0an", "C0ang", "C0ao", "C0e", "C0ei", "C0en", "C0eng", "C0er", "C0i", "C0ia", "C0ian", "C0iang", "C0iao", "C0ie", "C0ii", "C0iii", "C0in", "C0ing", "C0io", "C0iong", "C0iou", "C0ng", "C0o", "C0ong", "C0ou", "C0u", "C0ua", "C0uai", "C0uan", "C0uang", "C0uei", "C0uen", "C0ueng", "C0uo", "C0uor", "C0v", "C0van", "C0ve", "C0vn"]

YUNMU_ERHUA = ["C0air", "C0angr", "C0anr", "C0aor", "C0ar", "C0eir", "C0engr", "C0enr", "C0iangr", "C0ianr", "C0iaor", "C0iar", "C0ier", "C0iiir", "C0iir", "C0ingr", "C0inr", "C0iongr", "C0iour", "C0ir", "C0ongr", "C0or", "C0our", "C0uair", "C0uangr", "C0uanr", "C0uar", "C0ueir", "C0uengr", "C0uenr", "C0ur", "C0vanr", "C0ver", "C0vnr", "C0vr"]

# ALL_PHONE = SHENGMU + YUNMU + "C0＿"

ALL_PHONE = ["C0a", "C0ai", "C0air", "C0an", "C0ang", "C0angr", "C0anr", "C0ao", "C0aor", "C0ar", "C0b", "C0c", "C0ch", "C0d", "C0e", "C0ei", "C0eir", "C0en", "C0eng", "C0engr", "C0enr", "C0er", "C0f", "C0g", "C0h", "C0i", "C0ia", "C0ian", "C0iang", "C0iangr", "C0ianr", "C0iao", "C0iaor", "C0iar", "C0ie", "C0ier", "C0ii", "C0iii", "C0iiir", "C0iir", "C0in", "C0ing", "C0ingr", "C0inr", "C0io", "C0iong", "C0iongr", "C0iou", "C0iour", "C0ir", "C0j", "C0k", "C0l", "C0m", "C0n", "C0ng", "C0o", "C0ong", "C0ongr", "C0or", "C0ou", "C0our", "C0p", "C0q", "C0r", "C0s", "C0sh", "C0t", "C0u", "C0ua", "C0uai", "C0uair", "C0uan", "C0uang", "C0uangr", "C0uanr", "C0uar", "C0uei", "C0ueir", "C0uen", "C0ueng", "C0uengr", "C0uenr", "C0uo", "C0uor", "C0ur", "C0v", "C0van", "C0vanr", "C0ve", "C0ver", "C0vn", "C0vnr", "C0vr", "C0x", "C0z", "C0zh", "C0＿"]

PUNC_G2P = ["…", "、", "。", "《", "》", "【", "】", "！", "＂", "＃", "＄", "％", "＇", "＇＇", "（", "）", "＊", "，", "：", "；", "？", "＼", "＾", "＿", "｀", "｛", "｝", "～"]

PUNC = PUNC_G2P + list('.,?!~')

ENG_PHONE = ["E0aa", "E0ae", "E0ah", "E0ao", "E0aw", "E0ax", "E0ay", "E0b", "E0ch", "E0d", "E0dh", "E0eh", "E0ehr", "E0er", "E0ey", "E0f", "E0g", "E0hh", "E0ih", "E0iy", "E0iyr", "E0jh", "E0k", "E0l", "E0m", "E0n", "E0ng", "E0oh", "E0ow", "E0oy", "E0p", "E0r", "E0s", "E0sh", "E0t", "E0th", "E0uh", "E0uw", "E0uwr", "E0v", "E0w", "E0y", "E0z", "E0zh"]

PHONE_VOCAB = ["C0a", "C0ai", "C0air", "C0an", "C0ang", "C0angr", "C0anr", "C0ao", "C0aor", "C0ar", "C0b", "C0c", "C0ch", "C0d", "C0e", "C0ei", "C0eir", "C0en", "C0eng", "C0engr", "C0enr", "C0er", "C0f", "C0g", "C0h", "C0i", "C0ia", "C0ian", "C0iang", "C0iangr", "C0ianr", "C0iao", "C0iaor", "C0iar", "C0ie", "C0ier", "C0ii", "C0iii", "C0iiir", "C0iir", "C0in", "C0ing", "C0ingr", "C0inr", "C0io", "C0iong", "C0iongr", "C0iou", "C0iour", "C0ir", "C0j", "C0k", "C0l", "C0m", "C0n", "C0ng", "C0o", "C0ong", "C0ongr", "C0or", "C0ou", "C0our", "C0p", "C0q", "C0r", "C0s", "C0sh", "C0t", "C0u", "C0ua", "C0uai", "C0uair", "C0uan", "C0uang", "C0uangr", "C0uanr", "C0uar", "C0uei", "C0ueir", "C0uen", "C0ueng", "C0uengr", "C0uenr", "C0uo", "C0uor", "C0ur", "C0v", "C0van", "C0vanr", "C0ve", "C0ver", "C0vn", "C0vnr", "C0vr", "C0x", "C0z", "C0zh", "C0＿", "E0aa", "E0ae", "E0ah", "E0ao", "E0aw", "E0ax", "E0ay", "E0b", "E0ch", "E0d", "E0dh", "E0eh", "E0ehr", "E0er", "E0ey", "E0f", "E0g", "E0hh", "E0ih", "E0iy", "E0iyr", "E0jh", "E0k", "E0l", "E0m", "E0n", "E0ng", "E0oh", "E0ow", "E0oy", "E0p", "E0r", "E0s", "E0sh", "E0t", "E0th", "E0uh", "E0uw", "E0uwr", "E0v", "E0w", "E0y", "E0z", "E0zh", "sil", "…", "、", "。", "《", "》", "【", "】", "！", "＂", "＃", "＄", "％", "＇", "＇＇", "（", "）", "＊", "，", "：", "；", "？", "＼", "＾", "＿", "｀", "｛", "｝", "～"]

TONE_VOCAB = ["0", "1", "10", "11", "12", "13", "15", "17", "2", "3", "4", "5", "6", "7", "8", "9"]

def is_english(text):
    return bool(re.fullmatch(r'^[A-Za-z]+$', text))

def is_chinese(text: str):
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff" or "\u0030" <= ch <= "\u0039":
            return True
    return False

def isPUNC(c):
    for c_ in c:
        if c_.strip() and c_ in PUNC:
            return True
    return False

def lstrip(s: str, l: str):
    if len(s) < len(l):
        return s
    if s[:len(l)] == l:
        return s[len(l):]
    return s

def rstrip(s: str, r: str):
    if len(s) < len(r):
        return s
    if s[-len(r):] == r:
        return s[:-len(r)]
    return s
