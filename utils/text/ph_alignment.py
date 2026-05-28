from utils.text import YUNMU, SHENGMU, ALL_PHONE, PUNC, ENG_PHONE, YUNMU_WO_ERHUA, YUNMU_ERHUA, isPUNC
from utils.text.split_text import get_word_list
from utils.commons.seq_utils import seq_match, print_seq_match


def isEnglish(c):
    for c in c.split(' '):
        if not (c.isalnum() and not ('\u4e00' <= c <= '\u9fff') and not (c == 'sil')):
            return False
    return True

def align_word_phone(text, ph):
    if isinstance(text, str):
        text = get_word_list(text)
    if ph[0] == 'sil' and text[0] != 'sil':
        text = ['sil'] + text

    # 处理英文：将相邻的连续英文单词聚合变成单个单词，并将对应的英文音素都对应到该联合单词上
    if len(text) > 1:
        new_text = [text[0]]
        for i in range(1, len(text)):
            if isEnglish(text[i]):
                if isEnglish(new_text[-1]):
                    new_text[-1] = new_text[-1] + ' ' + text[i]
                else:
                    new_text.append(text[i])
            else:
                new_text.append(text[i])
        text = new_text
    
    if len(ph) < len(text):
        raise RuntimeError(f"音素的数量少于文本，可能遇到了特殊字符。\n文本：{text}\n音素：{ph}")

    ph2word = []
    word_idx = 0
    ph_output = []

    for ph_idx, p in enumerate(ph):
        if word_idx >= len(text):
            # break
            # FIXME: 目前这种情况只会发生在：raw text结尾没有句号，g2p加上了句号，导致多一个ph
            ph2word.append(-1)
            ph_output.append(p)
            continue
        if p in ALL_PHONE and isPUNC(text[word_idx]):
            # g2p可能会缺少一些标点
            ph_output.append(text[word_idx])
            ph2word.append(word_idx)
            word_idx += 1
        elif p in ALL_PHONE:
            if p in YUNMU_ERHUA:
                # FIXME: 如果遇到了儿化音，而且g2p还算正常，那么下一个应该还会有个“儿”字，需要往下走一个word再对齐，然后再走一个
                # 这种情况只有在带有声母的情况下成立。如果没有声母（例如“玩意儿”），会导致ph2word不再是单向的
                word_idx += 1
                ph2word.append(word_idx)
                if text[word_idx] != '儿':
                    raise RuntimeError(f"儿化音混乱。\n文本：{text}\n音素：{ph}")
                word_idx += 1
            else:
                ph2word.append(word_idx)
                if p in YUNMU_WO_ERHUA:
                    word_idx += 1
            ph_output.append(p)
        elif p == text[word_idx] or (isPUNC(p) and isPUNC(text[word_idx])):
            ph2word.append(word_idx)
            word_idx += 1
            ph_output.append(p)
        # elif p not in PUNC and text[word_idx] in PUNC:
        #     # FIXME: 这种情况很奇怪。text中某位置有标点符号，但是g2p后获得的phone没有。需要手动加上
        #     ph_output.append(text[word_idx])
        #     ph2word.append(word_idx)
        #     word_idx += 1
        elif p in ENG_PHONE and isEnglish(text[word_idx]):
            ph2word.append(word_idx)
            if ph_idx + 1 < len(ph) and ph[ph_idx + 1] not in ENG_PHONE:
                word_idx += 1
            ph_output.append(p)
        else:
            ph2word.append(word_idx)
            ph_output.append(p)
    # return text, ph, ph2word
    return text, ph_output, ph2word


def locate_ph_for_word(word_idx, ph2word):
    # excluded
    if word_idx > max(ph2word):
        return -1, -1
    for i in range(len(ph2word)):
        if ph2word[i] == word_idx:
            for j in range(i, len(ph2word)):
                if ph2word[j] != word_idx:
                    break
            return i, j
    return -1, -1


def print_align(text, ph, ph2word):
    from pypinyin import lazy_pinyin
    text = [lazy_pinyin(t)[0] if t != 'sil' and not isEnglish(t) else t for t in text]

    text_print = [f"{text[0]:<8s}"]
    ph_print = [f"{ph[0]:<8s}"]
    word_idx = 1
    for i in range(1, len(ph)):
        if ph2word[i] < 0:
            continue
        if ph2word[i] != ph2word[i-1]:
            text_width = max(8, len(text[word_idx]) + 1)
            text_print.append(f"{text[word_idx]:<{text_width}s}")
            ph_print.append(f"{ph[i]:<{text_width}s}")
            word_idx += 1
        else:
            text_print.append(f"{' ':<8s}")
            ph_print.append(f"{ph[i]:<8s}")
    
    import math
    to_print = ''
    n_words = 12
    for i in range(math.ceil(len(text_print) / n_words)):
        to_print = to_print + '|'.join(text_print[i * n_words: (i + 1) * n_words]) + '\n'
        to_print = to_print + '|'.join(ph_print[i * n_words: (i + 1) * n_words]) + '\n'
        to_print = to_print + '-' * n_words * 8 + '\n'
    print(to_print)


def is_word_match(x, y, repeat_dist_lambda=2, return_bool=False):
    # repeat_dist_lambda: for repeat words, control weight of score decay
    if isinstance(x, NumberedStr) and isinstance(y, NumberedStr):
        if is_word_match(str(x), str(y)):
            if return_bool:
                return x.number == y.number
            return 1 / (abs(x.number - y.number) + 1)**repeat_dist_lambda
        return False
    if x == y:
        return True
    if isPUNC(x) and isPUNC(y):
        return True
    return False

class NumberedStr(str):
    def __new__(cls, value, *args, **keywargs):
        return str.__new__(cls, value)
    def __init__(self, value, number=-1):
        self.number = number

def process_repeat_word(text):
    if len(text) <= 0:
        return []
    output = [text[0]]
    cnt = 0
    for i in range(1, len(text)):
        if (text[i] == text[i-1]) or (isPUNC(text[i]) and isPUNC(text[i-1])):
            if cnt == 0:
                output[i-1] = NumberedStr(output[i-1], cnt)
            cnt += 1
            output.append(NumberedStr(text[i], cnt))
        else:
            cnt = 0
            output.append(text[i])
    return output

# def merge_continuous_puncs(word_list):
#     ret = []
#     for i in range(len(word_list)):
#         if len(ret) > 0 and isPUNC(ret[-1]) and isPUNC(word_list[i]):
#             ret[-1] = ''.join([ret[-1], word_list[i]])
#         else:
#             ret.append(word_list[i])
#     return ret


def merge_norm_alignment(text, text_norm, match_track=None, debug=False):
    if isinstance(text, str):
        text = get_word_list(text)
    if isinstance(text_norm, str):
        text_norm = get_word_list(text_norm)

    # text = merge_continuous_puncs(text)     # 连续没merge的标点符号会导致下面的merge出错  # 现直接移到get_word_list中
    text = process_repeat_word(text)
    text_norm = process_repeat_word(text_norm)

    if match_track is None:
        # 需要更强的对角线权重（更大的step_punish）
        _, match_track = seq_match(
            text, 
            text_norm, 
            score_fn=is_word_match, step_punish=0.1,
            debug=debug
        )

    if debug:
        print_seq_match(text, text_norm, match_track)
        print('text', text)
        print('text_norm', text_norm)

    text_merged = []
    text_idx_merged = []
    text_norm_merged = []
    text_norm_idx_merged = []
    word_merged = []
    word_idx_merged = []
    word_norm_merged = []
    word_norm_idx_merged = []

    for match_idx, match in enumerate(match_track):
        if is_word_match(text[match[0]], text_norm[match[1]], return_bool=True):
            if len(word_merged) > 0 or len(word_norm_merged) > 0:
                text_merged.append(word_merged)
                text_idx_merged.append(word_idx_merged)
                text_norm_merged.append(word_norm_merged)
                text_norm_idx_merged.append(word_norm_idx_merged)
                word_merged = []
                word_idx_merged = []
                word_norm_merged = []
                word_norm_idx_merged = []
            text_merged.append(text[match[0]])
            text_idx_merged.append(match[0])
            text_norm_merged.append(text_norm[match[1]])
            text_norm_idx_merged.append(match[1])
        else:
            if match_idx == 0 or match[0] != match_track[match_idx-1][0]:
                word_merged.append(text[match[0]])
                word_idx_merged.append(match[0])
            if match_idx == 0 or match[1] != match_track[match_idx-1][1]:
                word_norm_merged.append(text_norm[match[1]])
                word_norm_idx_merged.append(match[1])
    return text_merged, text_norm_merged, text_idx_merged, text_norm_idx_merged


if __name__ == '__main__':
    # text_str = '你好~你好～你好~ 你好～ 2：3的比分，12～25天后，12~25天后，50%可能性，12¥的价格，我&他是这样，16块9毛钱，多了拍不了，Python3.9安装。'
    # words = ['你', '好', '你', '好', '你', '好', '你', '好', '二', '三', '的', '比', '分', ',', '十', '二', '二', '十', '五', '天', '后', ',', '十', '二', '到', '二', '十', '五', '天', '后', ',', '百', '分', '之', '五', '十', '可', '能', '性', ',', '十', '二', '的', '价', '格', ',', '我', '他', '是', '这', '样', ',', '十', '六', '块', '九', '毛', '钱', ',', '多', '了', '拍', '不', '了', ',']
    # text_str = '12～25天后，12~25天后，50%可能性'
    # words = ['十', '二', '二', '十', '五', '天', '后', ',', '十', '二', '到', '二', '十', '五', '天', '后', ',', '百', '分', '之', '五', '十', '可', '能', '性']
    # text_str = '三盒诺迪康胶囊要不要？（不要）那我再加两盒呢？（呃，我考虑一下。）别考虑了，这一箱诺迪康胶囊全部按活动价并且包邮送到家。（为什么你们这么便宜？）'
    # words = ['三', '盒', '诺', '迪', '康', '胶', '囊', '要', '不', '要', '?', '不', '要', '那', '我', '再', '加', '两', '盒', '呢', '?', '呃', ',', '我', '考', '虑', '一', '下', '.', '别', '考', '虑', '了', ',', '这', '一', '箱', '诺', '迪', '康', '胶', '囊', '全', '部', '按', '活', '动', '价', '并', '且', '包', '邮', '送', '到', '家', '.', '为', '什', '么', '你', '们', '这', '么', '便', '宜', '?']
    # text_str = '只要给它充足的阳光和适量的水分，它就能茁壮成长，开出绚丽的花朵，无论是阳台、庭院还是室内，三角梅都能成为你家的亮点，增添一抹生机与色彩'
    # words = ['只', '要', '给', '它', '充', '足', '的', '阳', '光', '和', '适', '量', '的', '水', '分', ',', '它', '就', '能', '茁', '壮', '成', '长', ',', '开', '出', '绚', '丽', '的', '花', '朵', ',', '无', '论', '是', '阳', '台', '、', '庭', '院', '还', '是', '室', '内', ',', '三', '角', '梅', '都', '能', '成', '为', '你', '家', '的', '亮', '点', ',', '增', '添', '一', '抹', '生', '机', '与', '色', '彩', ',,']
    # text_merged, text_norm_merged, text_idx_merged, text_norm_idx_merged = merge_norm_alignment(
    #     text_str, words, debug=True
    # )
    # print('words', words)
    # print('text_merged', text_merged)
    # print('text_norm_merged', text_norm_merged)

    # text = '厂家为了冲销量,八瓶只要两位数,还给你包邮到家一粒儿还不到二毛钱呢,价格简直太划算了,正宗的大豆磷脂软凝胶,不管你是职高、糖高、高压高管管堵,'
    # ph_tokens = ['sil', 'C0ch', 'C0ang', 'C0j', 'C0ia', 'C0uei', 'C0l', 'C0e', 'C0ch', 'C0ong', 'C0x', 'C0iao', 'C0l', 'C0iang', '，', 'C0b', 'C0a', 'C0p', 'C0ing', 'C0zh', 'C0iii', 'C0iao', 'C0l', 'C0iang', 'C0uei', 'C0sh', 'C0u', '，', 'C0h', 'C0ai', 'C0g', 'C0ei', 'C0n', 'C0i', 'C0b', 'C0ao', 'C0iou', 'C0d', 'C0ao', 'C0j', 'C0ia', 'C0i', 'C0l', 'C0ir', 'C0h', 'C0ai', 'C0b', 'C0u', 'C0d', 'C0ao', 'C0er', 'C0m', 'C0ao', 'C0q', 'C0ian', 'C0n', 'C0e', '，', 'C0j', 'C0ia', 'C0g', 'C0e', 'C0j', 'C0ian', 'C0zh', 'C0iii', 'C0t', 'C0ai', 'C0h', 'C0ua', 'C0s', 'C0uan', 'C0l', 'C0e', '，', 'C0zh', 'C0eng', 'C0z', 'C0ong', 'C0d', 'C0e', 'C0d', 'C0a', 'C0d', 'C0ou', 'C0l', 'C0in', 'C0zh', 'C0iii', 'C0r', 'C0uan', 'C0n', 'C0ing', 'C0j', 'C0iao', '，', 'C0b', 'C0u', 'C0g', 'C0uan', 'C0n', 'C0i', 'C0sh', 'C0iii', 'C0zh', 'C0iii', 'C0g', 'C0ao', '，', 'C0t', 'C0ang', 'C0g', 'C0ao', '，', 'C0g', 'C0ao', 'C0ia', 'C0g', 'C0ao', 'C0g', 'C0uan', 'C0g', 'C0uan', 'C0d', 'C0u', '。']
    # # text = '老百儿京儿的东西,就是地道,那玩意儿,真不是个东西'
    # # ph_tokens = ['sil', 'C0l', 'C0ao', 'C0b', 'C0ai', 'C0er', 'C0j', 'C0ing', 'C0er', 'C0d', 'C0e', 'C0d', 'C0ong', 'C0x', 'C0i', '，', 'C0j', 'C0iou', 'C0sh', 'C0iii', 'C0d', 'C0i', 'C0d', 'C0ao', '，', 'C0n', 'C0a', 'C0uan', 'C0ir', '，', 'C0zh', 'C0en', 'C0b', 'C0u', 'C0sh', 'C0iii', 'C0g', 'C0e', 'C0d', 'C0ong', 'C0x', 'C0i', '。']
    # # FIXME：这可能导致表现力下降，但如果不这么做，ph2word可能将不再是单向的
    # ph_output = []
    # for p in ph_tokens:
    #     if p in YUNMU_ERHUA:
    #         ph_output.append(p[:-1])
    #         ph_output.append("C0er")
    #     else:
    #         ph_output.append(p)
    # ph_tokens = ph_output
    # text_, ph_tokens, ph2word = align_word_phone(text, ph_tokens, remove_erhua=True)
    # print_align(text_, ph_tokens, ph2word)


    # g2p 可能会跳过标点符号
    text_str = '再也不用担心家里的卫生问题啦.朋友们,如果想要保持家门口的整洁,选它准没错!'
    ph_tokens = ['C0z', 'C0ai', 'C0ie', 'C0b', 'C0u', 'C0iong', 'C0d', 'C0an', 'C0x', 'C0in', 'C0j', 'C0ia', 'C0l', 'C0i', 'C0d', 'C0e', 'C0uei', 'C0sh', 'C0eng', 'C0uen', 'C0t', 'C0i', 'C0l', 'C0a', 'C0p', 'C0eng', 'C0iou', 'C0m', 'C0en', '，', 'C0r', 'C0u', 'C0g', 'C0uo', 'C0x', 'C0iang', 'C0iao', 'C0b', 'C0ao', 'C0ch', 'C0iii', 'C0j', 'C0ia', 'C0m', 'C0en', 'C0k', 'C0ou', 'C0d', 'C0e', 'C0zh', 'C0eng', 'C0j', 'C0ie', '，', 'C0x', 'C0van', 'C0t', 'C0a', 'C0zh', 'C0uen', 'C0m', 'C0ei', 'C0c', 'C0uo', '！']
    text_, ph_tokens, ph2word = align_word_phone(text_str, ph_tokens)
    print_align(text_, ph_tokens, ph2word)


