import re
from typing import List, Union, Iterable
from copy import deepcopy
from bs4 import BeautifulSoup, Tag, NavigableString
from utils.text import PUNC
from utils.commons.seq_utils import seq_match, print_seq_match
from utils.text.split_text import get_word_list, word_list_to_str, remove_space
from utils.text.ph_alignment import is_word_match, process_repeat_word

class SSML:
    def __init__(self, input_text='', rate=1, origin=None):
        if input_text == '':
            if (isinstance(rate, float) or isinstance(rate, int)) and rate != 1: 
                input_text = f'<speak rate="{rate}"></speak>'
            else:
                input_text = '<speak></speak>'
        else:
            if not (input_text.startswith('<speak') and input_text.endswith('</speak>')):
                input_text = f"<speak>{input_text}</speak>"
        self.soup = BeautifulSoup(input_text, 'html.parser')
        self.origin = self if origin is None else origin    # original ssml before normalization

    @property
    def rate(self):
        rate = self.soup.speak.attrs.get('rate')
        if rate is None or rate.strip() == '':
            rate = 1
        if isinstance(rate, str):
            rate = float(rate.strip())
        rate = max(0.01, min(2, rate))
        return rate
    
    @rate.setter
    def rate(self, value):
        if value is None:
            if 'rate' in self.soup.speak.attrs:
                del self.soup.speak['rate']
        else:
            self.soup.speak['rate'] = str(value)

    @property
    def pause_at_end(self):
        if not hasattr(self, '_pause_at_end'):
            return 0.0
        return self._pause_at_end
    
    @pause_at_end.setter
    def pause_at_end(self, sec: float):
        self._pause_at_end = sec
        
    @property
    def pause_at_start(self):
        if not hasattr(self, '_pause_at_start'):
            return 0.0
        return self._pause_at_start
    
    @pause_at_start.setter
    def pause_at_start(self, sec: float):
        self._pause_at_start = sec

    def __repr__(self):
        return self.ssml_str

    def __len__(self):
        return len(self.soup.speak.contents)
    
    def __getitem__(self, index):
        return self.soup.speak.contents[index]
    
    def __iter__(self):
        return iter(self.soup.speak.contents)
    
    @property
    def ssml_str(self):
        return str(self.soup)

    @property
    def text_str(self):
        return str(self.soup.text)
    
    @property
    def sa_ssml_str(self):
        # SSML representation for SA frontend
        contents = []
        for ele_idx, ele in enumerate(self):
            if isinstance(ele, Tag):
                if ele.name in ['phoneme']:
                    contents.append(str(ele))
                    # contents.append(ele.string)
                elif ele.string is not None:
                    contents.append(str(ele.string))
            elif isinstance(ele, NavigableString):
                contents.append(str(ele))
        return f'<speak>' + ''.join(contents) + '</speak>'
    
    @property
    def word_list(self):
        return get_word_list(str(self.soup.text))
    
    def prettify(self):
        return '\n'.join([str(ele) for ele in self.soup.speak.contents])
    
    @property
    def char_len(self):
        return len(str(self.soup.text))
    
    @property
    def char_lens(self):
        cnt = []
        for ele in self.soup.speak.contents:
            if ele.string is not None:
                cnt.append(len(ele.string))
            else:
                cnt.append(0)
        return cnt

    def check_char_index(self, index, unit='char'):
        char_len = self.char_len if unit == 'char' else self.word_len
        if index < 0:
            index = char_len + index
        if index >= char_len or index < 0:
            raise IndexError(f"{char_len}, {index}")
        return index
    
    def get_char(self, index, unit='char'):
        index = self.check_char_index(index, unit)
        if unit == 'char':
            return self.text_str[index]
        else:
            return self.word_list[index]
    
    def char_index2ele_index(self, index, unit='char'):
        # index: char index or word index
        # return: ele_idx: element index
        #         sub_idx: char index within the element
        index = self.check_char_index(index, unit)
        char_lens = self.char_lens if unit == 'char' else self.word_lens
        cur_len = 0
        ele_idx = -1
        while cur_len <= index:
            ele_idx += 1
            cur_len += char_lens[ele_idx]
        sub_idx = index - sum([char_lens[i] for i in range(ele_idx)])
        return ele_idx, sub_idx

    def ele_index2char_index(self, ele_idx, sub_idx, unit='char'):
        char_lens = self.char_lens if unit == 'char' else self.word_lens
        char_idx = 0
        for ele_idx_ in range(ele_idx):
            char_idx += char_lens[ele_idx_]
        char_idx += sub_idx
        return char_idx
    
    @property
    def word_len(self):
        return len(self.word_list)
    
    @property
    def word_lens(self):
        cnt = []
        for ele in self.soup.speak.contents:
            if ele.string is not None:
                cnt.append(len(get_word_list(ele.string)))
            else:
                cnt.append(0)
        return cnt
    
    def append(self, tag_name=None, content=None, **attrs):
        if tag_name is not None:
            new_tag = self.soup.new_tag(tag_name)
            new_tag.attrs.update(attrs)
            if content is not None:
                new_tag.string = content
            self.soup.speak.append(new_tag)
        else:
            assert content is not None
            new_string = self.soup.new_string(content)
            self.soup.speak.append(new_string)
    
    def insert(self, index, tag_name=None, content=None, **attrs):
        if tag_name is not None:
            new_tag = self.soup.new_tag(tag_name)
            new_tag.attrs.update(attrs)
            if content is not None:
                new_tag.string = content
            self.soup.speak.insert(index, new_tag)
        else:
            assert content is not None
            new_string = self.soup.new_string(content)
            self.soup.speak.insert(index, new_string)

    def remove_at(self, index):
        if 0 <= index < len(self):
            self.soup.speak.contents[index].decompose()
        else:
            raise IndexError("Index out of range")

    def find_all(self, tag_name):
        return self.soup.speak.find_all(tag_name)
    
    def clear_contents(self):
        self.soup.speak.clear()

    def normalize(self, normalize_func):
        self.origin = deepcopy(self.origin)
        for ele in self.soup.speak.contents:
            if isinstance(ele, Tag):
                if ele.string is not None:
                    ele.string.replace_with(normalize_func(ele.string))
            elif isinstance(ele, NavigableString):
                new_ele = NavigableString(normalize_func(str(ele)))
                ele.replace_with(new_ele)

    def apply_sub(self):
        self.origin = deepcopy(self.origin)
        for ele_idx, ele in enumerate(self.soup.speak.contents):
            if ele.name == 'sub' and isinstance(ele, Tag):
                alias = ele.get('alias')
                new_ele = NavigableString(alias)
                ele.replace_with(new_ele)

    @staticmethod
    def chunk_text(text, limit=60, language_type='zh', debug=False):
        if isinstance(text, str):
            text = SSML(text)
        assert isinstance(text, SSML)

        from utils.text.split_text import chunk_text_chinese, chunk_text_english, chunk_text_chinese_v2
        if language_type == 'en':
            text_segs = chunk_text_english(text.text_str, max_chars=limit)
        else:
            text_segs = chunk_text_chinese_v2(text.text_str, limit=limit)

        def get_text_org_segs():
            text_segs_words = []
            text_segs_end_idx = []
            for i in range(len(text_segs)):
                text_segs_words.extend(get_word_list(text_segs[i]))
                text_segs_end_idx.append(len(text_segs_words))

            text_org_words = get_word_list(text.origin.text_str)
            _, match_track = seq_match(
                process_repeat_word(text_segs_words), 
                process_repeat_word(text_org_words), 
                score_fn=is_word_match, debug=False)
            if debug:
                print_seq_match(text_segs_words, text_org_words, match_track)

            text_org_segs = []
            seg_idx = 0
            text_org_seg = []
            for match_idx, match in enumerate(match_track):
                if match[0] >= text_segs_end_idx[seg_idx]:
                    seg_idx += 1
                    text_org_segs.append(word_list_to_str(text_org_seg))
                    text_org_seg = []

                if match_idx == 0 or match[1] != match_track[match_idx-1][1]:
                    text_org_seg.append(text_org_words[match[1]])
            text_org_segs.append(word_list_to_str(text_org_seg))
            return text_org_segs

        text_org_segs = get_text_org_segs()
        if debug:
            # print('text_segs', text_segs)
            # print('text_org_segs', text_org_segs)
            print('text_segs vs text_org_segs:')
            for i in range(len(text_segs)):
                print(f'text_segs[{i}]    ', text_segs[i])
                print(f'text_org_segs[{i}]', text_org_segs[i])

        def _chunk_text(text: SSML, text_segs: List[str]):
            result = []
            word_lens = text.word_lens
            ele_idx = sub_idx = 0
            for seg_idx, text_seg in enumerate(text_segs):
                if ele_idx >= len(text):
                    break
                text_seg_n_words = len(get_word_list(text_seg))
                if word_lens[ele_idx] - sub_idx > text_seg_n_words:    # if rest len in current ele > current seg
                    new_item = SSML(rate=text.rate)
                    new_item = append_ele(new_item, text[ele_idx], sub_idx, sub_idx + text_seg_n_words)
                    result.append(new_item)
                    sub_idx = sub_idx + text_seg_n_words
                elif word_lens[ele_idx] - sub_idx < text_seg_n_words:
                    new_item = SSML(rate=text.rate)
                    len_rest = text_seg_n_words
                    new_item = append_ele(new_item, text[ele_idx], left=sub_idx)
                    len_rest -= word_lens[ele_idx] - sub_idx
                    ele_idx += 1
                    sub_idx = 0
                    while True:
                        if ele_idx >= len(text):
                            break
                        if word_lens[ele_idx] > len_rest:
                            new_item = append_ele(new_item, text[ele_idx], right=len_rest)
                            sub_idx = len_rest
                            len_rest = 0
                            break
                        elif word_lens[ele_idx] < len_rest:
                            new_item = append_ele(new_item, text[ele_idx])
                            len_rest -= word_lens[ele_idx]
                            ele_idx += 1
                        else:
                            new_item = append_ele(new_item, text[ele_idx])
                            ele_idx += 1
                            sub_idx = 0
                            len_rest = 0
                            break
                    result.append(new_item)
                else:
                    new_item = SSML(rate=text.rate)
                    append_ele(new_item, text[ele_idx], left=sub_idx)
                    result.append(new_item)
                    ele_idx += 1
                    sub_idx = 0
            return result

        text_segs = _chunk_text(text, text_segs)
        text_org_segs = _chunk_text(text.origin, text_org_segs)
        for i in range(len(text_segs)):
            text_segs[i].origin = text_org_segs[i]

        return text_segs
    
    @staticmethod
    def chunk_text_with_breaks(text, limit=60, language_type='zh', debug=False):
        if isinstance(text, str):
            text = SSML(text)
        assert isinstance(text, SSML)

        for ele in text:
            if ele.name == 'break' and isinstance(ele, Tag):
                break
        else:
            return SSML.chunk_text(text, limit, language_type, debug)
        
        def split_with_breaks(text: SSML) -> List[SSML]:
            new_items = []
            new_item = SSML(rate=text.rate)
            for ele in text:
                if ele.name == 'break' and isinstance(ele, Tag):
                    break_time = float(ele.get('time')[:-1])
                    if len(new_item) == 0:
                        new_item.pause_at_start = new_item.pause_at_start + break_time
                    else:
                        new_item.pause_at_end = break_time
                        new_items.append(new_item)
                        new_item = SSML(rate=text.rate)
                else:
                    append_ele(new_item, ele)
            if len(new_item) > 0:
                new_items.append(new_item)
            return new_items
        
        items = split_with_breaks(text)
        org_items = split_with_breaks(text)
        for i in range(len(items)):
            items[i].origin = org_items[i]

        text_segs = []
        for new_item in items:
            if new_item.text_str.strip() == '':
                text_segs_ = [new_item]
            else:
                text_segs_: List[SSML] = SSML.chunk_text(new_item, limit, language_type, debug)
            text_segs_[0].pause_at_start = new_item.pause_at_start
            text_segs_[-1].pause_at_end = new_item.pause_at_end
            text_segs.extend(text_segs_)

        return text_segs

    @staticmethod
    def replace_ph_tone(ssml, ph: List[str], tone: List[str], ph2word: List[int]):
        from utils.text.pinyin_utils import pinyin_to_phonemes
        from utils.text.ph_alignment import locate_ph_for_word
        word_lens = ssml.word_lens
        for ele_idx, ele in enumerate(ssml):
            if ele.name == 'phoneme' and isinstance(ele, Tag):
                pinyins_to_replace = ele.get('ph')
                if pinyins_to_replace is None or pinyins_to_replace == '':
                    continue

                pinyins_to_replace = pinyins_to_replace.strip().split()
                tones_to_replace = [p[-1] for p in pinyins_to_replace]
                pinyins_to_replace = [p[:-1] for p in pinyins_to_replace]
                # tones_to_replace = ling_dict['tone'].encode(' '.join(tones_to_replace))

                char_start_idx = sum(word_lens[:ele_idx])
                char_end_idx = sum(word_lens[:ele_idx + 1])

                for pinyin_idx, pinyin_to_replace in enumerate(pinyins_to_replace):
                    phonemes_to_replace = pinyin_to_phonemes(pinyin_to_replace)

                    char_idx = char_start_idx + pinyin_idx
                    ph_start_idx, ph_end_idx = locate_ph_for_word(char_idx, ph2word)
                    if ph_start_idx >= 0 and ph_end_idx >= 0:
                        ph = ph[:ph_start_idx] + phonemes_to_replace + ph[ph_end_idx:]
                        tone = tone[:ph_start_idx] + [tones_to_replace[pinyin_idx]] * len(phonemes_to_replace) + tone[ph_end_idx:]
                        ph2word = ph2word[:ph_start_idx] + [char_idx] * len(phonemes_to_replace) + ph2word[ph_end_idx:]

        return ph, tone, ph2word
    
    @staticmethod
    def add_breaks(ssml, ph: List, tone: List, ph2word: List, dur: List, 
                   break_token: Union[int, str], break_tone: Union[int, str], dur_timestep: float):
        """
        Deprecated: This function is implemented for old alignment strategy, and is deprecated for SA frontend.
        """
        from tts.utils.text_utils.ph_alignment import locate_ph_for_word
        for ele_idx, ele in enumerate(ssml):
            if ele.name == 'break' and isinstance(ele, Tag):
                break_time = ele.get('time')
                if break_time is None or break_time == '':
                    continue
                break_time = break_time.strip()
                if break_time.endswith('ms'):
                    break_unit = 0.001
                    break_time = float(break_time[:-2].strip()) * break_unit
                elif break_time.endswith('s'):
                    break_unit = 1
                    break_time = float(break_time[:-1].strip()) * break_unit
                else:
                    continue
                break_time = max(0, min(10, break_time))
                break_time = round(break_time / dur_timestep)
                # print('break_time', break_time)
                
                char_idx = ssml.ele_index2char_index(ele_idx, 0)
                ph_idx, _ = locate_ph_for_word(char_idx, ph2word)
                
                ph = ph[:ph_idx] + [break_token] + ph[ph_idx:]
                tone = tone[:ph_idx] + [break_tone] + tone[ph_idx:]
                # ph2word = ph2word[:ph_idx] + [ph2word[ph_idx-1] if ph_idx > 0 else ph2word[ph_idx]] + ph2word[ph_idx:]  # TODO
                ph2word = ph2word[:ph_idx] + [-1] + ph2word[ph_idx:]  # TODO: 暂时先用-1填充
                dur = dur[:ph_idx] + [break_time] + dur[ph_idx:]

                # # try ',' + 'sil'
                # ph = ph[:ph_idx] + [163, 145] + ph[ph_idx:]
                # tone = tone[:ph_idx] + [break_tone, break_tone] + tone[ph_idx:]
                # # ph2word = ph2word[:ph_idx] + [ph2word[ph_idx-1] if ph_idx > 0 else ph2word[ph_idx]] + ph2word[ph_idx:]  # TODO
                # ph2word = ph2word[:ph_idx] + [-1, -1] + ph2word[ph_idx:]  # TODO: 暂时先用-1填充
                # break_time_first = min(30, round(break_time * 0.3))
                # dur = dur[:ph_idx] + [break_time_first, break_time - break_time_first] + dur[ph_idx:]

        return ph, tone, ph2word, dur


def append_ele(new_ssml: SSML, ele: Union[Tag, NavigableString], left: int = None, right: int = None):
    if ele.string is not None:
        if left is None and right is None:
            content = ele.string
        else:
            old_words = get_word_list(ele.string)
            if left is None:
                left = 0
            if right is None:
                right = len(old_words)
            content = word_list_to_str(old_words[left: right])
        if hasattr(ele, 'attrs'):
            if ele.name == 'phoneme':
                # 不会遇到<sub>的问题：在chunk之前，就先用逻辑，把<sub>给替换了。
                phs = ele.attrs['ph'].split(' ')
                new_ssml.append(ele.name, content, alphabet='py', ph=' '.join(phs[left: right]))
            else:
                new_ssml.append(ele.name, content, **ele.attrs)
        else:
            new_ssml.append(ele.name, content)
    else:
        if hasattr(ele, 'attrs'):
            new_ssml.append(ele.name, **ele.attrs)
        else:
            new_ssml.append(ele.name)
    return new_ssml


if __name__ == '__main__':
    # # test chunk

    # ph_replace_table = {
    #     'en': {
    #         '@': 'at',
    #         '&': 'and'
    #     },
    #     'zh': {
    #         '@': '艾特',
    #         '&': '和'
    #     }
    # }
    # def _normalize_text_zh(text):
    #     text_norm = zh_normalizer.normalize(text)
    #     if ph_replace_table is not None:
    #         for src, tgt in ph_replace_table['zh'].items():
    #             text_norm = text_norm.replace(src, tgt)
    #     text_norm = common_process(text_norm)
    #     return text_norm

    # def common_process(text):
    #     pause_punc = ['~', '～', ':', '：', '$', '¥', '&', '（', '(', '）', ')', '%', '*']
    #     text_norm = batch_replace(text, pause_punc, tgt='')
    #     return text_norm
    
    # def batch_replace(text: str, src: Union[str, List], tgt: str = ','):
    #     for p in src:
    #         text = text.replace(p, tgt)
    #     return text

    # # text = '<speak>1，16块9毛钱，多<phoneme alphabet="py" ph="le1">了</phoneme>拍不了，Python3.9安装。2，16块9毛钱，多<phoneme alphabet="py" ph="le1">了</phoneme>拍不了，Python3.9安装。3，16块9毛钱，多<phoneme alphabet="py" ph="le1">了</phoneme>拍不了，Python3.9安装。4，16块9毛钱，多<phoneme alphabet="py" ph="le1">了</phoneme>拍不了，Python3.9安装。5，16块9毛钱，多<phoneme alphabet="py" ph="le1">了</phoneme>拍不了，Python3.9安装。6，16块9毛钱，多<phoneme alphabet="py" ph="le1">了</phoneme>拍不了，Python3.9安装。7，16块9毛钱，多<phoneme alphabet="py" ph="le1">了</phoneme>拍不了，Python3.9安装。8，16块9毛钱，多<phoneme alphabet="py" ph="le1">了</phoneme>拍不了，Python3.9安装。9，16块9毛钱，多<phoneme alphabet="py" ph="le1">了</phoneme>拍不了，Python3.9安装。10，16块9毛钱，多<phoneme alphabet="py" ph="le1">了</phoneme>拍不了，Python3.9安装。</speak>'
    # # text = '<speak>你好~你好～你好~ 你好～ 2：3的比分，12～25天后，12~25天后，50%可能性，12¥的价格，我&他是这样，16块9毛钱，多<phoneme alphabet="py" ph="le1">了</phoneme>拍不了，Python3.9安装。</speak>'
    # # text = '<speak>家人们，咱们基地三角梅开始发货了啊，你看它根系发达，苗型还特别好，而且呢，还有30<phoneme alphabet="py" ph="duo1 zhong3">多种</phoneme>不同的品种和颜色任你挑选，喜欢哪个咱就选哪个，咱家这个三角梅都是基地<phoneme alphabet="py" ph="zhi2 fa4">直发</phoneme>的啊，包成活，没有中间商赚差价，所以价格贼便宜，它一年循环开花四到五次，每次花期在一个半月到两个月左右，现在栽上两个月以后就能开花了，一开一大盆，特别漂亮，喜欢赏花的家人赶紧养起来<phoneme alphabet="py" ph="ba5">吧</phoneme>，花苗儿就在左下角小黄车里，,家人们，咱们的三角梅不仅好养活，而且特别适合新手花友，它耐旱耐热，对土壤要求也不高，只要给它充足的阳光和适量的水分，它就能茁壮成长，开出绚丽的花朵，无论是阳台、庭院还是室内，三角梅都能成为你家的亮点，增添一抹生机与色彩，,咱们基地的三角梅都是精心培育的，每一株都经过严格筛选，确保健康无病害，发货前我们还会进行专业的包装，确保花苗在运输过程中不受损伤，下面小黄车赶紧拍</speak>'
    # text = '这个9 9 9银螺胶囊，平时买你嫌不划算 ，现在华润三九官方正在做活动，不要399，也不要299，还有大额优惠券，现在只要这个数，就在视频下方链接里，错过可就不是这个价了。如果说你甘油三酯偏高，晚上睡不踏实，白天犯困没精神,总是手麻脚麻，胸闷气短,视力模糊，看不清,还伴有脑袋昏沉，容易忘事，千万别再拖了，你试试这个9 9 9银螺胶囊,它精选了营养价值丰富的淡水藻类螺旋藻，搭配银杏叶提取物，经过百倍浓缩萃取而成，每100克胶囊就含有银杏总黄酮3600毫克、螺旋藻多糖900毫克。有血脂问题的朋友趁着现在厂家有活动，赶紧点击下方链接带回家试试吧。你知道这个,9 9 9银螺胶囊,现在什么价格吗？我相信啊，你们一定很难理解，为什么你们买的,9 9 9银螺胶囊的价格,跟外面的差这么多，而且呢，买的更贵！这都是因为有中间差价！那在我这里呢？点击视频下方链接下单，厂家发货，不让你多花冤枉钱，如果说你这个血脂高问题已经打算放任不管了，那你刷到我呢，真的是很幸运。有血脂高的朋友可以行动起来了，很多人都不相信啊，刷到就直接划走了。总以为血脂问题光靠多运动就可以了，从开始的视力模糊，到后来的经常头晕犯困、脑袋昏沉、爱忘事等等问题。都可以试试它，这个9 9 9银螺胶囊，平时买你嫌不划算 ，今天还不买就真的亏了，现在华润三九官方正在做活动，不要399，也不要299，还有大额优惠券，现在只要这个数，就在视频下方链接里，错过可就不是这个价了。如果说你甘油三酯偏高，晚上睡不踏实，白天犯困没精神,总是手麻脚麻，胸闷气短,视力模糊，看不清,还伴有脑袋昏沉，容易忘事，千万别再拖了.你试试这个9 9 9银螺胶囊,它精选了营养价值丰富的淡水藻类螺旋藻，搭配银杏叶提取物，经过百倍浓缩萃取而成，每100克胶囊就含有银杏总黄酮3600毫克、螺旋藻多糖900毫克，有血脂问题的朋友趁着现在厂家有活动，赶紧点击下方链接带回家试试吧。你试试这个9 9 9银螺胶囊,它精选了营养价值丰富的淡水藻类螺旋藻，搭配银杏叶提取物，经过百倍浓缩萃取而成，有血脂问题的朋友趁着现在厂家有活动，赶紧点击下方链接带回家试试吧。你知道这个,9 9 9银螺胶囊,现在什么价格吗？我相信啊，你们一定很难理解，为什么你们买的,9 9 9银螺胶囊的价格,跟外面的差这么多，而且呢，买的更贵！那在我这里呢？点击视频下方链接下单，厂家发货，不让你多花一分冤枉钱。这个9 9 9银螺胶囊，平时买你嫌不划算 ，现在华润三九官方正在做活动，不要399，也不要299，还有大额优惠券，现在只要这个数，就在视频下方链接里，错过可就不是这个价了。如果说你甘油三酯偏高，晚上睡不踏实，白天犯困没精神,总是手麻脚麻，胸闷气短,视力模糊，看不清,还伴有脑袋昏沉，容易忘事，千万别再拖了，你试试这个9 9 9银螺胶囊,它精选了营养价值丰富的淡水藻类螺旋藻，搭配银杏叶提取物，经过百倍浓缩萃取而成，每100克胶囊就含有银杏总黄酮3600毫克、螺旋藻多糖900毫克。有血脂问题的朋友趁着现在厂家有活动，赶紧点击下方链接带回家试试吧。'
    # print(get_word_list(text))
    # # text = remove_space(text)
    # print(get_word_list(text))
    # text = SSML(text)

    # from tn.chinese.normalizer import Normalizer as ZhNormalizer
    # zh_normalizer = ZhNormalizer(overwrite_cache=False, remove_erhua=False, remove_interjections=False)
    # text.normalize(_normalize_text_zh)
    # text_segs = SSML.chunk_text(text, limit=60, language_type='zh', debug=True)

    # print('text', text)
    # print('text.origin', text.origin)

    # # print('text_segs[0]', text_segs[0])
    # # print('text_segs[0].origin', text_segs[0].origin)
    # # print('text_segs[1]', text_segs[1])
    # # print('text_segs[1].origin', text_segs[1].origin)

    # for i, text_seg in enumerate(text_segs):
    #     print(f'text_segs[{i}]       ', text_segs[i])
    #     print(f'text_segs[{i}].origin', text_segs[i].origin)


    # # test phone refine

    # from tts.utils.text_utils.ph_alignment import align_word_phone, print_align
    # text = '<speak>一,十六块九毛钱,多<phoneme alphabet="py" ph="le1">了</phoneme>拍不了,Python三点九安装.二,十六块九毛钱,多<phoneme alphabet="py" ph="le1">了</phoneme>拍不了,Python三点九安装.三,十六块九毛钱,多<phoneme alphabet="py" ph="le1">了</phoneme>拍不了,Python三点九安装.四,十六块九毛钱,</speak>'
    # ph_tokens = ['sil', 'C0i', '，', 'C0sh', 'C0iii', 'C0l', 'C0iou', 'C0k', 'C0uai', 'C0j', 'C0iou', 'C0m', 'C0ao', 'C0q', 'C0ian', '，', 'C0d', 'C0uo', 'C0l', 'C0e', 'C0p', 'C0ai', 'C0b', 'C0u', 'C0l', 'C0iao', '，', 'E0p', 'E0ay', 'E0th', 'E0aa', 'E0n', 'C0s', 'C0an', 'C0d', 'C0ian', 'C0j', 'C0iou', 'C0＿', 'C0an', 'C0zh', 'C0uang', '，', 'C0＿', 'C0er', '，', 'C0sh', 'C0iii', 'C0l', 'C0iou', 'C0k', 'C0uai', 'C0j', 'C0iou', 'C0m', 'C0ao', 'C0q', 'C0ian', '，', 'C0d', 'C0uo', 'C0l', 'C0e', 'C0p', 'C0ai', 'C0b', 'C0u', 'C0l', 'C0iao', '，', 'E0p', 'E0ay', 'E0th', 'E0aa', 'E0n', 'C0s', 'C0an', 'C0d', 'C0ian', 'C0j', 'C0iou', 'C0＿', 'C0an', 'C0zh', 'C0uang', '，', 'C0s', 'C0an', '，', 'C0sh', 'C0iii', 'C0l', 'C0iou', 'C0k', 'C0uai', 'C0j', 'C0iou', 'C0m', 'C0ao', 'C0q', 'C0ian', '，', 'C0d', 'C0uo', 'C0l', 'C0e', 'C0p', 'C0ai', 'C0b', 'C0u', 'C0l', 'C0iao', '，', 'E0p', 'E0ay', 'E0th', 'E0aa', 'E0n', 'C0s', 'C0an', 'C0d', 'C0ian', 'C0j', 'C0iou', 'C0＿', 'C0an', 'C0zh', 'C0uang', '，', 'C0s', 'C0ii', '，', 'C0sh', 'C0iii', 'C0l', 'C0iou', 'C0k', 'C0uai', 'C0j', 'C0iou', 'C0m', 'C0ao', 'C0q', 'C0ian', '。']
    # tone_tokens = ['0', '1', '0', '2', '2', '4', '4', '4', '4', '3', '3', '2', '2', '2', '2', '0', '1', '1', '5', '5', '1', '1', '4', '4', '3', '3', '0', '0', '0', '0', '0', '0', '1', '1', '6', '6', '3', '3', '1', '1', '1', '1', '0', '4', '4', '0', '2', '2', '4', '4', '4', '4', '3', '3', '2', '2', '2', '2', '0', '1', '1', '5', '5', '1', '1', '4', '4', '3', '3', '0', '0', '0', '0', '0', '0', '1', '1', '6', '6', '3', '3', '1', '1', '1', '1', '0', '1', '1', '0', '2', '2', '4', '4', '4', '4', '3', '3', '2', '2', '2', '2', '0', '1', '1', '5', '5', '1', '1', '4', '4', '3', '3', '0', '0', '0', '0', '0', '0', '1', '1', '6', '6', '3', '3', '1', '1', '1', '1', '0', '4', '4', '0', '2', '2', '4', '4', '4', '4', '3', '3', '2', '2', '2', '2', '0']
    # text = SSML(text)

    # print('text.prettify()', text.prettify())

    # text_, ph_, ph2word = align_word_phone(text.text_str, ph_tokens)
    # print_align(text_, ph_tokens, ph2word)
    # ph2word = [p-1 for p in ph2word]

    # # import pdb
    # # pdb.set_trace()
    # ph_tokens, tone_tokens, ph2word = SSML.replace_ph_tone(text, ph_tokens, tone_tokens, ph2word)
    # print('ph_tokens', ph_tokens)
    # print('tone_tokens', tone_tokens)

    # text = '<speak>16块9毛钱，多<phoneme alphabet="py" ph="le3">了</phoneme>拍不了，因为<break time="1.5s"/>今天这个价格<say-as interpret-as="cardinal">123</say-as>就是个体验价。</speak>'
    # text = SSML(text)
    # print(text.sa_ssml_str)


    text = '梅花（Titoni）瑞士手表男士机械表经典金色腕表生日礼物 宇宙系列 钢带表盘40MM 797 G-DB-306'
    # text = text.replace(' ', '')
    text = SSML(text)
    print('text', text)
    text_segs = SSML.chunk_text(text, limit=60, language_type='zh', debug=True)
    print('text_segs', text_segs)
