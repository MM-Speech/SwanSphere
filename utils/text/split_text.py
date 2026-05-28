# -*- coding: utf-8 -*-

import re
from utils.text import is_english, PUNC, isPUNC

def chunk_text_chinese(text, limit=60):
    # 中文字符匹配
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    # 标点符号匹配
    punctuation = "，。！？；：,\.!?;"
    
    result = []  # 存储断句结果
    current_chunk = []  # 当前片段
    chinese_count = 0  # 中文字符计数

    i = 0
    while i < len(text):
        char = text[i]
        current_chunk.append(char)
        if chinese_pattern.match(char):
            chinese_count += 1
        
        if chinese_count >= limit:  # 达到限制字符数
            # 从当前位置往前找最近的标点符号
            for j in range(len(current_chunk) - 1, -1, -1):
                if current_chunk[j] in punctuation:
                    result.append(''.join(current_chunk[:j + 1]))
                    current_chunk = current_chunk[j + 1:]
                    chinese_count = sum(1 for c in current_chunk if chinese_pattern.match(c))
                    break
            else:
                # 如果前面没有标点符号，则继续找后面的标点符号
                for k in range(i + 1, len(text)):
                    if text[k] in punctuation:
                        result.append(''.join(current_chunk)+text[i+1:k+1])
                        current_chunk = []
                        chinese_count = 0
                        i = k
                        break
        i+=1

    # 添加最后剩余的部分
    if current_chunk:
        result.append(''.join(current_chunk))

    return result


def chunk_text_chinese_v2(text, limit=60, look_ahead_limit=30):
    # https://github.com/bytedance/MegaTTS3/pull/68/files
    """
    将中文文本分成多个块，优先确保每个块以句号、感叹号或问号结尾，
    其次考虑逗号等其他标点符号，避免在无标点处断句
    
    参数:
        text: 要分块的文本
        limit: 每个块的中文字符数限制
        look_ahead_limit: 向后查找的最大字符数限制
    
    返回:
        分块后的文本列表
    """
    # 中文字符匹配
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')

    # 分级定义标点符号（优先级从高到低）
    primary_end_marks = "。！!？?"  # 首选：句号、感叹号、问号
    secondary_end_marks = "，.,；;："  # 次选：逗号、分号、冒号
    tertiary_end_marks = "、…—-~～"  # 再次：顿号、省略号、破折号等

    result = []  # 存储断句结果
    current_chunk = []  # 当前片段
    chinese_count = 0  # 中文字符计数

    text = get_word_list(text)

    def check_mark(char, marks):
        if len(char) <= 1:
            return char in marks
        else:
            for c in char:
                if c in marks:
                    return True
            return False

    i = 0
    while i < len(text):
        char = text[i]
        current_chunk.append(char)

        if chinese_pattern.match(char):
            chinese_count += 1

        if chinese_count >= limit:  # 达到字符数限制，需要寻找断句点
            found_end = False

            # 依次尝试不同优先级的断句策略

            # 1. 向后查找首选标点
            for k in range(1, min(look_ahead_limit, len(text) - i)):
                next_char = text[i + k]
                if check_mark(next_char, primary_end_marks):
                    result.append(word_list_to_str(current_chunk + text[i+1:i+k+1]))
                    current_chunk = []
                    chinese_count = 0
                    i = i + k
                    found_end = True
                    break

            if not found_end:
                # 2. 向前查找首选标点
                for j in range(len(current_chunk) - 1, -1, -1):
                    if check_mark(current_chunk[j], primary_end_marks):
                        result.append(word_list_to_str(current_chunk[:j + 1]))
                        current_chunk = current_chunk[j + 1:]
                        chinese_count = sum(1 for c in current_chunk if chinese_pattern.match(c))
                        found_end = True
                        break

            if not found_end:
                # 3. 向后查找次选标点
                for k in range(1, min(look_ahead_limit, len(text) - i)):
                    next_char = text[i + k]
                    if check_mark(next_char, secondary_end_marks):
                        result.append(word_list_to_str(current_chunk + text[i+1:i+k+1]))
                        current_chunk = []
                        chinese_count = 0
                        i = i + k
                        found_end = True
                        break

            if not found_end:
                # 4. 向前查找次选标点
                for j in range(len(current_chunk) - 1, -1, -1):
                    if check_mark(current_chunk[j], secondary_end_marks):
                        result.append(word_list_to_str(current_chunk[:j + 1]))
                        current_chunk = current_chunk[j + 1:]
                        chinese_count = sum(1 for c in current_chunk if chinese_pattern.match(c))
                        found_end = True
                        break

            if not found_end:
                # 5. 向后查找三级标点
                for k in range(1, min(look_ahead_limit, len(text) - i)):
                    next_char = text[i + k]
                    if check_mark(next_char, tertiary_end_marks):
                        result.append(word_list_to_str(current_chunk + text[i+1:i+k+1]))
                        current_chunk = []
                        chinese_count = 0
                        i = i + k
                        found_end = True
                        break

            if not found_end:
                # 6. 向前查找三级标点
                for j in range(len(current_chunk) - 1, -1, -1):
                    if check_mark(current_chunk[j], tertiary_end_marks):
                        result.append(word_list_to_str(current_chunk[:j + 1]))
                        current_chunk = current_chunk[j + 1:]
                        chinese_count = sum(1 for c in current_chunk if chinese_pattern.match(c))
                        found_end = True
                        break

            if not found_end:
                # 万不得已，在此处断句（这种情况很少见，因为汉语文本中通常会有标点）
                result.append(word_list_to_str(current_chunk))
                current_chunk = []
                chinese_count = 0

        i += 1

    # 添加最后剩余的部分
    if current_chunk:
        result.append(word_list_to_str(current_chunk))

    return result


def chunk_text_english(text, max_chars=130):
    """
    Splits the input text into chunks, each with a maximum number of characters.

    Args:
        text (str): The text to be split.
        max_chars (int): The maximum number of characters per chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    current_chunk = ""
    # Split the text into sentences based on punctuation followed by whitespace
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def remove_space(text: str):
    return word_list_to_str(get_word_list(text, del_unprintable=True))


def remove_unprintable(text: str):
    out = [c for c in text if c.isprintable()]
    return ''.join(out)


def merge_continuous_puncs(word_list, deduplicate_puncs=False):
    ret = []
    for i in range(len(word_list)):
        if len(ret) > 0 and isPUNC(ret[-1]) and isPUNC(word_list[i]):
            if deduplicate_puncs and word_list[i] in ',，':   # 这个连续的标点符号的第二个很有可能是特殊符号被norm成了','，直接忽略
                continue
            else:
                ret[-1] = ''.join([ret[-1], word_list[i]])
        else:
            ret.append(word_list[i])
    return ret


def get_word_list(text, deduplicate_puncs=False, del_unprintable=False):
    # cleaned_text = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff\s]', '', text)
    cleaned_text = text

    result = []
    current_word = ''
    
    for c in cleaned_text:
        if del_unprintable and not c.isprintable():
            continue
        if c == ' ':
            if current_word:
                result.append(current_word)
                current_word = ''
        elif c.isalnum() and not ('\u4e00' <= c <= '\u9fff'):
            current_word += c
        else:
            if current_word:
                result.append(current_word)
                current_word = ''
            result.append(c)
    
    if current_word:
        result.append(current_word)

    result = merge_continuous_puncs(result, deduplicate_puncs)
    
    return result


def word_list_to_str(word_list: list):
    if len(word_list) <= 0:
        return ''
    word_merge = []
    for i in range(len(word_list)):
        if len(word_merge) > 0 and \
            (
                is_english(word_merge[-1][-1]) and is_english(word_list[i][0]) or \
                word_merge[-1][-1].isdigit() and word_list[i][0].isdigit() or \
                is_english(word_merge[-1][-1]) and word_list[i][0].isdigit() or \
                word_merge[-1][-1].isdigit() and is_english(word_list[i][0])
            ):
            word_merge.append(' ')
        word_merge.append(word_list[i])
    return ''.join(word_merge)


def remove_spaces_between_chinese(text):
    """
    删除相邻中文字符之间的空格
    保留中文字符与非中文字符之间的空格及其他位置的空格
    """
    result = []  # 存储处理后的字符
    prev_char = None  # 上一个非空格字符
    
    for char in text:
        if char == ' ':  # 当前字符是空格
            result.append(char)  # 先暂时保留空格
        else:
            # 判断当前字符是否为中文（Unicode范围：\u4e00-\u9fff，包含常用中文字符）
            is_current_chinese = '\u4e00' <= char <= '\u9fff'
            
            # 当发现中文字符间的空格时删除
            if (prev_char is not None and 
                result and result[-1] == ' ' and  # 前一个字符是空格
                '\u4e00' <= prev_char <= '\u9fff' and  # 前一个非空格字符是中文
                is_current_chinese):  # 当前字符是中文
                result.pop()  # 删除空格
                
            result.append(char)  # 添加当前字符
            prev_char = char  # 更新上一个非空格字符
    
    return ''.join(result)


def get_word_list_advanced(text):
    # 优先匹配的复杂模式（保持整体性）
    priority_patterns = [
        (r'\d{1,2}:\d{2}', 'TIME'),        # 时间
        (r'\d+-\d+', 'SCORE'),             # 比分
        (r'\d+:\d+', 'RATIO'),             # 比例
        (r'\d+(?:\.\d+)?[元块斤]', 'PRICE'), # 价格模式（如16块、9.9元）
        (r'\d+\.\d+', 'FLOAT'),            # 浮点数
        (r'\d+', 'DIGIT'),                 # 连续数字
        (r'[a-zA-Z]+', 'ALPHABET')         # 英文单词
    ]

    # 构建复合正则表达式（使用命名组）
    combined_re = re.compile(
        '|'.join(f'(?P<{name}>{pattern})' for pattern, name in priority_patterns),
        flags=re.UNICODE
    )

    tokens = []
    last_pos = 0

    # 第一层：处理特殊模式
    for match in combined_re.finditer(text):
        start = match.start()
        # 处理特殊模式之前的普通字符
        if start > last_pos:
            raw_chars = list(text[last_pos:start])
            tokens.extend(filter(lambda x: x.strip(), raw_chars))
        
        # 处理匹配到的特殊模式
        tokens.append(match.group())
        last_pos = match.end()

    # 处理剩余字符
    if last_pos < len(text):
        raw_chars = list(text[last_pos:])
        tokens.extend(filter(lambda x: x.strip(), raw_chars))

    # 后处理：拆分带单位的组合（如"16块9"）
    final_tokens = []
    for token in tokens:
        # 处理 数字+单位+数字 的模式
        if re.match(r'(\d+)([^\d]+)(\d+)', token):
            parts = re.split(r'(\d+)([^\d]+)(\d+)', token)[1:-1]
            final_tokens.extend([p for p in parts if p])
        else:
            final_tokens.append(token)

    return final_tokens


if __name__ == '__main__':
    # print(chunk_text_chinese("哇塞！家人们，你们太好运了。我居然发现了一个宝藏零食大礼包，简直适合所有人的口味！有香辣的，让你舌尖跳舞；有盐焗的，咸香可口；还有五香的，香气四溢。就连怀孕的姐妹都吃得津津有味！整整三十包啊！什么手撕蟹柳、辣子鸡、嫩豆干、手撕素肉、鹌鹑蛋、小肉枣肠、猪肉腐、魔芋、魔芋丝等等，应有尽有。香辣土豆爽辣过瘾，各种素肉嚼劲十足，鹌鹑蛋营养美味，真的太多太多啦，...家人们，现在价格太划算了，赶紧下单。"))
    # print(chunk_text_english("Washington CNN When President Donald Trump declared in the House Chamber this week that executives at the nation’s top automakers were “so excited” about their prospects amid his new tariff regime, it did not entirely reflect the conversation he’d held with them earlier that day."))
    print(remove_unprintable('嗨\u200c你好呀'))
