import chardet


def get_encoding(file):
    with open(file, 'rb') as f:
        encoding = chardet.detect(f.read())['encoding']
    if encoding == 'GB2312':
        encoding = 'GB18030'
    return encoding


def full2half(s):
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        num = chr(num)
        n.append(num)
    return ''.join(n)
