DANCE_DOMAIN = [
    'bilibili_body_',
    'bilibili_dance_',
    'rf123',
    'pond5_hq',
    'Pexels',
]

ORAL_DOMAIN = [
    'baidunet',
    'bilibili230807',
    'cover_ted',
    'cover_ted_noclip',
    'internal_emotion_sanjia',
    'jichuang_',
    'jichuang_shengguang',
    'jichuang_avatar',
    'jichuang_haoxi',
    'jichuang_genai',
    'lingdong_avatar_240709',
    'mogong_avatar',
    'sinei',
    'youtube230807',
    'Ytb24_clips',
    'bilibili/230807'
    'ytb_zhengshu_2409',
]


def get_domain(item_name):
    # dance-0, oral-1
    for key in DANCE_DOMAIN:
        if key in item_name:
            return 0
    return 1
