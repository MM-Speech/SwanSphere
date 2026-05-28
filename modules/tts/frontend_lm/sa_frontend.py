#coding=utf-8
import base64
import json
import time
import uuid
import requests
import tenacity
import logging
logger = logging.getLogger(__name__)

from utils.text import PUNC, ENG_PHONE
from utils.commons.os_utils import handle_exacption

# appid = "ailab_tesla"
appid = "aw08fez3xbeozhyy"
access_token= "4qmn8Y9vEqbn5mWAPAzkuCziLcoGwdCJ"
# cluster = "test_gpu"
# cluster = 'parallel_test'
cluster = 'jichuang_frontend'

voice_type = "single_frontend"
# voice_type = "BV001_streaming"
host = "speech.byted.org"
api_url = f"https://{host}/api/v1/tts"

@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=8),
    after=tenacity.after_log(logger, logging.WARN))
def call_sa_frontend(text: str, debug=0, lang='zh', text_type='ssml'):
    # text = text.replace('"', '')

    header = {"Authorization": f"Bearer;{access_token}"}
    req_id = str(uuid.uuid4())
    request_json = {
        "app": {
            "appid": appid,
            "token": access_token,
            "cluster": cluster
        },
        "user": {
            "uid": "388808087185088"
        },
        "audio": {
            "voice": "other",
            "voice_type": voice_type,
            # "language": "en",
            "encoding": "mp3",
            "compression_rate": 1,
            "rate": 24000,
            "bits": 16,
            "channel": 1,
            "speed": 10,
            "volume": 10,
            "pitch": 10,
            "gender": 0,
        },
        "request": {
            "reqid": req_id,
            "text": text,
            # "text_type": "plain",
            "text_type": text_type,
            "operation": "query",
            "with_frontend": 1,
            "frontend_type": "unitTson",
            # "frontend_type": "tson",
        }
    }

    max_retry = 5
    retry_cnt = 0

    while retry_cnt < max_retry:

        resp = requests.post(api_url, json.dumps(request_json))

        if debug > 0:
            print('sa input text:', text)
            print('message', resp.json()['message'])
        if debug > 1:
            print(f"resp body: \n{resp.json()}")
            for key in resp.json():
                if key == "data":
                    print(key + ":" + str(len(resp.json()[key])))
                    continue
                print(key+":"+str(json.dumps(resp.json()[key]).replace("'", '"')))

        if resp.json()['message'] == 'Success' and 'addition' not in resp.json():
            print(f'输入文本段无法发音，可能有误：', text)
            return
        
        try:

            item = json.loads(resp.json()['addition']['description'].replace("\\", ""))[0]

            origin_text = item['unitTson']['origin_text']
            normed_text = item['unitTson']['text']
            unit_words = [unit['word'] for unit in item['unitTson']['unit']]

            ph_idx = 0
            alignment = []
            phone = []
            tone = []
            for unit in item['unitTson']['unit']:
                word = unit['word']
                if word == 'sp':
                    if lang == 'zh':
                        word = '，'
                    else:
                        word = ','
                p2w = {'word': word, 'phone': [], 'tone': [], 'phone_idx': []}
                for label in unit['label']:
                    ph = label['phone']
                    to = label['tone']
                    if ph in ENG_PHONE:
                        to = '0'
                    if ph == 'sp':
                        if lang == 'zh':
                            ph = '，'
                        else:
                            ph = ','
                    phone.append(ph)
                    tone.append(to)
                    p2w['phone'].append(ph)
                    p2w['tone'].append(to)
                    p2w['phone_idx'].append(ph_idx)
                    ph_idx += 1
                alignment.append(p2w)

            if debug > 0:
                print('origin_text', origin_text)
                print('normed_text', normed_text)
                print('unit_words', unit_words)
                print('phone', phone)
                print('tone', tone)
            if debug > 1:
                for p, t in zip(phone, tone):
                    print(p, t)


        except Exception as err:
            retry_cnt += 1
            print(f"SA frontend error, resp message: {resp.json()['message']}")
            handle_exacption(err)
            print(f"Retry: {retry_cnt}")
            time.sleep(0.1)
            continue
            # raise err

        break

    # print(item)

    # postprocess for MegaTTS3
    # for i in range(len(phone)):
    #     if phone[i] in ENG_PHONE:
    #         tone[i] = '0'
    #     if phone[i] in ['sp']:
    #         phone[i] = 'sil'
    #         if lang == 'zh':
    #             phone[i] = '，'
    #         else:
    #             phone[i] = ','

    return normed_text, phone, tone, alignment



if __name__ == '__main__':
    # text = 'https://v.kuaishou.com/2xp8rig 长按复制此条消息，打开【快手】直接观看！【圆圆优选好物】推荐：东北大酱黄豆酱。已售152.2万，98%好评，手慢无。'
    # text = '上领子压线就用大弹簧高低压脚，双弹簧压力大，压线转弯不发飘，可以轻松爬坡过坎。而且还是大头的卡槽身，可以压住正面的纸口，也能卡住下面的纸口，这样三层副领压线纸口非常标准。上好的领子，正面纸口非常标准，反面也不会有落坑线。有了它，上领副领真的太简单了，不管是上袖口、上门襟、上领子、上腰头，还是开口袋、定口袋压线用它都轻松搞定。一个好用的高低压脚，让我们干活更省心。压脚分左右，常用的选组高零点二。'
    # text = '他她（TATA）法式水钻高跟拖鞋女鞋气质透明底凉拖2025夏新款XEK04BT5 银色（蝴蝶款） 36'
    # text = '十块十块今天在降十块，皮尔卡丹桑禅丝polo衫，真的是really衣柜里不可或缺的存在，面料贴身穿柔软亲肤，细腻的触感就像夏日微风轻轻拂过肌肤，舒适感直接拉满，'
    # text = '这个北宁堂腰椎型筋骨贴，你可以在别处心动，但是一定要在这儿下单；你可以在别处咨询，但是你一定要在这儿成交。为什么呢？因为我怕你错过北宁堂品牌直发的腰椎帮扶活动。之前单盒要199，现在不要199，不要99，单盒不到一瓶水钱。你别看它价格便宜，但它价值却是很高的，敢向腰椎问题叫板。腰椎疼痛、腰肌劳损，贴它；腰椎间盘突出、下肢放射性疼痛，贴它；弯腰咔咔响、上下楼梯困难，贴它；腰椎酸麻肿痛僵，也贴它；走不了路，干不了活，都贴它。如果你尝试了各种方法，腰椎问题总是反反复复，贴它就对了。趁着现在工厂直发，包邮送到家，你赶紧点击视频下方链接拍几盒。腰椎改善了，身体轻松了，到时候你自然会感谢我。'
    # text = '<speak>16块9毛钱，多<phoneme alphabet="py" ph="le3">了</phoneme>拍不了，因为今天这个价格就是个体验价。</speak>'
    # text = '<speak rate="10">16块9毛钱，多<phoneme alphabet="py" ph="le2">了</phoneme>拍不了，因为今天2025这个价格就是个体验价。</speak>'
    # text = '<speak>16块9毛钱，多<phoneme alphabet="py" ph="le3">了</phoneme>拍不了，因为<break time="1.5s"/>今天这个价格<say-as interpret-as="cardinal">123</say-as>就是个体验价。</speak>'
    # text = '<speak>三分靠长相，七分靠打扮，女人的气质是打扮出来的。衣服要是穿的对，显得年轻好几岁。就这款超好看的蚂蚁腰防晒衬衣，穿在身上既优雅又有气质，还特别的舒服得劲，走到哪都是焦点。新品上市，厂家为了快速出一波店铺的销量，给咱姐妹们炸一波福利。今天下单还包邮送到家。姐妹们放心带回家试穿，喜欢满意就留下，不喜欢不满意，您直接退回来。精选轻薄透气面料，触感丝滑，穿上自带清凉感，仿佛给肌肤敷上冰膜，拒绝夏日闷热黏腻。版型设计独具匠心，巧妙运用公主线剪裁，勾勒出盈盈一握的蚂蚁腰，轻松展现女性曼妙身姿，谁穿谁是 “腰精”。短款衣长适配多种下装，搭配高腰裤或裙子，瞬间优化身材比例，大长腿既视感轻松拿捏。宽松的版型，胖瘦姐妹都能穿，简约的同时又不失精致。无论是日常出街、办公室久坐，还是户外游玩，这件防晒衬衣都能让你在防晒的同时，时尚感拉满，成为夏日里的靓丽风景线！不挑年龄不挑身材，能从30岁穿到五六十岁。关键价格不贵，还包邮送到家。但是福利数量不多了，喜欢的姐妹下方小黄车赶紧拼手速了。</speak>'
    # text = '<speak>梅花（Titoni）瑞士手表男士机械表经典金色腕表生日礼物 宇宙系列 钢带表盘40MM 797 G-DB-306</speak>'
    # text = '<speak>这个北宁堂腰椎型筋骨贴，你，（可以）在别处心动，但是一定要在这儿下单</speak>'
    # text = '<speak>梅花（Titoni）瑞士手表男士机械表经典金色腕表生日礼物 宇宙系列 钢带表盘40MM 797 G-DB-306</speak>'
    # text = '<speak>这可是国际救援中心认证的神器！不用插电，晒两三个小时就能亮40个小时，露营、停电应急都能用，角度亮度随便调，还有爆闪求救模式！想省钱又图安心的，赶紧点我头像进橱窗，多备几个准没错！</speak>'
    # text = '<speak>梅花手表99.999999%的比例</speak>'
    # text = '<speak>假期还不知道带孩子去哪儿的，一定要来青山上遇东坡画剧展，馆内几十个裸眼3D打卡点，直接把课本里的内容演活了，历史知识秒变视觉盛宴，让孩子感受宋式风韵，体验点茶、挂画、制香、插画，还有桌游和趣味拼图可以玩哦，关键是亲子票一大一小仅需59.9，<phoneme alphabet="py" ph="zhen1 de5">真的</phoneme>超划算，链接就在左下角，刷到赶紧囤~</speak>'
    text = '<speak>假期还不知道带孩子去哪儿的，一定要来青山上遇东坡画剧展，馆内几十个裸眼3D打卡点，直接把课本里的内容演活了，历史知识秒变视觉盛宴，让孩子感受宋式风韵，体验点茶、挂画、制香、插画，还有桌游和趣味拼图可以玩哦，关键是亲子票一大一小仅需59.9，<phoneme ph="zhen2 de4" alphabet="py">真的</phoneme>超划算，链接就在左下角，刷到赶紧囤~</speak>'
    # text = '<speak>告别冷水段!美的热水器双胆瞬热系统,晨起洗漱无需等待。活水抑菌技术守护水质,跟离子抗菌率99%,肌肤敏感星人也能安心冲澡。极简触控屏+远程APP操控,温暖总比你先到家。」</speak>'

    # from tts.utils.text_utils import ssml_utils
    # text = ssml_utils.SSML(text)
    # print(text.sa_ssml_str)

    # print(text)
    call_sa_frontend(text, debug=1)
