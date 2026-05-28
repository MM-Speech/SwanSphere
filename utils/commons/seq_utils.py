import numpy as np
import numba

def seq_match(seq1, seq2, score_fn=lambda x,y: x==y, metric='mean',
              scores=None, priority=0, step_punish=0, debug=False):
    # find the optimal path
    # metric: max for max score
    #         mean for maximal mean score (for path length)
    # priority: 0 for no priority (follow the seq length)
    #           1 for seq1 priority
    #           2 for seq2 priority

    @numba.njit
    def value(pre_value, cur_score, pre_step, cur_step=None):
        cur_step = pre_step + 1 if cur_step is None else cur_step
        if metric == 'max':
            return pre_value + cur_score - step_punish
        elif metric == 'mean':
            return (pre_value * pre_step + cur_score) / cur_step - step_punish

    dp = np.full((len(seq1), len(seq2)), -np.inf)
    steps = np.zeros((len(seq1), len(seq2)))
    parents = np.zeros((len(seq1), len(seq2), 2), dtype=int)
    if scores is None:
        scores = np.zeros((len(seq1), len(seq2)))
        for i in range(len(seq1)):
            for j in range(len(seq2)):
                scores[i, j] = score_fn(seq1[i], seq2[j])
    dp[0, 0] = scores[0, 0] - step_punish
    
    for i in range(1, len(seq1)):
        steps[i, 0] = steps[i-1, 0] + 1
        dp[i, 0] = value(dp[i-1, 0], scores[i, 0], steps[i-1, 0])
        parents[i, 0] = np.array([i-1, 0])
    for j in range(1, len(seq2)):
        steps[0, j] = steps[0, j-1] + 1
        dp[0, j] = value(dp[0, j-1], scores[0, j], steps[0, j-1])
        parents[0, j] = np.array([0, j-1])
    
    @numba.njit
    def compute_dp(dp, scores, steps, parents, priority):
        n, m = dp.shape
        priority_order = n > m if priority == 0 else priority == 1
        # 预定义候选偏移量（Numba需要静态结构）
        if priority_order:
            candidate_offsets = [(-1, -1), (-1, 0), (0, -1)]  # seq1优先
        else:
            candidate_offsets = [(-1, -1), (0, -1), (-1, 0)]  # seq2优先
        for i in range(1, n):
            for j in range(1, m):
                max_val = -np.inf
                best_idx = 0
                for k, (di, dj) in enumerate(candidate_offsets):
                    pi, pj = i + di, j + dj
                    current_val = value(dp[pi, pj], scores[i, j], steps[pi, pj], steps[pi, pj]+1)
                    if current_val > max_val:
                        max_val = current_val
                        best_idx = k
                di, dj = candidate_offsets[best_idx]
                pi, pj = i + di, j + dj
                dp[i, j] = max_val
                steps[i, j] = steps[pi, pj] + 1
                parents[i, j] = (pi, pj)
        return dp, steps, parents

    dp, steps, parents = compute_dp(dp, scores, steps, parents, priority)

    if debug:
        # max_n = min(15, len(seq1))
        max_n = len(seq1)
        print('score')
        for i in range(max_n):
            print(scores[i].astype(int).tolist())

    coord = [dp.shape[0] - 1, dp.shape[1] - 1]
    track = [coord]
    while True:
        if coord[0] == 0 and coord[1] == 0:
            break
        coord = parents[coord[0], coord[1]].tolist()
        track.append(coord)
    track = track[::-1]
    score = dp[-1, -1]

    if debug:
        board = np.zeros((len(seq1), len(seq2)))
        for i, coord in enumerate(track):
            board[coord[0], coord[1]] = i
        print('track')
        for i in range(max_n):
            # print(board[i, :max_n].astype(int).tolist())
            print(board[i].astype(int).tolist())
        print('dp')
        for i in range(max_n):
            # line = np.round(dp[i, :max_n], 3).tolist()
            line = np.round(dp[i], 3).tolist()
            line = [f"{l:.3f}" for l in line]
            print(line)

    return score, track


def seq_match_max_mean_score(seq1, seq2, score_fn=lambda x,y: x==y, scores=None, priority=0, debug=False):
    return seq_match(seq1, seq2, score_fn=score_fn, scores=scores, priority=priority, debug=debug)


def seq_match_A_star(seq1, seq2, score_fn=lambda x,y: x==y, scores=None, priority=0, step_punish=0.5, heuristic_weight=0.5, debug=False):
    # find the path that maximizes the mean score (punish longer detour) and also consider heuristic value
    # priority: 0 for no priority (follow the seq length)
    #           1 for seq1 priority
    #           2 for seq2 priority
    # step_punish: punish more steps (longer path)
    # heuristic_weight: control heuristic
    from heapq import heappush, heappop

    dp = np.full((len(seq1), len(seq2)), -np.inf)
    steps = np.zeros((len(seq1), len(seq2)))
    parents = np.zeros((len(seq1), len(seq2), 2), dtype=int)
    if scores is None:
        scores = np.array([[score_fn(a, b) for b in seq2] for a in seq1])

    def heuristic(i, j):
        return min(len(seq1) - i, len(seq2) - j) * (1 - heuristic_weight)

    heap = []

    dp[0, 0] = scores[0, 0] - step_punish
    steps[0, 0] = 1
    heappush(heap, (-(dp[0][0] + heuristic(0,0)), 0, 0))

    dir_priority = [
        (1, 1),   # 对角线
        (0, 1) if (priority==2 or (priority==0 and len(seq1) <= len(seq2))) else (1, 0),
        (1, 0) if (priority==1 or (priority==0 and len(seq1) > len(seq2))) else (0, 1)
    ]

    while heap:
        _, i, j = heappop(heap)
        if i == len(seq1) - 1 and j == len(seq2) - 1:
            break

        for di, dj in dir_priority:
            ni, nj = i + di, j + dj
            if ni >= len(seq1) or nj >= len(seq2):
                continue

            new_step = steps[i, j] + 1
            match_score = scores[ni, nj] if (di == 1 and dj == 1) else 0
            new_score = (dp[i, j] + match_score) - step_punish

            if new_score > dp[ni, nj]:
                dp[ni, nj] = new_score
                steps[ni, nj] = new_step
                parents[ni, nj] = (i, j)
                priority_val = -(new_score + heuristic(ni, nj))
                heappush(heap, (priority_val, ni, nj))
                
    if debug:
        max_n = min(15, len(seq1))
        print('score')
        for i in range(max_n):
            print(scores[i].astype(int).tolist())

    coord = [dp.shape[0] - 1, dp.shape[1] - 1]
    track = [coord]
    while True:
        if coord[0] == 0 and coord[1] == 0:
            break
        coord = parents[coord[0], coord[1]].tolist()
        track.append(coord)
    track = track[::-1]
    score = dp[-1, -1] + step_punish * steps[-1, -1]
    score = score / steps[-1, -1] if steps[-1, -1] else 0

    if debug:
        board = np.zeros((len(seq1), len(seq2)))
        for i, coord in enumerate(track):
            board[coord[0], coord[1]] = i
        print('track')
        for i in range(max_n):
            print(board[i, :max_n].astype(int).tolist())
        print('dp')
        for i in range(max_n):
            line = np.round(dp[i, :max_n], 3).tolist()
            line = [f"{l:.3f}" for l in line]
            print(line)

    return score, track


def print_seq_match(seq1, seq2, match_track, ele_width=8, line_len=12):
    width = max(ele_width, len(str(seq1[0])) + 2, len(str(seq2[0])) + 2)
    seq1_print = [f"{seq1[0]:^{width}s}"]
    seq2_print = [f"{seq2[0]:^{width}s}"]
    track_print = [f"{str(match_track[0]):^{width}s}"]
    for i in range(1, len(match_track)):
        if match_track[i][0] == match_track[i-1][0] and match_track[i][1] != match_track[i-1][1]:
            width = max(ele_width, len(str(seq2[match_track[i][1]])) + 2, len(str(match_track[i])) + 2)
            seq1_print.append(f"{'':^{width}s}")
            seq2_print.append(f"{seq2[match_track[i][1]]:^{width}s}")
        elif match_track[i][0] != match_track[i-1][0] and match_track[i][1] == match_track[i-1][1]:
            width = max(ele_width, len(str(seq1[match_track[i][0]])) + 2, len(str(match_track[i])) + 2)
            seq1_print.append(f"{seq1[match_track[i][0]]:^{width}s}")
            seq2_print.append(f"{'':^{width}s}")
        elif match_track[i][0] != match_track[i-1][0] and match_track[i][1] != match_track[i-1][1]:
            width = max(ele_width, len(str(seq1[match_track[i][0]])) + 2, len(str(seq2[match_track[i][1]])) + 2, len(str(match_track[i])) + 2)
            seq1_print.append(f"{seq1[match_track[i][0]]:^{width}s}")
            seq2_print.append(f"{seq2[match_track[i][1]]:^{width}s}")
        else:
            raise RuntimeError('somethings wrong')
        track_print.append(f"{str(match_track[i]):^{width}s}")
        
    import math
    to_print = ''
    for i in range(math.ceil(len(seq1_print) / line_len)):
        to_print = to_print + '|'.join(seq1_print[i * line_len: (i + 1) * line_len]) + '\n'
        to_print = to_print + '|'.join(seq2_print[i * line_len: (i + 1) * line_len]) + '\n'
        to_print = to_print + '|'.join(track_print[i * line_len: (i + 1) * line_len]) + '\n'
        to_print = to_print + '-' * line_len * ele_width + '\n'

    print(to_print)


def max_non_overlapping_with_indexes(intervals):
    # intervals: [(start1, end1), (start2, end2), ...]
    
    import bisect
    
    if not intervals:
        return 0, []
    
    ends = [end for start, end in intervals]
    n = len(intervals)
    
    dp = [0] * (n + 1)
    selected = [False] * (n + 1)
    
    for i in range(n):
        start, end = intervals[i]
        duration = end - start
        
        not_select = dp[i]
        
        idx = bisect.bisect_right(ends, start) - 1
        select = duration + (dp[idx + 1] if idx >= 0 else 0)
        
        if select > not_select:
            dp[i + 1] = select
            selected[i + 1] = True
        else:
            dp[i + 1] = not_select
            selected[i + 1] = False
    
    result_indexes = []
    current = n
    while current > 0:
        if selected[current]:
            result_indexes.append(current - 1)
            start, end = intervals[current - 1]
            idx = bisect.bisect_right(ends, start) - 1
            current = idx + 1
        else:
            current -= 1
    
    result_indexes.reverse()
    return dp[n], result_indexes


if __name__ == '__main__':
    # ph_g2p = ['sil', 'C0uo', 'C0j', 'C0iou', 'C0k', 'C0an', 'C0i', 'C0k', 'C0an', 'C0zh', 'C0ei', 'C0zh', 'C0ong', 'C0f', 'C0a', 'C0l', 'C0v', 'C0t', 'C0i', 'C0c', 'C0ai', 'C0d', 'C0e', 'C0d', 'C0ian', 'C0ing', '，', 'C0f', 'C0a', 'C0l', 'C0v', 'C0g', 'C0u', 'C0uen', 'C0sh', 'C0iii', 'C0sh', 'C0ei', '，', 'C0f', 'C0a', 'C0u', 'C0sh', 'C0iii', 'C0sh', 'C0ei', '？', 'C0k', 'C0e', 'C0i', 'C0sh', 'C0uo', 'C0f', 'C0ei', 'C0ch', 'C0ang', 'C0i', 'C0h', 'C0an', 'C0d', 'C0e', '。', 'C0m', 'C0ei', 'C0iou', 'C0zh', 'C0ao', 'C0d', 'C0ao', 'C0f', 'C0a', 'C0u', '，', 'C0ie', 'C0m', 'C0ei', 'C0zh', 'C0ao', 'C0d', 'C0ao', 'C0f', 'C0a', 'C0l', 'C0v', 'C0g', 'C0u', 'C0uen', '，', 'C0ie', 'C0j', 'C0iou', 'C0j', 'C0iang', 'C0zh', 'C0e', 'C0b', 'C0u', 'C0d', 'C0ian', 'C0ing', '。']
    # ph_mfa = ['sil', 'C0a', '，', 'C0uo', 'C0j', 'C0iou', 'C0k', 'C0an', 'C0i', 'C0k', 'C0an', '，', 'C0zh', 'C0e', 'C0zh', 'C0ong', 'C0f', 'C0a', 'C0l', 'C0v', 'C0t', 'C0i', 'C0c', 'C0ai', 'C0d', 'C0e', '，', 'C0d', 'C0ian', 'C0ing', '，', 'C0f', 'C0a', 'C0l', 'C0v', 'C0b', 'C0o', 'C0ie', 'C0sh', 'C0iii', 'C0sh', 'C0ei', '，', 'C0f', 'C0a', 'C0u', 'C0sh', 'C0iii', 'C0sh', 'C0ei', '，', 'C0h', 'C0ai', 'C0ie', 'C0sh', 'C0iii', 'C0f', 'C0ei', 'C0ch', 'C0ang', 'C0i', 'C0b', 'C0an', 'C0d', 'C0e', '，', 'C0m', 'C0ei', 'C0iou', 'C0zh', 'C0ao', 'C0d', 'C0ao', 'C0f', 'C0a', 'C0u', '，', 'C0ie', 'C0m', 'C0ei', 'C0zh', 'C0ao', 'C0d', 'C0ao', 'C0f', 'C0a', 'C0l', 'C0v', 'C0b', 'C0o', 'C0h', 'C0an', '，', 'C0ie', 'C0iou', 'C0iang', 'C0iang', '，', 'C0zh', 'C0e', 'C0b', 'C0u', 'C0d', 'C0ian', 'C0ing', '。']

    # print('ph_g2p', ph_g2p)
    # print('ph_mfa', ph_mfa)
    # match_score, match_track = seq_match_max_mean_score(ph_g2p, ph_mfa, score_fn=lambda x,y:x==y)
    # print('match_score', match_score)
    # print('match_track', match_track)
    # print_seq_match(ph_g2p, ph_mfa, match_track)

    from tts.utils.text_utils import PUNC
    puncs = PUNC + list('.,?!')
    def is_word_match(x, y):
        if x == y:
            return True
        if x in puncs and y in puncs:
            return True
        return False
    # text = ['Python3', '.', '9', '安', '装', '。', '8', '，', '16', '块', '9', '毛', '钱', '，', '多', '了', '拍', '不', '了', '，', 'Python3', '.', '9', '安', '装', '。', '9', '，', '16', '块', '9', '毛', '钱', '，', '多', '了', '拍', '不', '了', '，', 'Python3', '.', '9', '安', '装', '。', '10', '，', '16', '块', '9', '毛', '钱', '，', '多', '了', '拍', '不', '了', '，', 'Python3', '.', '9', '安', '装', '。']
    # text_norm = ['Python', '三', '点', '九', '安', '装', '.', '八', ',', '十', '六', '块', '九', '毛', '钱', ',', '多', '了', '拍', '不', '了', ',', 'Python', '三', '点', '九', '安', '装', '.', '九', ',', '十', '六', '块', '九', '毛', '钱', ',', '多', '了', '拍', '不', '了', ',', 'Python', '三', '点', '九', '安', '装', '.', '十', ',', '十', '六', '块', '九', '毛', '钱', ',', '多', '了', '拍', '不', '了', ',', 'Python', '三', '点', '九', '安', '装', '.']
    text = ['足', '足', '360ml', '的', '大', '容', '量', '，', '喝', '得', '那', '叫', '一', '个', '痛', '快', '！', '它', '用', '的', '是', '整', '颗', '小', '清', '柠', '榨', '汁', '，', '还', '额', '外', '添', '加', '了', '维', 'C', '，', '口', '感', '冰', '爽', '酸', '甜', '，', '简', '直', '让', '人', '欲', '罢', '不', '能', '。', '炎', '炎', '夏', '日', '，']
    text_norm = ['足', '足', '三', '百', '六', '十', '毫', '升', '的', '大', '容', '量', ',', '喝', '得', '那', '叫', '一', '个', '痛', '快', '!', '它', '用', '的', '是', '整', '颗', '小', '清', '柠', '榨', '汁', ',', '还', '额', '外', '添', '加', '了', '维', 'C', ',', '口', '感', '冰', '爽', '酸', '甜', ',', '简', '直', '让', '人', '欲', '罢', '不', '能', '.', '炎', '炎', '夏', '日', ',']
    match_score, match_track = seq_match(text, text_norm, score_fn=is_word_match, metric='mean', debug=True)
    # match_score, match_track = seq_match_A_star(text, text_norm, score_fn=is_word_match, steps_lambda=0.5, heuristic_weight=0.5, debug=True)
    print_seq_match(text, text_norm, match_track)

