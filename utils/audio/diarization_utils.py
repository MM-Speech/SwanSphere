def merge_segments(segments, max_overlap_duration=1.0, max_silence=2.0, min_voiced_duration=0.1, max_voiced_duration=60.0):
    if 'speaker' not in segments[0]:
        for segment in segments:
            segment['speaker'] = 'spk0'

    # remove overlapping
    segments_nonoverlapping = []
    segments = sorted(segments, key=lambda x: x['start'])
    idxs_to_remove = set()
    for i, segment in enumerate(segments):
        j = i + 1
        while j < len(segments):
            if segments[j]['start'] >= segments[i]['end']:
                break
            overlap_start = max(segments[i]['start'], segments[j]['start'])
            overlap_end = min(segments[i]['end'], segments[j]['end'])
            overlap_duration = overlap_end - overlap_start
            if overlap_duration > max_overlap_duration:
                idxs_to_remove.add(i)
                idxs_to_remove.add(j)
            j += 1
    for i, segment in enumerate(segments):
        if i not in idxs_to_remove:
            segments_nonoverlapping.append(segment)
    segments = segments_nonoverlapping

    # merge segments
    segments_merged = []
    for idx, segment in enumerate(segments):
        if len(segments_merged) == 0:
            segments_merged.append(segment)
            continue
        last_segment = segments_merged[-1]
        if (segment['end'] < last_segment['end'] or 
            segment['start'] < last_segment['start'] or
            last_segment['end'] - segment['start'] > max_overlap_duration):
            continue
        if segment['speaker'] == last_segment['speaker']:
            if (segment['start'] - last_segment['end'] > max_silence or
                segment['end'] - last_segment['start'] > max_voiced_duration):
                segments_merged.append(segment)
            else:
                last_segment['end'] = segment['end']
        else:
            segments_merged.append(segment)

    # remove short segments
    segments_merged_ = []
    for segment in segments_merged:
        if segment['end'] - segment['start'] < min_voiced_duration:
            continue
        segments_merged_.append(segment)
    segments_merged = segments_merged_

    return segments_merged


def merge_segments_multispk(segments, min_spk=2, max_spk=2, 
                            min_conversation_num=2, max_duration=80, 
                            max_silence=5, min_voiced_duration=4, 
                            sliding_window=True):
    """
    use this function after merge_segments()
    """
    class Segment:
        def __init__(self):
            self.segment_idxs = []
            self.start = 0
            self.end = 0
            self.spk_set = set()
            self.conversation_num = 0
            self.spks = []

        @property
        def duration(self):
            return self.end - self.start
        
        def is_empty(self):
            return len(self.segment_idxs) == 0

        def item(self):
            return {
                    'segment_idxs': self.segment_idxs,
                    'start': self.start,
                    'end': self.end,
                    'duration': self.end - self.start,
                    'spk_num': len(self.spk_set),
                    'speakers': self.spks,
                    'conversation_num': self.conversation_num // len(segment_merged.spk_set)
                }

    segments_merged = []

    def merge_segment(segment_merged: Segment, segment, segment_idx):
        cur_spk = segment['speaker']
        if (
                (not segment_merged.is_empty() and segment['end'] - segment_merged.start > max_duration) or
                (segment['end'] - segment['start'] > max_duration) or
                (cur_spk not in segment_merged.spk_set and len(segment_merged.spk_set) + 1 > max_spk) or
                (not segment_merged.is_empty() and segment['start'] - segment_merged.end > max_silence)
            ):
            if (len(segment_merged.spk_set) >= min_spk and 
                segment_merged.conversation_num // len(segment_merged.spk_set) >= min_conversation_num and 
                segment_merged.duration > min_voiced_duration):
                segments_merged.append(segment_merged.item())
            return True     # stop
        
        if len(segment_merged.spks) == 0 or segment_merged.spks[-1] != cur_spk:
            segment_merged.conversation_num += 1
        segment_merged.spks.append(cur_spk)
        if segment_merged.is_empty():
            segment_merged.start = segment['start']
        segment_merged.segment_idxs.append(segment_idx)
        segment_merged.end = segment['end']
        segment_merged.spk_set.add(cur_spk)
        return False

    start_segment_idx = 0
    while start_segment_idx < len(segments):
        segment_merged = Segment()
        for end_segment_idx in range(start_segment_idx, len(segments)):
            if merge_segment(segment_merged, segments[end_segment_idx], end_segment_idx):
                segment_merged = Segment()
                break

        if (len(segment_merged.spk_set) >= min_spk and 
            segment_merged.conversation_num // len(segment_merged.spk_set) >= min_conversation_num and 
            segment_merged.duration > min_voiced_duration):
            segments_merged.append(segment_merged.item())
            
        if sliding_window:
            start_segment_idx += 1
        else:
            start_segment_idx = end_segment_idx + 1

    return segments_merged


if __name__ == '__main__':
    segments = [
        {'start': 1.69, 'end': 14.27,},
        {'start': 14.78, 'end': 15.68,},
        {'start': 15.96, 'end': 22.335,},
        {'start': 22.335, 'end': 23.085,},
        {'start': 23.085, 'end': 37.335,},
        {'start': 37.335, 'end': 39.27,},
        {'start': 39.55, 'end': 46.675,},
        {'start': 46.675, 'end': 48.175,},
        {'start': 48.175, 'end': 49.97,},
        {'start': 51.81, 'end': 53.6,},
        {'start': 53.88, 'end': 54.79,},
        {'start': 55.48, 'end': 59.98,},
    ]
    
    segments_merged = merge_segments(segments, max_voiced_duration=20)
    
    for segment in segments_merged:
        print(segment)
    
    # segments = [
    #     {'start': 1.69, 'end': 14.27, 'speaker': '2'},
    #     {'start': 14.78, 'end': 15.68, 'speaker': '3'},
    #     {'start': 15.96, 'end': 22.335, 'speaker': '1'},
    #     {'start': 22.335, 'end': 23.085, 'speaker': '2'},
    #     {'start': 23.085, 'end': 37.335, 'speaker': '0'},
    #     {'start': 37.335, 'end': 39.27, 'speaker': '3'},
    #     {'start': 39.55, 'end': 46.675, 'speaker': '3'},
    #     {'start': 46.675, 'end': 48.175, 'speaker': '0'},
    #     {'start': 48.175, 'end': 49.97, 'speaker': '3'},
    #     {'start': 51.81, 'end': 53.6, 'speaker': '3'},
    #     {'start': 53.88, 'end': 54.79, 'speaker': '3'},
    #     {'start': 55.48, 'end': 59.98, 'speaker': '3'},
    #     # {'start': 1.69, 'end': 14.27, 'speaker': '2'},
    #     # {'start': 14.78, 'end': 15.68, 'speaker': '3'},
    #     # {'start': 15.96, 'end': 22.335, 'speaker': '1'},
    #     # {'start': 22.335, 'end': 23.085, 'speaker': '2'},
    #     # {'start': 23.085, 'end': 37.335, 'speaker': '0'},
    #     # {'start': 37.335, 'end': 46.675, 'speaker': '3'},
    #     # {'start': 46.675, 'end': 48.175, 'speaker': '0'},
    #     # {'start': 48.175, 'end': 59.98, 'speaker': '3'},
    # ]

    # segments_merged = merge_segments_multispk(segments, max_spk=4)

    # for segment in segments_merged:
    #     print(segment)
