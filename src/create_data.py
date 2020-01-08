import os
import re
import json
import regex
import pickle

import numpy as np

from utils import get_text, color_print

field_val = {
    'company': 1,
    'address': 2,
    'date': 3,
    'total': 4,
}


def get_files(root):
    json_files = sorted([os.path.join(root, f) for f in os.listdir(root) if f.endswith('.json')])
    txt_files = sorted([os.path.join(root, f) for f in os.listdir(root) if f.endswith('.txt')])

    assert (len(json_files) == len(txt_files))
    for f1, f2 in zip(json_files, txt_files):
        assert f1.replace('.json', '') == f2.replace('.txt', '')

    return json_files, txt_files


def create_data(root):
    json_files, txt_files = get_files(root)
    keys = [os.path.splitext(os.path.basename(f))[0] for f in json_files]

    data = {}
    for key, json_fn, txt_fn in zip(keys, json_files, txt_files):
        text = get_text(txt_fn)
        with open(json_fn, "r", encoding="utf-8") as fp:
            json_info = json.load(fp)

        text_space = regex.sub(r"[\t\n]", " ", text).upper()
        text_class = np.zeros(len(text), dtype=int)

        for field in json_info.keys():
            i = field_val[field]
            v = json_info[field]

            if field == "total":
                anchor = [i.start() for i in re.finditer('TOTAL', text_space)] + \
                         [i.start() for i in re.finditer('GROSS', text_space)] + \
                         [i.start() for i in re.finditer('AMOUNT', text_space)]

                points = [i.start() for i in re.finditer(v, text_space)]

                pos, dist = -1, 99999
                for p1 in points:
                    for p2 in anchor:
                        if p1 - p2 < 0:
                            continue
                        if '\n' in text[p2: p1]:
                            continue
                        if dist > p1 - p2:
                            dist = p1 - p2
                            pos = p1
                if pos == -1:
                    pos = text_space.find(v)
            else:
                pos = text_space.find(v)
                if pos == -1:
                    s = None
                    e = 0
                    while s is None and e < 3:
                        e += 1
                        s = regex.search("(" + v + "){e<=" + str(e) + "}", text_space)
                    if s is not None:
                        v = s[0]
                        pos = text_space.find(v)

            text_class[pos: pos + len(v)] = i

        data[key] = (text, text_class)
    return data


if __name__ == '__main__':
    data = create_data('path/to/data')

    with open('data/val_data.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)