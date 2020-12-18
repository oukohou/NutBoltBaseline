# -*- coding: utf-8 -*-

__author__ = 'kohou.wang'
__time__ = '20-12-17'
__email__ = 'oukohou@outlook.com'

# If this runs wrong, don't ask me, I don't know why;
# If this runs right, thank god, and I don't know why.
# Maybe the answer, my friend, is blowing in the wind.
# Well, I'm kidding... Always, Welcome to contact me.

"""Description for the script:
generate trian.csv according to training date.
"""

import os

label_json = {
    'neg': 0,
    'pos': 1,
}


def generate_train_csv(src_dir_, dst_csv_):
    global label_json
    
    dst_csv_ = open(dst_csv_, 'w')
    dst_csv_.write('{},{}\n'.format('filename', 'label'))
    count_ = 0
    for dirpath, dirnames, filenames in os.walk(src_dir_):
        for filename_ in filenames:
            pathname_ = os.path.basename(dirpath)
            label_ = label_json[pathname_]
            dst_csv_.write('{},{}\n'.format("{}{}{}".format(pathname_, os.sep, filename_), label_))
            count_ = count_ + 1
            if count_ % 100 == 0:
                print("{} processed ...".format(count_))
    print("{} processed ...".format(count_))


if __name__ == "__main__":
    src_dir = "path/to/datasets/螺母螺栓产品智能检测/螺栓质量检测-训练集/螺栓质量检测-训练集"
    train_file = "train.csv"
    generate_train_csv(src_dir, train_file)
