#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-03-25 11:17
# @Author  : zhangzhen
# @Site    : 
# @File    : data_utils.py
# @Software: PyCharm
import os
import json
import codecs
from pprint import pprint
from typing import Text


def data2ner(path: Text, filename: Text, to: Text):
    """
    NER——定义识别的实体, 并转化训练、测试语料
    :param path:
    :param file:
    :param to:
    :return:
    """
    # 组织 人名 地名 日期 作品 Number TEXT Other
    # ORG  PER  LOC DAT PRO NUM TXT OTH
    schemas = {
        "网络小说": "PRO",
        "企业": "ORG",
        "Number": "NUM",
        "电视综艺": "OTH",
        "地点": "LOC",
        "机构": "ORG",
        "语言": "OTH",
        "歌曲": "PRO",
        "城市": "LOC",
        "人物": "PER",
        "书籍": "PRO",
        "出版社": "ORG",
        "行政区": "LOC",
        "Text": "TXT",
        "影视作品": "PRO",
        "音乐专辑": "PRO",
        "学校": "ORG",
        "Date": "DAT",
        "图书作品": "PRO",
        "生物": "OTH",
        "学科专业": "OTH",
        "景点": "LOC",
        "网站": "OTH",
        "目": "OTH",
        "气候": "OTH",
        "作品": "PRO",
        "历史人物": "PER",
        "国家": "LOC"
    }
    i = 0
    tag2label = {
        "O": i
    }
    for val in set(schemas.values()):
        i += 1
        tag2label["{}-{}".format("B", val)] = i
        i += 1
        tag2label["{}-{}".format("I", val)] = i

    print(tag2label)

    with codecs.open(to, mode="w+", encoding="utf-8") as wf:
        with codecs.open(path + os.sep + filename, encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                rs = json.loads(line.strip())
                # print(rs['postag'])
                # print(rs['text'])
                # print(rs['spo_list'])
                txt = rs['text'].upper()  # 转成大写, 防止语料出错
                label_dict = {}
                for ent in rs['spo_list']:
                    # print(ent)
                    obj = ent['object'].upper()
                    sub = ent['subject'].upper()
                    obj_type = ent['object_type']
                    sub_type = ent['subject_type']
                    # print(txt.index(obj), len(obj))
                    # print(txt.index(sub), len(sub))
                    try:
                        label_dict[txt.index(obj)] = {
                            "length": len(obj),
                            "type": obj_type
                        }
                        label_dict[txt.index(sub)] = {
                            "length": len(sub),
                            "type": sub_type
                        }
                    except ValueError:
                        pass
                # print(label_dict)

                i = 0
                while i < len(txt):
                    w = txt[i]
                    if i in label_dict:
                        label = label_dict[i]
                        if label["length"]:
                            for ii in range(label["length"]):
                                w = txt[ii + i]
                                if w != " " and w != "\t" and w.strip() != "":
                                    tag = "{}-" + schemas[label["type"]]
                                    if ii == 0:
                                        # print(w, tag.format("B"))
                                        wf.write(w + "\t" + tag.format("B") + "\n")
                                    else:
                                        # print(w, tag.format("I"))
                                        wf.write(w + "\t" + tag.format("I") + "\n")
                            i += label['length'] - 1
                    else:
                        # print(w, "O")
                        if w != " " and w != "\t" and w.strip() != "":
                            wf.write(w + "\tO\n")
                    i += 1
                # print()
                wf.write("\n")


if __name__ == '__main__':
    path = './lic2019/original'
    filename = 'dev_data.json'
    to = './lic2019/dev_data.dat'
    data2ner(path, filename, to)
