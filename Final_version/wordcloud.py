import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from PIL import Image
import unidecode as uni
from os import path
import os
import random
import json


def fix_encoding(s):
    return uni.unidecode(s.encode('iso-8859-1').decode('utf-8'))

out = ""
dataset = []

paths = []
s = "D:\\Datasets\\facebook\\facebook-json-all-low"
files = os.listdir("{}\\messages\\inbox\\".format(s))
dataset = []
for file in files:
    conversations = os.listdir("{}\\messages\\inbox\\{}\\".format(s,file))
    for messages in conversations:
        if(str(messages).endswith('.json')):
            paths.append("{}\\messages\\inbox\\{}\\{}".format(s,file,messages))


for path in paths:
    with open(path,"r",encoding="utf-8") as f:
        data = json.load(f)
        for a in data['messages']:
            if("content" in a):
                for s in a["content"].split(" "):
                    out += fix_encoding(s).lower() + " "
                    dataset.append(fix_encoding(s).lower())


stop_words = ["za",
"fuck",
"jebe",
"kokot",
"debil",
"jebat",
"pici",
"skurveny",
"pica",
"picus",
"dopice",
"kokotina",
"debilina",
"jebnuty",
"jebnuta",
"kurva",
"dopici",
"ne",
"na",
"mam",
"do",
"ja,"
"ty",
"a",
"z",
"co",
"je",
"jak",
"to",
"ci",
"aj",
"ze",
"sa",
"si",
"kto",
"alebo",
"v",
"o",
"ti",
"mi",
"tam",
"ty",
"uz",
"ako",
"toto",
"na",
"jej",
"ma",
"ale",
"tu",
"da",
"bo",
"no",
"ho",
"som",
"by",
"d",
"ta",
"mas",
"kde",
"ani",
"len",
"s",
"ja",
"tak",
'hej',
"ked",
"este",
"asi",
"lebo",
]

#stop_words = ["jebe","kokot","debil","jebat"]


i = 0
words = {}

for word in out.split():
    if(word not in words):
        words[word] = 1
    else:
        words[word] += 1


wordcloud= WordCloud(max_words = 350,scale=3,stopwords=stop_words,background_color="#040916",collocation_threshold=30,min_font_size = 8,font_step = 1).generate(out)

wordcloud.to_file("messenger_all_wordcloud.png")
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
