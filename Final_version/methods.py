
import json
import unidecode as uni
from sklearn.decomposition import PCA
import pandas as pd
import re
import os
import csv



def flatten2D(lst):
    out = []
    for line in lst:
        for sent in line:
            out.append(sent)
    return out
    
#naive sentence split jff
def naive_sentenc_split(text):

    sites = ["sk","en","eu","com"]
    
    out = []
    text = text.split('?')
    for i in range(len(text)):
        if(not i == len(text)-1):
            out.append(text[i]+"? ")
      
     
    for part in range(len(out)):
        temp = []
        text = out[part].replace("!!!","!").split('!')
        for i in range(len(text)):
            if(not i == len(text)-1):
                temp.append(text[i]+"!")
            else:
                temp.append(text[i])
       
        out[part] = temp
      
    out =  flatten2D(out)

    for part in range(len(out)):
        temp = []

        text = out[part].replace("...",".").split('.')

        for i in range(len(text)):
            
            if("www" in text[i] or "http" in text[i]):
                s = ""
                if(i < len(text) -1):
                    for txt in text:
                        s += txt.strip()+ "."
                   
                
                temp.append(s)
                break

            
            else:
                if(not i == len(text)-1):
                    temp.append(text[i]+".")
                else:
                    temp.append(text[i])
        
        out[part] = temp
    out =  flatten2D(out)

    for line in out:
        line.strip().replace("  ", " ")
        line.replace("\n", "").replace("\t", "")
        if(line == ''):
            line.remove()
    
    return out
   

#removes all accents from words so we don't have to deal with them
def remove_accent(s):
    return uni.unidecode(s)

#json files save in the wrong encoding so we have to fix it 
def fix_encoding(s):
    #return s
    return uni.unidecode(s.encode('iso-8859-1').decode('utf-8'))

#experimental methods that check if the word contains accents
def contains_accent(s):
    if(remove_accent(s) != s):
        return True
    return False

#build a better intent classifier
#I need a better way to guess if something is a question
# an approach of trying to classify sentences into question, 
# didn't work much on this approach since I think there are better ones
def is_question(ans):
    question = ['nevedel by si','nemohla by si','nevedel by si',
    'mohol by si','mohla by si','vies mi','mozes mi',
    'by si mi','videl si','mozes',"nevies","prosim vedel","ako"]
    if("?" in ans):
        return True
    
    for ques in question:
        if(ques in ans):
            return True

    return False

def preprocess(sentence):
    s = ""
    for word in sentence.split(" "):
        s += re.sub("[^A-Za-z']+", ' ',remove_accent(word.lower()).strip().replace("  "," ").replace("..",".")) + " "
    s = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", s)
    return s.replace('\t',"")


def make_datasetv2(s):

    paths = []
    files = os.listdir("{}\\messages\\inbox\\".format(s))
    dataset = []
    for file in files:
        conversations = os.listdir("{}\\messages\\inbox\\{}\\".format(s,file))
        for messages in conversations:
            if(str(messages).endswith('.json')):
               paths.append("{}\\messages\\inbox\\{}\\{}".format(s,file,messages))

    
    
    for path in paths:
        print(path)
        f = open(path,"r")
        data = json.load(f)
        out = []

        user = "Jakub Komanicky"
        #gets the content of the messages from the messenger data
        for a in data['messages']:
            if("content" in a):
                for sentence in naive_sentenc_split(fix_encoding(a['content']).replace(".",". ")):
                    out.append([fix_encoding(a["sender_name"]),a["timestamp_ms"],sentence.replace("\n","") +" "])

        #reverses it so it is from oldst to youngest         
        out.reverse()

        #makes the dataset for now only by looking if the reply is withing a timeframe 
        #or looks if the reply is another quetsion
        #upgrade this method a lot but now not in the mood

        for line in range(len(out)):
            ttl = 3
            if(user != out[line][0]):
                question = preprocess(out[line][2])
                if(len(question) < 5):
                    # if it is just a question mark
                    continue
            else:
                continue

            answer = ""
            for answer_index in range(line+1,len(out)-2):
                if(user == out[answer_index][0] and ttl):
                    if(len(out[answer_index][2]) > 4):
                        answer += preprocess(out[answer_index][2])
                else:
                    line = answer_index 
                    if(answer != ""):
                        dataset.append([question.strip(),answer.replace("  "," ").strip()])
                    break
        

    return dataset

#writes the finished dataset to a json file
def write_dataset(dataset):

    json_object = json.dumps(dataset)
   
    header = ["question","answer"]
    with open("Final_version\sample.csv", "w") as csv_file:
        for line in header:
            csv_file.write(line+";")
        csv_file.write('\n')

        for line in dataset:
            for parameter in line:
                csv_file.write(parameter+";")
            csv_file.write('\n')

   

