> wordcloud z facebook konverzacii
![messenger_all_wordcloud](https://user-images.githubusercontent.com/56136477/167495622-f8bfcf30-4417-4bdf-b892-09356acac4d9.png)

# chatbotSK s využitím Seq2Seq

Seq2Seq je typ enkóder-dekóder modela využivajúci RNN, ktorý za použiva na interakciu s zariadeniami a prekladanie

Učením sa na veľkom množstve sekvencíi sa model naučí generovat odpovede
Výhoda daného modelu je, ze input a output nemusia byt rovnaké narozdiel od veľa iných modelov

![image](https://user-images.githubusercontent.com/56136477/167486693-3cef787d-c973-4e36-ac8d-de952f4a755a.png)

princíp činnosti trénovaného modelu
Model sa skladá z 2 LSTM (long short term memory) vrstiev (enkodera a dekodera).
Enkóder nám spracuje inputovú sekvenciu na embedovací vektor, ktorého cieľom je reprezentovať danú inputovú sekvenciu.
Tento embedovací vektor je input pre náš dekóder, ktorého cieľom je generovat pokračovanie sekvencie.
Na to využíva metódu nazývanú teacher forcing, ktorá ako input do dekódera dáva taktiež požadovanú výstupnú sekvenciu posunutú
o jeden krok dopredu

![image](https://user-images.githubusercontent.com/56136477/167488977-e054615f-b54b-443d-9222-aa3ad101f813.png)


## methods.py

>súbor metód na spracovanie raw messenegerových konverzácii na použitelný dataset

### metóda  make_datasetv2(FILE_PATH) ako parameter požíva cestu k súboru, ktorý obsahuje súbory s konverzáciami s ludmi na fecebooku
### a následne vyhľada všetky súbory konverzácii



```
paths = []
    files = os.listdir("{}\\messages\\inbox\\".format(s))
    dataset = []
    for file in files:
        conversations = os.listdir("{}\\messages\\inbox\\{}\\".format(s,file))
        for messages in conversations:
            if(str(messages).endswith('.json')):
               paths.append("{}\\messages\\inbox\\{}\\{}".format(s,file,messages))
```
### z json súborov, v ktorych su facebookove dáta uložene vyčitame požadované hodnoty, čas poslania,meno odosielateľa a obsah

```
for path in paths:
        print(path)
        f = open(path,"r")
        data = json.load(f)
        out = []

        user = "Jakub Komanicky"
        
        for a in data['messages']:
            if("content" in a):
                for sentence in naive_sentenc_split(fix_encoding(a['content']).replace(".",". ")):
                    out.append([fix_encoding(a["sender_name"]),a["timestamp_ms"],sentence.replace("\n","") +" "])

        out.reverse()
```        
### následne su dáta uložené do formátu otázka odpoveď

```
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
```

### metóda write_dataset() nám dataset vypíše do suboru csv 

## chatbot.py

> obsahuje metódy na vytvorenie inputov a outputov pre náš model a samotný model

### na začiatku importujeme potrebné knižnice
```
import numpy as np
import re
import random
from keras.callbacks import ModelCheckpoint,CSVLogger
from keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed
from keras.models import Model
from keras.utils.vis_utils import plot_model
import unidecode as uni
```
### načítame dáta z nášho csv súboru
```
data = []
with open("FILE_PATH","r") as f:
  data = f.read()
```  

### Vytvoríme input pre náš enkóder zložený z prvych 20 slov otázky a input pre náš dekóder, taktiež zložený z prvých 20 slov odpovede
```
for x in dataset:
    sent = ""
    max_words = 20
    for word in (x[0]).split(" "):
        if(word != ''):
          if(max_words == 0):
            break
          max_words -= 1
          sent += (word) + " "
    
    max_words = 20
    encoder_input_data.append(sent) 
    sent = ""
    for word in (x[1]).split(" "):
        if(word != ''):
            if(max_words == 0):
              break
            max_words -= 1
            sent += (word) + " "
    decoder_input_data.append(sent) 
``` 
```
#encoder_input_data
['ahoj ako sa mas ', 'serus ', 'cau ', 'serusco robis ', 'dobry den ', 'hello ', 'vitaj ', 'ahoj 
#decoder_input_data
['mam sa dobre ', 'serus ', 'ako sa mas ', 'teraz velmi nic ', 'ahoj ', 'ahoj ', 'ako to ide ', '
```
### poradie slov v inputoch pre náš dekóder prehodíme
``` 
 for i,sent in enumerate(decoder_input_data):
  tmp = ""
  for word in sent.split(" ")[::-1]:
    tmp += word + " "
  decoder_input_data[i] = tmp
```
```
#decoder
[' dobre sa mam ', ' serus ', ' mas sa ako ', ' nic velmi teraz 

```

### na začiatok a koniec našich odpovedí prídame speciálny znak, ktorý nam naznačuje, kde odpoveď začína a kde končí
```
bos = "<SOS>"
eos = "<EOS>"

for text in range(len(decoder_input_data)):
  decoder_input_data[text] = bos + decoder_input_data[text] + eos
``` 
```
#pridanie znakov
['<SOS> dobre sa mam <EOS>', '<SOS> serus <EOS>', '<SOS> mas sa ako <EOS>'

```
### vytvoríme zoznam všetkých slov a každému slovu pridelíme index
```
  def vocab_creater2(encoder_input_data,decoder_input_data):
       enword2idx = {}
       deword2idx = {}

       word2idx = {}
       deidx2word = {}
       i = 1
       enword2idx["<NOT>"] = 0
       for sent in encoder_input_data:
          for line in sent.split():
              if(not line in enword2idx and line != ""):
                   enword2idx[line] = i
                  
                   i += 1
       i = 1
       deword2idx["<NOT>"] = 0
       for sent in decoder_input_data:
         for line in sent.split():
              if(not line in deword2idx and line != ""):
                   deword2idx[line] = i
                   deidx2word[i] = line
                   i += 1

       return enword2idx,deword2idx,deidx2word
      
  enword2idx,deword2idx,deidx2word = vocab_creater2(encoder_input_data,decoder_input_data)
 
```
```
#enword2idx
{'<NOT>': 0, 'ahoj': 1, 'ako': 2, 'sa': 3, 'mas': 4, 'serus': 5, 'cau': 6, 'serusco': 7,
```
### One-hot-encodneme náš input a output jedinečné sekvencie núl a jednotiek
``` 
encoder_input = np.zeros(
       (len(encoder_input_data), MAX_EN_LEN, len(enword2idx)),
       dtype='float32')

decoder_input = np.zeros(
       (len(decoder_input_data), MAX_DE_LEN+2, len(deword2idx)),
        dtype='float32')

print(deword2idx)
decoder_target = np.zeros(
     (len(decoder_input_data), MAX_DE_LEN+2, len(deword2idx)), dtype='float32')


for line, (input_doc, target_doc) in enumerate(zip(encoder_input_data, decoder_input_data)):
    for word, token in enumerate(input_doc.split()):
        encoder_input[line, word, enword2idx[token]] = 1.
    
    for word, token in enumerate(target_doc.split()):
          decoder_input[line,word,deword2idx[token]] = 1.
          if word > 0:
              decoder_target[line,word - 1,deword2idx[token]] = 1.

  
 ``` 
 ```
 #tvar nasho enkoder inputu
 (1134, 20, 1985)
 #one-hot-encoder enkoder input
 [[0. 1. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]]
 
 ```
 
 ## Postavíme vrstvy nášho model,

 ### model sa skladá z inputovej vrstvy enkódera,samotného enkódera, inputovej vrstvy dekódera, samotného dekodera a nakoniec dense vrstvy
  
```  
EMBEDDING_DIM = 256
encoder_inputs = Input(shape=(None, len(enword2idx), ), name="Encoder_input")
encoder_LSTM = LSTM(EMBEDDING_DIM, return_state=True, name='Encoder_lstm')
encoder_outputs, state_h, state_c = encoder_LSTM(encoder_inputs)
encoder_states = [state_h,state_c]


decoder_inputs = Input(shape=(None,len(deword2idx) ),name = "Decoder_input")
decoder_LSTM = LSTM(EMBEDDING_DIM, return_state=True, return_sequences=True, name="Decoder_lstm")
decoder_outputs, _, _ = decoder_LSTM(decoder_inputs, initial_state= encoder_states)
decoder_dense = Dense(len(deword2idx), activation='softmax', name="Dense_layer") 
decoder_outputs = decoder_dense(decoder_outputs) 
```
  
### Postavíme  model,
```  
#ukladá váhy modela, aby sme vedeli pracovať už s natrénovanym modelom
filepath="WEIGHTS_FILEPATH"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, mode='max')
callbacks_list = [checkpoint]
  
#ukladá históriu trénovania
csv_logger = CSVLogger(HISTORY_SAVE_LOCATION, append=True)


model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer= "rmsprop", loss='categorical_crossentropy', metrics=['acc'],sample_weight_mode='temporal')

model.load_weights(WEIGHTS_FILEPATH)

 model.fit([encoder_input, decoder_input], decoder_target,
          batch_size=16,
          epochs=500,
          validation_split=0.20,
          callbacks=[callbacks_list,csv_logger])
  
 
  
``` 
```
#model.summary()
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 Encoder_input (InputLayer)     [(None, None, 1985)  0           []                               
                                ]                                                                 
                                                                                                  
 Decoder_input (InputLayer)     [(None, None, 2891)  0           []                               
                                ]                                                                 
                                                                                                  
 Encoder_lstm (LSTM)            [(None, 256),        2295808     ['Encoder_input[0][0]']          
                                 (None, 256),                                                     
                                 (None, 256)]                                                     
                                                                                                  
 Decoder_lstm (LSTM)            [(None, None, 256),  3223552     ['Decoder_input[0][0]',          
                                 (None, 256),                     'Encoder_lstm[0][1]',           
                                 (None, 256)]                     'Encoder_lstm[0][2]']           
                                                                                                  
 Dense_layer (Dense)            (None, None, 2891)   742987      ['Decoder_lstm[0][0]']           
                                                                                                  
==================================================================================================
Total params: 6,262,347
Trainable params: 6,262,347
Non-trainable params: 0
__________________________________________________________________________________________________
None

```
![image](https://user-images.githubusercontent.com/56136477/167493349-86d36bd0-624d-44ab-8e94-3b925f9d6bde.png)

### Keďže sme použili teacher forcing pre náS trénovací model musíme vytvorit interface model na generovanie sekvencií bez uz danej výstupnej sekvencie
```
encoder_model = Model(encoder_inputs, encoder_states) 

decoder_state_input_h = Input(shape=(EMBEDDING_DIM,), name="H_state_input") 
decoder_state_input_c = Input(shape=(EMBEDDING_DIM,), name="C_state_input") 
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c] 

decoder_outputs, state_h, state_c = decoder_LSTM(decoder_inputs,initial_state=decoder_states_inputs) 
decoder_states = [state_h, state_c] 
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
```  
```
decoder_model.summary()
Model: "model_3"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 Decoder_input (InputLayer)     [(None, None, 2891)  0           []                               
                                ]                                                                 
                                                                                                  
 H_state_input (InputLayer)     [(None, 256)]        0           []                               
                                                                                                  
 C_state_input (InputLayer)     [(None, 256)]        0           []                               
                                                                                                  
 Decoder_lstm (LSTM)            [(None, None, 256),  3223552     ['Decoder_input[0][0]',          
                                 (None, 256),                     'H_state_input[0][0]',          
                                 (None, 256)]                     'C_state_input[0][0]']          
                                                                                                  
 Dense_layer (Dense)            (None, None, 2891)   742987      ['Decoder_lstm[1][0]']           
                                                                                                  
==================================================================================================
Total params: 3,966,539
Trainable params: 3,966,539
Non-trainable params: 0
__________________________________________________________________________________________________
```
 ![image](https://user-images.githubusercontent.com/56136477/167493727-0ec14af9-a123-4abe-a386-ca1805ed656f.png)

# Grafy trénovania modla
 ![training_acc_training_loss](https://user-images.githubusercontent.com/56136477/167494318-498f9083-1f1b-4538-b11e-f36018535552.png)
![training![training_val_acc](https://user-images.githubusercontent.com/56136477/167494277-c696a7c2-d907-4090-8ab9-0c06ebc060dd.png)
![training_val_loss](https://user-images.githubusercontent.com/56136477/167494281-dc8c2225-7939-45e4-ac05-54887afe37ea.png)

# Výslená presnosť je 40%, čo je dosť nízka, ale dala by sa zvýšit lepšie urobeným datasetom a modelom

 ### Nakoniec na komunikáciu s naším modelom musíme náš input spracovať da dať do nášho interfacu

``` 
  def process(sentence):
  sentence = re.sub("[?!,.1-9@#$%/\(){}+-=*&^>'\"]_~", "", sentence)
  sentence = uni.unidecode(sentence).lower().replace("  "," ")

  return sentence


def decode_response(test_input):
   
    input = ""
    for word in test_input.split()[::-1]:
      input += word + " "

    input_matrix = np.zeros((1, MAX_EN_LEN, len(enword2idx)),dtype='float32')
      
    for timestep, token in enumerate(input.split()):
      if token in enword2idx:
        input_matrix[0, timestep, enword2idx[token]] = 1.
      else:
        input_matrix[0, timestep, enword2idx["<NOT>"]] = 1.
      

    states_value = encoder_model.predict(input_matrix)

    target_seq = np.zeros((1, 1, len(deword2idx)))

    target_seq[0, 0, deword2idx['<SOS>']] = 1.
    
    decoded_sentence = '' 
    stop_condition = False
    while not stop_condition:

      output_tokens, hidden_state, cell_state = decoder_model.predict([target_seq] + states_value)
      
     
      sampled_token_index = np.argmax(output_tokens[0, -1, :])
      #print(sampled_token_index)
      sampled_token = deidx2word[sampled_token_index]

      if (sampled_token == '<EOS>' or len(decoded_sentence) > 22):
        stop_condition = True
      else:
        decoded_sentence += " " + sampled_token
      target_seq = np.zeros((1, 1, len(deword2idx)))
      target_seq[0, 0, sampled_token_index] = 1.


      states_value = [hidden_state, cell_state]
    
    out = ""
    for word in decoded_sentence.split()[::-1]:
      out += word + " "
    
    return out

print("write exit to stop")
while True:
  query = process(input())

  stopwords = ["stop","koniec","exit","dost"]
  if query in stopwords:
      break
  print(decode_response(query))
 ```
 ```
write exit to stop
ahoj
dneska ako sa mas 
mam sa dobre a ty
just it it je it sounded 
Ako sa máš.
triggered co si das ze 

 ```
