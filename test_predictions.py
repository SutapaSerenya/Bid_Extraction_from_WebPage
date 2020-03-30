import nltk
nltk.download("stopwords")
from pandas import DataFrame
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Conv1D, MaxPooling1D,Bidirectional
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
STOPWORDS = set(stopwords.words('english'))
from bs4 import BeautifulSoup
import pickle
from selenium import webdriver
from selenium.webdriver.common.by import By
from unidecode import unidecode
from pprint import pprint
import time
def create_model():
    model = Sequential()
    model.add(Embedding(50000, 300, input_length=500))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=100, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = create_model()
model.load_weights("checkpoints")

with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

links = ["https://www.purcellvilleva.gov/bids.aspx?bidID=43", "https://www.purcellvilleva.gov/bids.aspx?bidID=49", "https://www.purcellvilleva.gov/bids.aspx?bidID=50", "http://www.fbgtx.org/bids.aspx?bidID=76", "http://www.fitchburgma.gov/bids.aspx?bidID=156", "http://www.fitchburgma.gov/bids.aspx?bidID=126", "http://www.fitchburgma.gov/bids.aspx?bidID=124", "http://www.fitchburgma.gov/bids.aspx?bidID=125", "http://www.fitchburgma.gov/bids.aspx?bidID=157", "http://www.fitchburgma.gov/bids.aspx?bidID=164", "http://www.fitchburgma.gov/bids.aspx?bidID=166", "http://www.fitchburgma.gov/bids.aspx?bidID=165", "https://www.haysusa.com/bids.aspx?bidID=31", "http://www.ci.greenville.tx.us/bids.aspx?bidID=385", "http://www.ci.greenville.tx.us/bids.aspx?bidID=377", "http://www.ci.greenville.tx.us/bids.aspx?bidID=386", "http://www.ci.greenville.tx.us/bids.aspx?bidID=375", "http://www.ci.greenville.tx.us/bids.aspx?bidID=348", "http://www.ci.greenville.tx.us/bids.aspx?bidID=374", "http://www.ci.greenville.tx.us/bids.aspx?bidID=346", "http://www.ci.greenville.tx.us/bids.aspx?bidID=344", "http://www.ci.greenville.tx.us/bids.aspx?bidID=382", "http://www.ci.greenville.tx.us/bids.aspx?bidID=361", "https://www.cityofsanmateo.org/bids.aspx?bidID=591", "https://www.cityofsanmateo.org/bids.aspx?bidID=590", "https://www.cityoforange.org/bids.aspx?bidID=120", "https://www.cityoforange.org/bids.aspx?bidID=118", "https://www.cityoforange.org/bids.aspx?bidID=122", "https://www.cityoforange.org/bids.aspx?bidID=119", "http://moline.il.us/bids.aspx?bidID=621", "http://moline.il.us/bids.aspx?bidID=623", "http://moline.il.us/bids.aspx?bidID=622", "http://ca-pismobeach2.civicplus.com/bids.aspx?bidID=71", "http://ca-pismobeach2.civicplus.com/bids.aspx?bidID=84", "http://ca-pismobeach2.civicplus.com/bids.aspx?bidID=86", "http://ca-pismobeach2.civicplus.com/bids.aspx?bidID=85", "https://cityofmhk.com/bids.aspx?bidID=898", "http://www.cityofdaytontx.com/bids.aspx?bidID=20", "http://www.cityofdaytontx.com/bids.aspx?bidID=19", "https://www.daytonohio.gov/bids.aspx?bidID=593", "https://www.daytonohio.gov/bids.aspx?bidID=592", "https://www.daytonohio.gov/bids.aspx?bidID=594", "https://www.daytonohio.gov/bids.aspx?bidID=598", "https://www.daytonohio.gov/bids.aspx?bidID=595", "https://www.daytonohio.gov/bids.aspx?bidID=597", "https://www.daytonohio.gov/bids.aspx?bidID=599", "https://www.westfordma.gov/bids.aspx?bidID=91", "https://www.westfordma.gov/bids.aspx?bidID=90", "https://www.westfordma.gov/bids.aspx?bidID=92", "https://www.cityofbowie.org/bids.aspx?bidID=177", "https://www.cityofbowie.org/bids.aspx?bidID=175", "http://bulverdetx.gov/bids.aspx?bidID=17", "http://www.ci.boerne.tx.us/bids.aspx?bidID=85", "http://orangecountyva.gov/bids.aspx?bidID=86", "http://orangecountyva.gov/bids.aspx?bidID=82", "http://orangecountyva.gov/bids.aspx?bidID=83", "http://orangecountyva.gov/bids.aspx?bidID=85", "https://hampton.gov/bids.aspx?bidID=736", "https://hampton.gov/bids.aspx?bidID=733", "https://hampton.gov/bids.aspx?bidID=732", "https://hampton.gov/bids.aspx?bidID=735", "https://hampton.gov/bids.aspx?bidID=734", "https://hampton.gov/bids.aspx?bidID=731", "https://hampton.gov/bids.aspx?bidID=729", "https://hampton.gov/bids.aspx?bidID=730", "https://www.fallschurchva.gov/bids.aspx?bidID=87", "https://www.fallschurchva.gov/bids.aspx?bidID=86", "http://www.danville-va.gov/bids.aspx?bidID=1802", "http://www.danville-va.gov/bids.aspx?bidID=1533", "http://www.danville-va.gov/bids.aspx?bidID=1511", "http://www.danville-va.gov/bids.aspx?bidID=2332", "http://www.danville-va.gov/bids.aspx?bidID=1985", "http://www.danville-va.gov/bids.aspx?bidID=1978", "http://www.danville-va.gov/bids.aspx?bidID=1792", "http://www.danville-va.gov/bids.aspx?bidID=2317", "http://www.danville-va.gov/bids.aspx?bidID=2322", "http://www.danville-va.gov/bids.aspx?bidID=2331", "http://www.danville-va.gov/bids.aspx?bidID=2330", "http://www.danville-va.gov/bids.aspx?bidID=2335", "http://www.danville-va.gov/bids.aspx?bidID=2334", "http://www.danville-va.gov/bids.aspx?bidID=1930", "http://www.danville-va.gov/bids.aspx?bidID=2321", "http://www.danville-va.gov/bids.aspx?bidID=2320", "http://www.danville-va.gov/bids.aspx?bidID=2329", "http://www.danville-va.gov/bids.aspx?bidID=2207", "http://www.danville-va.gov/bids.aspx?bidID=2323", "http://www.danville-va.gov/bids.aspx?bidID=2333", "http://www.danville-va.gov/bids.aspx?bidID=1526", "http://www.cityofgreenriver.org/bids.aspx?bidID=169", "http://www.cityofgreenriver.org/bids.aspx?bidID=168", "http://www.midlandtexas.gov/bids.aspx?bidID=506", "http://www.midlandtexas.gov/bids.aspx?bidID=505", "http://www.midlandtexas.gov/bids.aspx?bidID=507", "http://www.midlandtexas.gov/bids.aspx?bidID=488", "http://www.midlandtexas.gov/bids.aspx?bidID=508", "https://www.springfieldmo.gov/bids.aspx?bidID=987", "https://www.springfieldmo.gov/bids.aspx?bidID=988", "http://www.cityofweirton.com/bids.aspx?bidID=19", "http://morgantownwv.gov/bids.aspx?bidID=#TranslatePagee2252b69-f257-44fb-89e9-47acd2c8e8c4", "http://morgantownwv.gov/bids.aspx?bidID=#PrinterFriendlye2252b69-f257-44fb-89e9-47acd2c8e8c4"]
driver = webdriver.Chrome(executable_path="C:\\Users\\pmr11\\Downloads\\chromedriver")


for link in links:
    driver.get(link)

    soup = BeautifulSoup(driver.page_source, "lxml")
    desired_elements=["a","td","p","h1", "span"]
    dictionary = {"Subject":[]}

    for el_type in desired_elements:
        strings = soup.find_all(el_type)
        if strings:
            #print("Whats wrong buddy?")
            for tags in strings:
                line = tags.text
                line = line.lstrip()
                line = line.rstrip()
                line = line.rstrip("\r\n")
                line = line.replace("\t","").replace("\r", "").replace("  "," ").replace("\n","")
                # line = re.sub(r'<.*?>(.*?)<.*?>', '', line)	
                if line == '\n' or line == "":
                    continue
                if len(line) < 50 and len(line) > 2:
                    dictionary['Subject'].append(unidecode(line))

    data = DataFrame(dictionary)

    X = tokenizer.texts_to_sequences(data['Subject'].values)
    X = pad_sequences(X, maxlen=500)

    pprint(data.head())
    predictions = model.predict(X)
    pprint(predictions)
    positive_predictions = [p[1] for p in predictions]
    pprint(positive_predictions)
    max_index = positive_predictions.index(max(positive_predictions))
    print(str(max(positive_predictions)), ":", data.iloc[max_index, 0])
    time.sleep(7)
driver.quit()