import os
import json
import re
import nltk
import spacy
from collections import Counter
from scipy.sparse import hstack
from sklearn.model_selection import KFold
from scipy.sparse import hstack
import numpy as np
# nltk.download('all')  l'environnement marche pas
# from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, classification_report
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_class_weight



# tok_grm=re.compile(r"""
#     (?:etc.|p.ex.|cf.|M.)|
#     \w+(?=(?:-(?:je|tu|ils?|elles?|nous|vous|leur|lui|les?|ce|t-|même|ci|là)))|
#     [\w\-]+'?| # peut-être
#     .""",re.X)

tok_grm = re.compile(r"""
    (?:etc\.|p\.ex\.|cf\.|M\.)|                                     
    \b\w+(?:-\w+)*\b|                                               
    \b\w+'(?:\w+)?\b|                                               
    (?:(?<=\s)|^)[\w\-]+(?=[\s.,;!?]|\Z)|                           
    (?!^&amp;$)\b\w+\b|                                             
    \b(?!\d+\b)\w+\b                                               
    """, re.X | re.UNICODE)

def tokenize(text,tok_grm):
    return tok_grm.findall(text)

def info_author(data_dict)->int:
    # print(data_dict)
    author_forename = "" 
    author_surname = ""
    entete = data_dict.get("entête", {})
    author_info = entete.get("author", None)  # Use None as the default value
    # print(author_info)

    if author_info is None:  
        return 0
    elif isinstance(author_info, dict) and "persName" in author_info:
        # author_forename = author_info["persName"]["forename"]
        if type(author_info["persName"]) is str:  #dans certains,le type de valuer associée de persName est str
            return 0
        elif type(author_info["persName"]) is list:
            return 0
        else:
            author_forename = author_info["persName"].get("forename", "")
            author_surname = author_info["persName"].get("surname", "")
        # print(author_forename)
    elif type(author_info) is list:
        for item in author_info:
            if isinstance(item, dict) and "persName" in author_info:
                    if type(item["persName"]) is str:
                        return 0
                    else:
                        author_forename = author_info["persName"].get("forename", "")
                        author_surname = author_info["persName"].get("surname", "")
    # donner lable 1 au oeuvre de Claude
    if author_forename == "Claude" and author_surname == "Du Bosc de Montandré":
        label = 1  
    else:
        label = 0  
        
    return label

def load_stop_list(stopliste_file):
    with open(stopliste_file, 'r', encoding='utf-8') as file:
        stop_list = [line.strip() for line in file]
    return stop_list

def read_file_to_dict(file_path,stop_list)->dict:
    texte_dict={}
    with open(file_path,"r",encoding="utf-8") as f:
        print(f"Attempting to open file: {file_path}")
        data_json=f.read()
        data=json.loads(data_json)
        texte=data.get("texte", None)
        if not texte:
            return None
        else:
            texte_str=""
            for item in texte:
                texte_str+="".join(item)
            texte_tokenise=tokenize(texte_str,tok_grm)
            filtered_tokens = [token for token in texte_tokenise if token.strip() and token not in stop_list and token not in ["amp", "&amp"]]
            token_counts = Counter(filtered_tokens)
            # filtrer le hapax
            filtrage_tokens = [token for token in filtered_tokens if token_counts[token] > 1]
            texte_dict["label"]=info_author(data)
            texte_dict["texte"]=filtrage_tokens
    return texte_dict

def folder_to_data(folderpath,stopliste_filename)->list:
    stopliste=load_stop_list(stopliste_filename)
    data_set=[]
    for filename in os.listdir(folderpath):
        file_path = os.path.join(folderpath, filename)
        if os.path.isfile(file_path) and filename.endswith('.json'):
            texte_data=read_file_to_dict(file_path,stopliste)
            if texte_data:  # examiner 'texte_data' est None ou pas
                data_set.append(texte_data)
            # data_set.append(texte_data)
        elif os.path.isdir(file_path):
            for file in os.listdir(file_path):
                sub_file_path = os.path.join(file_path, file)
                if os.path.isfile(sub_file_path) and file.endswith('.json'):
                    texte_data = read_file_to_dict(sub_file_path,stopliste)
                    if texte_data:  
                        data_set.append(texte_data)
    #data_set.append(texte_data)
    print(len(data_set)) 
    return data_set

def extrafeature(dataset):
    texte_claude=[]
    count=0
    for j in dataset:
        if j["label"]==1:
            count+=1
            texte_claude.append(j)
    #print(texte_claude)
            # print(j)  
    print(count)

    nlp = spacy.load("fr_core_news_sm")
    nouns = []

    for item in texte_claude:
        #revenir au str l'ensemble
        doc_text = ' '.join(item["texte"])
        doc = nlp(doc_text)
    
        doc_nouns = [token.text for token in doc if token.pos_ == "NOUN"]
        nouns.extend(doc_nouns)
        conjunctions = [token.text for token in doc if token.pos_ == "CONJ" or token.pos_ == "SCONJ"]

    noun_freq = Counter(nouns)
    most_common_nouns = noun_freq.most_common(10)

    for noun, freq in most_common_nouns:
        print(f"{noun}: {freq}"+
              "########################")
# ##############

   
    conj_freq = Counter(conjunctions)
    most_common_conjs = conj_freq.most_common(10)

    for conj, freq in most_common_conjs:
        print(f"{conj}: {freq}"+
              "@@@@@@@@@@@@@@@@")

    return True

def tf_idf(data):
    texts = [' '.join(item["texte"]) for item in data]
    labels_y = np.array([item["label"] for item in data])

    # tf-idf 
    vectorizer_specific = TfidfVectorizer(vocabulary=['nous'])
    vectorizer_specific2 = TfidfVectorizer(vocabulary=['que','si','ſi'])
    vectorizer_specific3=TfidfVectorizer(vocabulary=['Eſtat','point','Princes','iamais','peuples'])
    X_specific = vectorizer_specific.fit_transform(texts)
    X_specific2 = vectorizer_specific2.fit_transform(texts)
    X_specific3 = vectorizer_specific3.fit_transform(texts)

    vectorizer_auto = TfidfVectorizer()
    X_auto = vectorizer_auto.fit_transform(texts)

    texts_x = hstack([X_specific,X_specific2,X_specific3, X_auto])

    return texts_x, labels_y

def calcul_ngram(data, ngram_range=(3, 5)):
    texts = [' '.join(item["texte"]) for item in data]
    labels = np.array([item["label"] for item in data])

    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    feature_matrix = vectorizer.fit_transform(texts)
 
    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, labels, test_size=0.2, random_state=42)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    mnb = MultinomialNB()
    mnb.fit(X_train, y_train, sample_weight=[class_weight_dict[y] for y in y_train])
    
    y_pred = mnb.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='binary')
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

    return True

def display_top_ngrams(texts, top_n=10):
    # 对于bigram、trigram和4-gram
    for n in range(2, 5):
        vectorizer = TfidfVectorizer(ngram_range=(n, n), analyzer='word', token_pattern=r'\b\w+\b')
        X = vectorizer.fit_transform(texts)
        
        sum_words = X.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_n]
        
        total_counts = sum([freq for _, freq in words_freq])
        print(f"\nTop {top_n} {n}-grams:")
        for word, count in words_freq:
            print(f"{word}: Count = {count}, frequency = {count / total_counts:.4f}")
    
    return True


def logic_regression(texts_x,labels_y):
    clf = LogisticRegression(class_weight='balanced')

    y_pred = cross_val_predict(clf, texts_x, labels_y, cv=5)

    cm = confusion_matrix(labels_y, y_pred)
    acc = accuracy_score(labels_y, y_pred)
    recall = recall_score(labels_y, y_pred, average='micro')
    f1 = f1_score(labels_y, y_pred, average='micro')

    print(cm)
    print(f"############\n"
      f"Accuracy: {acc}, Recall: {recall}, "
      f"F1 Score: {f1}")
    
    return True


def classifier_NB(texts_x, labels_y):
    X_train, X_test, y_train, y_test = train_test_split(texts_x, labels_y, test_size=0.2, random_state=42)
    
    # calcluer les poids de chaque classe
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    
    # donner chqaue échantillon une poids classe
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train, sample_weight=[class_weight_dict[y] for y in y_train])
    
    y_pred = mnb.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='binary')  # pour classification binaire
    f1 = f1_score(y_test, y_pred, average='binary')  
    report = classification_report(y_test, y_pred)
    

    print(f"Accuracy: {accuracy}")
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Classification Report:")
    print(report)
    
    # crosse validation
    scores = cross_val_score(mnb, texts_x, labels_y, cv=5)
    print("Accuracy scores for each fold:", scores)
    print("Mean cross-validation accuracy:", np.mean(scores))
    return True


    

if __name__=="__main__":
    dataset=folder_to_data("/Users/luoweiwei/Documents/projetML/Mazarinades_jsons","/Users/luoweiwei/Documents/projetML/stopliste.txt")
    texts_x, labels_y=tf_idf(dataset)
    classifier_NB(texts_x,labels_y)
    # logic_regression(texts_x,labels_y)

    # extrafeature(dataset)
    #print(dataset)


    # 1:
    # {'persName': {'forename': 'Jean-Olivier', 'surname': 'Du Sault'}}
    # 2:
    # {'@ref': 'isni:0000000061389452', '@source': 'Mazarine', '@role': 'alleged_author', 
    # 'persName': {'forename': 'Jacques de', 'surname': 'Lescornay'}}
    #3:
    # [{'@ref': 'isni:0000000119302891', '@source': 'Mazarine', 
    #'persName': {'forename': 'Armand de Bourbon', 'surname': 'Conti'}}, {'@ref': 'isni:0000000119572170', '@source': 'Mazarine', 'orgName': 'Parlement de Bordeaux'}]


#4:
#{'imprimatur': 'Cum licentia.', 'corrector': False, 
#'entête': {'form': 'prose', 'handwritten_note': False, 'table_of_content': False, 'illustration': False, 'subject': 'Conti, Armand de Bourbon (1629-1666\xa0; prince de)', 'creation': '1649-01-11', 
#'change': {'@status': 'corrected', '@when': '2022-02-07', '@who': 'AB'}, 
#'titre': 'Armandus armans.', 'dates': {'@type': 'file_creation', '@when': '2022-01-31', '#text': '31 janvier 2022'}, 
#'author': {'@source': 'Mazarine', '@role': 'alleged_author', 'persName': 'Mérigot'}, 'pubDate': [{'@source': 'Mazarine', '@when': '1649', '#text': '1649'}, {'@source': 'Carrier', '@when': '1649-01-15', '#text': 'au début du blocus de Paris, à la mi-janvier\n            1649'}]