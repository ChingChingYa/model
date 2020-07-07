
import os  
import sys  

from sklearn.decomposition import LatentDirichletAllocation
from load_data import load_data
from operator import itemgetter, attrgetter
import xlsxwriter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    
    alldata = load_data('./data/filedata1.pickle')
    train, test = train_test_split(alldata, test_size=0.2, random_state=1)


    data_x = []
    for i, sent in enumerate(train):
        doc = []
        for k, s in enumerate(sent):
            t = ''
            for j, word in enumerate(s):
                    if (k == 0) & (j == 0):
                        t += word
                    else:
                        t += ' '+word

            doc.append(t)
        # data_x.append(doc)
        data_x.extend(doc)

    corpus = data_x
    vectorizer = CountVectorizer(max_df = 0.9, min_df =2, stop_words = 'english')
    x = vectorizer.fit_transform(corpus)
    # 忽略在文章中佔了90%的文字(即去除高頻率字彙)
    # 文字至少出現在2篇文章中才進行向量轉換
    # n_components => 想分成幾群
    # random_state => 設定成42
    LDA = LatentDirichletAllocation(n_components=5, random_state=42)
    LDA.fit(x)

    for i, topic in enumerate(LDA.components_):
        print(f"TOP 5 WORDS PER TOPIC #{i}")
        # worksheet.write(i+1, 0, )
        print([vectorizer.get_feature_names()[index] for index in topic.argsort()[-5:]])

