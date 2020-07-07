
import json
import pickle
import nltk
import gensim
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from gensim.models import word2vec
import xlsxwriter
from nltk.corpus import stopwords
import numpy as np


# 使用nltk分词分句器
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()


# ============================FileData 類別=======================
# 一個文件多個句子將全部串接在一起分成一維視為一個文件(sequential)
# words:單詞全部串連在一起list(sequential)
# alllens:一維list單詞數量

#  一個文件包多個句子，一個句子包多個單詞，分成二維list視為一個文件
# sents:二維list，一個文件包多個句子，一個句子包多個單詞
# len_sents:二維句子數量
# len_words:二維單詞數量
# ===============================================================
class FileData():
    def __init__(self, words, alllens, sents, len_sents, len_words):
        self.sents = sents
        self.len_sents = len_sents
        self.words = words
        self.len_words = len_words
        self.alllens = alllens


# ============================Word2VecFun=======================
# Word2Vec訓練
# sents:二維list
# 儲存詞表

# return:none
# ==============================================================
def Word2VecFun(sents):
    model = gensim.models.Word2Vec(sents, size=300, min_count=5) # size:維度300，min_count:刪除詞頻小於5 
    model.save('./voc/word2vec.pickle') # 儲存詞表
    model = word2vec.Word2Vec.load('./voc/word2vec.pickle')
    # 方便可視版本(詞表)
    model.wv.save_word2vec_format('./voc/word2vec2.pickle', binary=False)


# ============================LoadFile=======================
# LoadFile讀取所需檔名的欄位的值，刪除停用詞，計算FileData所需資訊
# filename:檔名
# field:要擷取之欄位

# return:FileData(words, alllens, sents, len_sents, len_words)
# words:單詞全部串連在一起list(sequential)
# alllens:一維list單詞數量
# sents:二維list，一個文件包多個句子，一個句子包多個單詞
# len_sents:二維句子數量
# len_words:二維單詞數量
# ===============================================================
def LoadFile(filename, field):
    result_sents = [] #二維list
    len_sent = [] #二維句子數量
    len_word = [] #二維單詞數量
    result_words = [] #單詞全部串連在一起list(sequential)
    all_len = [] #一維list單詞數量

    # 停用詞集合
    stop_words = set(stopwords.words('english'))
    with open(filename, 'rb') as f:
        for line in f:
            word1 = []
            word2 = []
            # 讀取完整資料
            review = json.loads(line)
            # 一個文件擷取所需欄位之句子
            sents = nltk.sent_tokenize(review[field])
            # 計算二維句子數量len_sent
            len_sent.append(len(sents))
            
            # 判斷是否存在於詞表中
            for sentence in sents:
                newword = []
                # 一個句子逐字讀取一個單詞
                sentence = word_tokenizer.tokenize(sentence)
                # 計算二維單詞數量len_word
                len_word.append(len(sentence))
                # 變成小寫
                for i, w in enumerate(sentence):
                    # 如果不是停用詞，還原其詞性
                    if w not in stop_words:
                        w = w.lower()
                        newword.append(lemmatizer.lemmatize(''.join(w), pos='v'))
       
                word1.extend(newword) #分成一維，單詞全部串連在一起
                word2.append(newword) #分成二維 一個文件包多個句子，一個句子包多個單詞
 
            result_words.append(word1)
            # 計算一維單詞數量all_len
            all_len.append(len(word1))
            result_sents.append(word2)
        f.close()
    filedata = FileData(result_words, all_len, result_sents, len_sent, len_word)
    return filedata



# =========================LoadFileSingle=======================
# LoadFileSingle讀取所需檔名，回傳欄位的值。
# filename:檔名
# field:要擷取之欄位

# return: 回傳欄位的值(list)
# ===============================================================
def LoadFileSingle(filename, field):
    label = [] #欄位的值
    with open(filename, 'rb') as f:
        for line in f:
            review = json.loads(line)
            label.append((review[field]))
    return label


# ============================CutSents=======================
# 分割文本，計算90%長度，最終獲得"商品評分檔"、"文字評論檔"
# 商品評分檔(reviewerID/asin/overall)
# 文字評論檔(reviewText/overall)
# filename:資料集名稱
# model:詞表
# words:一維list
# sents:二維list
# len_all:ㄧ維最長長度
# len_sent:二維句子最長數量
# len_word:二維單詞最長數量

# return:none
# ===============================================================
def CutSents(filename, model, words, alllens, sents, len_sents, len_words):
    
    data_x = [] #data clean後的reviewText，已轉成詞表index
    data_y = [] #data clean後的overall
    vocab_word = [] #儲存詞表list
    prouduct_rating_data = [] #data clean後商品評分檔

    # 商品評分檔dict
    dict_reviewerID = {}
    dict_asin = {}
    # 讀取詞表
    for word in model.wv.index2word:
        vocab_word.append(word)
    
    # ===================讀取商品評分檔所需欄位=====================
    # 讀取商品評分檔所需欄位
    # overall_list 評分
    # reviewerID_list 消費者id
    # asin_list 商品id
    # ============================================================ 
    overall_list = (LoadFileSingle('./dataset/'+filename+'.json', 'overall'))
    reviewerID_list = (LoadFileSingle('./dataset/'+filename+'.json', 'reviewerID'))
    asin_list = (LoadFileSingle('./dataset/'+filename+'.json', 'asin'))
    
    # 排序長度，找出最大長度
    sorted_all = sorted(alllens)
    sorted_sents = sorted(len_sents)
    sorted_words = sorted(len_words)

    # 獲得涵蓋資料集100%的長度
    maxlen_all = sorted_all[-1]
    maxlen_sentences = sorted_sents[-1]
    maxlen_words = sorted_words[-1]

    # 獲得涵蓋資料集90%的長度
    maxlen_all_90 = sorted_all[int(len(sorted_all) * 0.9)]
    maxlen_sentences_90 = sorted_sents[int(len(sorted_sents) * 0.9)]
    maxlen_words_90 = sorted_words[int(len(sorted_words) * 0.9)]


    # ============================商品評分檔=====================
    # 輸出商評分檔prouduct_rating_data
    # words:一維list
    # ===========================================================
    for l, sent in enumerate(words):
        # overall_list[l]-1    
        # 為了計算方便，將評分1~5都減1，變成0~4對應到index
        data_y.append([int(overall_list[l])-1])
        prouduct_rating_data.append([reviewerID_list[l],
                                    asin_list[l], int(overall_list[l])-1])


    # ============================文本評論檔=====================
    # 輸出文本評論檔review_comment_data
    # sents:二維list
    # ===========================================================
    for i, sent in enumerate(sents):
        doc = np.zeros((maxlen_sentences_90, maxlen_words_90), dtype=np.int32)
        # 將所有的文件都轉每篇都有maxlen_sentences個句子，每個句子有maxlen_words個單詞，
        # 不够的補零，多餘的删除，最後輸出檔案
        # maxlen_sentences_90 90%句子數量
        # maxlen_words_90 90%單詞數量
        for k, s in enumerate(sent):
            if k < maxlen_sentences_90:
                for j, word in enumerate(s):
                    if j < maxlen_words_90:
                        if word in vocab_word:
                            # 對應到詞表的index
                            doc[k][j] = vocab_word.index(word)

        data_x.append(doc)

    # ============================商品評分檔=====================   
    # 為了計算方便，將商品評分檔prouduct_rating_data
    # 消費者id、商品id轉成數字表示，更新prouduct_rating_data
    # ===========================================================
    for i, t in enumerate(prouduct_rating_data):
        if len(dict_asin) == 0:
            dict_asin[t[1]] = i
        if t[1] not in dict_asin:
            dict_asin[t[1]] = len(dict_asin)
        tmp = prouduct_rating_data[i][1]
        prouduct_rating_data[i][1] = dict_asin.get(tmp)

        if len(dict_reviewerID) == 0:
            dict_reviewerID[t[0]] = i
        if t[0] not in dict_reviewerID:
            dict_reviewerID[t[0]] = len(dict_reviewerID)
        tmp = prouduct_rating_data[i][0]
        prouduct_rating_data[i][0] = dict_reviewerID.get(tmp)  

    # ============================輸出檔案=====================
    # 儲存"商品評分檔"、"文字評論檔"
    # ===========================================================
    pickle.dump((data_x, data_y),
                open('./data/review_comment_data_'+filename+'.pickle', 'wb'))   
    pickle.dump((prouduct_rating_data),
                open('./data/prouduct_rating_data_'+filename+'.pickle', 'wb'))


if __name__ == '__main__':


    # LoadFile:讀取分別資料集
    # return:FileData(words, alllens, sents, len_sents, len_words)
    # words:一維list單詞全部串連在一起list(sequential)
    # sents:二維list，一個文件包多個句子，一個句子包多個單詞
    # alllens:ㄧ維最長長度
    # len_sents:二維句子最長數量
    # len_words:二維單詞最長數量

    print("LoadFile:Musical Instruments.json")
    filedata_1 = ((LoadFile('./dataset/Musical_Instruments.json', 'reviewText')))

    print("LoadFile:Patio_Lawn_and_Garden.json")
    filedata_2 = ((LoadFile('./dataset/Patio_Lawn_and_Garden.json', 'reviewText')))

    print("LoadFile:Automotive")
    filedata_3 = ((LoadFile('./dataset/Automotive.json', 'reviewText')))

    print("LoadFile:Amazon_Instant_Video.json")
    filedata_4 = ((LoadFile('./dataset/Amazon_Instant_Video.json', 'reviewText')))

    print("LoadFile:Office_Products.json")
    filedata_5 = ((LoadFile('./dataset/Office_Products.json', 'reviewText')))

    print("LoadFile:Digital_Music.json")
    filedata_6 = ((LoadFile('./dataset/Digital_Music.json', 'reviewText')))

    print("==========================================")

    # Word2VecFun:將各個資料集二維list輸入，自行訓練詞庫，輸出詞表
    print("Word2Vec")
    Word2VecFun(filedata_1.words+filedata_2.words+filedata_3.words+filedata_4.words+filedata_5.words+filedata_6.words)

    # 讀取詞表
    model = word2vec.Word2Vec.load('./voc/word2vec.pickle')

    print("==========================================")
    print("計算各資料集90%長度，分割文本，獲得\"商品評分檔\"、\"文字評論檔\"")
    
    # CutSents:計算各資料集90%長度，分割文本，獲得"商品評分檔"、"文字評論檔"
    CutSents("Musical_Instruments", model, filedata_1.words, filedata_1.alllens, filedata_1.sents, filedata_1.len_sents, filedata_1.len_words)
    CutSents("Patio_Lawn_and_Garden", model, filedata_2.words, filedata_2.alllens, filedata_2.sents, filedata_2.len_sents, filedata_2.len_words)
    CutSents("Automotive", model, filedata_3.words, filedata_3.alllens, filedata_3.sents, filedata_3.len_sents, filedata_3.len_words)
    CutSents("Amazon_Instant_Video", model, filedata_4.words, filedata_4.alllens, filedata_4.sents, filedata_4.len_sents, filedata_4.len_words)
    CutSents("Office_Products", model, filedata_5.words, filedata_5.alllens, filedata_5.sents, filedata_5.len_sents, filedata_5.len_words)
    CutSents("Digital_Music", model, filedata_6.words, filedata_6.alllens, filedata_6.sents, filedata_6.len_sents, filedata_6.len_words)
   
    print("Save success")   
    print("==========================================")

