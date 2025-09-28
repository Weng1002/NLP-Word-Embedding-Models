#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install numpy==1.24.3')
get_ipython().system('pip install pandas==2.0.3')
get_ipython().system('pip install scikit-learn==1.3.0')
get_ipython().system('pip install matplotlib==3.7.2')
get_ipython().system('pip install gensim==4.3.1')
get_ipython().system('pip install tqdm==4.65.0')


# In[1]:


import pandas as pd
import numpy as np
import gensim.downloader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE   


# In[ ]:


get_ipython().system('wget http://download.tensorflow.org/data/questions-words.txt')


# # Part I: Data Pre-processing

# In[2]:


# Preprocess the dataset
file_name = "questions-words"
with open(f"{file_name}.txt", "r") as f:
    data = f.read().splitlines()


# In[3]:


# check data from the first 10 entries
for entry in data[:10]:
    print(entry)


# 語義類比類別 (5)：
# 
# - capital-common-countries: 常見國家首都關係
# 
# - capital-world: 世界各國首都關係
# 
# - city-in-state: 城市與州的關係
# 
# - currency: 貨幣關係
# 
# - family: 家庭關係
# 
# 句法類比類別 (9)：
# 
# - gram1-adjective-to-adverb: 形容詞轉副詞
# - gram2-opposite: 反義詞
# 
# - gram3-comparative: 比較級
# 
# - gram4-superlative: 最高級
# 
# - gram5-present-participle: 現在分詞
# 
# - gram6-nationality-adjective  國籍形容詞 
# 
# - gram7-past-tense  過去時態         
# 
# - gram8-plural  複數              
# 
# - gram9-plural-verbs   複數動詞

# ## TODO 1
# 
# 1. 讀取檔案行，分辨哪些是章節標頭，哪些是題目資料。
# 
# 2. 記錄章節資訊：
# 
#     檔案中前 5 個 ": " 開頭的章節 → 標記為 semantic
# 
#     之後的 9 個章節 → 標記為 syntactic
# 
# 3. 把 analogy 四元組轉成結構化格式：
# 
#     欄位設計：
# 
#         section（章節名稱，例如 capital-common-countries）
# 
#         type（semantic / syntactic）
# 
#         a, b, c, d（四個詞，對應 a:b :: c:d）
# 
# 4. 最後存成 DataFrame，方便後續做統計或送進 word2vec 測試。

# In[4]:


# TODO1: 處理資料為pd.DataFrame的程式碼
# 請注意前五個": "表示語義類別(semantic)，
# 其餘九個屬於句法類別(syntactic)

questions = []          
categories = []        
sub_categories = []    

current_category = None
category_count = 0
category_type = None

for line in data:
    if line.startswith(": "):
        current_category = line[2:]  # 移除 ": " 前綴
        category_count += 1
        
        # 根據順序確定是語義還是句法類別
        if category_count <= 5:
            category_type = "semantic"
        else:
            category_type = "syntactic"
    else:
        # 這是類比行
        if current_category and line.strip():  # 確保有類別且行不為空
            words = line.split()
            if len(words) == 4:  #
                # 將四個詞組合成一個問題字串
                question = f"{words[0]} {words[1]} {words[2]} {words[3]}"
                
                questions.append(question)
                categories.append(category_type)
                sub_categories.append(current_category)

# 創建DataFrame
df = pd.DataFrame(
    {
        "Question": questions,
        "Category": categories,
        "SubCategory": sub_categories,
    }
)

# 顯示資料集的基本資訊
print(f"總類比數量: {len(df)}")
print(f"語義類比: {len(df[df['Category'] == 'semantic'])}")
print(f"句法類比: {len(df[df['Category'] == 'syntactic'])}")
print(f"子類別數: {df['SubCategory'].nunique()}")
print("\n前幾行:")
print(df.head())
print("\n類別分佈:")
print(df.groupby(['Category', 'SubCategory']).size())

# 顯示DataFrame結構
print(f"\nDataFrame形狀: {df.shape}")
print(f"欄位名稱: {list(df.columns)}")

df.to_csv(f"{file_name}.csv", index=False)
print(f"\n資料已儲存至 {file_name}.csv")


# # Part II: Use pre-trained word embeddings

# In[5]:


data = pd.read_csv("questions-words.csv")

MODEL_NAME = "glove-wiki-gigaword-300"
# You can try other models.
# https://radimrehurek.com/gensim/models/word2vec.html#pretrained-models

# 載入預訓練模型（這裡使用GloVe向量）
model = gensim.downloader.load(MODEL_NAME)
print("The Gensim model loaded successfully!")


# ## TODO 2
# 
# 1. 類比解析：將問題字串分割成四個詞 word_a word_b word_c word_d
# 
# 2. 向量運算：
# 
# - 使用公式：word_b - word_a + word_c ≈ word_d
# - 在gensim中：positive=[word_b, word_c], negative=[word_a]
# 
# 3. 預測策略：
# 
# - 取前10個最相似的詞
# - 排除輸入的三個詞（避免trivial答案）
# - 選擇第一個有效的候選詞
# 
# 4. 錯誤處理：
# 
# - 處理詞彙表外的詞（OOV）
# - 處理其他可能的異常

# In[ ]:


# Do predictions and preserve the gold answers (word_D)
# 進行預測並保存正確答案 (word_D)
preds = []
golds = []

for analogy in tqdm(data["Question"]):
    # TODO2: Write your code here to use pre-trained word embeddings for getting predictions of the analogy task.
    # You should also preserve the gold answers during iterations for evaluations later.
    # TODO2: 使用預訓練詞嵌入進行類比任務預測的程式碼
    # 您也應該在迭代過程中保存正確答案以便後續評估
    # 解析類比問題 (例如: "man woman king queen")
    """ Hints
    # Unpack the analogy (e.g., "man", "woman", "king", "queen")
    # Perform vector arithmetic: word_b + word_c - word_a should be close to word_d
    # Source: https://github.com/piskvorky/gensim/blob/develop/gensim/models/keyedvectors.py#L776
    # Mikolov et al., 2013: big - biggest and small - smallest
    # Mikolov et al., 2013: X = vector(”biggest”) − vector(”big”) + vector(”small”).
    """
      
    words = analogy.split()
    word_a, word_b, word_c, word_d = words
    
    # GloVe模型的詞彙表都是小寫的!!!!
    word_a_lower = word_a.lower()
    word_b_lower = word_b.lower()
    word_c_lower = word_c.lower()
    word_d_lower = word_d.lower()
    
    # 保存正確答案
    golds.append(word_d_lower)

    if all(word in model.key_to_index for word in [word_a_lower, word_b_lower, word_c_lower]):
        # 執行向量運算: word_b - word_a + word_c ≈ word_d
        # 例如: queen = king - man + woman
        result = model.most_similar(
                positive=[word_b_lower, word_c_lower],  # woman, king
                negative=[word_a_lower],                # man
                topn=10
        )
        
        # 找到第一個不是輸入詞的預測結果
        prediction = None
        for candidate, similarity in result:
            if candidate not in [word_a_lower, word_b_lower, word_c_lower]:
                prediction = candidate
                break
        
        # 如果找到預測結果，則使用它；否則使用最相似的詞
        if prediction is None:
            prediction = result[0][0]
            
        preds.append(prediction)
        
    else:
        # 如果有詞不在詞彙表中，預測為空字串或特殊標記
        preds.append("<UNK>")

print(f"完成！處理了 {len(preds)} 個類比問題")
print(f"預測範例: {preds[:5]}")
print(f"正確答案範例: {golds[:5]}")


# In[ ]:


# Perform evaluations. You do not need to modify this block!!

def calculate_accuracy(gold: np.ndarray, pred: np.ndarray) -> float:
    return np.mean(gold == pred)

golds_np, preds_np = np.array(golds), np.array(preds)
data = pd.read_csv("questions-words.csv")

# Evaluation: categories
for category in data["Category"].unique():
    mask = data["Category"] == category
    golds_cat, preds_cat = golds_np[mask], preds_np[mask]
    acc_cat = calculate_accuracy(golds_cat, preds_cat)
    print(f"Category: {category}, Accuracy: {acc_cat * 100}%")

# Evaluation: sub-categories
for sub_category in data["SubCategory"].unique():
    mask = data["SubCategory"] == sub_category
    golds_subcat, preds_subcat = golds_np[mask], preds_np[mask]
    acc_subcat = calculate_accuracy(golds_subcat, preds_subcat)
    print(f"Sub-Category{sub_category}, Accuracy: {acc_subcat * 100}%")


# ## TODO 3

# In[ ]:


# Collect words from Google Analogy dataset
SUB_CATEGORY = "family"  

# TODO3: Plot t-SNE for the words in the SUB_CATEGORY `: family`

# 收集family子類別中的所有詞彙
family_data = data[data["SubCategory"] == SUB_CATEGORY]
print(f"Family子類別共有 {len(family_data)} 個類比問題")

# 從所有family類比中提取唯一的詞彙
all_family_words = set()
for question in family_data["Question"]:
    words = question.split()
    for word in words:
        all_family_words.add(word.lower())  

print(f"Family類別包含 {len(all_family_words)} 個唯一詞彙")
print(f"詞彙範例: {list(all_family_words)[:10]}")

valid_words = []
word_vectors = []

for word in all_family_words:
    if word in model.key_to_index:
        valid_words.append(word)
        word_vectors.append(model[word])

print(f"模型中找到 {len(valid_words)} 個詞彙")
word_vectors = np.array(word_vectors)

# 使用t-SNE降維
if len(valid_words) > 1:
    print("正在執行t-SNE降維...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(valid_words)-1))
    embeddings_2d = tsne.fit_transform(word_vectors)
    
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                alpha=0.7, s=100, c='steelblue')
    
    # 標註每個詞彙
    for i, word in enumerate(valid_words):
        plt.annotate(word, 
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, ha='left')
    
    plt.title("Word Relationships from Google Analogy Task (Family Category)", 
              fontsize=14, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.text(0.02, 0.98, f"Vocabulary size: {len(valid_words)}\nModel: {MODEL_NAME}", 
             transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== Family類比範例 ===")
    for i, row in family_data.head().iterrows():
        words = row["Question"].split()
        print(f"{words[0]} : {words[1]} = {words[2]} : {words[3]}")
        
else:
    print("詞彙數量不足，無法進行t-SNE分析")


# # Part III: Train your own word embeddings

# In[ ]:


# Download the split Wikipedia files
# Each file contain 562365 lines (articles).
get_ipython().system('gdown --id 1jiu9E1NalT2Y8EIuWNa1xf2Tw1f1XuGd -O wiki_texts_part_0.txt.gz')
get_ipython().system('gdown --id 1ABblLRd9HXdXvaNv8H9fFq984bhnowoG -O wiki_texts_part_1.txt.gz')
get_ipython().system('gdown --id 1z2VFNhpPvCejTP5zyejzKj5YjI_Bn42M -O wiki_texts_part_2.txt.gz')
get_ipython().system('gdown --id 1VKjded9BxADRhIoCzXy_W8uzVOTWIf0g -O wiki_texts_part_3.txt.gz')
get_ipython().system('gdown --id 16mBeG26m9LzHXdPe8UrijUIc6sHxhknz -O wiki_texts_part_4.txt.gz')


# In[ ]:


# Download the split Wikipedia files
# Each file contain 562365 lines (articles), except the last file.
get_ipython().system('gdown --id 17JFvxOH-kc-VmvGkhG7p3iSZSpsWdgJI -O wiki_texts_part_5.txt.gz')
get_ipython().system('gdown --id 19IvB2vOJRGlrYulnTXlZECR8zT5v550P -O wiki_texts_part_6.txt.gz')
get_ipython().system('gdown --id 1sjwO8A2SDOKruv6-8NEq7pEIuQ50ygVV -O wiki_texts_part_7.txt.gz')
get_ipython().system('gdown --id 1s7xKWJmyk98Jbq6Fi1scrHy7fr_ellUX -O wiki_texts_part_8.txt.gz')
get_ipython().system('gdown --id 17eQXcrvY1cfpKelLbP2BhQKrljnFNykr -O wiki_texts_part_9.txt.gz')
get_ipython().system('gdown --id 1J5TAN6bNBiSgTIYiPwzmABvGhAF58h62 -O wiki_texts_part_10.txt.gz')


# In[ ]:


Extract the downloaded wiki_texts_parts files.
get_ipython().system('gunzip -k w# iki_texts_part_*.gz')

# Combine the extracted wiki_texts_parts files.
get_ipython().system('cat wiki_texts_part_*.txt > wiki_texts_combined.txt')

# Check the first ten lines of the combined file
get_ipython().system('head -n 10 wiki_texts_combined.txt')


# In[6]:


# 檢查合併後的檔案
print("\n=== 檢查合併後的檔案 ===")
with open("wiki_texts_combined.txt", "r", encoding="utf-8") as f:
    lines = []
    for i, line in enumerate(f):
        lines.append(line.strip())
        if i >= 9:  # 只讀取前10行
            break
    
print("前10行內容:")
for i, line in enumerate(lines):
    print(f"{i+1}: {line[:100]}...")  # 只顯示前100個字符

# 計算總行數
print("\n=== 計算總文章數 ===")
total_lines = 0
with open("wiki_texts_combined.txt", "r", encoding="utf-8") as f:
    for line in f:
        total_lines += 1

print(f"總文章數: {total_lines:,}")


# ## TODO 4
# 
# 將合併後的文章檔案，進行5%、10%、20%抽樣

# In[ ]:


# Now you need to do sampling because the corpus is too big.
# You can further perform analysis with a greater sampling ratio.

import random
random.seed(4321)

# 多個抽樣比例
SAMPLING_RATIOS = [0.05, 0.10, 0.20]
wiki_txt_path = "wiki_texts_combined.txt"

for ratio in SAMPLING_RATIOS:
    output_path = f"wiki_texts_sampled_{int(ratio*100)}.txt"

    print(f"\n=== 開始抽樣 {ratio*100:.0f}% ===")

    sampled_count = 0
    total_processed = 0

    with open(wiki_txt_path, "r", encoding="utf-8") as f:
        with open(output_path, "w", encoding="utf-8") as output_file:
            for line in tqdm(f, desc=f"抽樣 {ratio*100:.0f}%"):
                total_processed += 1
                if random.random() < ratio:
                    output_file.write(line)
                    sampled_count += 1

    actual_ratio = sampled_count / total_processed if total_processed else 0
    print(f"總處理文章數: {total_processed:,}")
    print(f"抽樣文章數: {sampled_count:,}")
    print(f"實際抽樣比例: {actual_ratio:.4f} ({actual_ratio*100:.2f}%)")
    print(f"檔案已儲存為: {output_path}")

    print(f"檔案 {output_path} 前5行:")
    with open(output_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            print(f"  {i+1}: {line.strip()[:100]}...")
            if i >= 4:
                break


# ## TODO 5

# In[ ]:


pip install nltk


# In[7]:


import nltk

print("下載缺失的NLTK資源...")

# 下載新版本需要的資源
try:
    print("下載 punkt_tab...")
    nltk.download('punkt_tab', quiet=False)
    print("✓ punkt_tab 下載成功")
except Exception as e:
    print(f"✗ punkt_tab 下載失敗: {e}")

# 重新測試分詞功能
try:
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize("Hello world, this is a test.")
    print(f"✓ 分詞測試成功：{tokens}")
except Exception as e:
    print(f"✗ 分詞測試失敗: {e}")
    
    # 如果還是失敗，嘗試下載所有punkt相關資源
    punkt_resources = ['punkt', 'punkt_tab']
    for resource in punkt_resources:
        try:
            print(f"嘗試下載 {resource}...")
            nltk.download(resource, quiet=False)
        except:
            pass

try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("下載NLTK資源...")
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('omw-1.4', quiet=True)

print("資源下載完成！")


# In[8]:


import gensim
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import multiprocessing as mp
import os
import re
import string
from typing import Iterator, List
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import unicodedata


# In[ ]:


# TODO5: Train your own word embeddings with the sampled articles
# https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
# Hint: You should perform some pre-processing before training.

print("=== 開始訓練自己的詞嵌入模型 ===")

# ---------- 讓 BLAS/NumPy 不要把 CPU 線程開爆，避免 Jupyter 當機 ----------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ---------- 初始化工具 ----------
lemmatizer = WordNetLemmatizer()
english_stopwords = set(stopwords.words('english'))

# 添加自定義停用詞
custom_stopwords = {'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'will'}
english_stopwords.update(custom_stopwords)

# 正規表達式預編譯
_RE_SPACES = re.compile(r'\s+')
_RE_NONASCII = re.compile(r'[^\x00-\x7F]+')  # 移除非ASCII字符
_RE_DIGITS = re.compile(r'\d+')
_RE_SINGLE_CHAR = re.compile(r'\b\w\b')  # 單字符詞

def is_english_word(word: str) -> bool:
    return word.isalpha() and word.encode('ascii', 'ignore').decode('ascii') == word

def preprocess_text(line: str, use_lemmatization: bool = True, 
                           remove_stopwords: bool = True) -> List[str]:
    
    # 1. 基本清理
    text = line.lower().strip()
    if not text:
        return []
    
    # 移除非ASCII字符
    text = _RE_NONASCII.sub(' ', text)
    # 移除數字
    text = _RE_DIGITS.sub(' ', text)
    # 移除單字符詞
    text = _RE_SINGLE_CHAR.sub(' ', text)
    # 正規化空白字符
    text = _RE_SPACES.sub(' ', text).strip()
    
    # 2. 改進的分詞（Better tokenization）
    try:
        tokens = word_tokenize(text)  
    except:
        tokens = text.split()  
    
    processed_words = []
    
    for token in tokens:
        # 移除標點符號
        token = re.sub(r'[^\w]', '', token)
        if not token or len(token) < 2:
            continue
            
        # 3. 檢查是否為英文詞彙（Remove non-English words）
        if not is_english_word(token):
            continue
            
        # 4. 移除停用詞（Remove stop words）
        if remove_stopwords and token in english_stopwords:
            continue
            
        # 5. 詞元化（Lemmatization）
        if use_lemmatization:
            try:
                token = lemmatizer.lemmatize(token, pos='v')  # 動詞詞元化
                token = lemmatizer.lemmatize(token, pos='n')  # 名詞詞元化
            except:
                pass  # 如果詞元化失敗，保持原詞
        
        processed_words.append(token)
    
    return processed_words

class LineSentencePreprocessed:

    def __init__(self, path: str, min_len: int = 5, show_progress: bool = True,
                 use_lemmatization: bool = True, remove_stopwords: bool = True):
        self.path = path
        self.min_len = min_len
        self.show_progress = show_progress
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        
        try:
            with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
                self.total_lines = sum(1 for _ in f)
        except Exception:
            self.total_lines = None

    def __iter__(self) -> Iterator[List[str]]:
        with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
            iterator = f
            if self.show_progress:
                iterator = tqdm(f, total=self.total_lines, desc="進階預處理語料", leave=False)
            
            for line in iterator:
                line = line.strip()
                if not line:
                    continue
                    
                words = preprocess_text(
                    line, 
                    use_lemmatization=self.use_lemmatization,
                    remove_stopwords=self.remove_stopwords
                )
                
                if len(words) >= self.min_len:
                    yield words


class TqdmEpochLogger(CallbackAny2Vec):
    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.epoch = 0
        self.prev_cum_loss = 0.0
        self.pbar = tqdm(total=total_epochs, desc="訓練 Epoch", position=0)

    def on_epoch_end(self, model):
        cum_loss = model.get_latest_training_loss()
        delta = cum_loss - self.prev_cum_loss
        self.prev_cum_loss = cum_loss
        self.epoch += 1
        self.pbar.set_postfix(loss_delta=f"{delta:.2f}", cum_loss=f"{cum_loss:.2f}")
        self.pbar.update(1)
        if self.epoch >= self.total_epochs:
            self.pbar.close()


# 訓練設置
corpus_path = "wiki_texts_sampled_20.txt"
epochs = 10

# 比較不同預處理策略
preprocessing_configs = [
    {
        'name': '預處理版本1',
        'use_lemmatization': True,
        'remove_stopwords': True,
        'min_count': 3  # 因為移除了停用詞，可以降低最小詞頻
    }
]

models_results = {}

for config in preprocessing_configs:
    print(f"\n{'='*50}")
    print(f"訓練模型：{config['name']}")
    print(f"{'='*50}")
    
    word2vec_params = {
        "vector_size": 300,
        "window": 5,
        "min_count": config['min_count'],
        "workers": max(1, min(4, mp.cpu_count() // 2)),
        "sg": 0,  # CBOW
        "negative": 5,
        "sample": 1e-3,
        "seed": 4321,
    }
    
    print("訓練參數：")
    for k, v in word2vec_params.items():
        print(f"  {k}: {v}")
    print(f"  epochs: {epochs}")
    print(f"  詞元化: {config['use_lemmatization']}")
    print(f"  移除停用詞: {config['remove_stopwords']}")
    
    # 建立語料迭代器
    sentences = LineSentencePreprocessed(
        corpus_path, 
        min_len=5,
        show_progress=True,
        use_lemmatization=config['use_lemmatization'],
        remove_stopwords=config['remove_stopwords']
    )
    
    # 建立並訓練模型
    model = Word2Vec(**word2vec_params)
    print("\n=== 建立詞彙表 ===")
    model.build_vocab(sentences, progress_per=10000, keep_raw_vocab=False)
    print(f"詞彙表大小：{len(model.wv):,}")
    
    print("\n=== 開始訓練 ===")
    epoch_logger = TqdmEpochLogger(total_epochs=epochs)
    
    # 重新創建語料迭代器用於訓練
    training_sentences = LineSentencePreprocessed(
        corpus_path, 
        min_len=5,
        show_progress=False,
        use_lemmatization=config['use_lemmatization'],
        remove_stopwords=config['remove_stopwords']
    )
    
    model.train(
        corpus_iterable=training_sentences,
        total_examples=model.corpus_count,
        epochs=epochs,
        compute_loss=True,
        callbacks=[epoch_logger]
    )
    
    # 測試模型
    test_words = ["king", "queen", "man", "woman", "computer", "science", "rock", "stone"]
    in_vocab = [w for w in test_words if w in model.wv]
    
    print(f"\n=== 模型測試：{config['name']} ===")
    print(f"測試詞彙（存在於詞彙表）：{in_vocab}")
    
    if in_vocab:
        for w in in_vocab[:3]:
            print(f"\n與 '{w}' 最相似的詞：")
            try:
                for sw, sim in model.wv.most_similar(w, topn=5):
                    print(f"  {sw:20s} {sim:.3f}")
            except KeyError:
                print(f"  詞彙 '{w}' 不在模型中")
    
    # 保存模型
    os.makedirs("models", exist_ok=True)
    model_name = config['name'].replace('（', '_').replace('）', '').replace(' ', '_').replace('、', '_')
    model_save_path = f"./models/word2vec_{model_name}.model"
    vectors_save_path = f"./models/word_vectors_{model_name}.kv"
    
    model.save(model_save_path)
    model.wv.save(vectors_save_path)
    
    print(f"\n模型已保存：{model_save_path}")


# # Part IV

# ## TODO 6

# In[ ]:


data = pd.read_csv("questions-words.csv")

# Do predictions and preserve the gold answers (word_D)
print("=== 載入自訓練的詞嵌入模型 ===")

try:
    from gensim.models import Word2Vec
    my_model = Word2Vec.load("./models/word2vec_改進預處理_含詞元化和停用詞移除.model")
    print("✓ 成功載入完整模型")
except:
    try:
        from gensim.models import KeyedVectors
        my_model = KeyedVectors.load("./models/word_vectors_改進預處理_含詞元化和停用詞移除.kv")
        print("✓ 成功載入詞向量")
    except Exception as e:
        print(f"✗ 載入模型失敗: {e}")
        print("請確認模型檔案路徑正確")
        raise

print(f"模型詞彙表大小: {len(my_model.wv if hasattr(my_model, 'wv') else my_model):,}")

word_vectors = my_model.wv if hasattr(my_model, 'wv') else my_model

preds = []
golds = []
stats = {
    'total': 0,
    'oov_words': 0,  # 詞彙表外的詞
    'valid_predictions': 0,
    'oov_analogies': []  # 記錄有OOV詞的類比
}

print("\n=== 開始使用自訓練模型進行類比預測 ===")


for analogy in tqdm(data["Question"], desc="處理類比"):
      # TODO6: Write your code here to use your trained word embeddings for getting predictions of the analogy task.
      # You should also preserve the gold answers during iterations for evaluations later.
      """ Hints
      # Unpack the analogy (e.g., "man", "woman", "king", "queen")
      # Perform vector arithmetic: word_b + word_c - word_a should be close to word_d
      # Source: https://github.com/piskvorky/gensim/blob/develop/gensim/models/keyedvectors.py#L776
      # Mikolov et al., 2013: big - biggest and small - smallest
      # Mikolov et al., 2013: X = vector(”biggest”) − vector(”big”) + vector(”small”).
      """
      
      words = analogy.split()
      word_a, word_b, word_c, word_d = words
      
      # 轉換為小寫
      word_a_lower = word_a.lower()
      word_b_lower = word_b.lower()
      word_c_lower = word_c.lower()
      word_d_lower = word_d.lower()
      
      # 保存正確答案
      golds.append(word_d_lower)
      stats['total'] += 1
      
      # 檢查所有詞是否都在模型的詞彙表中
      if all(word in word_vectors.key_to_index for word in [word_a_lower, word_b_lower, word_c_lower]):
            
            # 執行向量運算: word_b - word_a + word_c ≈ word_d
            result = word_vectors.most_similar(
            positive=[word_b_lower, word_c_lower], 
            negative=[word_a_lower],               
            topn=10                                
            )
            
            # 找到第一個不是輸入詞的預測結果
            prediction = None
            for candidate, similarity in result:
                if candidate not in [word_a_lower, word_b_lower, word_c_lower]:
                    prediction = candidate
                    break
            
            # 如果找到預測結果，使用它；否則使用最相似的詞
            if prediction is None:
                prediction = result[0][0]
            
            preds.append(prediction)
            stats['valid_predictions'] += 1
            
      else:
            # 檢查哪些詞不在詞彙表中
            missing_words = [word for word in [word_a_lower, word_b_lower, word_c_lower] 
                        if word not in word_vectors.key_to_index]
            stats['oov_words'] += len(missing_words)
            stats['oov_analogies'].append({
            'analogy': analogy,
            'missing_words': missing_words
            })
            
            # 對於詞彙表外的詞，預測為未知
            preds.append("<OOV>")
            

      print(f"\n=== 預測完成統計 ===")
      print(f"總類比數量: {stats['total']:,}")
      print(f"有效預測: {stats['valid_predictions']:,} ({stats['valid_predictions']/stats['total']*100:.1f}%)")
      print(f"詞彙表外詞數: {stats['oov_words']:,}")
      print(f"包含OOV的類比: {len(stats['oov_analogies']):,}")

      print(f"\n預測範例: {preds[:10]}")
      print(f"正確答案範例: {golds[:10]}")

      # 分析詞彙覆蓋情況
      if stats['oov_analogies']:
        print(f"\n=== 詞彙表外詞分析（前10個） ===")
        for i, oov_info in enumerate(stats['oov_analogies'][:10]):
                print(f"{i+1}. {oov_info['analogy']} - 缺失: {oov_info['missing_words']}")


# ## TODO 7

# In[ ]:


# Collect words from Google Analogy dataset
SUB_CATEGORY = "family" 

# TODO7: Plot t-SNE for the words in the SUB_CATEGORY `family`

print(f"=== 使用自訓練模型視覺化 {SUB_CATEGORY} 類別詞彙關係 ===")

try:
    # 使用之前載入的自訓練模型
    word_vectors = my_model.wv if hasattr(my_model, 'wv') else my_model
    print(f"✓ 使用自訓練模型，詞彙表大小: {len(word_vectors):,}")
except:
    print("✗ 自訓練模型不可用，請先載入模型")
    raise

# 提取family子類別的數據
family_data = data[data["SubCategory"] == SUB_CATEGORY]
print(f"Family子類別共有 {len(family_data)} 個類比問題")


if len(family_data) == 0:
    print("⚠️ 未找到family類別資料，檢查可用的子類別:")
    print(data["SubCategory"].unique())
else:
    all_family_words = set()
    for question in family_data["Question"]:
        words = question.split()
        for word in words:
            all_family_words.add(word.lower()) 

    print(f"Family類別包含 {len(all_family_words)} 個唯一詞彙")
    print(f"詞彙範例: {list(all_family_words)[:10]}")

    # 檢查哪些詞在自訓練模型中
    valid_words = []
    word_vectors_list = []
    missing_words = []

    for word in all_family_words:
        if word in word_vectors.key_to_index:
            valid_words.append(word)
            word_vectors_list.append(word_vectors[word])
        else:
            missing_words.append(word)

    print(f"自訓練模型中找到 {len(valid_words)} 個詞彙")
    if missing_words:
        print(f"模型中缺失的詞彙: {missing_words}")

    # 進行t-SNE降維和視覺化
    if len(valid_words) > 1:
        print("正在執行t-SNE降維...")
        
        # 準備數據
        word_vectors_array = np.array(word_vectors_list)
        
        # 設置t-SNE參數
        perplexity = min(30, len(valid_words) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
                   n_iter=1000, learning_rate=200)
        
        # 執行t-SNE
        embeddings_2d = tsne.fit_transform(word_vectors_array)
        
        # 創建視覺化
        plt.figure(figsize=(14, 10))
        
        # 繪製散點圖
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            alpha=0.7, s=120, c='steelblue', edgecolors='darkblue')
        
        # 標註每個詞彙
        for i, word in enumerate(valid_words):
            plt.annotate(word, 
                        (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=11, ha='left', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.title("Word Relationships from Google Analogy Task\n(Family Category - Self-trained Model)", 
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("t-SNE Dimension 1", fontsize=12)
        plt.ylabel("t-SNE Dimension 2", fontsize=12)
        
        plt.grid(True, alpha=0.3)
        info_text = f"Vocabulary Size: {len(valid_words)}\nSelf-trained Model\nVector Dimension: {word_vectors.vector_size}"
        plt.text(0.02, 0.98, info_text, 
                transform=plt.gca().transAxes, fontsize=10, 
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        plt.show()
        
        print(f"\n=== Family類比範例 ===")
        for i, (_, row) in enumerate(family_data.head().iterrows()):
            words = row["Question"].split()
            print(f"{i+1}. {words[0]} : {words[1]} = {words[2]} : {words[3]}")
            
        # 分析詞彙間的相似性
        print(f"\n=== 詞彙相似性分析 ===")
        if len(valid_words) >= 4:
            sample_words = valid_words[:4]
            print("部分詞彙間的餘弦相似度:")
            for i, word1 in enumerate(sample_words):
                for word2 in sample_words[i+1:]:
                    try:
                        similarity = word_vectors.similarity(word1, word2)
                        print(f"  {word1} ↔ {word2}: {similarity:.3f}")
                    except:
                        print(f"  {word1} ↔ {word2}: 無法計算")
        
    else:
        print("⚠️ 詞彙數量不足，無法進行t-SNE分析")

print(f"\n=== 分析完成 ===")


# ## 比較不同的語料庫訓練

# ### 先下載不同的語料集

# In[ ]:


import requests
import nltk
from nltk.corpus import brown, reuters

def download_public_corpora():
    print("正在下載公開語料庫...")
    
    # 下載NLTK語料庫
    nltk.download('brown', quiet=True)
    nltk.download('reuters', quiet=True)
    
    # 創建新聞語料庫（使用Reuters）
    from nltk.corpus import reuters
    with open('news_corpus.txt', 'w', encoding='utf-8') as f:
        for fileid in reuters.fileids()[:1000]:  # 取前1000篇文章
            text = reuters.raw(fileid).lower()
            # 簡單清理
            lines = text.split('\n')
            for line in lines:
                if len(line.split()) > 5:  # 只保留足夠長的句子
                    f.write(line.strip() + '\n')
    
    print("✓ news_corpus.txt 創建完成")
    
    # 創建文學語料庫（使用Brown的文學部分）
    from nltk.corpus import brown
    with open('literature_corpus.txt', 'w', encoding='utf-8') as f:
        # Brown語料庫的文學類別
        literature_categories = ['romance', 'mystery', 'science_fiction', 'adventure']
        for category in literature_categories:
            if category in brown.categories():
                for sent in brown.sents(categories=category):
                    sentence = ' '.join(sent).lower()
                    if len(sentence.split()) > 5:
                        f.write(sentence + '\n')
    
    print("✓ literature_corpus.txt 創建完成")

    
    return True

success = download_public_corpora()


# In[18]:


from typing import Dict
import time

# 預處理相關函數
_RE_SPACES = re.compile(r'\s+')
_RE_NONWS = re.compile(r'[^\w\s]')
_RE_DIGITS = re.compile(r'\d+')

def preprocess_text(line: str):
    t = line.lower()
    t = _RE_SPACES.sub(' ', t)
    t = _RE_NONWS.sub(' ', t)
    t = _RE_DIGITS.sub('', t)  
    words = [w for w in t.split() if len(w) >= 2]
    return words

class FastLineSentencePreprocessed:
    def __init__(self, path: str, min_len: int = 5, max_lines: int = None):
        self.path = path
        self.min_len = min_len
        self.max_lines = max_lines
        self.sentences = self._load_sentences()
    
    def _load_sentences(self):
        print(f"載入語料庫: {self.path}")
        sentences = []      
        
        with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
            iterator = f
            if total_lines:
                iterator = tqdm(f, total=total_lines, desc="預處理語料", leave=True)
                
            for i, line in enumerate(iterator):
                if self.max_lines and i >= self.max_lines:
                    print(f"達到最大行數限制: {self.max_lines:,}")
                    break
                    
                line = line.strip()
                if line:
                    words = preprocess_text(line)
                    if len(words) >= self.min_len:
                        sentences.append(words)
        
        print(f"載入完成: {len(sentences):,} 句")
        return sentences
    
    def __iter__(self):
        return iter(self.sentences)
    
    def __len__(self):
        return len(self.sentences)

class MultiCorpusEvaluator:
    def __init__(self, analogy_data_path: str, max_corpus_size: int = 80000, eval_sample_size: int = 3000):
        self.analogy_data = pd.read_csv(analogy_data_path)
        self.models = {}
        self.results = {}
        self.max_corpus_size = max_corpus_size  
        self.eval_sample_size = eval_sample_size 
        print(f"設定加速參數:")
        print(f"   最大語料庫大小: {max_corpus_size:,} 句")
        print(f"   評估樣本大小: {eval_sample_size:,} 問題")
        
    def prepare_corpus_configs(self):
        corpus_configs = {
            'wikipedia_20': {
                'path': 'wiki_texts_sampled_20.txt',
                'description': '維基百科20%抽樣',
                'domain': '百科全書',
                'style': '正式學術'
            },
            'news_corpus': {
                'path': 'news_corpus.txt',
                'description': '新聞語料庫',
                'domain': '新聞時事',
                'style': '新聞體'
            },
            'literature_corpus': {
                'path': 'literature_corpus.txt', 
                'description': '文學作品語料庫',
                'domain': '文學創作',
                'style': '文學性'
            }
        }
        return corpus_configs
    
    def train_models_on_different_corpora(self):
        corpus_configs = self.prepare_corpus_configs()

        word2vec_params = {
            "vector_size": 300,
            "window": 5,
            "min_count": 5,
            "workers": 4,
            "sg": 0,  
            "epochs": 5, 
            "negative": 5,
            "sample": 1e-3,
            "seed": 4321,
        }
        
        print("=== 開始在不同語料庫上訓練模型 ===")
        print(f"加速設定: epochs={word2vec_params['epochs']}, 最大語料{self.max_corpus_size:,}句")
        
        total_corpora = len([c for c in corpus_configs.values() if os.path.exists(c['path'])])
        current_corpus = 0
        
        for corpus_name, config in corpus_configs.items():
            if not os.path.exists(config['path']):
                print(f"跳過 {corpus_name}：檔案 {config['path']} 不存在")
                continue
            
            current_corpus += 1
            print(f"\n[{current_corpus}/{total_corpora}] 訓練模型：{config['description']}")
            print(f"{'='*60}")
            
            start_time = time.time()

            sentences = FastLineSentencePreprocessed(
                config['path'], 
                min_len=5,
                max_lines=self.max_corpus_size  
            )
            
            # 建立模型
            print("建立Word2Vec模型...")
            model = Word2Vec(
                sentences=sentences,  
                **word2vec_params
            )
            
            training_time = time.time() - start_time
            
            self.models[corpus_name] = {
                'model': model.wv,
                'config': config,
                'training_time': training_time,
                'corpus_size': len(sentences)
            }
            
            print(f"{config['description']} 訓練完成!")
            print(f"   訓練時間: {training_time:.1f} 秒")
            print(f"   語料大小: {len(sentences):,} 句")
            

    def evaluate_model_on_analogies(self, model_name: str, model_wv) -> Dict:        
        print(f"\n評估模型：{model_name}")
        print(f"使用抽樣評估加速 (樣本大小: {self.eval_sample_size:,})")
        
        total_data = len(self.analogy_data)
        if total_data > self.eval_sample_size:
            eval_data = self.analogy_data.sample(n=self.eval_sample_size, random_state=42)
            print(f"抽樣 {self.eval_sample_size:,}/{total_data:,} 個問題進行評估")
        else:
            eval_data = self.analogy_data
            print(f"評估全部 {total_data:,} 個問題")
            
        results = {
            'total': len(eval_data),
            'correct': 0,
            'oov_count': 0,
            'category_results': {},
            'subcategory_results': {}
        }
        
        start_time = time.time()
        
        categories = eval_data['Category'].unique()
        
        for category in tqdm(categories, desc="評估類別"):
            category_data = eval_data[eval_data['Category'] == category]
            category_correct = 0
            category_valid = 0
            
            for _, row in tqdm(category_data.iterrows(), 
                             total=len(category_data), 
                             desc=f"  {category}",
                             leave=False):
                
                words = row['Question'].split()
                word_a, word_b, word_c, word_d = [w.lower() for w in words]
                
                if all(w in model_wv.key_to_index for w in [word_a, word_b, word_c]):
                    similar_words = model_wv.most_similar(
                        positive=[word_b, word_c],
                        negative=[word_a],
                        topn=5  
                    )
                    
                    prediction = None
                    for candidate, _ in similar_words:
                        if candidate not in [word_a, word_b, word_c]:
                            prediction = candidate
                            break
                    
                    if prediction == word_d.lower():
                        results['correct'] += 1
                        category_correct += 1
                    
                    category_valid += 1
                else:
                    results['oov_count'] += 1
            
            if category_valid > 0:
                results['category_results'][category] = {
                    'accuracy': category_correct / category_valid,
                    'correct': category_correct,
                    'total': category_valid
                }
        
        eval_time = time.time() - start_time
        accuracy = results['correct'] / results['total'] if results['total'] > 0 else 0
        coverage = (results['total'] - results['oov_count']) / results['total']
        
        print(f"評估完成!")
        print(f"   評估時間: {eval_time:.1f} 秒") 
        print(f"   準確率: {accuracy:.4f}")
        print(f"   覆蓋率: {coverage:.4f}")
        print(f"   正確: {results['correct']:,}")
        print(f"   OOV: {results['oov_count']:,}")
        
        return results
    
    def run_comprehensive_evaluation(self):
        total_start_time = time.time()
        print(f"開始綜合評估")
        print(f"最大語料庫大小: {self.max_corpus_size:,} 句")
        print(f"評估樣本大小: {self.eval_sample_size:,} 問題")
        
        print(f"\n{'='*60}")
        print("階段 1: 訓練模型")
        print(f"{'='*60}")
        self.train_models_on_different_corpora()
        
        # 評估所有模型
        print(f"\n{'='*60}")
        print("階段 2: 評估模型")
        print(f"{'='*60}")
        
        total_models = len(self.models)
        current_model = 0
        
        for model_name, model_info in self.models.items():
            current_model += 1
            print(f"\n[{current_model}/{total_models}] 評估模型: {model_name}")
            
            results = self.evaluate_model_on_analogies(
                model_name, 
                model_info['model']
            )
            self.results[model_name] = results
        
        print(f"\n{'='*60}")
        print("階段 3: 生成報告")
        print(f"{'='*60}")
        self.generate_comparison_report()
        
        total_time = time.time() - total_start_time
        print(f"\n綜合評估完成!")
        print(f"總耗時: {total_time:.1f} 秒 ({total_time/60:.1f} 分鐘)")
    
    def generate_comparison_report(self):
        print("\n" + "="*80)
        print("多語料庫比較報告")
        print("="*80)
        
        if not self.results:
            print("沒有評估結果可以比較")
            return
        
        print("\n整體性能比較：")
        print(f"{'模型名稱':<20} {'準確率':<10} {'詞彙表大小':<12} {'覆蓋率':<10} {'訓練時間':<10}")
        print("-" * 75)
        
        model_performances = []
        
        for model_name, results in self.results.items():
            accuracy = results['correct'] / results['total'] if results['total'] > 0 else 0
            coverage = (results['total'] - results['oov_count']) / results['total'] if results['total'] > 0 else 0
            training_time = self.models[model_name].get('training_time', 0)
            
            model_performances.append({
                'name': model_name,
                'accuracy': accuracy,
                'coverage': coverage,
                'training_time': training_time
            })

            print(f"{model_name:<20} {accuracy:<10.4f} {coverage:<10.4f} {training_time:<10.1f}s")

        # 按類別比較
        print(f"\n按主類別性能比較：")
        categories = set()
        for results in self.results.values():
            categories.update(results['category_results'].keys())
        
        for category in sorted(categories):
            print(f"\n{category} 類別：")
            print(f"{'模型':<20} {'準確率':<10} {'正確數/總數':<15}")
            print("-" * 50)
            
            category_performances = []
            for model_name, results in self.results.items():
                if category in results['category_results']:
                    cat_result = results['category_results'][category]
                    category_performances.append({
                        'model': model_name,
                        'accuracy': cat_result['accuracy'],
                        'correct': cat_result['correct'],
                        'total': cat_result['total']
                    })
            
            category_performances.sort(key=lambda x: x['accuracy'], reverse=True)
            for perf in category_performances:
                print(f"{perf['model']:<20} {perf['accuracy']:<10.4f} {perf['correct']}/{perf['total']}")
        
        print(f"\n最佳表現分析：")
        if model_performances:
            # 按準確率排序
            model_performances.sort(key=lambda x: x['accuracy'], reverse=True)
            best_accuracy = model_performances[0]
            
            # 按覆蓋率排序
            best_coverage = max(model_performances, key=lambda x: x['coverage'])
                   
            # 按訓練時間排序
            fastest_training = min(model_performances, key=lambda x: x['training_time'])
            
            print(f"最高準確率: {best_accuracy['name']} ({best_accuracy['accuracy']:.4f})")
            print(f"最高覆蓋率: {best_coverage['name']} ({best_coverage['coverage']:.4f})")
            print(f"最快訓練: {fastest_training['name']} ({fastest_training['training_time']:.1f}s)")
        
        print(f"\n報告生成完成!")


if __name__ == "__main__":
    evaluator = MultiCorpusEvaluator(
        "questions-words.csv",
        max_corpus_size=80000,    
        eval_sample_size=3000     
    )
    evaluator.run_comprehensive_evaluation()


# In[ ]:


jupyter nbconvert --to script main.ipynb

