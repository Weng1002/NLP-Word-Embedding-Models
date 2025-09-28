# NLP-Word-Embedding-Models
此為清大自然語言處理課程的 HW1: Practice word analogy (Google Analogy dataset) via 

(1) the pre-trained word embedding and also 

(2) the new word embedding trained by yourself.

# NLP Homework 1 Report

**Platform:** Local  
**Python version:** 3.10.18  
**Operating system:** Linux 6.8.0-65-generic  
**CPU:** x86_64 24核心  
**GPU:** NVIDIA GeForce RTX 3090  

---

## 1. Embedding Model & Preprocessing 

**使用的詞嵌入模型**  
- 預訓練模型：GloVe（`glove-wiki-gigaword-300`）  
- 自訓練模型：使用 20% 抽樣維基百科文章訓練的 Word2Vec  

**預處理步驟**  
- 基本處理：小寫化、移除標點符號/特殊字元、移除數字、過濾短詞、空白正規化  
- 改良處理：  
  - ASCII 過濾移除非英文詞  
  - 移除停用詞（NLTK stopwords）  
  - 詞形還原（WordNetLemmatizer）  
  - 精確斷詞（`nltk.word_tokenize`）  
  - 最小句長 ≥ 5  

**Word2Vec 超參數**  
- `vector_size=300, window=5, min_count=5, workers=4, sg=0, epochs=10`  
- `negative=5, sample=1e-3, seed=4321`  

---

## 2. Performance on Different Sampling Ratios 

**抽樣資料**  
- 5% → 281,253 篇文章  
- 10% → 563,068 篇文章  
- 20% → 1,123,661 篇文章  

**預期表現**  
- **5%**：詞彙覆蓋率 60-70%，類比準確率 15-25%，訓練時間最短但語義不完整  
- **10%**：詞彙覆蓋率 75-85%，準確率 25-35%，效能與資源平衡  
- **20%**：詞彙覆蓋率 85-90%，準確率 35-45%，最佳語義捕捉，但訓練時間最長  

➡ 效能隨抽樣比例遞增，但 **邊際效益遞減**。  

---

## 3. Corpus Comparison 

### 3.1 Results
| 語料 | 描述 | 準確率 | 詞彙覆蓋率 | 訓練時間 |
|------|------|--------|------------|----------|
| Wikipedia 20% | 百科全書文本 | **58.80%** | 99.77% | 311s |
| News Corpus  | 新聞語料 | 0.00% | 3.57% | 1s |
| Literature Corpus | 文學語料 | 0.03% | 11.13% | 2.9s |

**語義類比 (Semantic)**  
- Wiki: 62.24% (839/1348)  
- News: 0.00% (0/13)  
- Literature: 0.00% (0/27)  

**句法類比 (Syntactic)**  
- Wiki: 56.23% (925/1645)  
- Literature: 0.33% (1/307)  
- News: 0.00% (0/94)  

### 3.2 差異說明
- **Wikipedia**：大規模 (~80k 句)、正式、百科式知識 → 高多樣性詞彙  
- **News**：小語料，時事政治偏多，語言口語化  
- **Literature**：小語料，情感與修辭多，語言風格特殊  

### 3.3 原因分析
- Wikipedia 準確率高 → 覆蓋廣、語料大、內容一致性強  
- News/Literature 準確率低 → 資料少、領域偏差、風格差異大  

---

## 4. Word Similarity Results 

### 例子：Top-5 相似詞
- **king** → queen, prince, monarch, royal, kingdom  
- **computer** → technology, software, digital, machine, electronic  
- **father** → mother, parent, son, family, dad  
- **science** → research, scientific, technology, knowledge, study  
- **love** → passion, emotion, heart, relationship, feeling  

**觀察**  
- 語義聚類效果明顯（領域詞會聚在一起）  
- 詞性一致性：名詞對名詞、動詞對動詞  
- 學到同義、上下位、性別對應等語義關係  

---

## 5. Suggestions for Strengthening Report 

- **更多指標**：Precision/Recall/F1-score  
- **視覺化**：學習曲線、混淆矩陣、詞彙覆蓋熱圖  
- **可解釋性**：注意力分布、PCA 維度分析  
- **對比模型**：與 BERT、GPT 等上下文嵌入比較  
