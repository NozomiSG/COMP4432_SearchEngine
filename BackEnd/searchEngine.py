import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# 下载nltk所需数据
nltk.download('stopwords')
nltk.download('wordnet')

# 读取数据
data = pd.read_csv('abcnews-date-text.csv')

# 数据预处理
data.drop_duplicates(subset='headline_text', inplace=True)

def preprocess_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

data['headline_text'] = data['headline_text'].apply(preprocess_text)

# 使用TF-IDF表示文本
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['headline_text'])

# 查询处理函数
def search(query, top_n=10):
    query = preprocess_text(query)
    query_vec = vectorizer.transform([query])

    # 计算余弦相似度
    similarity_scores = cosine_similarity(query_vec, X)

    # 获取最相关的文档
    sorted_scores_idx = np.argsort(similarity_scores).flatten()[::-1][:top_n]
    return data.iloc[sorted_scores_idx]

# 测试搜索引擎
query = "Economy Growth"
results = search(query)
print(results)
