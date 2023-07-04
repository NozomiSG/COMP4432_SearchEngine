from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

app = Flask(__name__)

# 下载nltk所需数据
nltk.download('stopwords')
nltk.download('wordnet')

# 读取数据
data = pd.read_csv('abcnews-date-text.csv')
data = data.head(10000)
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


@app.route('/')
def index():
    return render_template('index.html')


# 创建一个API路由
@app.route('/search', methods=['GET'])
def search_api():
    query = request.args.get('query', '')
    results = search(query)
    return jsonify(results.to_dict())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
