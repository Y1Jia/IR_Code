代码源自[liucongg](https://github.com/liucongg/ZhiHu_Code/tree/master/bm25_code)，在tf-idf部分稍作改动

tf-idf权重计算公式为：<img src="https://latex.codecogs.com/svg.image?\sqrt{tf_{t,d}}\times&space;(\lg{\frac{numDocs}{df_t&plus;1}&plus;1})\times\frac{1}{\sqrt{length}}">

BM25权重计算公式为：<img src="https://latex.codecogs.com/svg.image?\lg{\frac{numDocs-df_t&plus;\frac{1}{2}}{df_t&plus;\frac{1}{2}}}&space;\times&space;\frac{(k_1&plus;1)tf_{t,d}}{k_1(1-b&plus;b\times\frac{length}{AvgLength})&plus;tf_{t,d}}\times\frac{(k_2&plus;1)tf_{t,q}}{k_2&plus;tf_{t,q}}&space;">

tf-idf code:

```python
class TF_IDF_Model(object):
    def __init__(self, documents_list):
        # 文本列表，内部每个文本需要事先分好词
        self.documents_list = documents_list
        # 文本总个数
        self.documents_number = len(documents_list)
        # 存储每个文本中每个词的词频
        self.tf = []
        # 存储每个词汇的逆文档频率
        self.idf = {}
        # 类初始化
        self.init()

    def init(self):
        df = {}
        for document in self.documents_list:
            temp = {}
            for word in document:
                # 存储每个文档中每个词的词频
                temp[word] = temp.get(word, 0) + 1  #raw count 直接使用词项出现次数
                # temp[word] = temp.get(word, 0) + 1/len(document)  #使用词项出现频率
            self.tf.append(temp)
            for key in temp.keys():
                df[key] = df.get(key, 0) + 1
        for key, value in df.items():
            # 每个词的逆文档频率
            self.idf[key] = np.log(self.documents_number / (value + 1) + 1)
            # self.idf[key] = np.log(self.documents_number / (value + 1))

    def get_score(self, index, query):
        score = 0.0
        for q in query:
            if q not in self.tf[index]:
                continue
            score += self.tf[index][q] * self.idf[q]
        score /= np.sqrt(len(self.tf[index]))   #考虑文档的长度
        return score

    def get_documents_score(self, query):
        score_list = []
        for i in range(self.documents_number):
            score_list.append(self.get_score(i, query))
        return score_list
```

BM25 code:

```python
class BM25_Model(object):
    def __init__(self, documents_list, k1=2, k2=1, b=0.5):
        # 文本列表，内部每个文本需要事先分好词
        self.documents_list = documents_list
        # 文本总个数
        self.documents_number = len(documents_list)
        # 文本库中文本的平均长度
        self.avg_documents_len = sum([len(document) for document in documents_list]) / self.documents_number
        # 存储每个文本中每个词的词频
        self.tf = []
        # 存储每个词汇的逆文档频率
        self.idf = {}
        self.k1 = k1
        self.k2 = k2
        self.b = b
        # 类初始化
        self.init()

    def init(self):
        df = {}
        for document in self.documents_list:
            temp = {}
            for word in document:
                # 存储每个文档中每个词的词频
                temp[word] = temp.get(word, 0) + 1
            self.tf.append(temp)
            for key in temp.keys():
                df[key] = df.get(key, 0) + 1
        for key, value in df.items():
            # 每个词的逆文档频率
            self.idf[key] = np.log((self.documents_number - value + 0.5) / (value + 0.5))

    def get_score(self, index, query):
        score = 0.0
        document_len = len(self.tf[index])
        qf = Counter(query)
        for q in query:
            if q not in self.tf[index]:
                continue
            score += self.idf[q] * (self.tf[index][q] * (self.k1 + 1) / (
                        self.tf[index][q] + self.k1 * (1 - self.b + self.b * document_len / self.avg_documents_len))) * (
                                 qf[q] * (self.k2 + 1) / (qf[q] + self.k2))

        return score

    def get_documents_score(self, query):
        score_list = []
        for i in range(self.documents_number):
            score_list.append(self.get_score(i, query))
        return score_list
```

注：使用BM25权重计算方法时，idf可能出现小于0的情况，可使用停用词表以缓解此问题。