import numpy as np
from scipy.special import gammaln, digamma
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_non_negative
from sklearn.utils import check_random_state
from scipy.special import kl_div
from gensim.models import CoherenceModel

class LatentDirichletAllocationWithCooccurrence(BaseEstimator, TransformerMixin):
    def __init__(self, lambda_param=1, n_components=10, doc_topic_prior=None, topic_word_prior=None,
                 max_iter=10, random_state=None, edges_threshold=0.5):
        self.n_components = n_components
        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior
        self.max_iter = max_iter
        self.random_state = random_state
        self.edges_threshold = edges_threshold
        self.lambda_param = lambda_param
        
    def _init_latent_vars(self, n_features, edge_dict, vocabulary, matrix):
        self.random_state_ = check_random_state(self.random_state)
        if self.doc_topic_prior is None:
            self.doc_topic_prior_ = np.ones(self.n_components) / self.n_components
        else:
            self.doc_topic_prior_ = self.doc_topic_prior

        if self.topic_word_prior is None:
            self.topic_word_prior_ = np.ones(n_features) / n_features
        else:
            self.topic_word_prior_ = self.topic_word_prior

        self.components_ = self.random_state_.gamma(100.0, 0.01, (self.n_components, n_features))
        self.exp_dirichlet_component_ = np.exp(self._dirichlet_expectation(self.components_))
        self.edge_dict_ = edge_dict  
        self.vocabulary_ = dict(vocabulary)  # Ensure vocabulary is a dictionary

        # Initialize count matrices
        self.nzw = np.zeros((self.n_components, n_features))  # word-topic count matrix
        self.nmz = np.zeros((n_features, self.n_components))  # document-topic count matrix
        self.nz = np.zeros(self.n_components)  # topic count
        self.nm = np.zeros(n_features)  # document count

        # Randomly assign topics to words in the initial state
        for m in range(matrix.shape[0]):
            for w_idx in np.where(matrix[m, :])[0]:
                z = self.random_state_.randint(self.n_components)  # Randomly select a topic
                self.nzw[z, w_idx] += 1
                self.nmz[w_idx, z] += 1
                self.nz[z] += 1
                self.nm[w_idx] += 1

    def _dirichlet_expectation(self, values):
        if values.ndim == 1:
            values = values[:, np.newaxis]
        return np.array([digamma(np.sum(values, axis=-1)) - digamma(values) for values in values.T]).T

    def _conditional_distribution(self, z, w_idx):
        # 原始的p_z计算
        left = (self.nzw[:, w_idx] + self.topic_word_prior_[w_idx]) / (self.nz + self.topic_word_prior_[w_idx])  # 词-主题
        right = (self.nmz[z, :] + self.doc_topic_prior_) / (self.nm[z] + self.doc_topic_prior_)  # 文档-主题
        p_z = left * right

        # 共现词的处理
        topic_assignment = np.zeros(self.n_components)  # 初始化主题分配参数
        parent = self.nzw[:, w_idx]  # 当前处理的共现词应该属于的主题
        if np.any(parent > 0):
            try:
                children_indices = [i for i in self.edge_dict_[w_idx] if i in self.edge_dict_[w_idx] and i < self.nzw.shape[1]]
                if children_indices:
                    children = self.nzw[:, children_indices]  # 获取共现词的word-topic count
                    for idx, i in enumerate(parent):
                        # 只累加当前主题对应的共现词的值
                        topic_assignment[idx] = np.sum(children[:, np.where(children_indices == i)[0][0]] if np.where(children_indices == i)[0].size > 0 else 0)
                    topic_assignment = topic_assignment / np.sum(topic_assignment) if np.sum(topic_assignment) > 0 else np.ones(self.n_components) / self.n_components
                    topic_assignment = np.exp(self.lambda_param * topic_assignment)
                    p_z *= topic_assignment
            except KeyError:
                pass

        p_z /= np.sum(p_z)
        return p_z
    
    def _e_step(self, X):
        n_samples, n_features = X.shape
        doc_topic_distr = np.random.dirichlet(self.doc_topic_prior_, n_samples)  # Initialize document-topic distribution

        for m in range(n_samples):
            for w_idx in np.where(X[m, :])[0]:  # For each word in the document
                p_z = self._conditional_distribution(m, w_idx)
                doc_topic_distr[m] = p_z
                self.nzw[:, w_idx] += X[m, w_idx] * p_z  # Update word-topic count matrix
                self.nmz[m, :] += p_z  # Update document-topic count matrix
                self.nz += p_z  # Update topic count
                self.nm[m] += 1  # Update document count
        return doc_topic_distr

    def _m_step(self, X, doc_topic_distr):
        n_samples, n_features = X.shape
        topic_word_suff_stats = np.zeros_like(self.components_) 
        
        for m in range(n_samples):
            for w_idx in np.where(X[m, :])[0]:
                topic_word_suff_stats[:, w_idx] += X[m, w_idx] * doc_topic_distr[m, :]
        
        self.components_ = (self.topic_word_prior_ + topic_word_suff_stats) / (self.topic_word_prior_ + X.sum(axis=0))
        self.exp_dirichlet_component_ = np.exp(self._dirichlet_expectation(self.components_))
        
        if len(self.edge_dict_) > 0:
            edge_sum = 0
            for edge in self.edge_dict_.values():
                if len(edge) >= 2 and edge[0] < self.nzw.shape[1] and edge[1] < self.nzw.shape[1]:
                    edge_sum += self.nzw[:, edge[0]] @ self.nzw[:, edge[1]]
                else:
                    #print("Invalid edge found:", edge)  # 打印无效的边
                    pass
            self.lambda_ = np.log(edge_sum / len(self.edge_dict_)) if edge_sum > 0 else 1
        else:
            self.lambda_ = 1
    
    def getTopKWords(self, K, vocab):

        pseudocounts = np.copy(self.nzw.T)  # 词-主题分布的伪计数
        normalizer = np.sum(pseudocounts, axis=0)  # 每个词的总伪计数
        pseudocounts /= normalizer[np.newaxis, :]  # 归一化伪计数
        worddict = {}
        for t in range(self.n_components):  # 遍历每个主题
            topWordIndices = np.argsort(-pseudocounts[:, t])[:K]  # 找到概率最高的K个词汇的索引

            index_to_word = {index: word for word, index in vocab.items()}
            worddict[t] = [index_to_word[i] for i in topWordIndices]  
        return worddict   
    
    def fit(self, X, y=None, edge_dict=None, vocabulary=None,matrix=None):
        check_non_negative(X, "LDA")
        n_samples, n_features = X.shape
        print("Starting fit with shape:", n_samples, n_features)
        self._init_latent_vars(n_features, edge_dict, vocabulary,X)
        
        for iteration in range(self.max_iter):
            print("Iteration:", iteration)
            doc_topic_distr = self._e_step(X)
            self._m_step(X, doc_topic_distr)
            print("Lambda value at iteration {}: {}".format(iteration, self.lambda_))
        return self

    def transform(self, X):
        check_non_negative(X, "LDA")
        doc_topic_distr = self._e_step(X)
        doc_topic_distr /= doc_topic_distr.sum(axis=1, keepdims=True)
        return doc_topic_distr

    def _more_tags(self):
        return {
            "preserves_dtype": [np.float64, np.float32],
            "requires_positive_X": True,
        }
    
def word_indices(vec):

    for idx in vec.nonzero()[0]:
        for i in range(int(vec[idx])):
            yield idx