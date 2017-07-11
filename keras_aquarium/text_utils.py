from sklearn.feature_extraction.text import CountVectorizer, defaultdict
import re

class TextIndicesTransformer():

    def __init__(self, min_df=4, sents_mode=True):
        self.cv = CountVectorizer(stop_words=None, min_df=min_df, token_pattern=u'(?u)\\b\\w\\w+\\b'+u"|[\\.]+")
        self.min_df = min_df
        self.sents_mode = sents_mode

    def fit(self, texts):
        return self.fit_transform(texts)

    def fit_transform(self, texts):
        cv = self.cv

        vocabulary = defaultdict(lambda : 1 + vocabulary.__len__())
        if self.sents_mode:
            self.sents_sep = set([vocabulary["."], ])
        analyze = cv.build_analyzer()
        min_df = cv.min_df

        df_counter = defaultdict(int)
        words_li = []

        for doc in texts:
            word_set = set()
            text_in_idx = []

            words = analyze(doc)
            words_li.append(words)
            for word in words:
                wi = vocabulary[word]
                word_set.add(wi)

            for wi in word_set:
                df_counter[wi] += 1

        for wi, c in df_counter.items():
            if df_counter[wi] < min_df:
                del df_counter[wi]

        for word, wi in vocabulary.items():
            if wi not in df_counter:
                del vocabulary[word]

        texts_in_indices = []
        for words in words_li:
            text_in_idx = []
            for w in words:
                wi = vocabulary.get(w, None)
                if wi is not None:
                    text_in_idx.append(wi)

            if self.sents_mode:
                text_in_idx = [list(y) for x, y in itertools.groupby(text_in_idx, lambda z: z in self.sents_sep)
                               if not x]

            texts_in_indices.append(filter(lambda x: len(x) > 3, text_in_idx))

        self.analyze = analyze
        self.vocabulary = vocabulary
        self.df_counter = df_counter

        return texts_in_indices

    def transform(self, texts):
        texts_in_indices = []
        vocabulary = self.vocabulary
        analyze = self.analyze

        for doc in texts:
            text_in_idx = []

            for word in analyze(doc):
                wi = vocabulary.get(w, None)
                if wi is not None:
                    text_in_idx.append(wi)

            texts_in_indices.append(text_in_idx)

        return texts_in_indices
