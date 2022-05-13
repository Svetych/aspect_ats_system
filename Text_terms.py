# -*- coding: utf-8 -*-

from nltk.stem.snowball import RussianStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import pymorphy2

from collections import Counter
import itertools
import regex as re

# нормализация термина
def normalize(term):
    return normalize.stemmer.stem(normalize.lemmatizer_ru.parse(term)[0].normal_form)
normalize.lemmatizer_ru = pymorphy2.MorphAnalyzer()
normalize.stemmer = RussianStemmer()

# возвращает список стем в виде [[[],[],[]],[[],[]]], где второй уровень - абзацы, третий - предложения
def text_segmentor(text):
    return [sent_tokenize(paragraph) for paragraph in re.split(r"[\r\n]+", text)]

# поиск имен собственных
def lookForProper(structured_stems):
        """
        Наивный метод поиска имен собственных.
        Они нужны при добавлении дополнительных весов предложениям и словам.
        Выбираются все слова с большой буквы, если они
        стоят не в начале предложения.
        """
        proper_nouns = set()
        stemmed_pnn = set()
        for paragraph in structured_stems:
            for sentence in paragraph:
                for stem, word in sentence[1:]:
                    if word.split('-')[0].istitle():
                        proper_nouns.add(word)
                        stemmed_pnn.add(stem)
        return stemmed_pnn

# подсчет tf-idf стем
def simpleTermFreqCount(big_lst):
    """
    Метод получает на вход список стем.
    В словаре stemfreqs считаются частотности стем.
    Подсчитывается общее количество стем в словаре (total_stems_in_text).
    Вычисляется среднее арифметическое (mean_freq).
    Стемы с частотностью выше среднего арифм. выбираются в список termsfreq.
    Список termsfreq - это кортежи с парами (слово, относительная частота)
    """
    stemfreqs = Counter(big_lst)
    total_stems_in_text = sum(stemfreqs.values())
    mean_freq = total_stems_in_text / len(stemfreqs.values())
    termsfreq = [
        (word, freq / total_stems_in_text)
        for word, freq in stemfreqs.items()
        if freq >= mean_freq
    ]
    return termsfreq


# текст
class StructuredText():
    def __init__(self, text):
        self.text = text
        self.len = len(list(itertools.chain.from_iterable(self.text)))
    
    def sentences(self):
        return list(itertools.chain.from_iterable(self.text))
    
    def map_sent(self, func):
        return [
            [
                func(sentence)
                for sentence in paragraph
            ]
            for paragraph in self.text
        ]