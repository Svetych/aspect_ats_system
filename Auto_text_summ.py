# -*- coding: utf-8 -*-

import itertools

from Text_terms import *
from Symmetrical_summ import *
from Templates import *

# пересчет весов для предложений
def countFinalWeights(tf_weights, stemmed_text, stemmed_pnn):
    weighted_terms = dict(tf_weights.items())
    total_sents_in_text = stemmed_text.len
    
    # список первых и последних предложений
    collection_of_first_last_sents = list(itertools.chain.from_iterable(
        (paragraph[0], paragraph[-1]) if len(paragraph) > 1 else (paragraph[0],)
        for paragraph in stemmed_text.text if paragraph))
    
    # список слов первых и последних предложений абзацев
    pairs_collection_of_first_last_sents = list(
        itertools.chain.from_iterable(collection_of_first_last_sents))
    stems_collection_of_first_last_sents = {
        pair[0] for pair in pairs_collection_of_first_last_sents }
    
    # количество слов в первых и последних предложениях
    total_stems_in_first_last = len(stems_collection_of_first_last_sents)
    
    # количество слов из словаря в первых и последних предложениях
    total_dictwords_in_first_last = 0
    for s1 in stems_collection_of_first_last_sents:
        if s1 in tf_weights:
            total_dictwords_in_first_last += 1
    
    # количество слов в тексте
    total_stems_in_text = len(list(itertools.chain.from_iterable(stemmed_text.sentences())))
    
    # среднее количество слов из словаря в первых и последних предложениях абзацев
    avg_dictwords_in_first_last = total_dictwords_in_first_last / total_stems_in_first_last
    
    # среднее количество слов в первых и последних предложениях абзацев
    avg_stems_in_first_last = total_stems_in_first_last / total_stems_in_text
    
    # список вопросительных и восклиц. предложений
    collection_of_q_excl_sents = [
        sentence
        for sentence in stemmed_text.sentences()
        if sentence and sentence[-1][0] in {'?', '!'}]
    
    # список слов из вопросительных и восклицательных предложений
    stems_collection_of_q_excl_sents = {tpl[0] for tpl in itertools.chain.from_iterable(collection_of_q_excl_sents) if tpl[0] not in '?!'}
    
    # количество вопросительных и восклицательных предложений в тексте
    num_of_q_excl_sents = len(collection_of_q_excl_sents)
    
    """
    если термины есть в первых и последн. предложениях абзацев, то вес термина
    умножаем на частное среднего кол-ва терминов из словаря в первых и посл. предл.
    и среднего кол-ва терминов в первых и последн. предл-х.
    """
    for term in stems_collection_of_first_last_sents:
        if term in weighted_terms:
            weighted_terms[term] *= avg_dictwords_in_first_last / avg_stems_in_first_last
    """
    если термины есть в вопросительных и восклицательных предложениях,
    то умножаем вес термина на частное от кол-ва таких предложений
    и общего кол-ва предложений текста
    """
    for term in stems_collection_of_q_excl_sents:
        if term in weighted_terms:
            weighted_terms[term] *= num_of_q_excl_sents / total_sents_in_text
    """
    если термины из словаря - это "имена собственные", то умножаем вес
    термина на частное среднего кол-ва терминов из словаря в первых и посл. предл.
    и среднего кол-ва терминов в первых и последн. предл-х.
    """
    for term in stemmed_pnn:
        if term in weighted_terms:
            weighted_terms[term] *= avg_dictwords_in_first_last / avg_stems_in_first_last
    
    mean_weight = sum(weighted_terms.values()) / len(weighted_terms)
    sorted_tf = {term : weight
            for term, weight in weighted_terms.items()
            if weight > mean_weight}
    return sorted_tf

# пересчет предложений по весам
def convertFinalWeights(symmetry, symm_weights, ordinary_sents, indicators = True, adj = False):
    if indicators:
        """
        Здесь стоит надстройка, что пересчитывает веса в зависимости от индикаторов
        """
        processor = TextProcessor(flag = adj)
        result = []
        for (counter, weight),\
            (index, original),\
            ( _ , sentence)\
        in \
            zip(symm_weights,
                enumerate(ordinary_sents),
                processor.parse(ordinary_sents)
            ):
            search_result = processor.apply(sentence)
            weight_indicator = 1
            if search_result:
                for lst in search_result:
                    if lst:
                        pattern_len = list(lst[0].values())[1][2]
                        word_num = list(lst[0].values())[1][1]
                        if pattern_len > 3:
                            if word_num >= 3:
                                weight_indicator += 1
                            elif word_num == 2:
                                weight_indicator += 0.5
                        elif pattern_len == 3:
                            if word_num >= 2:
                                weight_indicator += 1
                            elif word_num == 1:
                                weight_indicator += 0.5                                
                        else:
                            if word_num >= pattern_len:
                                weight_indicator += 1
            if len(counter) > 6:
                result.append((original, weight * weight_indicator, index))
        return sorted(result, key=lambda x: x[1], reverse=True)
    
    else:
        return symmetry.convertSymmetryToOrdinary(symm_weights, ordinary_sents)

# процентная выборка
def selectFinalSents(converted_sents, percentage=10):
    """
    Метод выбирает n первых предложений из списка.
    n определяется указанным процентом. Список сортируется
    по позиции предложения в оригинальном тексте, таким образом
    возвращается оригинальная последовательность, чтобы хоть
    как-то сохранить связность.
    """
    compression_rate = int(len(converted_sents) * percentage / 100 + 0.5)
    sorted_salient_sentences = sorted(converted_sents[:compression_rate], key=lambda w: w[2])
    return sorted_salient_sentences


# суммаризатор
class SUMMARIZER():

    def __init__(self):
        self.language = 'ru'
        self.stopwords = list(stopwords.words('russian'))
        self.re_term = re.compile("[\wа-яА-Я]+\-[\wа-яА-Я]+|[\wа-яА-Я]+|[!?]")
        self.symmetry = SymmetricalSummarizationWeightCount()
    
    def check(self, term):
        if term in self.stopwords:
            return False
        elif re.match(self.re_term, term):
            return True
        else:
            return False
    
    def summarize(self, file_name, output_name, indicators = True, adj = False, percentage = 10):
        file = open(file_name, 'r')
        text = StructuredText(text_segmentor(file.read()))
        res = ''
        
        if text.len >= 3:
            # стемминг предложений
            # текст без стоп-слов: (стема, слово), предложения сгруппированны по абзацам
            STEMMED_SENTENCES = StructuredText(text.map_sent(
                    lambda sentence:
                    [(normalize(term), term)
                         for term
                         in word_tokenize(sentence)
                         if self.check(term)] ))
            # список всех стем
            BIG_LIST_OF_PAIRS = list(itertools.chain.from_iterable(STEMMED_SENTENCES.sentences()))
            LIST_OF_STEMS = [pair1[0] for pair1 in BIG_LIST_OF_PAIRS]

            if LIST_OF_STEMS:
                # список кортежей (слово, его относительная частота), усечённый по средней частоте
                TOTAL_STEM_COUNT = dict(simpleTermFreqCount(LIST_OF_STEMS))
                # список "имён собственных"
                STEMMED_PNN = lookForProper(STEMMED_SENTENCES.text)
                
                # список терминов с весовыми коэффициентами
                SORTED_TFIDF = countFinalWeights(TOTAL_STEM_COUNT, STEMMED_SENTENCES, STEMMED_PNN)
                SORTED_TFIDF = sorted(SORTED_TFIDF.items(), key=lambda w: w[1], reverse=True)
                
                # словари каждого предложения с частотностью по словам
                S_with_termfreqs = [Counter([word[0] for word in sentence]) for sentence in STEMMED_SENTENCES.sentences()]
                # общее количество стем в тексте
                TOTAL_STEMS_IN_TEXT = len(LIST_OF_STEMS)
                # общее количество предложений в тексте
                TOTAL_SENTS_IN_TEXT = len(text.sentences())
                
                # пересчет весов
                SYMMETRICAL_WEIGHTS = self.symmetry.countFinalSymmetryWeight(
                        SORTED_TFIDF, S_with_termfreqs,
                        TOTAL_STEMS_IN_TEXT, TOTAL_SENTS_IN_TEXT,
                        STEMMED_PNN)
                
                # отбор предложений
                ORIGINAL_SENTENCES = convertFinalWeights(
                        self.symmetry,
                        SYMMETRICAL_WEIGHTS,
                        text.sentences(),
                        indicators, adj)
                
                #print(ORIGINAL_SENTENCES)
                
                res = selectFinalSents(ORIGINAL_SENTENCES, percentage)

            else:
                print("There are no words to process!")

        else:
            print("Text should be at least 3 sentences long.")
            
        # результат записан в отдельный файл
        with open(output_name, 'w') as output_file:
            for sent3 in range(len(res)):
                #output_file.write(str(res[sent3][2]) + ',') # для выдачи № выбранных предложений
                output_file.write(res[sent3][0] + '\n')
                print(res[sent3][0])
            '''
            output_file.write(str(len(text.sentences()))) # для выдачи № выбранных предложений
            
            print(len(res)) # для подсчета длины реферата
            for i, x in enumerate(text.sentences()): # для получения разбиения на предложения
                output_file.write(str(i+1) + '. ' + x + '\n')'''

                
if __name__ == '__main__':

    Sum = SUMMARIZER()
    Sum.summarize('text.txt', 'output.txt', indicators = True, adj = True, percentage = 60)