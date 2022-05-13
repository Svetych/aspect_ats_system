# -*- coding: utf-8 -*-
import math
import regex as re

# пересчет весов для предложений
class SymmetricalSummarizationWeightCount():
    """
    Проводится начисление весов предложениям по методике симметричного реферирования
    (Яцко В.А. Симметричное реферирование: теоретические основы и методика
     // Научно-техническая информация. Сер.2. - 2002. -  № 5).
    """

    def rightLinksCount(self, tfidf_terms, sents_with_termsfreqs):
        """
        Метод производит поиск связей между предложениями вправо.
        Принимает на вход список tf-idf, и список словарей с частотами
        для каждого предложения. Берется предложение (словарь), если в нем есть
        термин из tf-idf, то ищется вхождение этого термина в предложениях справа.
        Если термин встретился в предложении справа, то выбирается его наибольшая
        частота (т.е. либо из исходного предложения, либо из правого), эта
        частота суммируется с общим весом предложения. Последнее предложение
        скидывается в список с нулевым весом, т.к. справа от него ничего нет.
        Возвращается список кортежей, в котором предложениям (словараям)
        приписаны веса [({sentence1}, вес), ({sentence2}, вес)]
        Параллельно для текущего предложения суммируются веса входящих
        в него ключевых слов, сумма прибавляется к весу предложения.
        Дополнительно вычисляется позиционный коэффициент, т.е. чем выше
        предложение, тем больше вес. Тоже прибавляется к общему весу.
        """
        tf_dict = dict(tfidf_terms)
        w_sent = []
        for line, left in enumerate(sents_with_termsfreqs):
            pscore = 10 / (line+1)
            right_context = sents_with_termsfreqs[line+1:] + [dict()]
            own_weight = sum(tf_dict.get(word, 0)*count for word, count in left.items())
            context_weight = sum(
                max(value, right[word])
                for word, value in left.items()
                for right in right_context
                if word in right
            )
            w_sent.append((left, own_weight + context_weight + pscore))     
        return w_sent

    def leftLinksCount(self, tfidf_terms, sents_with_termsfreqs):
        """
        Метод производит поиск связей между предложениями влево.
        Тот же алгоритм, что и при поиске вправо, только с нулевым
        весом скидывается первое предложение. Чтобы реализовать
        поиск влево, список предложений переворачивается.
        Возвращается список кортежей, в котором предложениям (словараям)
        приписаны веса [({sentence1}, вес), ({sentence2}, вес)]
        """
        tf_dict = dict(tfidf_terms)
        w_sent = []
        for line, right in enumerate(sents_with_termsfreqs):
            pscore = 10 / (line+1)
            left_context = [dict()] + sents_with_termsfreqs[:line]
            own_weight = sum(tf_dict.get(word, 0)*count for word, count in right.items())
            context_weight = sum(
                max(value, left[word])
                for word, value in right.items()
                for left in left_context
                if word in left
            )
            w_sent.append((right, own_weight + context_weight + pscore))
        return w_sent

    def countSymmetry(self, tfidf_terms, sents_with_termsfreqs):
        """
        Метод складывает два списка, полученных при поиске
        вправо и влево. Принимает на вход так же список tf-idf,
        и список предложений-словарей. Внутри явно вызываются
        функции установления правых и левых связей, порядок
        предложений в двух списках выравнивается и их веса складываются.
        Возвращается список кортежей предложений-словарей с весами.
        """
        return [(left[0], left[1] + right[1])
            for left, right in zip(
            self.leftLinksCount(tfidf_terms, sents_with_termsfreqs),
            self.rightLinksCount(tfidf_terms, sents_with_termsfreqs)) ]

    def countFinalSymmetryWeight(
        self,
        tfidf_terms,
        sents_with_termsfreqs,
        total_stems_in_text,
        total_sents_in_text,
        stemmed_pnn,
    ):
        """
        Метод добавляет к весам предложения дополнительный
        коэффициент ASL (average sentence length), чтобы
        длинные предложения не набрали большой вес.
        Принимает на вход список tf-idf, список предложений-словарей,
        общее кол-во стем в текте и общее количество предлоежний в тексте.
        Явно вычисляется функция countSymmetry(), затем asl и в цикле
        весу каждого предложения добавляется доп. коэффициент.
        Также каждый вес умножается на кол-во имен собственных и цифр.
        """
        w_sent1 = self.countSymmetry(tfidf_terms, sents_with_termsfreqs)
        
        # average sentence length
        asl = total_stems_in_text / total_sents_in_text
        
        # список для хранения частот "имен собственных" в каждом предложении
        proper_counts = [
            sum(pnn in sentence for pnn in stemmed_pnn)
            for sentence, weight in w_sent1 ]
        w_sent2 = [
            (sentence, weight * (1 + math.log(proper, 2) if proper else 1))
            for (sentence, weight), proper in zip(w_sent1, proper_counts) ]
        
        f_digits = re.compile(r"[0-9]+([\.\,\:][0-9]+)*")
        digits = [
            sum([bool(re.fullmatch(f_digits, word)) * value
                for word, value in sentence.items()])
            for sentence, weight in w_sent2 ]
        
        w_sent3 = [
            (sentence, weight * (1 + math.log(dg, 2) if dg else 1))
            for (sentence, weight), dg in zip(w_sent2, digits) ]
        
        w_sent4 = [
            (sentence, (asl * weight) / len(sentence) if len(sentence) > 5 else weight)
            for (sentence, weight), dg in zip(w_sent3, digits) ]
        
        return w_sent4

    def convertSymmetryToOrdinary(self, symm_weights, ordinary_sents):
        """
        Метод получает на вход список кортежей предложений-словарей с весами
        и список оригинальных предложений, т.е. неподвергшихся токенизации
        и стеммингу. Из последнего списка выбираются предложения, которые
        соответствуют словарям с весами. Стоит ограничение на длину показываемого
        предложения. Она должна быть не меньше 6 (и не больше 50 токенов) (UPD:
        если выходной список пустой, то берутся все предложения, вне зависимости
        от длины).
        Возвращается отсортированный по убыванию список предложений
        из оригинального текста с весом и его позицией в тексте.
        [(sentence, weight), (sentence, weight)]
        """
        result = []
        for (counter, weight),\
            (index, original)\
        in \
            zip(symm_weights,
                enumerate(ordinary_sents)
            ):
            if len(counter) > 6:
                result.append((original, weight, index))
        return sorted(result, key=lambda x: x[1], reverse=True)
        
