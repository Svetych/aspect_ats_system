# -*- coding: utf-8 -*-

from nltk.tokenize import sent_tokenize, word_tokenize
from pymorphy2 import MorphAnalyzer
import re


class TextProcessor():
    def __init__(self, flag = False):
        self.aspects = ["Aim", "Method", "Problem", "Relevance", "Result"]
        self.templates = {}
        for aspect in self.aspects:
            if not flag:
                self.make_dict(f"./templates_1/{aspect.lower()}_all_patterns", aspect) 
            else:
                self.make_dict(f"./templates_2/{aspect.lower()}_all_patterns_a", aspect)            
        self.morph = MorphAnalyzer()
        self.translator = Translator()
    
    def make_dict(self, file_name, aspect):
        with open(file_name, 'r') as file:
            self.templates[aspect] = Template(file.readlines(), aspect)        
    
    def parse(self, text):
        words = map(word_tokenize, text)
        for sentence in words:
            result = []
            for w in sentence:
                parsed = self.morph.parse(self.morph.parse(w)[0].normal_form)[0]
                pos, norm = self.translator.apply(parsed)
                if pos:
                    result.append((pos, norm))
            yield ' '.join(sentence), ''.join([f"{pos}<{norm}>" for pos, norm in result])
    
    def apply(self, sentence):
        extracted_aspects = [self.templates[aspect].analyze(sentence) for aspect in self.aspects]
        return extracted_aspects

class Translator():
    def __init__(self):
        self.pm2lspl_pos = {
            'NOUN' : 'N',   #существительное
            'ADJF' : 'A',   #прилагательное(полное)
            'ADJS' : 'A',   #прилагательное(краткое)
            'COMP' : 'A',   #компаратив
            'VERB' : 'V',   #глагол(личная', #форма)
            'INFN' : 'V',   #глагол(инфинитив)
            'PRTF' : 'Pa',  #причастие(полное)
            'PRTS' : 'Pa',  #причастие(краткое)
            'GRND' : 'Ap',  #деепричастие
            'NUMR' : 'Num', #числительное
            'ADVB' : 'Av',  #наречие
            'NPRO' : 'Pn',  #местоимение
            'PRED' : 'Av',  #предикатив
            'PREP' : 'Pr',  #предлог
            'CONJ' : 'Cn',  #союз
            'PRCL' : 'Pt',  #частица
            'INTJ' : 'Int', #междометие
        }
    
    def apply(self, parsed):
        pos = parsed.tag.POS
        return (None, parsed.normal_form) \
          if pos is None \
          else (self.pm2lspl_pos[pos], parsed.normal_form)

class Template():
    def __init__(self, text, aspect):
        self.dct = {}
        self.dct_len = {}
        self.aspect = aspect
        self.pt = re.compile("\w*<\w*>")  # для слов
        self.pr = re.compile('\[(\w*)\]') # для группы слов
        
        for pos_group in ['A', 'Ap', 'Av', 'Cn', 'Int', 'N', 'Num', 
                          'Pa', 'Pn', 'Pr', 'Pt', 'V']:
                    self.dct[pos_group] = f"{pos_group}<\w*>"   
        
        for line in text:
            if line.strip():
                left, right = line.split('=')
                name = left.strip()
                
                result = '|'.join([
                    ''.join([self.prepare(part.strip()) for part in alternative.split() if part]) # part = Pos<слово> / -группа слов- / Pattern№
                    for alternative in right.split('|')
                ])
                # result = группа слов: Pos<слово>|... / группы слов: (...)?... / шаблон: (?P<Pattern№>(...)?...)
                
                if "Dict" in name or "Pos" in name:
                    self.dct[name] = f"{result}" # result = группа слов: Pos<слово>|...
                else:
                    self.dct[name] = f"(?P<{name}>{result})" # result = группы слов: (...)?... / шаблон: (?P<Pattern№>(...)?...)
                    if result[0:2] != '(?':
                        self.dct_len[name] = result.count('?') + 1
        self.regexp = re.compile(self.dct[self.aspect])
    
    def key(self, text):
        key = re.match(self.pr, text)
        
        if key is None:
            s = self.dct.get(text, None)
            # None, если Pos<слово>; шаблон для обязательной группы слов, если группа обязательных слов; шаблон, если Pattern№
            return f"({s})" if s else None
        else:
            s = self.dct.get(key.group(1), None) # "(группа слов)?"
            return f"({s})?" if s else None
    
    def prepare(self, text):
        res = self.key(text)
        if res is not None:
            return res
        else:
            return text
    
    def analyze(self, sentence):
        dct_list = [
            {name : (s, len(list(self.pt.finditer(s))), self.dct_len[name] if name != self.aspect else 1)
             for name, s in res.groupdict().items() if s is not None}
            for res in self.regexp.finditer(sentence)
        ]
        # name = название шаблона; s = строка; № = количество слов

        dct_list = sorted(dct_list, key = lambda dct : max([i[1] for i in dct.values()]), reverse=True)

        return dct_list


if __name__ == '__main__':    
    file = open('text.txt', 'r')
    text = file.read()
    
    processor = TextProcessor(flag = True)
    for i, (_, new) in enumerate(processor.parse(sent_tokenize(text, language='russian'))):
        search_result = processor.apply(new)
        print(f"Sentence: {i+1}")
        print(new)
        for lst in search_result:
            if lst:
                weight_indicator = 0.0
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
                
                print(f"Aspect: {list(lst[0].keys())[0]}",
                      f"Weight: {weight_indicator}",
                      f"Max words: {list(lst[0].values())[0][1]} with pattern length: {pattern_len}",
                      "Patterns:",
                      sep='\n')
                
                for dct in lst:
                    for p, res in dct.items():
                        print("{:>9}".format(f"{p}"), f": {res}")
                print('')
        print('')