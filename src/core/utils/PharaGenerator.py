import re
import logging
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer

log = logging.getLogger('pharaGeneratorLogger')

from core.metrics_and_models.benchmarks import tf_idf_similarity

tfidf = TfidfVectorizer(stop_words='english') # vectorizer
model_path = r'C:\Users\morichp\huggingface\sentence-transformers\parrot_paraphraser_on_T5'

class PharaGenerator:
    '''
        This class is used to generate pharaprase of a given sentence.

        Attributes
        ----------
        model: object
            Pretrained model as PharaGenerator 

        tokenizer: object
            Pretrained to tokenize a sentence
    '''

    def __init__(self):
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)

    def get_phrase(self, text:str, min:float = None, max:float = None) -> tuple:
        '''
            This is retrieve a paraphase for a given text, the paraphase has to 
            respect a similarity range constrain.

            Parameters
            ----------      
            text: str
                Text from a ticket
            
            min: float
                Minimum value for similarity range
            
            max: float
                Maximum value for similarity range

            Returns
            -------
                tuple
                paraphase and its similarity score
        '''

        log.debug('Get phrases sequences')
        list_phrases = self.get_phrases_sequence(text, num_return_sequences=10)

        log.debug('Compute similarity scores for phrases')
        sims_dict = {}
        for phrase in list_phrases:
            sim = self._similiarity_score(text, phrase)
            if min and max and self._in_similarity_range(sim, min, max):
                sims_dict[phrase] = sim

        log.debug('Get median among the scores')
        paraphrase, sim = self._get_median(sims_dict)
        return paraphrase, sim

    def _in_similarity_range(self, sim:float, min:float, max:float) -> bool:
        '''
            This is a support method and it is used to check if the similarity score
            of a current paraphrase is in the established similarity range.

            Parameters
            ----------      
            sim: float
                Similarity score for current paraphrase
            
            min: float
                Minimum value for similarity range
            
            max: float
                Maximum value for similarity range

            Returns
            -------
                bool
                True if sim is in the similarity score range, False otherwise
        '''

        return min <= sim and sim <= max

    def _similiarity_score(self, text:str, phrase:str) -> float:
        '''
            This is a support method and it is used to compute the tf_idf
            similarity score between a pair of text tickets.

            Parameters
            ----------      
            text: str
                Original ticket text
            
            phrase: str
                Paraphrase of the ticket text

            Returns
            -------
            sim: float
                Similarity score
        '''

        tfidf_matrix = tfidf.fit_transform(pd.Series([text, phrase]))
        sent1, sent2 = tfidf_matrix[0], tfidf_matrix[1]
        sim = tf_idf_similarity(sentence1=sent1, sentence2=sent2)
        return round(sim, 2)

    def _get_median(self, sims_dict:dict) -> tuple:
        '''
            This is a support method and it is used to compute the tf_idf
            similarity score between a pair of text tickets.

            Parameters
            ----------      
            text: str
                Original ticket text
            
            phrase: str
                Paraphrase of the ticket text

            Returns
            -------
            sim: float
                Similarity score
        '''

        if len(sims_dict) == 0:
            return '', 0
        
        sims_dict = {k: v for k, v in sorted(sims_dict.items(), key=lambda item: item[1])}
        list_sims_vals = list(sims_dict.values())
        median_score = list_sims_vals[int(len(list_sims_vals)/2)]

        key = list(sims_dict.keys())[list_sims_vals.index(median_score)]
        return key, median_score

    def get_phrases_sequence(self, text:str, num_return_sequences:int=5) -> list:
        '''
            This is the core of the class, here paraphrases of text are generated.

            Parameters
            ----------      
            text: str
                Original ticket text
            
            num_return_sequences: int
                NUmber of paraphrases to generate for text

            Returns
            -------
            out_list: list
                List of generated paraphrases
        '''

        out_list = [''] * num_return_sequences

        # TODO: [BUG] - text does not have any '.' because of the pre-processing!
        sentences = self._split_text(text)

        for sentence in sentences:
            sentence = self._filter_spaces(sentence)

            if len(sentence) == 0:
                for idx, i in enumerate(sentences):
                    print(idx, i)
                raise IndexError('Error in PharaGenerator.get_phrases_sequence(): sentence is empty')

            # tokenize the text to be form of a list of token IDs
            inputs = self.tokenizer([sentence], truncation=True, padding="longest", return_tensors="pt")

            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                top_k=50,
                top_p=.95,
                max_length=int(len(sentence) + (len(sentence) * .15)),
                num_return_sequences=num_return_sequences,
                temperature=0.75,
                no_repeat_ngram_size=2
            )
            """
                do_sample:
                Language generation using random sampling.

                top_k: 
                In Top-K sampling, the K most likely next words are filtered and the 
                probability mass is redistributed among only those K next words.

                top_p:
                In Top-p sampling chooses from the smallest possible set of words whose 
                cumulative probability exceeds the probability p. The probability mass 
                is then redistributed among this set of words.

                temperature:
                Make the distribution P(w|w_{1:t-1})P(w∣w 1:t−1) sharper 
                (increasing the likelihood of high probability words and decreasing 
                the likelihood of low probability words) by lowering 
                the so-called temperature of the softmax.

                no_repeat_ngram_size:
                Most common n-grams penalty makes sure that no n-gram appears twice by 
                manually setting the probability of next words that could create an already seen n-gram to 0.

                num_return_sequences:
                NUmber of generated senteces to return
            """
            # decode the generated sentences using the tokenizer to get them back to text
            out = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # add '.' at the end of each phrase
            out = [o + '.' if o[-1] != '.' else o for o in out] 
            # concatenate text phrase sentences and add space after '.'
            out_list = [out_list[i] + ' ' + p if i != 0 else out_list[i] + p for i, p in enumerate(out)]
        
        return out_list

    def _split_text(self, text:str) -> list:
        '''
            This is a support method and it is used to 
            perform some simple text manipulation.

            Parameters
            ----------      
            text: str
                Original ticket text

            Returns
            -------
                list
                List of splitted texts
        '''

        splitted_text = text.replace('\n', ' ')
        splitted_text = re.split('[.;]', splitted_text)
        return list(filter(None, splitted_text))

    def _filter_spaces(self, sentence:str) -> str:
        '''
            This is a support method and it is used to 
            perform some simple text manipulation.

            Parameters
            ----------      
            text: str
                Original ticket text

            Returns
            -------
                str
                A manipulated sentece
        '''

        sentence = re.sub('\s+', ' ', sentence)
        sentence = re.sub('^\s+', '', sentence)
        return re.sub('\s+$', '', sentence)
