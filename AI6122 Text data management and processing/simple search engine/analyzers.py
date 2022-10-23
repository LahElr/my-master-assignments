import lucene
from java.util import HashMap
from org.apache.lucene.analysis import Analyzer, StopFilter
from org.apache.lucene.analysis.classic import ClassicFilter, ClassicTokenizer
from org.apache.lucene.analysis.core import (KeywordAnalyzer, LowerCaseFilter,
                                             StopAnalyzer, WhitespaceTokenizer)
from org.apache.lucene.analysis.en import PorterStemFilter
from org.apache.lucene.analysis.miscellaneous import (LengthFilter,
                                                      LimitTokenCountFilter,
                                                      PerFieldAnalyzerWrapper)
from org.apache.lucene.analysis.standard import (StandardAnalyzer,
                                                 StandardTokenizer)
from org.apache.lucene.util import Version
from org.apache.pylucene.analysis import PythonAnalyzer
from org.apache.lucene.analysis.en import EnglishAnalyzer

r'''
Ref:  https://stackoverflow.com/questions/2012843/writing-a-custom-anaylzer-in-pylucene-inheritance-using-jcc
https://stackoverflow.com/questions/65884397/how-to-create-a-custom-analyzer-in-pylucene-8-6-1
'''


class ReferenceStandardAnalyzer(PythonAnalyzer):
    '''
    This is an example of a custom Analyzer, should be equal to `StandardAnalyzer`
    '''

    def tokenStream(self, fieldName, reader):
        # StandardTokenizer, LowerCaseFilter, StopFilter
        result = StandardTokenizer(Version.LUCENE_CURRENT, reader)
        result = LowerCaseFilter(result)
        result = StopFilter(result, EnglishAnalyzer.ENGLISH_STOP_WORDS_SET)
        return result


class ReferencePorterStemmerAnalyzer(PythonAnalyzer):

    def tokenStream(self, fieldName, reader):
        #There can only be 1 tokenizer in each Analyzer
        result = StandardTokenizer(Version.LUCENE_CURRENT, reader)
        result = LowerCaseFilter(result)
        result = PorterStemFilter(result)
        result = StopFilter(result, EnglishAnalyzer.ENGLISH_STOP_WORDS_SET)

        return result


def StopFilterFunc(stream):
    return StopFilter(stream, EnglishAnalyzer.ENGLISH_STOP_WORDS_SET)


def IdentityFilter(stream):
    return stream


def LengthFilterFunc(stream):
    return LengthFilter(stream, 0, 50)


def LimitTokenCountFilterFunc(stream):
    return LimitTokenCountFilter(stream, 20)


class CustomAnalyzer(PythonAnalyzer):

    def __init__(self, options):
        '''
        To customize analyzer, list of integers, the first indicates tokenizer, the left indicates filters
        '''
        PythonAnalyzer.__init__(self)

        self.tokenizer = {
            0: StandardTokenizer,
            1: WhitespaceTokenizer,
            2: ClassicTokenizer
        }[options[0]]

        self.options = options[1:]
        self.filter_config = {
            0: IdentityFilter,
            1: LowerCaseFilter,
            2: StopFilterFunc,
            3: PorterStemFilter,
            4: ClassicFilter,
            5: LengthFilterFunc,
            6: LimitTokenCountFilterFunc
        }

    # def tokenStream(self, fieldName, reader):
    #     #There can only be 1 tokenizer in each Analyzer
    #     result = self.tokenizer(Version.LUCENE_CURRENT, reader)
    #     for option in self.options:
    #         result = self.filter_config[option](result)
    #     return result
    
    def createComponents(self, fieldName):
        source = self.tokenizer()
        result = source
        for option in self.options:
            result = self.filter_config[option](result)
        return Analyzer.TokenStreamComponents(source, result)


# stemming,case folding,stop word remove


class FieldAnalyzerMapper:

    def __init__(self, default_analyzer):
        # "reviewText", "summary" field are not here
        self.special_analyzers = {
            "reviewerID": KeywordAnalyzer(),
            "asin": KeywordAnalyzer(),
            # "reviewerName": StandardAnalyzer(),
            "overall": KeywordAnalyzer(),
            "unixReviewTime": KeywordAnalyzer()
        }
        self.default_return = default_analyzer

    def __getitem__(self, field):
        try:
            return self.special_analyzers[field]
        except KeyError:
            return self.default_return


def PerFieldAnalyzerWrapperFactory(default_analyzer):
    per_field = HashMap()
    field_analyzer_mapper = FieldAnalyzerMapper(StandardAnalyzer)
    for field, analyzer in field_analyzer_mapper.special_analyzers.items():
        per_field.put(field, analyzer)
    if type(default_analyzer) is type:
        ret = PerFieldAnalyzerWrapper(default_analyzer(), per_field)
    else:
        ret = PerFieldAnalyzerWrapper(default_analyzer, per_field)

    return ret


field_names = [
    "reviewerID", "asin", "reviewerName", "reviewText", "overall", "summary",
    "unixReviewTime"
]
