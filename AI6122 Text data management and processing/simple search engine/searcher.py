import os
import sys
from datetime import datetime

import lucene
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import FloatPoint, IntPoint, LongPoint
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
# from org.apache.lucene.document import FloatRangeSlowRangeQuery
from org.apache.lucene.search import (IndexSearcher, PointRangeQuery, Sort,
                                      SortField, TopFieldCollector)
from org.apache.lucene.store import NIOFSDirectory
from org.apache.lucene.util import Version

import analyzers

default_index_path = "index"
default_analyzer = StandardAnalyzer
lucene.initVM()


class Searcher:

    def __init__(self,
                 index_path=default_index_path,
                 analyzer=StandardAnalyzer(),
                 max_retrieve=50):
        self.fsDir = NIOFSDirectory(Paths.get(index_path))
        self.searcher = IndexSearcher(DirectoryReader.open(self.fsDir))
        self.analyzer = analyzer
        self.max_retrieve = max_retrieve
        self.analyzer_mapper = analyzers.FieldAnalyzerMapper(self.analyzer)

        self.numeric_fields = {
            "overall": FloatPoint,
            "unixReviewTime": LongPoint
        }
        self.numeric_mappers = {"overall": float, "unixReviewTime": int}

        # self.sorter = Sort([
        #     SortField.FIELD_SCORE,
        #     SortField('origin_score', SortField.Type.FLOAT, True),
        #     SortField('en_sent_lenth', SortField.Type.INT, True)
        # ])
        # self.collector = TopFieldCollector.create(self.sorter,
        #                                           self.max_retrieve,
        #                                           self.max_retrieve)

    def search(self, query, field="reviewText", default_operator="and"):
        # This parser actully cannot handle numeric queries about "overall" and "unixReviewTime" field,
        # because for some reason, QueryParser uses TermRangeQuery for range queries,
        # which can only handle string comparisons
        parser = QueryParser(
            field, analyzers.PerFieldAnalyzerWrapperFactory(default_analyzer)
        )  # the `field` here is merely the default field
        if default_operator == "and":
            parser.setDefaultOperator(QueryParser.Operator.AND)
        elif default_operator == "or":
            parser.setDefaultOperator(QueryParser.Operator.OR)
        query = parser.parse(query)
        # print(query)

        scoreDocs = self.searcher.search(query, self.max_retrieve).scoreDocs
        # self.searcher.search(query, self.collector)
        # scoreDocs = self.collector.topDocs().scoreDocs
        return scoreDocs

    def range_search(self, min, max, field="overall"):
        r'''
        The function to handle numeric queries
        '''
        if min == "*":
            min = "-1"
        min = self.numeric_mappers[field](min)

        if max == "*":
            max = 9999999999
        max = self.numeric_mappers[field](max)

        query = self.numeric_fields[field].newRangeQuery(f"{field}_", min, max)
        scoreDocs = self.searcher.search(query, self.max_retrieve).scoreDocs
        # self.searcher.search(query, self.collector)
        # scoreDocs = self.collector.topDocs().scoreDocs
        return scoreDocs


def print_doc(doc):
    print("-" * 7)
    for field in analyzers.field_names:
        print(f"<{field}>: {doc[field]}")
    print("-" * 7)

def print_help():
    print('''Query grammar: 
    Query  ::= ( Clause )*
    Clause ::= ["+", "-"] [<TERM> ":"] ( <TERM> | "(" Query ")" )
eg.: `+field:keyword1 keyword2 -field:"a phrase"` 
\033[33mFeatures supported:\033[0m
    wildcard query(`?` and `*`) is possible, just should not be at the start of a word
    regexp query: `/RegExp/`
    fuzzy query: `fuzzy~2`, default edit distance is 2, modifiable
    numeric query: for exact query, use it like it's a keyword;
        for range query, use this: `field:{start TO end]`,
        where `{}` means exclusive, `[]` means inclusive,
\033[33mFields supported:\033[0m
    "reviewerID": the id of the reviewer, identical match only
    "asin": the id of the product, identical match only
    "reviewerName": the name of the reviewer
    "reviewText": the review given by the reviewer
    "overall": the rate given by the reviewer, RegEx`([0-9]\.[0-9])`
    "summary": a simple summary of the review
    "unixReviewTime": the unix time stamp of the review time, RegEx`([0-9]{10})`
With this program, you can also add `AND` or `OR` at the start of the query, to specify the boolean operators
Please note that the "overall" and "unixReviewTime" field are stored as strings and thus the number of digits must correpond
You can put "NUM" as a query to invoke range search methods with numbers.
\033[33mSimilarly, type `HELP` can bring this tutorial to you again.\033[0m
Default operator is `AND`, and default field is "reviewText".
\033[33mYou can use Crtl+C to exit anytime.\033[0m

eg.: `OR "Kit Kat" chocolate `
eg.: `chocolate cheap overall:{3.0 TO *}`
''')

# -User Interface--------------------
print(f"Welcome to ReviewSearcher.searcher.")

print("Please specify index path, press Enter to use default")
index_path = input()
if index_path == "":
    index_path = default_index_path

print(
    "Please specify how many files to retrieve at max, press Enter to use default 50."
)
max_retrieve = input()
if max_retrieve == "":
    max_retrieve = 50
else:
    max_retrieve = int(max_retrieve)

print(
    "Initializing indices, you can read about the advanced search grammer now:"
)

print_help()

searcher = Searcher(index_path=index_path,
                    analyzer=default_analyzer(),
                    max_retrieve=max_retrieve)
# scoreDocs = searcher.search(r'''reviewText:flavor taste AND reviewText:"Kit Kat"''')
# scoreDocs = searcher.search(r'''unixReviewTime:01370044800''')
# scoreDocs = searcher.search(r'''unixReviewTime:{01370044790 TO 01370044810]''')
# scoreDocs = searcher.range_search(1370044800,1370044800,"unixReviewTime")

print("Initializing completed")
while True:
    query = input("Please give your query:")
    if query == "":
        break
    elif query.startswith("HELP"):
        print_help
        continue
    elif query.startswith("NUM"):
        field = input("On what field would you like to search?")
        min = input("The min value?")
        max = input("The max value?")
        start = datetime.now()
        scoreDocs = searcher.range_search(min, max, field)
        duration = datetime.now() - start
    else:
        if query.startswith("AND"):
            query = query[3:]
            default_operator = "and"
        elif query.startswith("OR"):
            query = query[2:]
            default_operator = "or"
        else:
            default_operator = "and"
        # query = process_query_numeric(query)
        start = datetime.now()
        scoreDocs = searcher.search(query, default_operator=default_operator)
        duration = datetime.now() - start
    print(
        f"\033[33m{len(scoreDocs)} docs have been retrieved in {duration} time.\033[0m"
    )
    for i, hit in enumerate(scoreDocs):
        # score: {hit.score}
        print(f"\033[33mThe No.{i+1} hit, score: {hit.score}, doc id: {hit.doc}\033[0m")
        doc = searcher.searcher.doc(hit.doc)
        print_doc(doc)
        # print(doc.get("reviewText").encode("utf-8"))

print("Thanks for using :)")