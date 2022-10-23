import json
import os
import sys
import time
from datetime import datetime

import lucene
from java.nio.file import Paths
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import (Document, Field, FieldType,
                                        FloatDocValuesField, FloatPoint,
                                        IntPoint, LongPoint,
                                        NumericDocValuesField, StoredField,
                                        StringField, TextField)
from org.apache.lucene.index import (IndexOptions, IndexWriter,
                                     IndexWriterConfig, Term)
from org.apache.lucene.store import NIOFSDirectory

import analyzers

original_data_paths = ["Grocery_and_Gourmet_Food_5.json", "Electronics_5.json"]
# original_data_paths = [
#     "/home/lahelr/ReviewSearcher/test_data/TenthDataset.json"
# ]
default_index_path = "index"

lucene.initVM()

default_analyzer = StandardAnalyzer
# default_analyzer = analyzers.CustomAnalyzer([0, 1, 3])


class ObjectIndexBuilder:

    def __init__(self, index_path: str = default_index_path, analyzer=None):
        self.directory = NIOFSDirectory(Paths.get(index_path))
        if analyzer is None:
            # StandardTokenizer, LowerCaseFilter, StopFilter
            # removes stop words and lowercases the generated tokens
            self.analyzer = StandardAnalyzer()
        else:
            self.analyzer = analyzer
        self.config = IndexWriterConfig(self.analyzer)
        self.writer = IndexWriter(self.directory, self.config)

        # self.fieldType = FieldType();
        # self.fieldType.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS);
        # self.fieldType.setStored(True)
        # self.fieldType.setTokenized(True)
        # self.fieldType.freeze()

    def build_index_for_obj(self, obj):
        r'''eg.:
        {"reviewerID": "A1VEELTKS8NLZB",
        "asin": "616719923X",
        "reviewerName": "Amazon Customer",
        "helpful": [0, 0],
        "reviewText": "Just another flavor of Kit Kat but the taste is unique and a bit different.  The only thing that is bothersome is the price.  I thought it was a bit expensive....",
        "overall": 4.0,
        "summary": "Good Taste",
        "unixReviewTime": 1370044800,
        "reviewTime": "06 1, 2013"}
        '''
        r'''
        String indexing ref: https://stackoverflow.com/questions/21347377/how-to-index-a-string-in-lucene
        String matching ref: https://blog.csdn.net/mbgmbg/article/details/4310181
        '''

        def float_format_str(num: float):
            return f"{float(num):0=1.1f}"

        def int_format_str(num: int):
            return f"{int(num):0=10}"

        doc = Document()
        try:
            doc.add(
                StringField("reviewerID", obj["reviewerID"], Field.Store.YES))
        except KeyError:
            pass
        try:
            doc.add(StringField("asin", obj["asin"], Field.Store.YES))
        except KeyError:
            pass
        try:
            doc.add(
                Field("reviewerName", obj["reviewerName"],
                      TextField.TYPE_STORED))
        except KeyError:
            pass
        try:
            doc.add(
                Field("reviewText", obj["reviewText"], TextField.TYPE_STORED))
        except KeyError:
            pass
        try:
            doc.add(FloatPoint("overall_", float(obj["overall"])))
        except KeyError:
            pass
        try:
            doc.add(
                StringField("overall", float_format_str(obj["overall"]),
                            Field.Store.YES))
        except KeyError:
            pass
        try:
            doc.add(Field("summary", obj["summary"], TextField.TYPE_STORED))
        except KeyError:
            pass
        try:
            doc.add(LongPoint("unixReviewTime_", obj["unixReviewTime"]))
        except KeyError:
            pass
        try:
            doc.add(
                StringField("unixReviewTime",
                            int_format_str(obj["unixReviewTime"]),
                            Field.Store.YES))
        except KeyError:
            pass

        self.writer.addDocument(doc)

    def __call__(self, obj):
        self.build_index_for_obj(obj)

    def close(self):
        self.writer.flush()
        self.writer.commit()
        self.writer.close()


special_analyzer = analyzers.PerFieldAnalyzerWrapperFactory(
    default_analyzer=default_analyzer)
index_builder = ObjectIndexBuilder(analyzer=special_analyzer)

start = datetime.now()
for data_file in original_data_paths:
    with open(data_file, "r") as f:
        while True:
            if os.path.getsize(data_file) == f.tell():
                break
            obj = json.loads(f.readline())
            index_builder.build_index_for_obj(obj)

index_builder.close()
duration = datetime.now() - start
print(
    f"ReviewSearcher.buildIndex: Index for {original_data_paths} built at {default_index_path} in {duration} time."
)
