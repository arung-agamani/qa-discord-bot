""" 
    13518005 Arung Agamani Budi Putera
    13518036 Fikra Hadi Ramadhan
    13518140 Hansel Grady Daniel Thamrin
"""
from elasticsearch import Elasticsearch

# Definisi dari class ElasticSearchAgent
# Adalah agent yang menjadi Document Retrieval dengan melakukan search
# terhadap koleksi dokumen yang ada pada database
class ElasticSearchAgent():
    def __init__(self, host, port, index_name) -> None:
        self.index_name = index_name
        self.config = {
            'host': host,
            'port': port
        }
        print("Initializing ElasticSearch Agent")
        self.es = Elasticsearch([self.config])
        print("ElasticSearch Initialization Complete")
    
    # Pencaharian akan berdasarkan pertanyaan yang diberikan dengan mengembalikan
    # N buah dokumen yang paling relevan dengan pertanyaan yang diberikan.
    def search(self, question, n_result):
        query = {
            'query': {
                'match': {
                    'content': question
                }
            }
        }
        res = self.es.search(index=self.index_name, body=query, size=n_result)
        return res['hits']['hits']