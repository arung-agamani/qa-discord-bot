from elasticsearch import Elasticsearch

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