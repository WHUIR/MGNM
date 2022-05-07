from elasticsearch import Elasticsearch
from tqdm import tqdm


class ESClient():
    def __init__(self):
        self.es = Elasticsearch('172.28.6.8:9200')
        self.index_name = 'se1'
        self.doc_type = 'politics'
        self.mapping = {
            'properties': {
                'title': {
                    'type': 'text',
                    'analyzer': 'whitespace',
                    'search_analyzer': 'whitespace'
                }
            }
        }

    def add_data(self, se_pool, se_user):
        # only run at the first time
        self.es.indices.create(index=self.index_name, ignore=400)
        self.es.indices.put_mapping(index=self.index_name, doc_type=self.doc_type, body=self.mapping)
        ## ************************* ##

        for idx, d in tqdm(enumerate(se_pool)):
            body_tmp = {'user': se_user[idx], 'hist': ' '.join(map(str, d))}
            self.es.index(index=self.index_name, doc_type=self.doc_type, body=body_tmp)
        return

    def delete_index(self, index):
        self.es.indices.delete(index=index, ignore=[400, 404])
        return

    def search(self, query):
        ## query(list)
        dsl = {
            'query': {
                'multi_match': {
                    'query': ' '.join(map(str, query)),
                    'fields': ['hist']
                }
            }
        }
        result = self.es.search(index=self.index_name, doc_type=self.doc_type, body=dsl)
        se_item_set = set()
        for id in result['hits']['hits']:
            print(list(map(int, id['_source']['hist'].split(' '))))
            se_item_set.update(list(map(int, id['_source']['hist'].split(' '))))
        return se_item_set


if __name__ == '__main__':
    se = ESClient()
    res = se.search('1 2 3 4 5')
    #res = se.delete_index('se1')#add_data([[-1], [-2]], [0, 1])
    print(res)


