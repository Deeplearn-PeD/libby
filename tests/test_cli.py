import unittest
from libbydbot.cli import LibbyInterface
import os

class TestLibbyInterface(unittest.TestCase):

    def setUp(self):
        self.libby = LibbyInterface()

    def test_initialization_default(self):
        self.assertEqual(self.libby.name, 'Libby D. Bot')
        self.assertEqual(self.libby.languages, ['pt_BR', 'en'])
        self.assertEqual(self.libby.model, 'gpt-4o')


    def test_initialization_custom(self):
        custom_libby = LibbyInterface(name='Custom Bot', languages=['en'], model='gpt-3', dburl='sqlite:///custom.db')
        self.assertEqual(custom_libby.name, 'Custom Bot')
        self.assertEqual(custom_libby.languages, ['en'])
        self.assertEqual(custom_libby.model, 'gpt-3')

    def test_embed(self):
        de = self.libby.embed(corpus_path='test_corpus', collection_name='test_embeddings')
        self.assertIsNotNone(de)
        self.assertEqual(de.collection_name, 'test_embeddings')


    def test_answer(self):
        response = self.libby.answer(question='Who is Moby Dick?', collection_name='test_collection')
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)

if __name__ == '__main__':
    unittest.main()