import unittest
from libbydbot.cli import LibbyInterface
import os

class TestLibbyInterface(unittest.TestCase):

    def setUp(self):
        self.libby = LibbyInterface(collection_name='test_collection')

    def test_initialization_default(self):
        self.assertEqual(self.libby.name, 'Libby D. Bot')
        self.assertEqual(self.libby.languages, ['pt_BR', 'en'])
        self.assertEqual(self.libby.model, 'llama3.2')


    def test_initialization_custom(self):
        custom_libby = LibbyInterface(name='Custom Bot', languages=['en'], model='Gemma', dburl='sqlite:///custom.db')
        self.assertEqual(custom_libby.name, 'Custom Bot')
        self.assertEqual(custom_libby.languages, ['en'])
        self.assertEqual(custom_libby.model, 'gemma3')

    def test_invalid_model(self):
        with self.assertRaises(ValueError):
            LibbyInterface(model='invalid_model')

    def test_default_model(self):
        libby = LibbyInterface()
        self.assertEqual(libby.model, 'llama3.2')  # Default model from config

    def test_embed(self):
        de = self.libby.embed(corpus_path='tests/test_corpus')
        self.assertIsNotNone(de)
        self.assertEqual(de.collection_name, 'test_collection')


    def test_answer(self):
        response = self.libby.answer(question='Who is Moby Dick?', collection_name='test_collection')
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)

    def test_generate_console(self):
        prompt = "Write a haiku about programming"
        response = self.libby.generate(prompt)
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)

    def test_generate_file(self):
        prompt = "Write a haiku about programming"
        test_file = "test_output.txt"
        response = self.libby.generate(prompt, output_file=test_file)
        
        # Check if file was created and contains the response
        self.assertTrue(os.path.exists(test_file))
        with open(test_file, 'r') as f:
            file_content = f.read()
        self.assertEqual(response, file_content)
        
        # Cleanup
        os.remove(test_file)

if __name__ == '__main__':
    unittest.main()
