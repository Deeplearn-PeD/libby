'''
This module is used to talk to the user. It uses the `piper` text to speech command to do so.
The class Speaker is used to configure the vois and language of the speaker.
The method `say` is used to speak the text.
'''
import subprocess as sp
import shlex
import loguru

logger = loguru.logger

piper_languages = {
    'pt_BR': 'faber-medium',
    'en_US': 'amy-medium'
}

introductions = {
    'pt_BR': [
        'Olá, eu sou a Libby, sua gestora de conhecimento.',
        'Posso ajudá-lo com a integração de conhecimento, provenientede corpora textuais.',

    ],
    'en_US': [
        "Hello, I'm Libby, your knowledge manager.,",
        "I can help you with knowledge integration from textual corpora.",
    ]
}

corpora_questions = {
    'pt_BR': [
        'Me fale sobre o corpus que você deseja integrar.',
        'Me informe o caminho para o corpus:',
    ],
    'en_US': [
        'Tell me about the corpus you wish to integrate.',
        'Inform me the path to the corpus:',
    ]
}


class Speaker:
    def __init__(self, language='pt_BR'):
        try:
            self.voice = piper_languages[language]
        except KeyError:
            logger.warning(f'Language {language} not supported, using en_US instead')
            self.voice = piper_languages['en_US']
        self.language = language
        self.model = language + '-' + self.voice
        self.outfile = '/tmp/speech.wav'

    def say(self, text):
        args = shlex.split(f'piper --model {self.model} --output-file {self.outfile} ')
        sp.Popen(args, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE).communicate(input=text.encode())
        sp.call(shlex.split(f'play {self.outfile}'))