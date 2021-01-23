from setuptools import setup, find_packages

import ontonotes5


long_description = '''
Ontonotes-5-Parsing
===================

A simple parser of the famous Ontonotes 5 dataset
https://catalog.ldc.upenn.edu/LDC2013T19

This dataset is very useful for experiments with NER, i.e. Named Entity
Recognition. Besides, Ontonotes 5 includes three languages (English,
Arabic, and Chinese), and this fact increases interest to use it in
experiments with multi-lingual NER. But the source format of Ontonotes 5
is very intricate, in my view. Conformably, the goal of this project is
the creation of a special parser to transform Ontonotes 5 into a simple
JSON format. In this format, each annotated sentence is represented as
a dictionary with five keys: text, morphology, syntax, entities, and
language. In their's turn, morphology, syntax, and entities are
specified as dictionaries too, where each dictionary describes labels
(part-of-speech labels, syntactical tags, or entity classes) and their
bounds in the corresponded text.
'''

setup(
    name='ontonotes-5-parsing',
    version=ontonotes5.__version__,
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    description='Ontonotes-5-parsing: parser of Ontonotes 5.0 '
                'to transform this corpus to a simple JSON format.',
    long_description=long_description,
    url='https://github.com/nsu-ai/ontonotes-5-parsing',
    author='Ivan Bondarenko',
    author_email='i.bondarenko@g.nsu.ru',
    license='Apache License Version 2.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Linguistic',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords=['ontonotes', 'ontonotes5', 'ontonotes-5', 'ner', 'nlp',
              'multi-lingual'],
    install_requires=['tqdm', 'numpy'],
    test_suite='tests',
    entry_points={
        'console_scripts': ['ontonotes5_to_json = ontonotes5_to_json:main']
    }
)
