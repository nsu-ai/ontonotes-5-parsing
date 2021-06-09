[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/bond005/impartial_text_cls/blob/master/LICENSE)
![Python 3.6, 3.7](https://img.shields.io/badge/python-3.6%20%7C%203.7-green.svg)

# Ontonotes-5-Parsing

**Ontonotes-5-Parsing**: parser of [Ontonotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19) to transform this corpus to a simple JSON format.

[Ontonotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19) is very useful for experiments with NER, i.e. Named Entity Recognition. There are many papers devoted to various NER architectures, and these architectures are checked on Ontonotes 5 (for example, see [Papers With Code](https://paperswithcode.com/sota/named-entity-recognition-ner-on-ontonotes-v5)). Besides, Ontonotes 5 includes three languages (English, Arabic, and Chinese), and this fact increases interest to use it in experiments with multi-lingual NER. But the source format of Ontonotes 5 is very intricate, in my view. Conformably, **the goal of this project** is the creation of a special parser to transform Ontonotes 5 into a simple JSON format. In this format, each annotated sentence is represented as a dictionary with five keys: text, morphology, syntax, entities, and language. In their's turn, morphology, syntax, and entities are specified as dictionaries too, where each dictionary describes labels (part-of-speech labels, syntactical tags, or entity classes) and their bounds in the corresponded text.

## Installing


For installation you need to Python 3.6 or later. To install this project on your local machine, you should run the following commands in the Terminal:

```
git clone https://github.com/nsu-ai/ontonotes-5-parsing.git
cd ontonotes-5-parsing
pip install .
```

If you want to install the **Ontonotes-5-Parsing** into a some virtual environment, than you don't need to use `sudo`, but before installing you have to activate this virtual environment (for example, using `source /path/to/your/python/environment/bin/activate` in the command prompt).

You can also run the tests

```
python setup.py test
```

Also, you can install the **Ontonotes-5-Parsing** from the [PyPi](https://pypi.org/project/ontonotes-5-parsing) using the following command:

```
pip install ontonotes-5-parsing
```

## Usage

### The building of JSON-formatted dataset

**Ontonotes-5-Parsing** can be used as a Python package in your projects after its installing. But the main use case is using as a command-line tool. For transforming source Ontonotes 5 data to the JSON format, you have to run such command:

```shell
ontonotes5_to_json \
    -s /path/to/directory/with/source/ontonotes-release-5.0_LDC2013T19.tgz \
    -d /path/to/directory/with/parsing/result/ontonotes5.json \
    -i /path/to/directory/with/ontonotes/indexing \
    -r 42
```

where:

- `/path/to/directory/with/source/ontonotes-release-5.0_LDC2013T19.tgz` is the path to the source Ontonotes 5 data in `*.tgz` format (it can be downloaded from https://catalog.ldc.upenn.edu/LDC2013T19);

- `/path/to/directory/with/parsing/result/ontonotes5.json` is the path to the JSON file, which will be created as a result of source data parsing;

- `/path/to/directory/with/ontonotes/indexing` is the path to the directory with indexing of source data by subsets for training, development (validation), and testing accordingly paper [Towards Robust Linguistic Analysis using OntoNotes](http://www.aclweb.org/anthology/W13-3516); this content can be download from https://cemantix.org/conll/2012/download/ids/, but if this directory is not specified, then all source data will be parsed as training one without selecting of any subsets for validation and testing.

- `42` is a random seed, but any other integer can be specified as a random seed instead of this value (if random seed is not specified, then system timer is used for its generating).

The above-mentioned TGZ-archive with source data contains many subdirectories and files. But all in all, such facts are significant:

1) all language-specific sub-corpuses of Ontonotes 5 are located in the `ontonotes-release-5/data/files/data` sub-directory (now there are three languages in this sub-directory: Arabic, Chinese, and English);

2) each language sub-directory contains two other subdirectories: `annotations` and `metadata`, but `metadata` is not interesting for us;
   
3) each sample is represented by 8 files: `*.cored`, `*.name`, `*.onf`, `*.parallel`, `*.parse`, `*.prop`, `*.sense`, and `*.speaker`, but only `*.onf` file includes all necessary information about the corresponded sample and its annotation.

You can see a small fragment of generated result in the JSON format below:

```json
{
    "TRAINING": [
        {
            "text": "In the summer of 2005, a picture that people have long been looking forward to started emerging with frequency in various major Hong Kong media.",
            "language": "english",
            "morphology": {
                "IN": [[0, 3], [14, 17], [76, 79], [96, 101], [111, 114]],
                "NN": [[7, 14], [25, 33], [101, 111]],
                "DT": [[3, 7], [23, 25]],
                "CD": [[17, 21]],
                ",": [[21, 23]],
                "WDT": [[33, 38]],
                "NNS": [[38, 45], [138, 143]],
                "VBP": [[45, 50]],
                "RB": [[50, 55], [68, 76]],
                "VBN": [[55, 60]],
                "VBG": [[60, 68], [87, 96]],
                "VBD": [[79, 87]],
                "JJ": [[114, 122], [122, 128]],
                "NNP": [[128, 133], [133, 138]],
                ".": [[143, 144]]
            },
            "entities": {
                "DATE": [[3, 21]],
                "GPE": [[128, 138]]
            },
            "syntax": {
                "PP-TMP": [[0, 21]],
                "NP": [[3, 21], [23, 33], [101, 111], [114, 143]],
                "PP": [[14, 21], [76, 79]],
                "NP-SBJ-2": [[23, 79]],
                "SBAR": [[33, 79]],
                "WHNP-1": [[33, 38]],
                "NP-SBJ": [[38, 45]],
                "VP": [[45, 143]],
                "ADVP-DIR": [[68, 79]],
                "PP-MNR": [[96, 111]],
                "PP-LOC": [[111, 143]],
                "NML": [[128, 138]],
                "ADVP-TMP": [[50, 55]]
            }
        },
        {
            "text": "With their unique charm, these well-known cartoon images once again caused Hong Kong to be a focus of worldwide attention.",
            "language": "english",
            "morphology": {
                "IN": [[0, 5], [99, 102]],
                "PRP$": [[5, 11]],
                "JJ": [[11, 18], [102, 112]],
                "NN": [[18, 23], [42, 50], [93, 99], [112, 121]],
                ",": [[23, 24]],
                ".": [[121, 122]],
                "DT": [[25, 31], [91, 93]],
                "RB": [[31, 35], [57, 62], [62, 68]],
                "HYPH": [[35, 36]],
                "VBN": [[36, 42]],
                "NNS": [[50, 57]],
                "VBD": [[68, 75]],
                "NNP": [[75, 80], [80, 85]],
                "TO": [[85, 88]],
                "VB": [[88, 91]]
            },
            "entities": {
                "GPE": [[75, 85]]
            },
            "syntax": {
                "PP": [[0, 23], [99, 121]],
                "NP": [[5, 23], [91, 99], [102, 121]],
                "NP-SBJ": [[25, 57], [75, 85]],
                "ADJP": [[31, 42]],
                "ADVP-TMP": [[57, 68]],
                "VP": [[68, 121]],
                "NP-PRD": [[91, 121]]
            }
        },
        {
            "text": "و ص ف, رويترز, أب",
            "language": "arabic",
            "morphology": {
                "PUNC": [[5, 6], [13, 14]],
                "ABBREV": [[0, 1], [2, 3], [4, 5], [15, 17]],
                "NOUN_PROP": [[7, 13]]
            },
            "syntax": {"NP": [[0, 17]]},
            "entities": {
                "ORG": [[7, 13]]
            }
        }
    ]
}
```

### The linguistic entities reduction in JSON-formatted dataset

Also, if you want to train a machine learning algorithm which understands morphology or syntax, then we can get some problem with morphological and syntactical annotations in Ontonotes 5: there are many various morphological and syntactical tags in some texts, especially in Arabic, and these tags describe small nuances of linguistics. But this fact highly extends the number of classes if we solve the linguistic analysis problem as a classification task. You can use a special command to reduce the linguistic classes number (this command unites low-frequent linguistic tags with similar ones, which are more frequent):

```shell
reduce_entities \
    -s /path/to/directory/with/parsing/result/ontonotes5.json \
    -d /path/to/directory/with/parsing/result/ontonotes5_reduced.json \
    -n 50
```

where:

- `/path/to/directory/with/parsing/result/ontonotes5.json` is the path to the JSON file with source Ontonotes 5.0 data (this file can be created using the abovementioned `ontonotes5_to_json` command).

- `/path/to/directory/with/parsing/result/ontonotes5_reduced.json` is the path to the analogous JSON file, into which all Ontonotes 5.0 data will be written after linguistic entities reduction.

- `50` is a maximal number of linguistic entity classes (such as part-of-speech tags, syntactical tags in a dependency tree, or named entities), which will be obtained after reduction. This value can be any integer value greater than 2.

### Printing information about parsed Ontonotes 5.0

The abovementioned script for parsing `ontonotes5_to_json` not only parses the specified corpus but shows statistics about Ontonotes 5.0 after parsing ends. But if you want to recollect these statistics a long time after parsing, then you can run a special script for statistics printing:

```shell
show_statistics \
    -s /path/to/directory/with/parsing/result/ontonotes5.json
```

where:

- `/path/to/directory/with/parsing/result/ontonotes5.json` is the path to the JSON file with source Ontonotes 5.0 data (this file can be created using the `ontonotes5_to_json` command and re-built using the `reduce_entities` command, as mentioned above).

## Breaking Changes

**Breaking changes in version 0.0.5**
- clustering of linguistic entities (syntactical tags, PoS tags) has been improved.

**Breaking changes in version 0.0.4**
- tokenization bug for hieroglyphic languages has been fixed.

**Breaking changes in version 0.0.3**
- documentation has been updated.

**Breaking changes in version 0.0.2**
- tokenization bug for Arabic texts has been fixed.

**Breaking changes in version 0.0.1**
- initial (alpha) version of the **Ontonotes-5-Parsing** has been released.

## License

The **Ontonotes-5-Parsing** (`ontonotes-5-parsing`) is Apache 2.0 - licensed.
