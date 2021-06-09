import codecs
import json
import os
import random
import re
import unittest

from ontonotes5.utils import get_plain_text, strip_bounds
from ontonotes5.utils import get_token_bounds_fuzzy, get_token_bounds
from ontonotes5.utils import parse_tree, parse_named_entities_labeling
from ontonotes5.utils import megre_bounds, parse_file
from ontonotes5.utils import split_filename_by_parts, check_onf_name
from ontonotes5.utils import parse_splitting
from ontonotes5.utils import find_subword_bounds
from ontonotes5.utils import get_language_by_filename
from ontonotes5.utils import is_item_in_sequence
from ontonotes5.utils import unite_overlapped_bounds, check_bounds
from ontonotes5.utils import insert_new_bounds, calculate_distance
from ontonotes5.utils import tokenize_any_text


class TestUtils(unittest.TestCase):
    identifiers_for_testing = dict()

    @classmethod
    def setUpClass(cls) -> None:
        random.seed(42)
        cls.identifiers_for_testing = {
            'test': sorted(list(map(
                lambda it1: os.path.join(*it1),
                [
                    ('data', 'english', 'annotations', 'bc', 'cctv', '00',
                     'cctv_0005'),
                    ('data', 'english', 'annotations', 'bc', 'cnn', '00',
                     'cnn_0008'),
                    ('data', 'english', 'annotations', 'bc', 'msnbc', '00',
                     'msnbc_0007')
                ]
            ))),
            'development': sorted(list(map(
                lambda it2: os.path.join(*it2),
                [
                    ('data', 'english', 'annotations', 'wb', 'sel', '63',
                     'sel_6380'),
                    ('data', 'english', 'annotations', 'pt', 'nt', '43',
                     'nt_4320')
                ]
            ))),
            'train': sorted(list(map(
                lambda it3: os.path.join(*it3),
                [
                    ('data', 'english', 'annotations', 'bc', 'p2.5_a2e',
                     '00', 'p2.5_a2e_0006'),
                    ('data', 'english', 'annotations', 'bc', 'cnn', '00',
                     'cnn_0007'),
                    ('data', 'english', 'annotations', 'bn', 'cnn', '01',
                     'cnn_0144')
                ]
            )))
        }

    def test_get_plain_text_pos01(self):
        source_text = ['123 fkj 4fkl 2']
        true_text = '123 fkj 4fkl 2'
        start_idx = 0
        end_idx = 1
        predicted = get_plain_text(source_text, start_idx, end_idx)
        self.assertEqual(true_text, predicted)

    def test_get_plain_text_pos02(self):
        source_text = ['123 fkj 4fkl 2', ' k/fs klf; 89p3k ka\';',
                       'pslok\' kyhj7-0', 'dfd ']
        true_text = '123 fkj 4fkl 2 k/fs klf; 89p3k ka\'; pslok\' kyhj7-0 dfd'
        start_idx = 0
        end_idx = 4
        predicted = get_plain_text(source_text, start_idx, end_idx)
        self.assertEqual(true_text, predicted)

    def test_get_plain_text_pos03(self):
        source_text = ['123 fkj 4fkl 2', ' k/fs klf; 89p3k ka\';',
                       'pslok\' kyhj7-0', 'dfd ']
        true_text = 'k/fs klf; 89p3k ka\'; pslok\' kyhj7-0'
        start_idx = 1
        end_idx = 3
        predicted = get_plain_text(source_text, start_idx, end_idx)
        self.assertEqual(true_text, predicted)

    def test_get_plain_text_pos04(self):
        source_text = ['و ص ف , رُويْتِرز , أب']
        true_text = 'و ص ف , رُويْتِرز , أب'
        start_idx = 0
        end_idx = 1
        predicted = get_plain_text(source_text, start_idx, end_idx)
        self.assertEqual(true_text, predicted)

    def test_strip_bounds_pos01(self):
        text = '123 fkj 4fkl'
        source_bounds = [(0, 4), (4, 8), (8, 12)]
        true_bounds = [(0, 3), (4, 7), (8, 12)]
        calc_bounds = strip_bounds(text, source_bounds)
        self.assertEqual(true_bounds, calc_bounds)

    def test_strip_bounds_pos02(self):
        text = '123 fkj 4fkl'
        source_bounds = [(0, 5), (5, 8), (8, 12)]
        true_bounds = [(0, 5), (5, 7), (8, 12)]
        calc_bounds = strip_bounds(text, source_bounds)
        self.assertEqual(true_bounds, calc_bounds)

    def test_strip_bounds_pos03(self):
        text = '123 fkj 4fkl'
        source_bounds = [(0, 3), (4, 7), (8, 12)]
        true_bounds = [(0, 3), (4, 7), (8, 12)]
        calc_bounds = strip_bounds(text, source_bounds)
        self.assertEqual(true_bounds, calc_bounds)

    def test_strip_bounds_pos04(self):
        text = '123, fkj-4fkl '
        source_bounds = [(0, 3), (3, 5), (5, 8), (8, 9), (9, 14)]
        true_bounds = [(0, 3), (3, 4), (5, 8), (8, 9), (9, 13)]
        calc_bounds = strip_bounds(text, source_bounds)
        self.assertEqual(true_bounds, calc_bounds)

    def test_strip_bounds_pos05(self):
        text = '123,fkj-4fkl'
        source_bounds = [(0, 3), (3, 4), (4, 7), (7, 8), (8, 12)]
        true_bounds = [(0, 3), (3, 4), (4, 7), (7, 8), (8, 12)]
        calc_bounds = strip_bounds(text, source_bounds)
        self.assertEqual(true_bounds, calc_bounds)

    def test_strip_bounds_pos06(self):
        text = '123 fkj 4fkl'
        source_bounds = [(0, 3), (3, 7), (8, 12)]
        true_bounds = [(0, 3), (4, 7), (8, 12)]
        calc_bounds = strip_bounds(text, source_bounds)
        self.assertEqual(true_bounds, calc_bounds)

    def test_parse_tree_pos01(self):
        source_tree = ''
        true_res = []
        predicted = parse_tree(source_tree)
        self.assertEqual(true_res, predicted)

    def test_parse_tree_pos02(self):
        source_tree = '(DT the)'
        true_res = [('the', ['DT'])]
        predicted = parse_tree(tree=source_tree)
        self.assertEqual(true_res, predicted)

    def test_parse_tree_pos03(self):
        source_tree = '(PP (IN by) (NP-LGS (DT the) (JJ Israeli) (NNP Army)))'
        true_res = [('by', ['PP', 'IN']), ('the', ['PP', 'NP-LGS', 'DT']),
                    ('Israeli', ['PP', 'NP-LGS', 'JJ']),
                    ('Army', ['PP', 'NP-LGS', 'NNP'])]
        predicted = parse_tree(tree=source_tree)
        self.assertEqual(true_res, predicted)

    def test_parse_tree_pos04(self):
        source_tree = '(TOP (S (PP (IN With) (NP (PRP$ their) (JJ unique) ' \
                      '(NN charm))) (, ,) (NP-SBJ (DT these) (ADJP (RB well) ' \
                      '(HYPH -) (VBN known)) (NN cartoon) (NNS images)) ' \
                      '(ADVP-TMP (RB once) (RB again)) (VP (VBD caused) ' \
                      '(S (NP-SBJ (NNP Hong) (NNP Kong)) (VP (TO to) ' \
                      '(VP (VB be) (NP-PRD (NP (DT a) (NN focus)) ' \
                      '(PP (IN of) (NP (JJ worldwide) (NN attention)))))))) ' \
                      '(. .)))'
        true_res = [
            ('With', ['TOP', 'S', 'PP', 'IN']),
            ('their', ['TOP', 'S', 'PP', 'NP', 'PRP$']),
            ('unique', ['TOP', 'S', 'PP', 'NP', 'JJ']),
            ('charm', ['TOP', 'S', 'PP', 'NP', 'NN']),
            (',', ['TOP', 'S', ',']),
            ('these', ['TOP', 'S', 'NP-SBJ', 'DT']),
            ('well', ['TOP', 'S', 'NP-SBJ', 'ADJP', 'RB']),
            ('-', ['TOP', 'S', 'NP-SBJ', 'ADJP', 'HYPH']),
            ('known', ['TOP', 'S', 'NP-SBJ', 'ADJP', 'VBN']),
            ('cartoon', ['TOP', 'S', 'NP-SBJ', 'NN']),
            ('images', ['TOP', 'S', 'NP-SBJ', 'NNS']),
            ('once', ['TOP', 'S', 'ADVP-TMP', 'RB']),
            ('again', ['TOP', 'S', 'ADVP-TMP', 'RB']),
            ('caused', ['TOP', 'S', 'VP', 'VBD']),
            ('Hong', ['TOP', 'S', 'VP', 'S', 'NP-SBJ', 'NNP']),
            ('Kong', ['TOP', 'S', 'VP', 'S', 'NP-SBJ', 'NNP']),
            ('to', ['TOP', 'S', 'VP', 'S', 'VP', 'TO']),
            ('be', ['TOP', 'S', 'VP', 'S', 'VP', 'VP', 'VB']),
            ('a', ['TOP', 'S', 'VP', 'S', 'VP', 'VP', 'NP-PRD', 'NP', 'DT']),
            ('focus', ['TOP', 'S', 'VP', 'S', 'VP', 'VP', 'NP-PRD', 'NP',
                       'NN']),
            ('of', ['TOP', 'S', 'VP', 'S', 'VP', 'VP', 'NP-PRD', 'PP', 'IN']),
            ('worldwide', ['TOP', 'S', 'VP', 'S', 'VP', 'VP', 'NP-PRD', 'PP',
                           'NP', 'JJ']),
            ('attention', ['TOP', 'S', 'VP', 'S', 'VP', 'VP', 'NP-PRD', 'PP',
                           'NP', 'NN']),
            ('.', ['TOP', 'S', '.'])
        ]
        predicted = parse_tree(tree=source_tree)
        self.assertEqual(true_res, predicted)

    def test_parse_named_entities_labeling_pos01(self):
        source_text = [
            '    0   Protesting',
            '           prop:  protest.01',
            '            v          * -> 0:0,  Protesting',
            '            ARG0       * -> 1:0,  Palestinians',
            '    1   Palestinians',
            '           name:  NORP               1-1    Palestinians',
            '    2   are',
            '           prop:  be.03',
            '            v          * -> 2:0,  are',
            '    3   being',
            '           prop:  be.03',
            '            v          * -> 3:0,  being',
            '    4   met',
            '           sense: meet-v.5',
            '           prop:  meet.03',
            '            v          * -> 4:0,  met',
            '            ARG1       * -> 5:0,  *-1 -> 0:1, Protesting '
            'Palestinians',
            '            ARGM-MNR   * -> 6:1,  with heavier firepower',
            '            ARG0       * -> 9:1,  by the Israeli Army',
            '            ARGM-TMP   * -> 13:1, as violence in the West Bank '
            'and Gaza Strip '
            'escalates',
            '    5   *-1',
            '    6   with',
            '    7   heavier',
            '    8   firepower',
            '    9   by',
            '    10  the',
            '           coref: IDENT        3     10-12  the Israeli Army',
            '           name:  ORG                10-12  the Israeli Army',
            '    11  Israeli',
            '    12  Army',
            '    13  as',
            '    14  violence',
            '    15  in',
            '    16  the',
            '    17  West',
            '           coref: IDENT        28    17-18  West Bank',
            '           name:  GPE                17-18  West Bank',
            '    18  Bank',
            '    19  and',
            '    20  Gaza',
            '           coref: IDENT        29    20-21  Gaza Strip',
            '           name:  GPE                20-21  Gaza Strip',
            '    21  Strip',
            '    22  escalates',
            '           sense: escalate-v.1',
            '           prop:  escalate.01',
            '            v          * -> 22:0, escalates',
            '            ARG1       * -> 14:2, violence in the West Bank and '
            'Gaza Strip',
            '    23  ,',
            '    24  and',
            '    25  as',
            '    26  the',
            '    27  tension',
            '    28  grows',
            '           sense: grow-v.4',
            '           prop:  grow.01',
            '            v          * -> 28:0, grows',
            '            ARG1       * -> 26:1, the tension',
            '    29  ,',
            '    30  so',
            '    31  does',
            '           prop:  do.01',
            '            v          * -> 31:0, does',
            '    32  *T*-2',
            '    33  the',
            '           coref: IDENT        7     33-35  the death toll',
            '    34  death',
            '           sense: death-n.1',
            '    35  toll',
            '           sense: toll-n.2',
            '    36  .'
        ]
        true_tokens = [
            'Protesting',
            'Palestinians',
            'are',
            'being',
            'met',
            '*-1',
            'with',
            'heavier',
            'firepower',
            'by',
            'the',
            'Israeli',
            'Army',
            'as',
            'violence',
            'in',
            'the',
            'West',
            'Bank',
            'and',
            'Gaza',
            'Strip',
            'escalates',
            ',',
            'and',
            'as',
            'the',
            'tension',
            'grows',
            ',',
            'so',
            'does',
            '*T*-2',
            'the',
            'death',
            'toll',
            '.'
        ]
        true_res = [
            'O',
            'B-NORP',
            'O',
            'O',
            'O',
            'O',
            'O',
            'O',
            'O',
            'O',
            'B-ORG',
            'I-ORG',
            'I-ORG',
            'O',
            'O',
            'O',
            'O',
            'B-GPE',
            'I-GPE',
            'O',
            'B-GPE',
            'I-GPE',
            'O',
            'O',
            'O',
            'O',
            'O',
            'O',
            'O',
            'O',
            'O',
            'O',
            'O',
            'O',
            'O',
            'O',
            'O'
        ]
        calc_res = parse_named_entities_labeling(source_text, true_tokens)
        self.assertEqual(true_res, calc_res)

    def test_parse_named_entities_labeling_pos02(self):
        source_text = [
            '    0   Hello',
            '    1   ,',
            '    2   world',
            '    3   !'
        ]
        true_tokens = [
            'Hello',
            ',',
            'world',
            '!'
        ]
        true_res = [
            'O',
            'O',
            'O',
            'O'
        ]
        calc_res = parse_named_entities_labeling(source_text, true_tokens)
        self.assertEqual(true_res, calc_res)

    def test_parse_named_entities_labeling_pos03(self):
        source_text = [
            '0 Protesting',
            'prop: protest.01',
            'v * -> 0:0, Protesting',
            'ARG0 * -> 1:0, Palestinians',
            '1 Palestinians',
            'name: NORP 1-1 Palestinians',
            '2 are',
            'prop: be.03',
            'v * -> 2:0, are',
            '3 being',
            'prop: be.03',
            'v * -> 3:0, being',
            '4 met',
            'sense: meet-v.5',
            'prop: meet.03',
            'v * -> 4:0, met',
            'ARG1 * -> 5:0, *-1 -> 0:1, Protesting Palestinians',
            'ARGM-MNR * -> 6:1, with heavier firepower',
            'ARG0 * -> 9:1, by the Israeli Army',
            'ARGM-TMP * -> 13:1, as violence in the West Bank and Gaza Strip '
            'escalates',
            '5 *-1',
            '6 with',
            '7 heavier',
            '8 firepower',
            '9 by',
            '10 the',
            'coref: IDENT 3 10-12 the Israeli Army',
            'name: ORG 10-12 the Israeli Army',
            '11 Israeli',
            '12 Army',
            '13 as',
            '14 violence',
            '15 in',
            '16 the',
            '17 West',
            'coref: IDENT 28 17-18 West Bank',
            'name: GPE 17-18 West Bank',
            '18 Bank',
            '19 and',
            '20 Gaza',
            'coref: IDENT 29 20-21 Gaza Strip',
            'name: GPE 20-21 Gaza Strip',
            '21 Strip',
            '22 escalates',
            'sense: escalate-v.1',
            'prop: escalate.01',
            'v * -> 22:0, escalates',
            'ARG1 * -> 14:2, violence in the West Bank and Gaza Strip',
            '23 ,',
            '24 and',
            '25 as',
            '26 the',
            '27 tension',
            '28 grows',
            'sense: grow-v.4',
            'prop: grow.01',
            'v * -> 28:0, grows',
            'ARG1 * -> 26:1, the tension',
            '29 ,',
            '30 so',
            '31 does',
            'prop: do.01',
            'v * -> 31:0, does',
            '32 *T*-2',
            '33 the',
            'coref: IDENT 7 33-35 the death toll',
            '34 death',
            'sense: death-n.1',
            '35 toll',
            'sense: toll-n.2',
            '36 .'
        ]
        true_tokens = [
            'Protesting',
            'Palestinians',
            'are',
            'being',
            'met',
            '*-1',
            'with',
            'heavier',
            'firepower',
            'by',
            'the',
            'Israeli',
            'Army',
            'as',
            'violence',
            'in',
            'the',
            'West',
            'Bank',
            'and',
            'Gaza',
            'Strip',
            'escalates',
            ',',
            'and',
            'as',
            'the',
            'tension',
            'grows',
            ',',
            'so',
            'does',
            '*T*-2',
            'the',
            'death',
            'toll',
            '.'
        ]
        true_res = [
            'O',
            'B-NORP',
            'O',
            'O',
            'O',
            'O',
            'O',
            'O',
            'O',
            'O',
            'B-ORG',
            'I-ORG',
            'I-ORG',
            'O',
            'O',
            'O',
            'O',
            'B-GPE',
            'I-GPE',
            'O',
            'B-GPE',
            'I-GPE',
            'O',
            'O',
            'O',
            'O',
            'O',
            'O',
            'O',
            'O',
            'O',
            'O',
            'O',
            'O',
            'O',
            'O',
            'O'
        ]
        calc_res = parse_named_entities_labeling(source_text, true_tokens)
        self.assertEqual(true_res, calc_res)

    def test_megre_bounds_pos01(self):
        src = [(68, 121), (85, 121), (88, 121)]
        true_dst = [(68, 121)]
        self.assertEqual(true_dst, megre_bounds(src))

    def test_megre_bounds_pos02(self):
        src = [(0, 23), (68, 121), (85, 121), (88, 121), (121, 122)]
        true_dst = [(0, 23), (68, 121), (121, 122)]
        self.assertEqual(true_dst, megre_bounds(src))

    def test_megre_bounds_pos03(self):
        src = [(18, 23), (42, 50), (93, 99), (112, 121)]
        true_dst = [(18, 23), (42, 50), (93, 99), (112, 121)]
        self.assertEqual(true_dst, megre_bounds(src))

    def test_parse_file(self):
        src_file_name = os.path.join(os.path.dirname(__file__), 'data',
                                     'sample_of_data.onf')
        dst_file_name = os.path.join(os.path.dirname(__file__), 'data',
                                     'sample_of_res.json')
        global_true_keys = {'text', 'morphology', 'entities', 'syntax'}
        with codecs.open(dst_file_name, mode='r', encoding='utf-8',
                         errors='ignore') as fp:
            true_data = json.load(fp)
        calculated_data, err_msg = parse_file(src_file_name)
        self.assertEqual(err_msg, '')
        self.assertIsInstance(calculated_data, list)
        self.assertEqual(len(true_data), len(calculated_data))
        for idx, (cur_calc, cur_true) in enumerate(zip(calculated_data,
                                                       true_data)):
            err_msg = 'Sample {0}: sample is not a dictionary.'.format(idx)
            self.assertIsInstance(cur_calc, dict, msg=err_msg)
            err_msg = 'Sample {0}: {1} != {2}'.format(
                idx, set(cur_calc.keys()), global_true_keys
            )
            self.assertEqual(set(cur_calc.keys()), global_true_keys,
                             msg=err_msg)
            err_msg = 'Sample {0}: text differs from target one.'.format(idx)
            self.assertEqual(cur_true['text'], cur_calc['text'], msg=err_msg)
            for part in ['morphology', 'syntax', 'entities']:
                err_msg = 'Sample {0}, part "{1}": information about this ' \
                          'part is not a dictionary.'.format(idx, part)
                self.assertIsInstance(cur_calc[part], dict, msg=err_msg)
                true_keys = sorted(list(cur_true[part].keys()))
                calculated_keys = sorted(list(cur_calc[part].keys()))
                err_msg = 'Sample {0}, part "{1}": {2} != {3}'.format(
                    idx, part, true_keys, calculated_keys
                )
                self.assertEqual(true_keys, calculated_keys, msg=err_msg)
                for tag in cur_calc[part]:
                    true_bounds_list = strip_bounds(cur_true['text'],
                                                    cur_true[part][tag])
                    err_msg = 'Sample {0}, part "{1}", tag {2}: information ' \
                              'about bounds is not a list.'.format(
                        idx, part, tag
                    )
                    self.assertIsInstance(cur_calc[part][tag], list,
                                          msg=err_msg)
                    err_msg = 'Sample {0}, part "{1}", tag {2}: ' \
                              '{3} != {4}. {5}'.format(
                        idx, part, tag, len(cur_calc[part][tag]),
                        len(true_bounds_list), cur_calc[part][tag]
                    )
                    self.assertEqual(len(cur_calc[part][tag]),
                                     len(true_bounds_list),
                                     msg=err_msg)
                    for t, (calc_bounds, true_bounds) in enumerate(zip(
                            cur_calc[part][tag], true_bounds_list
                    )):
                        err_msg = 'Sample {0}, part "{1}", tag {2}, item {3}:' \
                                  ' bounds are not a tuple.'.format(
                            idx, part, tag, t
                        )
                        self.assertIsInstance(calc_bounds, tuple, msg=err_msg)
                        err_msg = 'Sample {0}, part "{1}", tag {2}, item {3}:' \
                                  ' {4} != 2.'.format(
                            idx, part, tag, t, len(calc_bounds)
                        )
                        self.assertEqual(len(calc_bounds), 2, msg=err_msg)
                        err_msg = 'Sample {0}, part "{1}", tag {2}, item {3}:' \
                                  ' {4} != {5}.'.format(
                            idx, part, tag, t, true_bounds, calc_bounds
                        )
                        self.assertEqual(true_bounds, calc_bounds, msg=err_msg)

    def test_split_filename_by_parts_pos01(self):
        source_name = 'abc'
        true_parts = ['abc']
        self.assertEqual(true_parts, split_filename_by_parts(source_name))

    def test_split_filename_by_parts_pos02(self):
        source_name = 'abc.txt'
        true_parts = ['abc.txt']
        self.assertEqual(true_parts, split_filename_by_parts(source_name))

    def test_split_filename_by_parts_pos03(self):
        source_name = os.path.join('hahaha', 'abc.txt')
        true_parts = ['hahaha', 'abc.txt']
        self.assertEqual(true_parts, split_filename_by_parts(source_name))

    def test_split_filename_by_parts_pos04(self):
        source_name = os.path.join('123', 'hahaha', 'abc.txt')
        true_parts = ['123', 'hahaha', 'abc.txt']
        self.assertEqual(true_parts, split_filename_by_parts(source_name))

    def test_check_onf_name_pos01(self):
        source_name = os.path.join(
            'ontonotes-release-5.0', 'data', 'files', 'data', 'english',
            'annotations', 'bn', 'cnn', '01', 'cnn_0144.onf'
        )
        true_key = 'train'
        found_key = check_onf_name(source_name, self.identifiers_for_testing)
        self.assertEqual(true_key, found_key)

    def test_check_onf_name_pos02(self):
        source_name = os.path.join(
            'ontonotes-release-5.0', 'data', 'files', 'data', 'english',
            'annotations', 'pt', 'nt', '43', 'nt_4320.onf'
        )
        true_key = 'development'
        found_key = check_onf_name(source_name, self.identifiers_for_testing)
        self.assertEqual(true_key, found_key)

    def test_check_onf_name_pos03(self):
        source_name = os.path.join(
            'ontonotes-release-5.0', 'data', 'files', 'data', 'english',
            'annotations', 'bc', 'msnbc', '00', 'msnbc_0007.onf'
        )
        true_key = 'test'
        found_key = check_onf_name(source_name, self.identifiers_for_testing)
        self.assertEqual(true_key, found_key)

    def test_check_onf_name_neg01(self):
        source_name = os.path.join(
            'ontonotes-release-5.0', 'data', 'files', 'data', 'arabic',
            'annotations', 'nw', 'ann', '00', 'ann_0001.onf'
        )
        with self.assertRaises(ValueError):
            _ = check_onf_name(source_name, self.identifiers_for_testing)

    def test_parse_splitting(self):
        src_dir_name = os.path.join(os.path.dirname(__file__), 'data',
                                    'identifiers')
        loaded = parse_splitting(src_dir_name)
        self.assertIsInstance(loaded, dict)
        true_keys = ['train', 'test', 'development']
        self.assertEqual(set(true_keys), set(loaded.keys()))
        for cur in true_keys:
            self.assertIsInstance(loaded[cur], list)
            loaded_names = sorted(loaded[cur])
            self.assertEqual(self.identifiers_for_testing[cur], loaded_names)

    def test_get_token_bounds_pos01(self):
        source_text = 'With their unique charm, these well-known cartoon ' \
                      'images once again caused Hong Kong to be a focus of ' \
                      'worldwide attention.'
        tokenized = [
            ('With', ['TOP', 'S', 'PP', 'IN'], 'O'),
            ('their', ['TOP', 'S', 'PP', 'NP', 'PRP$'], 'O'),
            ('unique', ['TOP', 'S', 'PP', 'NP', 'JJ'], 'O'),
            ('charm', ['TOP', 'S', 'PP', 'NP', 'NN'], 'O'),
            (',', ['TOP', 'S', ','], 'O'),
            ('these', ['TOP', 'S', 'NP-SBJ', 'DT'], 'O'),
            ('well', ['TOP', 'S', 'NP-SBJ', 'ADJP', 'RB'], 'O'),
            ('-', ['TOP', 'S', 'NP-SBJ', 'ADJP', 'HYPH'], 'O'),
            ('known', ['TOP', 'S', 'NP-SBJ', 'ADJP', 'VBN'], 'O'),
            ('cartoon', ['TOP', 'S', 'NP-SBJ', 'NN'], 'O'),
            ('images', ['TOP', 'S', 'NP-SBJ', 'NNS'], 'O'),
            ('once', ['TOP', 'S', 'ADVP-TMP', 'RB'], 'O'),
            ('again', ['TOP', 'S', 'ADVP-TMP', 'RB'], 'O'),
            ('caused', ['TOP', 'S', 'VP', 'VBD'], 'O'),
            ('Hong', ['TOP', 'S', 'VP', 'S', 'NP-SBJ', 'NNP'], 'B-GPE'),
            ('Kong', ['TOP', 'S', 'VP', 'S', 'NP-SBJ', 'NNP'], 'I-GPE'),
            ('to', ['TOP', 'S', 'VP', 'S', 'VP', 'TO'], 'O'),
            ('be', ['TOP', 'S', 'VP', 'S', 'VP', 'VP', 'VB'], 'O'),
            ('a', ['TOP', 'S', 'VP', 'S', 'VP', 'VP', 'NP-PRD', 'NP', 'DT'],
             'O'),
            ('focus', ['TOP', 'S', 'VP', 'S', 'VP', 'VP', 'NP-PRD', 'NP',
                       'NN'], 'O'),
            ('of', ['TOP', 'S', 'VP', 'S', 'VP', 'VP', 'NP-PRD', 'PP', 'IN'],
             'O'),
            ('worldwide', ['TOP', 'S', 'VP', 'S', 'VP', 'VP', 'NP-PRD', 'PP',
                           'NP', 'JJ'], 'O'),
            ('attention', ['TOP', 'S', 'VP', 'S', 'VP', 'VP', 'NP-PRD', 'PP',
                           'NP', 'NN'], 'O'),
            ('.', ['TOP', 'S', '.'], 'O')
        ]
        true_bounds = [
            (0, 4),  # With
            (5, 10),  # their
            (11, 17),  # unique
            (18, 23),  # charm
            (23, 24),  # ,
            (25, 30),  # these
            (31, 35),  # well
            (35, 36),  # -
            (36, 41),  # known
            (42, 49),  # cartoon
            (50, 56),  # images
            (57, 61),  # once
            (62, 67),  # again
            (68, 74),  # caused
            (75, 79),  # Hong
            (80, 84),  # Kong
            (85, 87),  # to
            (88, 90),  # be
            (91, 92),  # a
            (93, 98),  # focus
            (99, 101),  # of
            (102, 111),  # worldwide
            (112, 121),  # attention
            (121, 122)  # .
        ]
        calculated_bounds = get_token_bounds(source_text, tokenized)
        self.assertEqual(true_bounds, calculated_bounds)

    def test_get_token_bounds_neg01(self):
        source_text = 'With their unque charm, these well-known cartoon ' \
                      'images once again caused Hong Kong to be a focus of ' \
                      'worldwide attention.'
        tokenized = [
            ('With', ['TOP', 'S', 'PP', 'IN'], 'O'),
            ('their', ['TOP', 'S', 'PP', 'NP', 'PRP$'], 'O'),
            ('unique', ['TOP', 'S', 'PP', 'NP', 'JJ'], 'O'),
            ('charm', ['TOP', 'S', 'PP', 'NP', 'NN'], 'O'),
            (',', ['TOP', 'S', ','], 'O'),
            ('these', ['TOP', 'S', 'NP-SBJ', 'DT'], 'O'),
            ('well', ['TOP', 'S', 'NP-SBJ', 'ADJP', 'RB'], 'O'),
            ('-', ['TOP', 'S', 'NP-SBJ', 'ADJP', 'HYPH'], 'O'),
            ('known', ['TOP', 'S', 'NP-SBJ', 'ADJP', 'VBN'], 'O'),
            ('cartoon', ['TOP', 'S', 'NP-SBJ', 'NN'], 'O'),
            ('images', ['TOP', 'S', 'NP-SBJ', 'NNS'], 'O'),
            ('once', ['TOP', 'S', 'ADVP-TMP', 'RB'], 'O'),
            ('again', ['TOP', 'S', 'ADVP-TMP', 'RB'], 'O'),
            ('caused', ['TOP', 'S', 'VP', 'VBD'], 'O'),
            ('Hong', ['TOP', 'S', 'VP', 'S', 'NP-SBJ', 'NNP'], 'B-GPE'),
            ('Kong', ['TOP', 'S', 'VP', 'S', 'NP-SBJ', 'NNP'], 'I-GPE'),
            ('to', ['TOP', 'S', 'VP', 'S', 'VP', 'TO'], 'O'),
            ('be', ['TOP', 'S', 'VP', 'S', 'VP', 'VP', 'VB'], 'O'),
            ('a', ['TOP', 'S', 'VP', 'S', 'VP', 'VP', 'NP-PRD', 'NP', 'DT'],
             'O'),
            ('focus', ['TOP', 'S', 'VP', 'S', 'VP', 'VP', 'NP-PRD', 'NP',
                       'NN'], 'O'),
            ('of', ['TOP', 'S', 'VP', 'S', 'VP', 'VP', 'NP-PRD', 'PP', 'IN'],
             'O'),
            ('worldwide', ['TOP', 'S', 'VP', 'S', 'VP', 'VP', 'NP-PRD', 'PP',
                           'NP', 'JJ'], 'O'),
            ('attention', ['TOP', 'S', 'VP', 'S', 'VP', 'VP', 'NP-PRD', 'PP',
                           'NP', 'NN'], 'O'),
            ('.', ['TOP', 'S', '.'], 'O')
        ]
        with self.assertRaises(ValueError):
            _ = get_token_bounds(source_text, tokenized)

    def test_get_token_bounds_fuzzy_pos01(self):
        source_text = 'With their unique charm, these well-known cartoon ' \
                      'images once again caused Hong Kong to be a focus of ' \
                      'worldwide attention.'
        tokenized = [
            ('With', ['TOP', 'S', 'PP', 'IN'], 'O'),
            ('their', ['TOP', 'S', 'PP', 'NP', 'PRP$'], 'O'),
            ('unique', ['TOP', 'S', 'PP', 'NP', 'JJ'], 'O'),
            ('charm', ['TOP', 'S', 'PP', 'NP', 'NN'], 'O'),
            (',', ['TOP', 'S', ','], 'O'),
            ('these', ['TOP', 'S', 'NP-SBJ', 'DT'], 'O'),
            ('well', ['TOP', 'S', 'NP-SBJ', 'ADJP', 'RB'], 'O'),
            ('-', ['TOP', 'S', 'NP-SBJ', 'ADJP', 'HYPH'], 'O'),
            ('known', ['TOP', 'S', 'NP-SBJ', 'ADJP', 'VBN'], 'O'),
            ('cartoon', ['TOP', 'S', 'NP-SBJ', 'NN'], 'O'),
            ('images', ['TOP', 'S', 'NP-SBJ', 'NNS'], 'O'),
            ('once', ['TOP', 'S', 'ADVP-TMP', 'RB'], 'O'),
            ('again', ['TOP', 'S', 'ADVP-TMP', 'RB'], 'O'),
            ('caused', ['TOP', 'S', 'VP', 'VBD'], 'O'),
            ('Hong', ['TOP', 'S', 'VP', 'S', 'NP-SBJ', 'NNP'], 'B-GPE'),
            ('Kong', ['TOP', 'S', 'VP', 'S', 'NP-SBJ', 'NNP'], 'I-GPE'),
            ('to', ['TOP', 'S', 'VP', 'S', 'VP', 'TO'], 'O'),
            ('be', ['TOP', 'S', 'VP', 'S', 'VP', 'VP', 'VB'], 'O'),
            ('a', ['TOP', 'S', 'VP', 'S', 'VP', 'VP', 'NP-PRD', 'NP', 'DT'],
             'O'),
            ('focus', ['TOP', 'S', 'VP', 'S', 'VP', 'VP', 'NP-PRD', 'NP',
                       'NN'], 'O'),
            ('of', ['TOP', 'S', 'VP', 'S', 'VP', 'VP', 'NP-PRD', 'PP', 'IN'],
             'O'),
            ('worldwide', ['TOP', 'S', 'VP', 'S', 'VP', 'VP', 'NP-PRD', 'PP',
                           'NP', 'JJ'], 'O'),
            ('attention', ['TOP', 'S', 'VP', 'S', 'VP', 'VP', 'NP-PRD', 'PP',
                           'NP', 'NN'], 'O'),
            ('.', ['TOP', 'S', '.'], 'O')
        ]
        true_bounds = [
            (0, 4),  # With
            (5, 10),  # their
            (11, 17),  # unique
            (18, 23),  # charm
            (23, 24),  # ,
            (25, 30),  # these
            (31, 35),  # well
            (35, 36),  # -
            (36, 41),  # known
            (42, 49),  # cartoon
            (50, 56),  # images
            (57, 61),  # once
            (62, 67),  # again
            (68, 74),  # caused
            (75, 79),  # Hong
            (80, 84),  # Kong
            (85, 87),  # to
            (88, 90),  # be
            (91, 92),  # a
            (93, 98),  # focus
            (99, 101),  # of
            (102, 111),  # worldwide
            (112, 121),  # attention
            (121, 122)  # .
        ]
        calculated_bounds = get_token_bounds_fuzzy(source_text, tokenized)
        self.assertEqual(true_bounds, calculated_bounds)

    def test_get_token_bounds_fuzzy_pos02(self):
        source_text = 'With their unque charm, these well-known cartoon ' \
                      'images once again caused Hong Kong to be a focus of ' \
                      'worldwide attention.'
        tokenized = [
            ('With', ['TOP', 'S', 'PP', 'IN'], 'O'),
            ('their', ['TOP', 'S', 'PP', 'NP', 'PRP$'], 'O'),
            ('unique', ['TOP', 'S', 'PP', 'NP', 'JJ'], 'O'),
            ('charm', ['TOP', 'S', 'PP', 'NP', 'NN'], 'O'),
            (',', ['TOP', 'S', ','], 'O'),
            ('these', ['TOP', 'S', 'NP-SBJ', 'DT'], 'O'),
            ('well', ['TOP', 'S', 'NP-SBJ', 'ADJP', 'RB'], 'O'),
            ('-', ['TOP', 'S', 'NP-SBJ', 'ADJP', 'HYPH'], 'O'),
            ('known', ['TOP', 'S', 'NP-SBJ', 'ADJP', 'VBN'], 'O'),
            ('cartoon', ['TOP', 'S', 'NP-SBJ', 'NN'], 'O'),
            ('images', ['TOP', 'S', 'NP-SBJ', 'NNS'], 'O'),
            ('once', ['TOP', 'S', 'ADVP-TMP', 'RB'], 'O'),
            ('again', ['TOP', 'S', 'ADVP-TMP', 'RB'], 'O'),
            ('caused', ['TOP', 'S', 'VP', 'VBD'], 'O'),
            ('Hong', ['TOP', 'S', 'VP', 'S', 'NP-SBJ', 'NNP'], 'B-GPE'),
            ('Kong', ['TOP', 'S', 'VP', 'S', 'NP-SBJ', 'NNP'], 'I-GPE'),
            ('to', ['TOP', 'S', 'VP', 'S', 'VP', 'TO'], 'O'),
            ('be', ['TOP', 'S', 'VP', 'S', 'VP', 'VP', 'VB'], 'O'),
            ('a', ['TOP', 'S', 'VP', 'S', 'VP', 'VP', 'NP-PRD', 'NP', 'DT'],
             'O'),
            ('focus', ['TOP', 'S', 'VP', 'S', 'VP', 'VP', 'NP-PRD', 'NP',
                       'NN'], 'O'),
            ('of', ['TOP', 'S', 'VP', 'S', 'VP', 'VP', 'NP-PRD', 'PP', 'IN'],
             'O'),
            ('worldwide', ['TOP', 'S', 'VP', 'S', 'VP', 'VP', 'NP-PRD', 'PP',
                           'NP', 'JJ'], 'O'),
            ('atention', ['TOP', 'S', 'VP', 'S', 'VP', 'VP', 'NP-PRD', 'PP',
                          'NP', 'NN'], 'O'),
            ('.', ['TOP', 'S', '.'], 'O')
        ]
        true_bounds = [
            (0, 4),  # With
            (5, 10),  # their
            (11, 16),  # unque
            (17, 22),  # charm
            (22, 23),  # ,
            (24, 29),  # these
            (30, 34),  # well
            (34, 35),  # -
            (35, 40),  # known
            (41, 48),  # cartoon
            (49, 55),  # images
            (56, 60),  # once
            (61, 66),  # again
            (67, 73),  # caused
            (74, 78),  # Hong
            (79, 83),  # Kong
            (84, 86),  # to
            (87, 89),  # be
            (90, 91),  # a
            (92, 97),  # focus
            (98, 100),  # of
            (101, 110),  # worldwide
            (111, 120),  # attention
            (120, 121)  # .
        ]
        calculated_bounds = get_token_bounds_fuzzy(source_text, tokenized)
        self.assertEqual(true_bounds, calculated_bounds)

    def test_get_token_bounds_fuzzy_pos03(self):
        source_text = 'بعد ما تضاربت المعلومات الاسبوع الماضي عن عدد الجنود ' \
                      'الاميركيين الذين س يشاركون في بعثة تدريبية ل مكافحة ' \
                      'الارهاب في الفيليبين, في إطار جهود واشنطن ل مكافحة ' \
                      'الارهاب في العالم, أفاد أمس مسؤولون عسكريون في مانيلا ' \
                      'ان نحو 650 جندياً اميركياً س ينضمون تباعاً الى قوات ' \
                      'فيليبينية ل تعزيز قدرات ها الدفاعية من أجل القضاء ' \
                      'على جماعة " أبو سياف " التي تربط ها صلات ب تنظيم " ' \
                      'القاعدة " الذي يتزعم ه اسامة بن لادن, م ما س يتيح ل هم ' \
                      'الانتقال الى مناطق القتال في جنوب البلاد.'
        some_list = ['linguistic', 'information']
        tokenized = [
            ("بَعْدَ-", some_list, "anything"),
            ("-ما", some_list, "anything"),
            ("تَضارَبَت", some_list, "anything"),
            ("المَعْلُوماتُ", some_list, "anything"),
            ("الأُسْبُوعَ", some_list, "anything"),
            ("الماضِيَ", some_list, "anything"),
            ("عَن", some_list, "anything"),
            ("عَدَدِ", some_list, "anything"),
            ("الجُنُودِ", some_list, "anything"),
            ("الأَمِيرْكِيِّينَ", some_list, "anything"),
            ("الَّذِينَ", some_list, "anything"),
            ("سَ-", some_list, "anything"),
            ("-يُشارِكُونَ", some_list, "anything"),
            ("فِي", some_list, "anything"),
            ("بِعْثَةٍ", some_list, "anything"),
            ("تَدْرِيبِيَّةٍ", some_list, "anything"),
            ("لِ-", some_list, "anything"),
            ("-مُكافَحَةِ", some_list, "anything"),
            ("الإِرْهابِ", some_list, "anything"),
            ("الفِيلِيبِّين", some_list, "anything"),
            (",", some_list, "anything"),
            ("فِي", some_list, "anything"),
            ("إِطارِ", some_list, "anything"),
            ("جُهُودِ", some_list, "anything"),
            ("واشِنْطُن", some_list, "anything"),
            ("لِ-", some_list, "anything"),
            ("-مُكافَحَةِ", some_list, "anything"),
            ("الإِرْهابِ", some_list, "anything"),
            ("فِي", some_list, "anything"),
            ("العالَمِ", some_list, "anything"),
            (",", some_list, "anything"),
            ("أَفادَ", some_list, "anything"),
            ("أَمْسِ", some_list, "anything"),
            ("مَسْؤُولُونَ", some_list, "anything"),
            ("عَسْكَرِيُّونَ", some_list, "anything"),
            ("فِي", some_list, "anything"),
            ("مانِيلا", some_list, "anything"),
            ("أَنَّ", some_list, "anything"),
            ("نَحْوَ", some_list, "anything"),
            ("650", some_list, "anything"),
            ("جُنْدِيّاً", some_list, "anything"),
            ("أَمِيرْكِيّاً", some_list, "anything"),
            ("سَ-", some_list, "anything"),
            ("-يَنْضَمُّونَ", some_list, "anything"),
            ("تِباعاً", some_list, "anything"),
            ("إِلَى", some_list, "anything"),
            ("قُوّاتٍ", some_list, "anything"),
            ("فِلِيبِّينِيَّةٍ", some_list, "anything"),
            ("لِ-", some_list, "anything"),
            ("-تَعْزِيزِ", some_list, "anything"),
            ("قُدْراتِ-", some_list, "anything"),
            ("-ها", some_list, "anything"),
            ("الدِفاعِيَّةِ", some_list, "anything"),
            ("مِن", some_list, "anything"),
            ("أَجْلِ", some_list, "anything"),
            ("القَضاءِ", some_list, "anything"),
            ("عَلَى", some_list, "anything"),
            ("جَماعَةِ", some_list, "anything"),
            ("\"", some_list, "anything"),
            ("أَبُو", some_list, "anything"),
            ("سَيّاف", some_list, "anything"),
            ("\"", some_list, "anything"),
            ("الَّتِي", some_list, "anything"),
            ("تَرْبِطُ-", some_list, "anything"),
            ("-ها", some_list, "anything"),
            ("صِلاتٌ", some_list, "anything"),
            ("بِ-", some_list, "anything"),
            ("-تَنْظِيمِ", some_list, "anything"),
            ("\"", some_list, "anything"),
            ("القاعِدَةِ", some_list, "anything"),
            ("\"", some_list, "anything"),
            ("الَّذِي", some_list, "anything"),
            ("يَتَزَعَّمُ-", some_list, "anything"),
            ("-هُ", some_list, "anything"),
            ("أُسامَة", some_list, "anything"),
            ("بِن", some_list, "anything"),
            ("لادِن", some_list, "anything"),
            (",", some_list, "anything"),
            ("مِن-", some_list, "anything"),
            ("-ما", some_list, "anything"),
            ("سَ-", some_list, "anything"),
            ("-يُتِيحُ", some_list, "anything"),
            ("لَ-", some_list, "anything"),
            ("-هُم", some_list, "anything"),
            ("ال{ِنْتِقالَ", some_list, "anything"),
            ("إِلَى", some_list, "anything"),
            ("مَناطِقِ", some_list, "anything"),
            ("القِتالِ", some_list, "anything"),
            ("فِي", some_list, "anything"),
            ("جَنُوبِ", some_list, "anything"),
            ("البِلادِ", some_list, "anything"),
            (".", some_list, "anything")
        ]
        calculated_bounds = get_token_bounds_fuzzy(source_text, tokenized)
        self.assertIsInstance(calculated_bounds, list)
        self.assertEqual(len(tokenized), len(calculated_bounds))
        prev_pos = 0
        for token_idx, token_bounds in enumerate(calculated_bounds):
            err_msg = 'Token[{0}] = {1} is wrong! All bounds list = {2}'.format(
                token_idx, tokenized[token_idx][0], calculated_bounds
            )
            self.assertIsInstance(token_bounds, tuple, msg=err_msg)
            self.assertEqual(len(token_bounds), 2, msg=err_msg)
            token_start, token_end = token_bounds
            self.assertLess(token_start, token_end, msg=err_msg)
            self.assertGreaterEqual(token_start, prev_pos, msg=err_msg)
            self.assertLessEqual(token_end, len(source_text), msg=err_msg)
            if token_start > prev_pos:
                space_text = source_text[prev_pos:token_start].strip()
                self.assertEqual('', space_text, msg=err_msg)
            token_text = source_text[token_start:token_end]
            self.assertGreater(len(token_text), 0, msg=err_msg)
            self.assertEqual(token_text, token_text.strip(), msg=err_msg)
            prev_pos = token_end
        self.assertEqual('', source_text[prev_pos:].strip())

    def test_get_token_bounds_fuzzy_pos04(self):
        source_text = 'و في الاجمال, س يشارك 500 جندي اميركي في عمليات " ' \
                      'دعم و صيانة ", بينما س يسمح ل لاخرين, و هم اعضاء في ' \
                      'القوات الخاصة, ب مشاركة الجنود الفيليبينيين أحياناً ' \
                      'في مطاردة المتطرفين الاسلاميين في جزيرة باسيلان في ' \
                      'جنوب البلاد.'
        some_list = ['linguistic', 'information']
        tokenized = [
            ("وَ-", some_list, "anything"),
            ("-فِي", some_list, "anything"),
            ("الإِجْمالِ", some_list, "anything"),
            (",", some_list, "anything"),
            ("سَ-", some_list, "anything"),
            ("-يُشارِكُ", some_list, "anything"),
            ("500", some_list, "anything"),
            ("جُنْدِيٍّ", some_list, "anything"),
            ("أَمِيرْكِيٍّ", some_list, "anything"),
            ("فِي", some_list, "anything"),
            ("عَمَلِيّاتِ", some_list, "anything"),
            ("\"", some_list, "anything"),
            ("دَعْمٍ", some_list, "anything"),
            ("وَ-", some_list, "anything"),
            ("-صِيانَةٍ", some_list, "anything"),
            ("\"", some_list, "anything"),
            (",", some_list, "anything"),
            ("بَيْنَما", some_list, "anything"),
            ("سَ-", some_list, "anything"),
            ("-يُسْمَحُ", some_list, "anything"),
            ("لِ-", some_list, "anything"),
            ("-الآخِرِينَ", some_list, "anything"),
            (",", some_list, "anything"),
            ("وَ-", some_list, "anything"),
            ("-هُم", some_list, "anything"),
            ("أَعْضاءٌ", some_list, "anything"),
            ("فِي", some_list, "anything"),
            ("القُوّاتِ", some_list, "anything"),
            ("الخاصَّةِ", some_list, "anything"),
            (",", some_list, "anything"),
            ("بِ-", some_list, "anything"),
            ("-مُشارَكَةِ", some_list, "anything"),
            ("الجُنُودِ", some_list, "anything"),
            ("الفِلِيبِّينِيِّينَ", some_list, "anything"),
            ("أَحْياناً", some_list, "anything"),
            ("فِي", some_list, "anything"),
            ("مُطارَدَةِ", some_list, "anything"),
            ("المُتَطَرِّفِينَ", some_list, "anything"),
            ("الإِسْلامِيِّينَ", some_list, "anything"),
            ("فِي", some_list, "anything"),
            ("جَزِيرَةِ", some_list, "anything"),
            ("باسِيلان", some_list, "anything"),
            ("فِي", some_list, "anything"),
            ("جَنُوبِ", some_list, "anything"),
            ("البِلادِ", some_list, "anything"),
            (".", some_list, "anything")
        ]
        calculated_bounds = get_token_bounds_fuzzy(source_text, tokenized)
        self.assertIsInstance(calculated_bounds, list)
        self.assertEqual(len(tokenized), len(calculated_bounds))
        prev_pos = 0
        for token_idx, token_bounds in enumerate(calculated_bounds):
            err_msg = 'Token[{0}] = {1} is wrong! All bounds list = {2}'.format(
                token_idx, tokenized[token_idx][0], calculated_bounds
            )
            self.assertIsInstance(token_bounds, tuple, msg=err_msg)
            self.assertEqual(len(token_bounds), 2, msg=err_msg)
            token_start, token_end = token_bounds
            self.assertLess(token_start, token_end, msg=err_msg)
            self.assertGreaterEqual(token_start, prev_pos, msg=err_msg)
            self.assertLessEqual(token_end, len(source_text), msg=err_msg)
            if token_start > prev_pos:
                space_text = source_text[prev_pos:token_start].strip()
                self.assertEqual('', space_text, msg=err_msg)
            token_text = source_text[token_start:token_end]
            self.assertGreater(len(token_text), 0, msg=err_msg)
            self.assertEqual(token_text, token_text.strip(), msg=err_msg)
            prev_pos = token_end
        self.assertEqual('', source_text[prev_pos:].strip())

    def test_get_token_bounds_fuzzy_pos05(self):
        source_text = 'و أضاف : " ان هما لجنتان فنيتان س تناقشان وقفاً ' \
                      'ل لنار في جبال النوبة فقط...ان ها ليست محادثات سلام ".'
        some_list = ['linguistic', 'information']
        tokenized = [
            ("وَ-", some_list, "anything"),
            ("-أَضافَ", some_list, "anything"),
            (":", some_list, "anything"),
            ("\"", some_list, "anything"),
            ("إِنَّ-", some_list, "anything"),
            ("-هُما", some_list, "anything"),
            ("لَجْنَتانِ", some_list, "anything"),
            ("فَنِّيَّتانِ", some_list, "anything"),
            ("سَ-", some_list, "anything"),
            ("-تُناقِشانِ", some_list, "anything"),
            ("وَقْفاً", some_list, "anything"),
            ("لِ-", some_list, "anything"),
            ("-النارِ", some_list, "anything"),
            ("فِي", some_list, "anything"),
            ("جِبالِ", some_list, "anything"),
            ("النوبة", some_list, "anything"),
            ("فَقَط", some_list, "anything"),
            (".", some_list, "anything"),
            (".", some_list, "anything"),
            (".", some_list, "anything"),
            ("إِنَّ-", some_list, "anything"),
            ("-ها", some_list, "anything"),
            ("لَيْسَت", some_list, "anything"),
            ("مُحادَثاتُ", some_list, "anything"),
            ("سَلامٍ", some_list, "anything"),
            ("\"", some_list, "anything"),
            (".", some_list, "anything")
        ]
        calculated_bounds = get_token_bounds_fuzzy(source_text, tokenized)
        self.assertIsInstance(calculated_bounds, list)
        self.assertEqual(len(tokenized), len(calculated_bounds))
        prev_pos = 0
        for token_idx, token_bounds in enumerate(calculated_bounds):
            err_msg = 'Token[{0}] = {1} is wrong! All bounds list = {2}'.format(
                token_idx, tokenized[token_idx][0], calculated_bounds
            )
            self.assertIsInstance(token_bounds, tuple, msg=err_msg)
            self.assertEqual(len(token_bounds), 2, msg=err_msg)
            token_start, token_end = token_bounds
            self.assertLess(token_start, token_end, msg=err_msg)
            self.assertGreaterEqual(token_start, prev_pos, msg=err_msg)
            self.assertLessEqual(token_end, len(source_text), msg=err_msg)
            if token_start > prev_pos:
                space_text = source_text[prev_pos:token_start].strip()
                self.assertEqual('', space_text, msg=err_msg)
            token_text = source_text[token_start:token_end]
            self.assertGreater(len(token_text), 0, msg=err_msg)
            self.assertEqual(token_text, token_text.strip(), msg=err_msg)
            prev_pos = token_end
        self.assertEqual('', source_text[prev_pos:].strip())

    def test_get_language_by_filename_pos01(self):
        src_name = os.path.join('data', 'files', 'data', 'arabic',
                                'annotations', 'nw', 'ann', '00',
                                'ann_0001.onf')
        self.assertEqual('arabic', get_language_by_filename(src_name))

    def test_get_language_by_filename_pos02(self):
        src_name = os.path.join('english', 'annotations', 'bn', 'pri', '01',
                                'pri_0100.onf')
        self.assertEqual('english', get_language_by_filename(src_name))

    def test_get_language_by_filename_neg01(self):
        src_name = os.path.join('data', 'files', 'data', 'arabic',
                                'nw', 'ann', '00', 'ann_0001.onf')
        with self.assertRaises(ValueError):
            _ = get_language_by_filename(src_name)

    def test_is_item_in_sequence_pos01(self):
        re_for_special_token = re.compile('^\-[A-Z]+\-$')
        sequence = [
            '-LRB-',
            'و',
            'ص',
            'ف',
            ',',
            'رُويْتِرز',
            ',',
            'أب',
            '-RRB-'
        ]
        self.assertTrue(is_item_in_sequence(re_for_special_token, sequence))

    def test_is_item_in_sequence_pos02(self):
        re_for_special_token = re.compile('^\-[A-Z]+\-$')
        sequence = [
            'و',
            'ص',
            'ف',
            ',',
            'رُويْتِرز',
            ',',
            'أب',
            '-RRB-'
        ]
        self.assertTrue(is_item_in_sequence(re_for_special_token, sequence))

    def test_is_item_in_sequence_pos03(self):
        re_for_special_token = re.compile('^\-[A-Z]+\-$')
        sequence = [
            'و',
            'ص',
            'ف',
            ',',
            'رُويْتِرز',
            ',',
            'أب'
        ]
        self.assertFalse(is_item_in_sequence(re_for_special_token, sequence))

    def test_insert_new_bounds_pos01(self):
        new_bounds = (0, 5)
        old_bounds_list = [(7, 9), (13, 20), (27, 32)]
        true_bounds_list = [(0, 5), (7, 9), (13, 20), (27, 32)]
        calculated_bounds_list = insert_new_bounds(new_bounds, old_bounds_list)
        self.assertEqual(true_bounds_list, calculated_bounds_list)

    def test_insert_new_bounds_pos02(self):
        new_bounds = (0, 7)
        old_bounds_list = [(7, 9), (13, 20), (27, 32)]
        true_bounds_list = [(0, 9), (13, 20), (27, 32)]
        calculated_bounds_list = insert_new_bounds(new_bounds, old_bounds_list)
        self.assertEqual(true_bounds_list, calculated_bounds_list)

    def test_insert_new_bounds_pos03(self):
        new_bounds = (34, 40)
        old_bounds_list = [(7, 9), (13, 20), (27, 32)]
        true_bounds_list = [(7, 9), (13, 20), (27, 32), (34, 40)]
        calculated_bounds_list = insert_new_bounds(new_bounds, old_bounds_list)
        self.assertEqual(true_bounds_list, calculated_bounds_list)

    def test_insert_new_bounds_pos04(self):
        new_bounds = (11, 12)
        old_bounds_list = [(7, 9), (13, 20), (27, 32)]
        true_bounds_list = [(7, 9), (11, 12), (13, 20), (27, 32)]
        calculated_bounds_list = insert_new_bounds(new_bounds, old_bounds_list)
        self.assertEqual(true_bounds_list, calculated_bounds_list)

    def test_insert_new_bounds_pos05(self):
        new_bounds = (11, 13)
        old_bounds_list = [(7, 9), (13, 20), (27, 32)]
        true_bounds_list = [(7, 9), (11, 20), (27, 32)]
        calculated_bounds_list = insert_new_bounds(new_bounds, old_bounds_list)
        self.assertEqual(true_bounds_list, calculated_bounds_list)

    def test_insert_new_bounds_pos06(self):
        new_bounds = (8, 13)
        old_bounds_list = [(7, 9), (13, 20), (27, 32)]
        true_bounds_list = [(7, 20), (27, 32)]
        calculated_bounds_list = insert_new_bounds(new_bounds, old_bounds_list)
        self.assertEqual(true_bounds_list, calculated_bounds_list)

    def test_insert_new_bounds_neg01(self):
        new_bounds = (8, 13)
        old_bounds_list = [(7, 9), (13, 20), (27, 27)]
        with self.assertRaises(ValueError):
            _ = insert_new_bounds(new_bounds, old_bounds_list)

    def test_insert_new_bounds_neg02(self):
        new_bounds = (8, 13)
        old_bounds_list = [(7, 9), (9, 20), (27, 32)]
        with self.assertRaises(ValueError):
            _ = insert_new_bounds(new_bounds, old_bounds_list)

    def test_calculate_distance_pos01(self):
        syntax1 = 'PV+PVSUFF_SUBJ:2MP'
        syntax2 = 'PV+PVSUFF_SUBJ:2MP'
        self.assertEqual(0, calculate_distance(syntax1, syntax2))

    def test_calculate_distance_pos02(self):
        syntax1 = 'PV+PVSUFF_SUBJ:2MP'
        syntax2 = 'PV'
        self.assertEqual(2, calculate_distance(syntax1, syntax2))

    def test_calculate_distance_pos03(self):
        syntax1 = 'PP-TMP'
        syntax2 = 'SBAR-PRD'
        self.assertEqual(2 + 7 * 10, calculate_distance(syntax1, syntax2))

    def test_calculate_distance_pos04(self):
        syntax1 = 'VBD'
        syntax2 = 'WRB'
        self.assertEqual(2 + 3 * 10, calculate_distance(syntax1, syntax2))

    def test_calculate_distance_pos05(self):
        syntax1 = 'DET+NOUN+CASE_DEF_GEN'
        syntax2 = 'DET+NOUN+CASE_DEF_ACC'
        self.assertEqual(2 + 3, calculate_distance(syntax1, syntax2))

    def test_unite_overlapped_bounds_pos01(self):
        source_bounds = [(0, 5), (6, 11), (13, 20)]
        true_bounds = [(0, 5), (6, 11), (13, 20)]
        calculated_bounds = unite_overlapped_bounds(source_bounds)
        self.assertEqual(true_bounds, calculated_bounds)

    def test_unite_overlapped_bounds_pos02(self):
        source_bounds = [(0, 5), (4, 11), (13, 20)]
        true_bounds = [(0, 11), (13, 20)]
        calculated_bounds = unite_overlapped_bounds(source_bounds)
        self.assertEqual(true_bounds, calculated_bounds)

    def test_unite_overlapped_bounds_pos03(self):
        source_bounds = [(0, 5), (6, 11), (11, 20)]
        true_bounds = [(0, 5), (6, 20)]
        calculated_bounds = unite_overlapped_bounds(source_bounds)
        self.assertEqual(true_bounds, calculated_bounds)

    def test_unite_overlapped_bounds_neg01(self):
        source_bounds = [(0, 5), (6, 11), (5, 20)]
        with self.assertRaises(ValueError):
            _ = unite_overlapped_bounds(source_bounds)

    def test_check_bounds_pos01(self):
        source_text = 'With their unique charm, these well-known cartoon ' \
                      'images once again caused Hong Kong to be a focus of ' \
                      'worldwide attention.'
        source_bounds = [
            (0, 4),  # With
            (5, 10),  # their
            (11, 17),  # unique
            (18, 23),  # charm
            (23, 24),  # ,
            (25, 30),  # these
            (31, 35),  # well
            (35, 36),  # -
            (36, 41),  # known
            (42, 49),  # cartoon
            (50, 56),  # images
            (57, 61),  # once
            (62, 67),  # again
            (68, 74),  # caused
            (75, 79),  # Hong
            (80, 84),  # Kong
            (85, 87),  # to
            (88, 90),  # be
            (91, 92),  # a
            (93, 98),  # focus
            (99, 101),  # of
            (102, 111),  # worldwide
            (112, 121),  # attention
            (121, 122)  # .
        ]
        res = check_bounds(text=source_text, bounds=source_bounds)
        self.assertEqual('', res)

    def test_check_bounds_pos02(self):
        source_text = 'With their unique charm, these well-known cartoon ' \
                      'images once again caused Hong Kong to be a focus of ' \
                      'worldwide attention.'
        source_bounds = [
            (0, 4),  # With
            (5, 10),  # their
            (11, 17),  # unique
            (18, 23),  # charm
            (23, 24),  # ,
            (25, 30),  # these
            (31, 35),  # well
            (35, 36),  # -
            (36, 41),  # known
            (42, 49),  # cartoon
            (50, 56),  # images
            (57, 61),  # once
            (62, 67),  # again
            (68, 74),  # caused
            (75, 79),  # Hong
            (80, 84),  # Kong
            (85, 87),  # to
            (88, 90),  # be
            (91, 92),  # a
            (93, 98),  # focus
            (99, 101),  # of
            (102, 111),  # worldwide
            (112, 121),  # attention
            (121, 132)  # .
        ]
        res = check_bounds(text=source_text, bounds=source_bounds)
        self.assertGreater(len(res), 0)

    def test_check_bounds_pos03(self):
        source_text = 'With their unique charm, these well-known cartoon ' \
                      'images once again caused Hong Kong to be a focus of ' \
                      'worldwide attention.'
        source_bounds = [
            (0, 4),  # With
            (5, 10),  # their
            (11, 17),  # unique
            (18, 23),  # charm
            (23, 24),  # ,
            (25, 30),  # these
            (29, 35),  # well
            (35, 36),  # -
            (36, 41),  # known
            (42, 49),  # cartoon
            (50, 56),  # images
            (57, 61),  # once
            (62, 67),  # again
            (68, 74),  # caused
            (75, 79),  # Hong
            (80, 84),  # Kong
            (85, 87),  # to
            (88, 90),  # be
            (91, 92),  # a
            (93, 98),  # focus
            (99, 101),  # of
            (102, 111),  # worldwide
            (112, 121),  # attention
            (121, 122)  # .
        ]
        res = check_bounds(text=source_text, bounds=source_bounds)
        self.assertGreater(len(res), 0)

    def test_check_bounds_pos04(self):
        source_text = 'With their unique charm, these well-known cartoon ' \
                      'images once again caused Hong Kong to be a focus of ' \
                      'worldwide attention.'
        source_bounds = [
            (0, 4),  # With
            (4, 5),
            (5, 10),  # their
            (11, 17),  # unique
            (18, 23),  # charm
            (23, 24),  # ,
            (25, 30),  # these
            (31, 35),  # well
            (35, 36),  # -
            (36, 41),  # known
            (42, 49),  # cartoon
            (50, 56),  # images
            (57, 61),  # once
            (62, 67),  # again
            (68, 74),  # caused
            (75, 79),  # Hong
            (80, 84),  # Kong
            (85, 87),  # to
            (88, 90),  # be
            (91, 92),  # a
            (93, 98),  # focus
            (99, 101),  # of
            (102, 111),  # worldwide
            (112, 121),  # attention
            (121, 122)  # .
        ]
        res = check_bounds(text=source_text, bounds=source_bounds)
        self.assertGreater(len(res), 0)

    def test_find_subword_bounds_pos01(self):
        word = '12345'
        subwords = ['1', '2', '3', '4', '5']
        true_bounds = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        calculated_bounds, _ = find_subword_bounds(word, subwords)
        self.assertEqual(true_bounds, calculated_bounds)

    def test_find_subword_bounds_pos02(self):
        word = '12345'
        subwords = ['1', '2', '3', '4a', '5']
        true_bounds = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        calculated_bounds, _ = find_subword_bounds(word, subwords)
        self.assertEqual(true_bounds, calculated_bounds)

    def test_find_subword_bounds_pos03(self):
        word = '1234a5'
        subwords = ['1', '2', '3', '4', '5']
        true_bounds = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6)]
        calculated_bounds, _ = find_subword_bounds(word, subwords)
        self.assertEqual(true_bounds, calculated_bounds)

    def test_find_subword_bounds_pos04(self):
        word = '1234a5'
        subwords = ['1', '2', '3', '4b', '5']
        true_bounds = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 6)]
        calculated_bounds, _ = find_subword_bounds(word, subwords)
        self.assertEqual(true_bounds, calculated_bounds)

    def test_find_subword_bounds_neg01(self):
        word = '1234'
        subwords = ['1', '2', '3', '4', '5']
        with self.assertRaises(ValueError):
            _ = find_subword_bounds(word, subwords)

    def test_tokenize_any_word_pos01(self):
        s = 'Hello, world!'
        words = ['Hello', ',', 'world', '!']
        self.assertEqual(words, tokenize_any_text(s))

    def test_tokenize_any_word_pos02(self):
        s = '天地方益権'
        words = ['天', '地', '方', '益', '権']
        self.assertEqual(words, tokenize_any_text(s))

    def test_tokenize_any_word_pos03(self):
        s = 'hello?天地方3 d gh益権, world!'
        words = ['hello', '?', '天', '地', '方', '3', 'd', 'gh', '益', '権', ',',
                 'world', '!']
        self.assertEqual(words, tokenize_any_text(s))


if __name__ == '__main__':
    unittest.main(verbosity=2)
