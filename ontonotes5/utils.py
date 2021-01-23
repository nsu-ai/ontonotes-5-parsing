import codecs
import os
import re
from typing import Dict, Pattern, List, Tuple

import numpy as np


def get_plain_text(all_lines: List[str], start_idx: int, end_idx: int) -> str:
    res = all_lines[start_idx].strip()
    for idx in range(start_idx + 1, end_idx):
        res += (' ' + all_lines[idx].strip())
    return res.strip()


def parse_tree(tree: str) -> List[Tuple[str, List[str]]]:
    if len(tree.strip()) == 0:
        return []
    err_msg = '"{0}" is wrong syntax tree!'.format(tree)
    depth = 0
    re_for_bracket = re.compile('(\(|\))')
    prev_pos = -1
    prev_bracket = ''
    tags = []
    tokens_with_tags = []
    for search_res in re_for_bracket.finditer(tree):
        bracket_pos = search_res.start()
        bracket = tree[bracket_pos]
        if bracket not in {'(', ')'}:
            raise ValueError(err_msg)
        if bracket == '(':
            depth += 1
            if prev_pos >= 0:
                if prev_bracket == '':
                    raise ValueError(err_msg)
                text_between_brackets = tree[(prev_pos + 1):bracket_pos].strip()
                if prev_bracket == '(':
                    if len(text_between_brackets) == 0:
                        raise ValueError(err_msg)
                    text_parts = text_between_brackets.split()
                    if len(text_parts) != 1:
                        raise ValueError(tree)
                    tags.append(text_parts[0])
                else:
                    if len(text_between_brackets) != 0:
                        raise ValueError(err_msg)
            else:
                if prev_bracket != '':
                    raise ValueError(err_msg)
        else:
            if depth < 1:
                raise ValueError(err_msg)
            if (prev_pos < 0) or (prev_bracket == ''):
                raise ValueError(err_msg)
            text_between_brackets = tree[(prev_pos + 1):bracket_pos].strip()
            if prev_bracket == ')':
                if len(text_between_brackets) != 0:
                    raise ValueError(err_msg)
                tags = tags[:-1]
            else:
                text_parts = text_between_brackets.split()
                if len(text_parts) != 2:
                    raise ValueError(tree)
                tokens_with_tags.append((text_parts[1], tags + [text_parts[0]]))
            depth -= 1
        prev_bracket = bracket
        prev_pos = bracket_pos
    return tokens_with_tags


def parse_named_entities_labeling(lines: List[str], true_tokens: List[str],
                                  onf_name: str = '') -> List[str]:
    re_for_token = re.compile('^\d+\s+.+')
    re_for_name = re.compile('^name\:\s+\w+\s+\d+\-\d+')
    token_idx = 0
    start_pos = -1
    token_bounds = []
    n_tokens = len(true_tokens)
    for line_idx, cur_line in enumerate(lines):
        prep_line = cur_line.strip()
        if len(onf_name) == 0:
            err_msg = 'Description "{0}" contains incorrect line. ' \
                      'Line "{1}" is wrong!'.format(
                ' '.join(list(map(lambda it: it.strip(), lines))),
                prep_line
            )
        else:
            err_msg = 'File "{0}": Description "{1}" contains incorrect ' \
                      'line. Line "{2}" is wrong!'.format(
                onf_name, ' '.join(list(map(lambda it: it.strip(), lines))),
                prep_line
            )
        search_res = re_for_token.search(prep_line)
        if search_res is not None:
            if (search_res.start() == 0) and (search_res.end() > 0):
                if token_idx >= len(true_tokens):
                    raise ValueError(err_msg)
                text_parts = prep_line[:search_res.end()].split()
                if text_parts[0].isdigit():
                    if (int(text_parts[0]) == token_idx) and \
                            (text_parts[1] == true_tokens[token_idx]):
                        if start_pos >= 0:
                            token_bounds.append((start_pos, line_idx))
                        start_pos = line_idx
                        token_idx += 1
    if start_pos >= 0:
        token_bounds.append((start_pos, len(lines)))
    if len(token_bounds) != n_tokens:
        if len(onf_name) == 0:
            err_msg = 'Description "{0}" contains incorrect line. Number of ' \
                      'tokens does not correspond to number of entity labels!' \
                      ' {1} != {2}'.format(
                ' '.join(list(map(lambda it: it.strip(), lines))), n_tokens,
                len(token_bounds)
            )
        else:
            err_msg = 'File "{0}": Description "{1}" contains incorrect line.' \
                      ' Number of tokens does not correspond to number of ' \
                      'entity labels! {2} != {3}'.format(
                onf_name, ' '.join(list(map(lambda it: it.strip(), lines))),
                n_tokens, len(token_bounds)
            )
        raise ValueError(err_msg)
    entities = []
    token_idx = 0
    while token_idx < n_tokens:
        name_idx = -1
        entity_type = ''
        if len(onf_name) == 0:
            err_msg = 'Description "{0}" contains incorrect line. ' \
                      'Token {1} is wrong!'.format(
                ' '.join(list(map(lambda it: it.strip(), lines))), token_idx
            )
        else:
            err_msg = 'File "{0}": Description "{1}" contains incorrect line.' \
                      ' Token {2} is wrong'.format(
                onf_name, ' '.join(list(map(lambda it: it.strip(), lines))),
                token_idx
            )
        token_start, token_end = token_bounds[token_idx]
        entity_bounds = (-1, -1)
        for line_idx in range(token_start, token_end):
            cur_line = lines[line_idx].strip()
            search_res = re_for_name.search(cur_line)
            if search_res is not None:
                start_pos = search_res.start()
                end_pos = search_res.end()
                if (start_pos == 0) and (end_pos > 0):
                    name_idx = line_idx
                    line_parts = cur_line[start_pos:end_pos].strip().split()
                    if len(line_parts) != 3:
                        raise ValueError(err_msg)
                    entity_type = line_parts[1]
                    if not entity_type.isupper():
                        raise ValueError(err_msg)
                    entity_bounds = line_parts[2].split('-')
                    if len(entity_bounds) != 2:
                        raise ValueError(err_msg)
                    if (not entity_bounds[0].isdigit()) or \
                            (not entity_bounds[1].isdigit()):
                        raise ValueError(err_msg)
                    entity_bounds = (
                        int(entity_bounds[0]),
                        int(entity_bounds[1])
                    )
                    if entity_bounds[0] > entity_bounds[1]:
                        raise ValueError(err_msg)
                    if entity_bounds[0] != token_idx:
                        raise ValueError(err_msg)
                    break
        if name_idx < 0:
            entities.append('O')
            token_idx += 1
        else:
            entities.append('B-' + entity_type)
            for _ in range(entity_bounds[1] - entity_bounds[0]):
                entities.append('I-' + entity_type)
            token_idx += (entity_bounds[1] - entity_bounds[0] + 1)
    return entities


def get_token_bounds(source_text: str,
                     tokens_with_info: List[Tuple[str, List[str], str]]) -> \
        List[Tuple[int, int]]:
    bounds = []
    start_pos = 0
    for token, lingvo_data, named_entity in tokens_with_info:
        found_idx = source_text[start_pos:].find(token)
        if found_idx < 0:
            raise ValueError(
                'Token "{0}" is not found in the text "{1}"! '
                'start_pos = {2}'.format(token, source_text, start_pos)
            )
        found_idx += start_pos
        bounds.append((found_idx, found_idx + len(token)))
        start_pos = found_idx + len(token)
    return bounds


def get_token_bounds_fuzzy(
        source_text: str,
        tokens_with_info: List[Tuple[str, List[str], str]]
) -> List[Tuple[int, int]]:
    N_s = len(source_text)
    char_indices = list(range(N_s))
    token_words = [cur[0] for cur in tokens_with_info]
    token_chars = ' '.join(token_words)
    token_indices = [0 for _ in range(len(token_words[0]))]
    for idx, word in enumerate(token_words[1:]):
        token_indices.append(-1)
        token_indices += [(idx + 1) for _ in range(len(word))]
    N_t = len(token_chars)
    err_msg = '{0} != {1}'.format(N_t, len(token_indices))
    assert N_t == len(token_indices), err_msg
    D_matrix = np.zeros((N_s + 1, N_t + 1), dtype=np.int32)
    for idx_s in range(1, N_s + 1):
        D_matrix[idx_s, 0] = idx_s
    for idx_t in range(1, N_t + 1):
        D_matrix[0, idx_t] = idx_t
    for idx_s in range(1, N_s + 1):
        for idx_t in range(1, N_t + 1):
            cur_dist = 0 if (source_text[idx_s - 1] == token_chars[idx_t - 1]) \
                else 1
            D_matrix[idx_s, idx_t] = min(
                D_matrix[idx_s, idx_t - 1] + 1,
                D_matrix[idx_s - 1, idx_t] + 1,
                D_matrix[idx_s - 1, idx_t - 1] + cur_dist
            )
    idx_s = N_s
    idx_t = N_t
    optimal_path = [(idx_s - 1, idx_t - 1)]
    while (idx_s > 1) and (idx_t > 1):
        if D_matrix[idx_s - 1, idx_t - 1] < D_matrix[idx_s - 1, idx_t]:
            if D_matrix[idx_s - 1, idx_t - 1] < D_matrix[idx_s, idx_t - 1]:
                idx_s -= 1
                idx_t -= 1
            else:
                idx_t -= 1
        else:
            if D_matrix[idx_s - 1, idx_t] < D_matrix[idx_s, idx_t - 1]:
                idx_s -= 1
            else:
                idx_t -= 1
        optimal_path.insert(0, (idx_s - 1, idx_t - 1))
    if idx_s > 1:
        idx_s -= 1
        optimal_path.insert(0, (idx_s - 1, idx_t - 1))
        while idx_s > 1:
            idx_s -= 1
            optimal_path.insert(0, (idx_s - 1, idx_t - 1))
    elif idx_t > 1:
        idx_t -= 1
        optimal_path.insert(0, (idx_s - 1, idx_t - 1))
        while idx_t > 1:
            idx_t -= 1
            optimal_path.insert(0, (idx_s - 1, idx_t - 1))
    del D_matrix
    prev_token_idx = -1
    start_pos = -1
    bounds = []
    for idx_s, idx_t in optimal_path:
        char_idx = char_indices[idx_s]
        token_idx = token_indices[idx_t]
        if token_idx != prev_token_idx:
            if prev_token_idx >= 0:
                if start_pos >= 0:
                    bounds.append((start_pos, char_idx))
                if token_idx >= 0:
                    err_msg = '{0} != {1}'.format(token_idx, prev_token_idx + 1)
                    assert token_idx == (prev_token_idx + 1), err_msg
                    start_pos = char_idx
                    prev_token_idx = token_idx
                else:
                    start_pos = -1
            else:
                if token_idx >= 0:
                    start_pos = char_idx
                    prev_token_idx = token_idx
                else:
                    start_pos = -1
    if (prev_token_idx >= 0) and (start_pos >= 0):
        bounds.append((start_pos, N_s))
    return bounds


def megre_bounds(source: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    new_bounds = []
    prev_bounds = None
    for cur_bounds in source:
        if prev_bounds is None:
            new_bounds.append(cur_bounds)
            prev_bounds = cur_bounds
        else:
            if cur_bounds[0] >= prev_bounds[1]:
                new_bounds.append(cur_bounds)
                prev_bounds = cur_bounds
    return new_bounds


def strip_bounds(text: str,
                 bounds: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    new_bounds = []
    for start_pos, end_pos in bounds:
        end_pos_ = end_pos - 1
        while end_pos_ > start_pos:
            if not text[end_pos_].isspace():
                break
            end_pos_ -= 1
        new_bounds.append((start_pos, end_pos_ + 1))
    return new_bounds


def is_item_in_sequence(re_for_item: Pattern, sequence: List[str]) -> bool:
    res = False
    for cur in sequence:
        search_res = re_for_item.search(cur)
        if search_res is None:
            continue
        if (search_res.start() < 0) or (search_res.end() < 0):
            continue
        res = True
        break
    return res


def parse_file(onf_name: str, src_name_for_log: str = '') -> \
        Tuple[List[Dict[str, Dict[str, Tuple[int, int]]]], str]:
    re_for_special_token = re.compile(r'^\-[A-Z]+\-$')
    file_name_for_log = onf_name if (len(src_name_for_log) == 0) \
        else src_name_for_log
    with codecs.open(onf_name, mode='r', encoding='utf-8',
                     errors='ignore') as fp:
        all_lines = list(map(lambda it: it.strip(), fp.readlines()))
    all_data = []
    if len(all_lines) == 0:
        return [], 'File "{0}" is empty!'.format(file_name_for_log)
    global_separator = '-----------------------------------------------------' \
                       '-----------------------------------------------------' \
                       '--------------'
    final_separator = '======================================================' \
                      '======================================================' \
                      '============'
    plain_sentence_head = 'Plain sentence:'
    plain_sentence_separator = '---------------'
    treebank_sentence_head = 'Treebanked sentence:'
    treebank_sentence_separator = '--------------------'
    tree_head = 'Tree:'
    tree_separator = '-----'
    leaves_head = 'Leaves:'
    leaves_separator = '-------'
    try:
        start_idx = all_lines.index(global_separator)
    except:
        start_idx = -1
    err_msg = 'File "{0}" has bad content!'.format(file_name_for_log)
    if start_idx < 0:
        return [], err_msg
    ok = True
    while start_idx >= 0:
        try:
            end_idx = all_lines[(start_idx + 1):].index(global_separator) + \
                      start_idx + 1
        except:
            try:
                end_idx = all_lines[(start_idx + 1):].index(final_separator) + \
                          start_idx + 1
            except:
                end_idx = -1
        if end_idx < 0:
            ok = False
            break
        try:
            plain_sentence_idx = all_lines[start_idx:end_idx].index(
                plain_sentence_head)
        except:
            plain_sentence_idx = -1
        if plain_sentence_idx < 0:
            ok = False
            break
        plain_sentence_idx += start_idx
        if all_lines[plain_sentence_idx + 1] != plain_sentence_separator:
            ok = False
            break
        try:
            treebank_sentence_idx = all_lines[start_idx:end_idx].index(
                treebank_sentence_head)
        except:
            treebank_sentence_idx = -1
        if treebank_sentence_idx < 0:
            ok = False
            break
        treebank_sentence_idx += start_idx
        if all_lines[treebank_sentence_idx + 1] != treebank_sentence_separator:
            ok = False
            break
        try:
            tree_idx = all_lines[start_idx:end_idx].index(tree_head)
        except:
            tree_idx = -1
        if tree_idx < 0:
            ok = False
            break
        tree_idx += start_idx
        if all_lines[tree_idx + 1] != tree_separator:
            ok = False
            break
        try:
            leaves_idx = all_lines[start_idx:end_idx].index(leaves_head)
        except:
            leaves_idx = -1
        if leaves_idx < 0:
            ok = False
            break
        leaves_idx += start_idx
        if all_lines[leaves_idx + 1] != leaves_separator:
            ok = False
            break
        if plain_sentence_idx <= start_idx:
            ok = False
            break
        if treebank_sentence_idx <= plain_sentence_idx:
            ok = False
            break
        if tree_idx <= treebank_sentence_idx:
            ok = False
            break
        if leaves_idx <= tree_idx:
            ok = False
            break
        plain_text = get_plain_text(all_lines, plain_sentence_idx + 2,
                                    treebank_sentence_idx)
        if len(plain_text) <= 0:
            ok = False
            break
        tree_text = get_plain_text(all_lines, tree_idx + 2, leaves_idx)
        try:
            tokens_with_lingvo_data = parse_tree(tree_text)
            named_entities_as_bio = parse_named_entities_labeling(
                all_lines[(leaves_idx + 2):end_idx],
                [cur[0] for cur in tokens_with_lingvo_data],
                file_name_for_log
            )
        except ValueError as e:
            tokens_with_lingvo_data = []
            named_entities_as_bio = []
            ok = False
            err_msg = str(e)
        if not ok:
            break
        tokens_with_labels = list(map(
            lambda it2: (it2[0][0], it2[0][1], it2[1]),
            filter(
                lambda it1: not is_item_in_sequence(re_for_special_token,
                                                    it1[0][1]),
                zip(tokens_with_lingvo_data, named_entities_as_bio)
            )
        ))
        if len(tokens_with_labels) == 0:
            ok = False
            break
        del tokens_with_lingvo_data, named_entities_as_bio, tree_text
        try:
            bounds_of_tokens = get_token_bounds(plain_text, tokens_with_labels)
        except:
            bounds_of_tokens = get_token_bounds_fuzzy(plain_text,
                                                      tokens_with_labels)
        token_idx = 0
        while token_idx < len(tokens_with_labels):
            token_text, _, _ = tokens_with_labels[token_idx]
            search_res = re_for_special_token.search(token_text)
            if search_res is None:
                token_idx += 1
            else:
                if (search_res.start() < 0) or (search_res.end() < 0):
                    token_idx += 1
                else:
                    token_start, token_end = bounds_of_tokens[token_idx]
                    plain_text = plain_text[:token_start] + \
                                 plain_text[token_end:]
                    del tokens_with_labels[token_idx]
                    for next_idx in range(token_idx + 1, len(bounds_of_tokens)):
                        bounds_of_tokens[next_idx] = (
                            bounds_of_tokens[next_idx][0] - token_end,
                            bounds_of_tokens[next_idx][1] - token_end
                        )
                    del bounds_of_tokens[token_idx]
        if len(tokens_with_labels) == 0:
            ok = False
            break
        syntactic_tags = set()
        new_data = {
            'text': plain_text,
            'morphology': dict(),
            'syntax': dict(),
            'entities': dict()
        }
        previous_entity = 'O'
        entity_start = -1
        for (token, lingvo, named_ent), cur_bounds in zip(tokens_with_labels,
                                                          bounds_of_tokens):
            if len(lingvo) > 1:
                syntactic_tags |= set(lingvo[:-1])
            if lingvo[-1] not in new_data['morphology']:
                new_data['morphology'][lingvo[-1]] = []
            new_data['morphology'][lingvo[-1]].append(cur_bounds)
            if named_ent == 'O':
                if previous_entity != 'O':
                    if previous_entity not in new_data['entities']:
                        new_data['entities'][previous_entity] = []
                    new_data['entities'][previous_entity].append(
                        (entity_start, cur_bounds[0]))
                    entity_start = -1
                    previous_entity = 'O'
            else:
                if named_ent.startswith('B-'):
                    if previous_entity != 'O':
                        if previous_entity not in new_data['entities']:
                            new_data['entities'][previous_entity] = []
                        new_data['entities'][previous_entity].append(
                            (entity_start, cur_bounds[0]))
                    entity_start = cur_bounds[0]
                    previous_entity = named_ent[2:]
        if previous_entity != 'O':
            if previous_entity not in new_data['entities']:
                new_data['entities'][previous_entity] = []
            new_data['entities'][previous_entity].append(
                (entity_start, len(plain_text)))
        if len(syntactic_tags) > 0:
            whole_sentence_tag = None
            for cur_tag in sorted(list(syntactic_tags)):
                new_data['syntax'][cur_tag] = []
                tag_text = ''
                tag_start = -1
                for (_, lingvo, _), cur_bounds in zip(tokens_with_labels,
                                                      bounds_of_tokens):
                    if cur_tag in lingvo[:-1]:
                        if tag_text != cur_tag:
                            if len(tag_text) > 0:
                                new_data['syntax'][cur_tag].append(
                                    (tag_start, cur_bounds[0])
                                )
                            tag_text = cur_tag
                            tag_start = cur_bounds[0]
                    else:
                        if len(tag_text) > 0:
                            new_data['syntax'][cur_tag].append(
                                (tag_start, cur_bounds[0])
                            )
                            tag_text = ''
                            tag_start = -1
                if len(tag_text) > 0:
                    new_data['syntax'][cur_tag].append(
                        (tag_start, len(plain_text))
                    )
                new_data['syntax'][cur_tag] = sorted(
                    new_data['syntax'][cur_tag],
                    key=lambda it: (it[0], it[0] - it[1])
                )
                while len(new_data['syntax'][cur_tag]) > 0:
                    tag_list = new_data['syntax'][cur_tag]
                    start_pos, end_pos = tag_list[0]
                    if (start_pos != 0) or (end_pos != len(plain_text)):
                        break
                    if whole_sentence_tag is None:
                        whole_sentence_tag = {cur_tag: [(0, len(plain_text))]}
                    new_data['syntax'][cur_tag] = tag_list[1:]
                    del tag_list
                new_data['syntax'][cur_tag] = megre_bounds(
                    new_data['syntax'][cur_tag])
                if len(new_data['syntax'][cur_tag]) == 0:
                    del new_data['syntax'][cur_tag]
            if len(new_data['syntax']) == 0:
                if whole_sentence_tag is None:
                    ok = False
                    break
                for cur_tag in whole_sentence_tag:
                    new_data['syntax'][cur_tag] = whole_sentence_tag[cur_tag]
        for data_key in new_data:
            if data_key != 'entities':
                if len(new_data[data_key]) == 0:
                    ok = False
                    err_msg += ' The {0} list is empty!'.format(data_key)
                    break
        del plain_text, tokens_with_labels
        for data_key in new_data:
            if data_key == 'text':
                continue
            for lingvo_key in new_data[data_key]:
                new_data[data_key][lingvo_key] = strip_bounds(
                    new_data['text'],
                    new_data[data_key][lingvo_key]
                )
        all_data.append(new_data)
        if all_lines[end_idx] == final_separator:
            start_idx = -1
        else:
            start_idx = end_idx
    if not ok:
        return all_data, err_msg
    return all_data, ''


def load_identifiers(file_name: str) -> List[str]:
    with codecs.open(file_name, mode='r', encoding='utf-8',
                     errors='ignore') as fp:
        lines = list(filter(
            lambda it2: len(it2) > 0,
            map(
                lambda it1: it1.strip(), fp.readlines()
            )
        ))
    if len(lines) == 0:
        err_msg = 'File "{0}" is empty!'.format(file_name)
        raise ValueError(err_msg)
    return [os.path.join(*split_filename_by_parts(cur)) for cur in lines]


def parse_splitting(dir_name: str) -> Dict[str, List[str]]:
    base_name = os.path.basename(dir_name)
    if len(base_name) == 0:
        raise ValueError('A directory name is empty!')
    identifiers = dict()
    if base_name == 'all':
        train_file_name = os.path.join(dir_name, 'train.id')
        development_file_name = os.path.join(dir_name, 'development.id')
        test_file_name = os.path.join(dir_name, 'test.id')
        if not os.path.isfile(train_file_name):
            err_msg = 'File "{0}" does not exist!'.format(train_file_name)
            raise ValueError(err_msg)
        if not os.path.isfile(development_file_name):
            err_msg = 'File "{0}" does not exist!'.format(development_file_name)
            raise ValueError(err_msg)
        if not os.path.isfile(test_file_name):
            err_msg = 'File "{0}" does not exist!'.format(test_file_name)
            raise ValueError(err_msg)
        if 'train' not in identifiers:
            identifiers['train'] = []
        identifiers['train'] += load_identifiers(train_file_name)
        if 'development' not in identifiers:
            identifiers['development'] = []
        identifiers['development'] += load_identifiers(development_file_name)
        if 'test' not in identifiers:
            identifiers['test'] = []
        identifiers['test'] += load_identifiers(test_file_name)
    else:
        subdirs = list(filter(
            lambda it3: os.path.isdir(it3),
            map(
                lambda it2: os.path.join(dir_name, it2),
                filter(
                    lambda it1: it1 not in {'.', '..'},
                    os.listdir(dir_name)
                )
            )
        ))
        for cur_subdir in subdirs:
            subdir_identifiers = parse_splitting(cur_subdir)
            if len(subdir_identifiers) > 0:
                if 'train' not in identifiers:
                    identifiers['train'] = []
                identifiers['train'] += subdir_identifiers['train']
                if 'development' not in identifiers:
                    identifiers['development'] = []
                identifiers['development'] += subdir_identifiers['development']
                if 'test' not in identifiers:
                    identifiers['test'] = []
                identifiers['test'] += subdir_identifiers['test']
            del subdir_identifiers
    return identifiers


def split_filename_by_parts(file_name: str) -> List[str]:
    left_part = os.path.dirname(file_name).strip()
    right_part = os.path.basename(file_name).strip()
    if len(right_part) == 0:
        err_msg = 'File name "{0}" cannot be splitted!'.format(file_name)
        raise ValueError(err_msg)
    parts = [right_part]
    if len(left_part) > 0:
        parts = split_filename_by_parts(left_part) + parts
    return parts


def check_onf_name(onf_name: str,
                   identifiers_for_splitting: Dict[str, List[str]]) -> str:
    keys_of_identifiers = set(identifiers_for_splitting.keys())
    if keys_of_identifiers != {'train', 'development', 'test'}:
        err_msg = '{0} are wrong subsets of the ' \
                  'Ontonotes 5.0!'.format(sorted(list(keys_of_identifiers)))
        raise ValueError(err_msg)
    prepared_onf_name = os.path.join(*split_filename_by_parts(onf_name))
    point_pos = prepared_onf_name.rfind('.')
    if point_pos >= 0:
        prepared_onf_name = prepared_onf_name[0:point_pos].strip()
        if len(prepared_onf_name) == 0:
            err_msg = 'File name "{0}" is wrong!'.format(onf_name)
            raise ValueError(err_msg)
    found_key = ''
    for idx, value in enumerate(identifiers_for_splitting['train']):
        if prepared_onf_name.endswith(value):
            found_key = 'train'
            break
    if len(found_key) == 0:
        for idx, value in enumerate(identifiers_for_splitting['development']):
            if prepared_onf_name.endswith(value):
                found_key = 'development'
                break
    if len(found_key) == 0:
        for idx, value in enumerate(identifiers_for_splitting['test']):
            if prepared_onf_name.endswith(value):
                found_key = 'test'
                break
    if len(found_key) == 0:
        raise ValueError('File name "{0}" is not found!'.format(onf_name))
    return found_key


def get_language_by_filename(onf_name: str) -> str:
    err_msg = '"{0}" is a wrong name for the ONF file! ' \
              'Language cannot be detected.'.format(onf_name)
    name_parts = split_filename_by_parts(onf_name)
    if len(name_parts) < 3:
        raise ValueError(err_msg)
    if 'annotations' not in name_parts:
        raise ValueError(err_msg)
    found_idx = name_parts.index('annotations')
    if (found_idx < 1) or (found_idx >= (len(name_parts) - 1)):
        raise ValueError(err_msg)
    if len(name_parts[found_idx - 1].strip()) == 0:
        raise ValueError(err_msg)
    return name_parts[found_idx - 1].strip()
