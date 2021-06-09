import codecs
import json
import os
import re
from typing import Dict, Pattern, List, Tuple, Union

from Levenshtein import distance
from nltk import wordpunct_tokenize
import numpy as np


def tokenize_any_text(s: str) -> List[str]:
    re_for_cjk = re.compile(
        "([\uac00-\ud7a3]+|[\u3040-\u30ff]+|[\u4e00-\u9FFF]+)"
    )
    ok = True
    cjk_bounds = []
    start_pos = 0
    while ok:
        search_res = re_for_cjk.search(s[start_pos:])
        if search_res is None:
            ok = False
        elif search_res.start() < 0:
            ok = False
        elif search_res.end() <= search_res.start():
            ok = False
        else:
            cjk_bounds.append(
                (
                    start_pos + search_res.start(),
                    start_pos + search_res.end()
                )
            )
            start_pos = start_pos + search_res.end()
    if len(cjk_bounds) == 0:
        tokens = wordpunct_tokenize(s)
    else:
        tokens = []
        start_pos = 0
        for cur in cjk_bounds:
            if len(s[start_pos:cur[0]].strip()) > 0:
                tokens += wordpunct_tokenize(s[start_pos:cur[0]].strip())
            tokens += [s[char_idx:(char_idx + 1)]
                       for char_idx in range(cur[0], cur[1])]
            start_pos = cur[1]
        cur = cjk_bounds[-1]
        if len(s[cur[1]:].strip()) > 0:
            tokens += wordpunct_tokenize(s[cur[1]:])
    return tokens


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
    for token, _, _ in tokens_with_info:
        found_idx = source_text[start_pos:].find(token)
        if found_idx < 0:
            err_msg = 'Token "{0}" is not found in the text "{1}"! ' \
                      'start_pos = {2}'.format(token, source_text, start_pos)
            raise ValueError(err_msg)
        found_idx += start_pos
        bounds.append((found_idx, found_idx + len(token)))
        start_pos = found_idx + len(token)
    return bounds


def find_subword_bounds(word: str, subwords: List[str] ) -> \
        Tuple[List[Tuple[int, int]], int]:
    err_msg = 'Word {0} does not correspond to sub-words {1}.'.format(
        word, subwords
    )
    if len(word) < len(subwords):
        raise ValueError(err_msg)
    if len(word) == len(subwords):
        bounds = [(idx, idx + 1) for idx in range(len(word))]
        dist = sum(map(
            lambda it: distance(word[it[0][0]:it[0][1]], it[1]),
            zip(bounds, subwords)
        ))
        return bounds, dist
    if len(subwords) == 1:
        bounds = [(0, len(word))]
        dist = distance(word, subwords[0])
        return bounds, dist
    start_pos = 0
    variants_of_end_pos = list(range(1, len(word) - len(subwords) + 2))
    best_end_pos = variants_of_end_pos[0]
    best_dist = distance(word[start_pos:best_end_pos], subwords[0])
    next_bounds, next_dist = find_subword_bounds(word[best_end_pos:],
                                                 subwords[1:])
    best_dist += next_dist
    best_bounds = [(start_pos, best_end_pos)] + \
                  [(it[0] + best_end_pos, it[1] + best_end_pos)
                   for it in next_bounds]
    del next_bounds
    for cur_end_pos in variants_of_end_pos[1:]:
        cur_dist = distance(word[start_pos:cur_end_pos], subwords[0])
        next_bounds, next_dist = find_subword_bounds(word[cur_end_pos:],
                                                     subwords[1:])
        cur_dist += next_dist
        cur_bounds = [(start_pos, cur_end_pos)] + \
                     [(it[0] + cur_end_pos, it[1] + cur_end_pos)
                      for it in next_bounds]
        del next_bounds
        if cur_dist < best_dist:
            best_dist = cur_dist
            best_bounds = cur_bounds
        del cur_bounds
    return best_bounds, best_dist


def get_token_bounds_fuzzy(
        source_text: str,
        tokens_with_info: List[Tuple[str, List[str], str]],
        insertion_cost: float = 3.0,
        deletion_cost: float = 1.0
) -> List[Tuple[int, int]]:
    if len(source_text.strip()) == 0:
        return []
    source_tokens_with_info = list(map(
        lambda it3: (it3, ['1', '2'], 'O'),
        filter(
            lambda it2: len(it2) > 0,
            map(
                lambda it1: it1.strip(),
                tokenize_any_text(source_text)
            )
        )
    ))
    token_bounds = get_token_bounds(source_text, source_tokens_with_info)
    n1 = len(source_tokens_with_info)
    n2 = len(tokens_with_info)
    if n1 == n2:
        return token_bounds
    source_tokens = [cur[0] for cur in source_tokens_with_info]
    del source_tokens_with_info
    treebanked_tokens = [cur[0] for cur in tokens_with_info]
    N_s = n1
    N_t = n2
    D_matrix = np.zeros((N_s + 1, N_t + 1), dtype=np.float32)
    for idx_t in range(1, N_t + 1):
        D_matrix[0, idx_t] = D_matrix[0, idx_t - 1] + insertion_cost
    for idx_s in range(1, N_s + 1):
        D_matrix[idx_s, 0] = D_matrix[idx_s - 1, 0] + deletion_cost
    for idx_s in range(1, N_s + 1):
        for idx_t in range(1, N_t + 1):
            if source_tokens[idx_s - 1] == treebanked_tokens[idx_t - 1]:
                cur_dist = 0.0
            else:
                cur_dist = float(distance(
                    source_tokens[idx_s - 1],
                    treebanked_tokens[idx_t - 1]
                ))
                cur_dist /= float(max(
                    len(source_tokens[idx_s - 1]),
                    len(treebanked_tokens[idx_t - 1])
                ))
            D_matrix[idx_s, idx_t] = min(
                D_matrix[idx_s - 1, idx_t] + deletion_cost,
                D_matrix[idx_s, idx_t - 1] + insertion_cost,
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
    pairs = [
        [
            [optimal_path[0][0]],
            [optimal_path[0][1]]
        ]
    ]
    for idx_s, idx_t in optimal_path[1:]:
        if idx_s > pairs[-1][0][-1]:
            if idx_t > pairs[-1][1][-1]:
                pairs.append(
                    [
                        [idx_s],
                        [idx_t]
                    ]
                )
            else:
                pairs[-1][0].append(idx_s)
        else:
            if idx_t > pairs[-1][1][-1]:
                pairs[-1][1].append((idx_t))
    bounds = []
    for cur_pair in pairs:
        source_indices = cur_pair[0]
        tokenized_indices = cur_pair[1]
        span_start = token_bounds[source_indices[0]][0]
        span_end = token_bounds[source_indices[-1]][1]
        if len(tokenized_indices) > 1:
            start_idx = tokenized_indices[0]
            end_idx = tokenized_indices[-1]
            next_bounds, _ = find_subword_bounds(
                source_text[span_start:span_end],
                treebanked_tokens[start_idx:(end_idx + 1)]
            )
            bounds += [(it[0] + span_start, it[1] + span_start)
                       for it in next_bounds]
        else:
            bounds.append((span_start, span_end))
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
        source_token = text[start_pos:end_pos]
        stripped_token = source_token.strip()
        start_pos_ = source_token.find(stripped_token)
        assert start_pos_ >= 0
        end_pos_ = start_pos_ + len(stripped_token)
        new_bounds.append((start_pos + start_pos_, start_pos + end_pos_))
    return new_bounds


def unite_overlapped_bounds(bounds: List[Tuple[int, int]]) -> \
        List[Tuple[int, int]]:
    new_bounds = []
    if len(bounds) == 0:
        return new_bounds
    new_bounds = [bounds[0]]
    for span_start, span_end in bounds[1:]:
        if span_start > new_bounds[-1][1]:
            new_bounds.append((span_start, span_end))
        else:
            if span_start < new_bounds[-1][0]:
                err_msg = 'Item {0} in bounds list {1} is wrong!'.format(
                    (span_start, span_end),
                    bounds
                )
                raise ValueError(err_msg)
            new_bounds[-1] = (new_bounds[-1][0], span_end)
    return new_bounds


def check_bounds(text: str,
                 bounds: List[Tuple[int, int]]) -> str:
    res = ''
    prev_pos = 0
    for start_pos, end_pos in sorted(bounds, key=lambda it: (it[0], it[1])):
        err_msg = 'Item {0} in the bounds list {1} is wrong!'.format(
            (start_pos, end_pos),
            bounds
        )
        if start_pos >= end_pos:
            res = err_msg
            res += ' start_pos={0} >= end_pos={1}'.format(start_pos, end_pos)
            break
        if start_pos < prev_pos:
            res = err_msg
            res += ' start_pos={0} < prev_pos={1}'.format(start_pos, prev_pos)
            break
        if end_pos > len(text):
            res = err_msg
            res += ' end_pos={0} > len(text)={1}'.format(end_pos, len(text))
            break
        span_text = text[start_pos:end_pos]
        if len(span_text.strip()) == 0:
            res = err_msg
            res += ' text[{0}:{1}] is empty!'.format(start_pos, end_pos)
            break
        if span_text != span_text.strip():
            res = err_msg
            res += ' text[{0}:{1}] != text[{0}:{1}].strip()'.format(start_pos,
                                                                    end_pos)
            break
        prev_pos = end_pos
    return res


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
    number_of_tokenization_restarts = 5
    special_token_re = re.compile(r'^(\-[A-Z]+\-|EDITED)$')
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
                lambda it1: not is_item_in_sequence(special_token_re,
                                                    it1[0][1]),
                zip(tokens_with_lingvo_data, named_entities_as_bio)
            )
        ))
        if len(tokens_with_labels) == 0:
            ok = False
            break
        del tokens_with_lingvo_data, named_entities_as_bio, tree_text
        tokens_with_labels_ = []
        for it in tokens_with_labels:
            search_res = special_token_re.search(it[0])
            if search_res is not None:
                if (search_res.start() >= 0) and (search_res.end() >= 0):
                    plain_text = plain_text.replace(it[0], ' ')
                else:
                    tokens_with_labels_.append(it)
            else:
                tokens_with_labels_.append(it)
        del tokens_with_labels
        tokens_with_labels = tokens_with_labels_
        plain_text_ = plain_text.replace('  ', ' ')
        while plain_text_ != plain_text:
            plain_text = plain_text_
            plain_text_ = plain_text.replace('  ', ' ')
        del plain_text_
        plain_text = plain_text.strip()
        can_tokenize = True
        bounds_of_tokens = []
        try:
            bounds_of_tokens = get_token_bounds(plain_text, tokens_with_labels)
        except:
            insertion_cost = 1.0
            deletion_cost = 1.0
            try:
                bounds_of_tokens = get_token_bounds_fuzzy(
                    plain_text,
                    tokens_with_labels,
                    insertion_cost=insertion_cost, deletion_cost=deletion_cost
                )
            except:
                can_tokenize = False
            restart_counter = 1
            while not can_tokenize:
                insertion_cost += 1.0
                try:
                    bounds_of_tokens = get_token_bounds_fuzzy(
                        plain_text,
                        tokens_with_labels,
                        insertion_cost=insertion_cost,
                        deletion_cost=deletion_cost
                    )
                    can_tokenize = (check_bounds(plain_text,
                                                 bounds_of_tokens) == '')
                except:
                    can_tokenize = False
                restart_counter += 1
                if restart_counter > number_of_tokenization_restarts:
                    break
            if not can_tokenize:
                insertion_cost = 1.0
                while not can_tokenize:
                    deletion_cost += 1.0
                    try:
                        bounds_of_tokens = get_token_bounds_fuzzy(
                            plain_text,
                            tokens_with_labels,
                            insertion_cost=insertion_cost,
                            deletion_cost=deletion_cost
                        )
                        can_tokenize = (check_bounds(plain_text,
                                                     bounds_of_tokens) == '')
                    except:
                        can_tokenize = False
                    restart_counter += 1
                    if restart_counter > number_of_tokenization_restarts:
                        break
        if not can_tokenize:
            ok = False
            break
        if len(tokens_with_labels) == 0:
            ok = False
            break
        if len(tokens_with_labels) != len(bounds_of_tokens):
            err_msg3 = err_msg
            err_msg3 += ' Sample {0} cannot be tokenized! {1} != {2}'.format(
                len(all_data), len(tokens_with_labels),
                len(bounds_of_tokens)
            )
            raise ValueError(err_msg3)
        for token_text, (token_start, token_end) in zip(
                map(lambda it: it[0], tokens_with_labels),
                bounds_of_tokens
        ):
            if token_start >= token_end:
                err_msg3 = err_msg
                err_msg3 += ' Sample {0} cannot be tokenized! ' \
                            'Token {1} has wrong bounds {2}'.format(
                    len(all_data), token_text, (token_start, token_end)
                )
                raise ValueError(err_msg3)
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
            max_depth = len(tokens_with_labels[0][1]) - 1
            for (_, lingvo, _) in tokens_with_labels[1:]:
                if len(lingvo[:-1]) > max_depth:
                    max_depth = len(lingvo[:-1])
            if max_depth > 0:
                for depth_idx in range(max_depth):
                    tag_text = ''
                    tag_start = -1
                    tag_end = -1
                    for (_, lingvo, _), cur_bounds in zip(tokens_with_labels,
                                                          bounds_of_tokens):
                        if len(lingvo[:-1]) > depth_idx:
                            if tag_text == lingvo[depth_idx]:
                                tag_end = cur_bounds[1]
                            else:
                                if len(tag_text) > 0:
                                    if tag_text not in new_data['syntax']:
                                        new_data['syntax'][tag_text] = []
                                    new_data['syntax'][tag_text].append(
                                        (tag_start, tag_end)
                                    )
                                tag_text = lingvo[depth_idx]
                                tag_start = cur_bounds[0]
                                tag_end = cur_bounds[1]
                        else:
                            if len(tag_text) > 0:
                                if tag_text not in new_data['syntax']:
                                    new_data['syntax'][tag_text] = []
                                new_data['syntax'][tag_text].append(
                                    (tag_start, tag_end)
                                )
                                tag_text = ''
                                tag_start = -1
                                tag_end = -1
                    if len(tag_text) > 0:
                        if tag_text not in new_data['syntax']:
                            new_data['syntax'][tag_text] = []
                        new_data['syntax'][tag_text].append(
                            (tag_start, tag_end)
                        )
            whole_sentence_tag = None
            for cur_tag in sorted(list(new_data['syntax'].keys())):
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
                new_bounds = strip_bounds(
                    new_data['text'],
                    new_data[data_key][lingvo_key]
                )
                err_msg2 = check_bounds(new_data['text'], new_bounds)
                if len(err_msg2) > 0:
                    err_msg3 = err_msg + ' Sample {0}, {1} in {2}: {3}'.format(
                        len(all_data), lingvo_key, data_key,
                        err_msg2
                    )
                    err_msg3 += ' Text is {0}'.format(new_data['text'])
                    raise ValueError(err_msg3)
                new_data[data_key][lingvo_key] = unite_overlapped_bounds(
                    sorted(new_bounds)
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


def insert_new_bounds(new_bounds: Tuple[int, int],
                      old_bounds_list: List[Tuple[int, int]]) \
        -> List[Tuple[int, int]]:
    if len(old_bounds_list) == 0:
        return [new_bounds]
    old_bounds_list_ = sorted(old_bounds_list, key=lambda it: (it[0], it[1]))
    prev_pos = -1
    for span_start, span_end in old_bounds_list_:
        err_msg = 'Item {0} of bounds list {1} is wrong!'.format(
            (span_start, span_end), old_bounds_list_
        )
        if span_start >= span_end:
            raise ValueError(err_msg)
        if span_start <= prev_pos:
            raise ValueError(err_msg)
        prev_pos = span_end
    if new_bounds[1] < old_bounds_list_[0][0]:
        return [new_bounds] + old_bounds_list_
    if new_bounds[0] > old_bounds_list_[-1][1]:
        return old_bounds_list_ + [new_bounds]
    n = max(new_bounds[1], old_bounds_list_[-1][1])
    indices = np.zeros(shape=(n,), dtype=np.int32)
    for span_start, span_end in old_bounds_list_:
        for idx in range(span_start, span_end):
            indices[idx] = 1
    del old_bounds_list_
    span_start, span_end = new_bounds
    for idx in range(span_start, span_end):
        indices[idx] = 1
    new_bounds_list = []
    span_start = -1
    for idx in range(n):
        if indices[idx] > 0:
            if span_start < 0:
                span_start = idx
        else:
            if span_start >= 0:
                new_bounds_list.append((span_start, idx))
                span_start = -1
    if span_start >= 0:
        new_bounds_list.append((span_start, n))
    del indices
    return new_bounds_list


def calculate_distance(lingvo1: str, lingvo2: str) -> int:
    if lingvo1.strip() == lingvo2.strip():
        return 0
    re_for_alpha = re.compile(r'\w+')
    search_res = re_for_alpha.search(lingvo1)
    if search_res is None:
        has_alpha1 = False
    elif (search_res.start() >= 0) and (search_res.end() > search_res.start()):
        has_alpha1 = True
    else:
        has_alpha1 = False
    search_res = re_for_alpha.search(lingvo2)
    if search_res is None:
        has_alpha2 = False
    elif (search_res.start() >= 0) and (search_res.end() > search_res.start()):
        has_alpha2 = True
    else:
        has_alpha2 = False
    if (has_alpha1 and (not has_alpha2)) or (has_alpha2 and (not has_alpha1)):
        return (len(lingvo1) + len(lingvo2)) * 10
    re_for_splitting = re.compile('[\.\+\:\-\=]')
    parts1 = list(filter(
        lambda it2: len(it2) > 0,
        map(
            lambda it1: it1.strip(),
            re_for_splitting.split(lingvo1)
        )
    ))
    parts2 = list(filter(
        lambda it2: len(it2) > 0,
        map(
            lambda it1: it1.strip(),
            re_for_splitting.split(lingvo2)
        )
    ))
    if (len(parts1) == 0) and (len(parts2) == 0):
        return 0
    if len(parts1) == 0:
        return len('-'.join(parts2)) + 2
    if len(parts2) == 0:
        return len('-'.join(parts1)) + 2
    s1 = '-'.join(parts1)
    s2 = '-'.join(parts2)
    if s1 == s2:
        return 1
    if s1.startswith(s2) or s2.startswith(s1):
        return 2
    identical_parts_number = 0
    for idx in range(min(len(parts1), len(parts2))):
        if parts1[idx] != parts2[idx]:
            break
        identical_parts_number = idx + 1
    if identical_parts_number > 0:
        s1 = '-'.join(parts1[identical_parts_number:])
        s2 = '-'.join(parts2[identical_parts_number:])
        return distance(s1, s2) + 2
    return distance(s1, s2) * 10 + 2


def load_ontonotes5_from_json(file_name: str) -> Dict[
    str,
    List[Dict[str, Union[str, List[Tuple[int, int]]]]]
]:
    with codecs.open(file_name, mode='r', encoding='utf-8') as fp:
        source_data = json.load(fp)
    if not isinstance(source_data, dict):
        err_msg = 'File "{0}" contains wrong data! Expected {1}, ' \
                  'got {2}.'.format(file_name, type({'a': 1, 'b': 2}),
                                    type(source_data))
        raise ValueError(err_msg)
    true_entity_classes = {'syntax', 'morphology', 'entities'}
    prepared_data = dict()
    for data_part in source_data:
        samples = source_data[data_part]
        prepared_data[data_part] = []
        if not isinstance(samples, list):
            err_msg = 'The {0} part in the file "{1}" contains wrong data! ' \
                      'Expected {2}, got {3}.'.format(
                data_part, file_name,
                type([1, 2]), type(samples)
            )
            raise ValueError(err_msg)
        for sample_idx, cur_sample in enumerate(samples):
            max_end_pos = 0
            prepared_sample = dict()
            if not isinstance(cur_sample, dict):
                err_msg = 'Sample {0} of the {1} part in the file "{2}" ' \
                          'contains wrong data! Expected {3}, got {4}.'.format(
                    sample_idx, data_part, file_name,
                    type({'a': 1, 'b': 2}), type(cur_sample)
                )
                raise ValueError(err_msg)
            if 'text' not in cur_sample:
                err_msg = 'Sample {0} of the {1} part in the file "{2}" ' \
                          'contains wrong data! Information about text ' \
                          'is not specified.'.format(
                    sample_idx, data_part, file_name
                )
                raise ValueError(err_msg)
            unknown_keys = true_entity_classes - set(cur_sample.keys())
            if len(unknown_keys) > 0:
                unknown_keys = sorted(list(unknown_keys))
                if len(unknown_keys) > 1:
                    unknown_keys_description = ', '.join(unknown_keys[:-1])
                    unknown_keys_description += (', and ' + unknown_keys[-1])
                else:
                    unknown_keys_description = unknown_keys[0]
                err_msg = 'Sample {0} of the {1} part in the file "{2}" ' \
                          'contains wrong data! Information about {3} is not ' \
                          'specified.'.format(
                    sample_idx, data_part, file_name,
                    unknown_keys_description
                )
                raise ValueError(err_msg)
            for entity_class in true_entity_classes:
                prepared_sample[entity_class] = dict()
                if not isinstance(cur_sample[entity_class], dict):
                    err_msg = 'Sample {0} of the {1} part in the file "{2}" ' \
                              'contains wrong data! Information about {3} is ' \
                              'specified incorrectly! Expected {4}, ' \
                              'got {5}.'.format(
                        sample_idx, data_part, file_name, entity_class,
                        type({'a': 1, 'b': 2}), type(cur_sample[entity_class])
                    )
                    raise ValueError(err_msg)
                for entity_type in cur_sample[entity_class]:
                    old_bounds_of_spans = cur_sample[entity_class][entity_type]
                    prepared_bounds = []
                    if not isinstance(old_bounds_of_spans, list):
                        err_msg = 'Sample {0} of the {1} part in the file ' \
                                  '"{2}" contains wrong data! Bounds of {3} ' \
                                  'in the {4} are specified incorrectly! ' \
                                  'Expected {5}, got {6}.'.format(
                            sample_idx, data_part, file_name,
                            entity_type, entity_class,
                            type([1, 2]), type(cur_sample[entity_class])
                        )
                        raise ValueError(err_msg)
                    if len(old_bounds_of_spans) == 0:
                        err_msg = 'Sample {0} of the {1} part in the file ' \
                                  '"{2}" contains wrong data! Bounds of {3} ' \
                                  'in the {4} are specified incorrectly! ' \
                                  'Expected a non-empty list, got an ' \
                                  'empty one.'.format(
                            sample_idx, data_part, file_name,
                            entity_type, entity_class
                        )
                        raise ValueError(err_msg)
                    prev_pos = -1
                    old_bounds_of_spans.sort(key=lambda it: (it[0], it[1]))
                    for cur_bounds in old_bounds_of_spans:
                        err_msg = 'Sample {0} of the {1} part in the file ' \
                                  '"{2}" contains wrong data! Bounds of {3} ' \
                                  'in the {4} are specified incorrectly! ' \
                                  'Item {5} in the bounds list {6} ' \
                                  'is inadmissible.'.format(
                            sample_idx, data_part, file_name,
                            entity_type, entity_class,
                            cur_bounds, old_bounds_of_spans
                        )
                        if (not isinstance(cur_bounds, list)) and \
                                (not isinstance(cur_bounds, tuple)):
                            raise ValueError(err_msg)
                        if len(cur_bounds) != 2:
                            raise ValueError(err_msg)
                        start_pos = cur_bounds[0]
                        end_pos = cur_bounds[1]
                        if start_pos >= end_pos:
                            raise ValueError(err_msg)
                        if start_pos <= prev_pos:
                            raise ValueError(err_msg)
                        if end_pos > len(cur_sample['text']):
                            raise ValueError(err_msg)
                        span_text = cur_sample['text'][start_pos:end_pos]
                        err_msg = 'Sample {0} of the {1} part in the file ' \
                                  '"{2}" contains wrong data! Bounds of {3} ' \
                                  'in the {4} are specified incorrectly! ' \
                                  'Item {5} in the bounds list {6} is ' \
                                  'inadmissible, because its text is ' \
                                  'empty.'.format(
                            sample_idx, data_part, file_name,
                            entity_type, entity_class,
                            cur_bounds, old_bounds_of_spans
                        )
                        if len(span_text.strip()) == 0:
                            raise ValueError(err_msg)
                        stripped_span_text = span_text.strip()
                        if span_text != stripped_span_text:
                            found_idx = span_text.find(stripped_span_text)
                            if found_idx > 0:
                                start_pos += found_idx
                                end_pos = start_pos + len(stripped_span_text)
                        prepared_bounds.append((start_pos, end_pos))
                        prev_pos = end_pos
                        if end_pos > max_end_pos:
                            max_end_pos = end_pos
                    prepared_sample[entity_class][entity_type] = prepared_bounds
                    del prepared_bounds
            for sample_key in (set(cur_sample.keys()) - true_entity_classes):
                prepared_sample[sample_key] = cur_sample[sample_key]
            prepared_data[data_part].append(prepared_sample)
            del prepared_sample
            if max_end_pos > len(cur_sample['text']):
                err_msg = 'Sample {0} of the {1} part in the file "{2}" ' \
                          'contains wrong data! List of span bounds does not ' \
                          'correspond to the text. {3} > {4}'.format(
                    sample_idx, data_part, file_name,
                    max_end_pos, len(cur_sample['text'])
                )
                raise ValueError(err_msg)
    return prepared_data


def get_language_frequencies(
        data: List[Dict[str, Union[str, Dict[str, Tuple[int, int]]]]]
) -> List[Tuple[str, int]]:
    languages_dict = dict()
    for sample_idx, sample_data in enumerate(data):
        if 'language' not in sample_data:
            raise ValueError('Sample {0} is wrong! The "language" key is '
                             'not found!'.format(sample_idx))
        cur_lang = sample_data['language']
        languages_dict[cur_lang] = languages_dict.get(cur_lang, 0) + 1
    languages = []
    for cur_lang in languages_dict:
        languages.append((cur_lang, languages_dict[cur_lang]))
    return sorted(languages, key=lambda it: (-it[1], it[0]))


def get_entity_frequencies(
        data: List[Dict[str, Union[str, Dict[str, Tuple[int, int]]]]],
        language: str = ''
) -> List[Tuple[str, int]]:
    entities_dict = dict()
    for sample_idx, sample_data in enumerate(data):
        if 'language' not in sample_data:
            raise ValueError('Sample {0} is wrong! The "language" key is '
                             'not found!'.format(sample_idx))
        if 'entities' not in sample_data:
            raise ValueError('Sample {0} is wrong! The "entities" key is '
                             'not found!'.format(sample_idx))
        cur_lang = sample_data['language']
        cur_entities = sample_data['entities']
        if (len(language) > 0) and (language != cur_lang):
            continue
        for entity_type in cur_entities:
            entity_bounds = cur_entities[entity_type]
            entities_dict[entity_type] = entities_dict.get(entity_type, 0) + \
                                         len(entity_bounds)
    entities = []
    for entity_type in entities_dict:
        entities.append((entity_type, entities_dict[entity_type]))
    return sorted(entities, key=lambda it: (-it[1], it[0]))
