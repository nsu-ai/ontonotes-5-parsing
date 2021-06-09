from argparse import ArgumentParser
import codecs
import copy
import json
import os
from typing import List

from ontonotes5.utils import load_ontonotes5_from_json
from ontonotes5.utils import insert_new_bounds, calculate_distance


def find_similary_item(it: str, all_items: List[str]) -> int:
    found_idx = 0
    best_dist = calculate_distance(all_items[0], it)
    for idx, val in enumerate(all_items[1:]):
        cur_dist = calculate_distance(val, it)
        if cur_dist < best_dist:
            best_dist = cur_dist
            found_idx = idx + 1
    return found_idx


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-s',
        '--src',
        dest='source_file', type=str, required=True,
        help='The source *.json file with texts and their annotations '
             '(named entities, morphology and syntax).'
    )
    parser.add_argument(
        '-d',
        '--dst',
        dest='dst_file', type=str, required=True,
        help='The destination *.json file with texts and their annotations '
             '(named entities, morphology and syntax) after reducing of '
             'annotation classes.'
    )
    parser.add_argument(
        '-n',
        '--number',
        dest='maximal_number_of_enttity_types', type=int, required=True,
        help='A maximal number of entity types for each annotation class '
             '(morphology, syntax and named entities).'
    )
    cmd_args = parser.parse_args()

    err_msg = '{0} is too small value for maximal number of ' \
              'entity types.'.format(cmd_args.maximal_number_of_enttity_types)
    assert cmd_args.maximal_number_of_enttity_types >= 3, err_msg

    src_file_name = os.path.normpath(cmd_args.source_file)
    err_msg = 'File "{0}" does not exist!'.format(src_file_name)
    assert os.path.isfile(src_file_name), err_msg

    dst_file_name = os.path.normpath(cmd_args.dst_file)
    dst_file_dir = os.path.dirname(dst_file_name)
    if len(dst_file_dir) > 0:
        err_msg = 'Directory "{0}" does not exist!'.format(dst_file_dir)
        assert os.path.isdir(dst_file_dir), err_msg

    source_data = load_ontonotes5_from_json(src_file_name)
    entity_freq = dict()
    for data_part in source_data:
        source_samples = source_data[data_part]
        for idx, cur in enumerate(source_samples):
            for entity_class in {'morphology', 'syntax', 'entities'}:
                entities = cur[entity_class]
                if not isinstance(entities, dict):
                    err_msg = 'File "{0}", part {1}, sample {2}: the {3} is ' \
                              'specified incorrectly! Expected {4}, ' \
                              'got {5}.'.format(src_file_name, data_part, idx,
                                                entity_class,
                                                type({'a': 1, 'b': 2}),
                                                type(entities))
                    raise ValueError(err_msg)
                for entity_type in entities:
                    n_entities = len(entities[entity_type])
                    if n_entities > 0:
                        if entity_class not in entity_freq:
                            entity_freq[entity_class] = dict()
                        if entity_type in entity_freq[entity_class]:
                            entity_freq[entity_class][entity_type] += n_entities
                        else:
                            entity_freq[entity_class][entity_type] = n_entities

    rules = dict()
    clusters = dict()
    for entity_class in entity_freq:
        entities = [(entity_type, entity_freq[entity_class][entity_type])
                    for entity_type in entity_freq[entity_class]]
        entities.sort(key=lambda it: (-it[1], it[0]))
        rules[entity_class] = dict()
        clusters[entity_class] = dict()
        n = min(len(entities), cmd_args.maximal_number_of_enttity_types)
        for entity_type, freq in entities[0:n]:
            clusters[entity_class][entity_type] = {entity_type}
        if len(entities) > cmd_args.maximal_number_of_enttity_types:
            for entity_type, freq in entities[n:]:
                index_of_similar_item = find_similary_item(
                    entity_type,
                    [it[0] for it in entities[0:n]]
                )
                similar_entity_type = entities[index_of_similar_item][0]
                clusters[entity_class][similar_entity_type].add(entity_type)
        del entities
        entity_types = sorted(list(clusters[entity_class].keys()))
        new_clusters_of_entity_class = dict()
        for entity_type in entity_types:
            values = sorted(list(clusters[entity_class][entity_type]),
                            key=lambda it: (len(it), it))
            new_clusters_of_entity_class[values[0]] = set(values)
            for val in values:
                rules[entity_class][val] = values[0]
        clusters[entity_class] = new_clusters_of_entity_class
        del new_clusters_of_entity_class, entity_types
    del entity_freq
    assert set(clusters.keys()) == set(rules.keys())
    for entity_class in sorted(list(clusters.keys())):
        print('====================')
        print('Clusters of {0}:'.format(entity_class))
        print('====================')
        max_width = 0
        for entity_type in sorted(list(clusters[entity_class].keys())):
            if len(entity_type) > max_width:
                max_width = len(entity_type)
        for entity_type in sorted(list(clusters[entity_class].keys())):
            print('')
            print('  {0:<{1}} {2}'.format(
                entity_type + ':', max_width + 1,
                sorted(list(clusters[entity_class][entity_type]))
            ))
        print('')
    del clusters

    prepared_data = dict()
    for data_part in source_data:
        prepared_samples = []
        for sample_idx, old_sample in enumerate(source_data[data_part]):
            new_sample = dict()
            for entity_class in old_sample:
                if entity_class in rules:
                    old_entities = old_sample[entity_class]
                    new_entities = dict()
                    if not isinstance(old_entities, dict):
                        err_msg = 'File "{0}", part {1}, sample {2}: the {3} ' \
                                  'is specified incorrectly! Expected {4}, ' \
                                  'got {5}.'.format(src_file_name, data_part,
                                                    sample_idx, entity_class,
                                                    type({'a': 1, 'b': 2}),
                                                    type(old_entities))
                        raise ValueError(err_msg)
                    for entity_type in old_entities:
                        if not isinstance(old_entities[entity_type], list):
                            err_msg = 'Sample {0} of {1}: item "{2}" must be ' \
                                      'specified as {3}, but it is specified ' \
                                      'as a {4}'.format(
                                sample_idx, data_part, entity_type,
                                type([1, 2]), type(old_entities[entity_type])
                            )
                            raise ValueError(err_msg)
                        other_type = rules[entity_class][entity_type]
                        if other_type in new_entities:
                            for span_bounds in old_entities[entity_type]:
                                new_entities[other_type] = insert_new_bounds(
                                    new_bounds=span_bounds,
                                    old_bounds_list=new_entities[other_type]
                                )
                        else:
                            new_entities[other_type] = copy.deepcopy(
                                old_entities[entity_type]
                            )
                    new_sample[entity_class] = new_entities
                    del new_entities, old_entities
                else:
                    new_sample[entity_class] = old_sample[entity_class]
            prepared_samples.append(new_sample)
        prepared_data[data_part] = prepared_samples
        del prepared_samples

    with codecs.open(dst_file_name, mode='w', encoding='utf-8',
                     errors='ignore') as fp:
        json.dump(prepared_data, fp=fp, ensure_ascii=False, indent=4,
                  sort_keys=True)


if __name__ == '__main__':
    main()
