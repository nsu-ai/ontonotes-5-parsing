from argparse import ArgumentParser
import codecs
import gc
import json
import os
import random
import tarfile
from tempfile import NamedTemporaryFile

from tqdm import tqdm

from ontonotes5.utils import parse_file, parse_splitting, check_onf_name
from ontonotes5.utils import get_language_by_filename
from ontonotes5.utils import get_language_frequencies, get_entity_frequencies


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-s',
        '--src',
        dest='source_file', type=str, required=True,
        help='The source *.tgz file with gzipped Ontonotes 5 dataset (see '
             'https://catalog.ldc.upenn.edu/LDC2013T19).'
    )
    parser.add_argument(
        '-d',
        '--dst',
        dest='dst_file', type=str, required=True,
        help='The destination *.json file with texts and their annotations '
             '(named entities, morphology and syntax).'
    )
    parser.add_argument(
        '-i',
        '--ids',
        dest='train_dev_test_ids', type=str, required=False, default=None,
        help='The directory with identifiers list, which is described the '
             'Ontonotes 5 splitting by subsets for training, development '
             '(validation) and final testing (see '
             'http://conll.cemantix.org/2012/download/ids/).'
    )
    parser.add_argument(
        '-r',
        '--random',
        dest='random_seed', type=int, required=False, default=None,
        help='A random seed.'
    )
    cmd_args = parser.parse_args()

    if cmd_args.random_seed is not None:
        random.seed(cmd_args.random_seed)

    src_file_name = os.path.normpath(cmd_args.source_file)
    err_msg = 'File "{0}" does not exist!'.format(src_file_name)
    assert os.path.isfile(src_file_name), err_msg

    dst_file_name = os.path.normpath(cmd_args.dst_file)
    dst_file_dir = os.path.dirname(dst_file_name)
    if len(dst_file_dir) > 0:
        err_msg = 'Directory "{0}" does not exist!'.format(dst_file_dir)
        assert os.path.isdir(dst_file_dir), err_msg

    if cmd_args.train_dev_test_ids is None:
        ids_dir_name = None
    else:
        ids_dir_name = os.path.normpath(cmd_args.train_dev_test_ids)
        err_msg = 'Directory "{0}" does not exist!'.format(ids_dir_name)
        assert os.path.isdir(dst_file_dir), err_msg

    data_for_training = []
    data_for_validation = []
    data_for_testing = []
    if ids_dir_name is None:
        splitting = None
    else:
        splitting = parse_splitting(ids_dir_name)
        assert len(set(splitting['train']) & set(splitting['test'])) == 0
        assert len(set(splitting['train']) & set(splitting['development'])) == 0
        assert len(set(splitting['development']) & set(splitting['test'])) == 0
    files_with_errors = []
    with tarfile.open(src_file_name, mode='r:*', encoding='utf-8') as tgz_fp:
        onf_names = list(map(
            lambda it2: it2.name,
            filter(
                lambda it1: it1.isfile() and it1.name.endswith('.onf'),
                tgz_fp.getmembers()
            )
        ))
        number_of_members = len(onf_names)
        err_msg = 'There are no labeled texts with *.onf extension in the ' \
                  '"{0}"!'.format(src_file_name)
        assert number_of_members > 0, err_msg
        for cur_name in tqdm(onf_names):
            language = get_language_by_filename(cur_name)
            tmp_name = None
            try:
                with NamedTemporaryFile(mode='w', delete=False) as tmp_fp:
                    tmp_name = tmp_fp.name
                binary_stream = tgz_fp.extractfile(cur_name)
                if binary_stream is not None:
                    binary_data = binary_stream.read()
                    with open(tmp_name, 'wb') as tmp_fp:
                        tmp_fp.write(binary_data)
                    del binary_data, binary_stream
                    parsed, err_msg_2 = parse_file(tmp_name, cur_name)
                    if err_msg_2 != '':
                        files_with_errors.append((cur_name, err_msg_2))
                    n = len(parsed)
                    if n > 0:
                        for idx in range(n):
                            parsed[idx]['language'] = language
                        if splitting is None:
                            data_for_training += parsed
                        else:
                            dst_key = check_onf_name(cur_name, splitting)
                            if dst_key == 'train':
                                data_for_training += parsed
                            elif dst_key == 'development':
                                data_for_validation += parsed
                            elif dst_key == 'test':
                                data_for_testing += parsed
            finally:
                if tmp_name is not None:
                    if os.path.isfile(tmp_name):
                        os.remove(tmp_name)
            gc.collect()

    with codecs.open(dst_file_name, mode='w', encoding='utf-8',
                     errors='ignore') as fp:
        random.shuffle(data_for_training)
        res = {'TRAINING': data_for_training}
        if splitting is None:
            assert len(data_for_validation) == 0
            assert len(data_for_testing) == 0
        else:
            assert len(data_for_validation) > 0
            assert len(data_for_testing) > 0
            random.shuffle(data_for_validation)
            res['VALIDATION'] = data_for_validation
            random.shuffle(data_for_testing)
            res['TESTING'] = data_for_testing
        json.dump(res, fp=fp, ensure_ascii=False, indent=4, sort_keys=True)

    print('{0} files are processed.'.format(number_of_members))
    n_errors = len(files_with_errors)
    if n_errors > 0:
        print('{0} files from them contain some errors.'.format(n_errors))
        print('They are:')
        for filename, err_msg in files_with_errors:
            print('    file name "{0}"'.format(filename))
            print('        error "{0}"'.format(err_msg))
    assert len(data_for_training) > 0
    if splitting is None:
        print('{0} samples are loaded...'.format(len(data_for_training)))
        languages_for_training = get_language_frequencies(data_for_training)
        print('By languages:')
        for lang, freq in languages_for_training:
            entity_stat = get_entity_frequencies(data_for_training, lang)
            print('  {0}:'.format(lang))
            print('    {0} samples;'.format(freq))
            print('    {0} entities, among them:'.format(
                sum([cur[1] for cur in entity_stat])
            ))
            max_width = max([len(cur[0]) for cur in entity_stat])
            for entity_type, entity_freq in entity_stat:
                print('      {0:>{1}} {2}'.format(entity_type, max_width,
                                                  entity_freq))
    else:
        for goal in res:
            print('===============')
            print('  {0}'.format(goal))
            print('===============')
            print('')
            print('{0} samples are loaded...'.format(len(res[goal])))
            languages_for_training = get_language_frequencies(res[goal])
            print('By languages:')
            for lang, freq in languages_for_training:
                entity_stat = get_entity_frequencies(res[goal], lang)
                print('  {0}:'.format(lang))
                print('    {0} samples;'.format(freq))
                print('    {0} entities, among them:'.format(
                    sum([cur[1] for cur in entity_stat])
                ))
                max_width = max([len(cur[0]) for cur in entity_stat])
                for entity_type, entity_freq in entity_stat:
                    print('      {0:>{1}} {2}'.format(entity_type, max_width,
                                                      entity_freq))
            print('')


if __name__ == '__main__':
    main()
