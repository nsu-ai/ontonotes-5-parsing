from argparse import ArgumentParser
import os

from ontonotes5.utils import load_ontonotes5_from_json
from ontonotes5.utils import get_language_frequencies, get_entity_frequencies


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-s',
        '--src',
        dest='source_file', type=str, required=True,
        help='The source *.json file with texts and their annotations '
             '(named entities, morphology and syntax).'
    )
    cmd_args = parser.parse_args()

    src_file_name = os.path.normpath(cmd_args.source_file)
    err_msg = 'File "{0}" does not exist!'.format(src_file_name)
    assert os.path.isfile(src_file_name), err_msg

    source_data = load_ontonotes5_from_json(src_file_name)

    for goal in source_data:
        print('===============')
        print('  {0}'.format(goal))
        print('===============')
        print('')
        print('{0} samples are loaded...'.format(len(source_data[goal])))
        languages_for_training = get_language_frequencies(source_data[goal])
        print('By languages:')
        for lang, freq in languages_for_training:
            entity_stat = get_entity_frequencies(source_data[goal], lang)
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
