

def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text

meta = load_filepaths_and_text('./ljs.txt')

from data_util.text import cleaned_text_to_sequence, cleaners

def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)
    return text

cleaner = ["english_cleaners2"]
phoneme = _clean_text('alphabet', cleaner)
# phones = cleaned_text_to_sequence(phoneme)