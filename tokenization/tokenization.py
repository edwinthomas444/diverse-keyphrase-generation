# define Tokenization class
class WhitespaceTokenizer(object):
    def tokenize(self, text, truncate):
        return whitespace_tokenize(text, truncate)


def whitespace_tokenize(text, truncate=None):
    text = text.strip()
    if not text:
        return []
    tokens = text.split(" ")

    # truncate tokens
    if truncate:
        tokens = tokens[:truncate]

    return tokens


# data tokenizer for keyphrase units (Seq2Set)
class UnitTokenizer(object):
    def tokenize(self, text, max_kps):
        return unit_tokenizer(text, max_kps)


def unit_tokenizer(text, max_kps):
    text = text.strip()
    if not text:
        return []

    unit_phrases = text.split(" ; ")
    if max_kps:
        unit_phrases = unit_phrases[:max_kps]  # trim to include only max_kps
    return unit_phrases
