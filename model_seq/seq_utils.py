def combine(a, b):
    if a == 'O' and b == 'O':
        return 'O'
    return a + '-' + b


def symb_seq_to_spans(symb_seq):
    """
    Converts sequence of symbol labels (IOBES) format to spans
    """

    chunks = []
    current = None

    for i, label in enumerate(symb_seq):
        if label.startswith('B-'):

            if current is not None:
                chunks.append('@'.join(current))
            current = [label.replace('B-', ''), '%d' % i]

        elif label.startswith('S-'):

            if current is not None:
                chunks.append('@'.join(current))
                current = None
            base = label.replace('S-', '')
            chunks.append('@'.join([base, '%d' % i]))

        elif label.startswith('I-'):

            if current is not None:
                base = label.replace('I-', '')
                if base == current[0]:
                    current.append('%d' % i)
                else:
                    chunks.append('@'.join(current))
                    current = [base, '%d' % i]

            else:
                current = [label.replace('I-', ''), '%d' % i]

        elif label.startswith('E-'):

            if current is not None:
                base = label.replace('E-', '')
                if base == current[0]:
                    current.append('%d' % i)
                    chunks.append('@'.join(current))
                    current = None
                else:
                    chunks.append('@'.join(current))
                    current = [base, '%d' % i]
                    chunks.append('@'.join(current))
                    current = None

            else:
                current = [label.replace('E-', ''), '%d' % i]
                chunks.append('@'.join(current))
                current = None
        else:
            if current is not None:
                chunks.append('@'.join(current))
            current = None

    if current is not None:
        chunks.append('@'.join(current))

    return set(chunks)
