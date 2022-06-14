INTERNAL_ERROR_MESSAGE = 'Internal error. Please, open an issue.'


def _all_or_none(values):
    return all(x is None for x in values) or all(x is not None for x in values)
