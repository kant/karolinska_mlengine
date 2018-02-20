"""Operations that apply to lists."""
def chunks(l, n):
    """Yield successive n-sized chunks from l.

    Parameters
    ----------
    l : list
        The list to split into N chunks
    n : int
        How many chunks to return (they are not necessarily even)

    Yields
    ------
    list
        sub lists from the original list
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def chunker_list(seq, size):
    return (seq[i::size] for i in range(size))
