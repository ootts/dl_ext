def safe_zip(*args):
    n = len(args[0])
    for a in args:
        assert len(a) == n, f'{len(a)}!={n}'
    return zip(*args)
