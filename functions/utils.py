import pickle
PICKLE_PROTOCOL = 4


def get_file(filename):
    with open(filename, "rb") as f:
        result = pickle.load(f)
    return result


def save_file(object_name, adr, append=False):
    flag = 'ab' if append else 'wb'
    with open(adr, flag) as f:
        pickle.dump(object_name, f, PICKLE_PROTOCOL)
    return 'Done'
