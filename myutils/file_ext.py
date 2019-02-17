import numpy as np
import csv


def read_csv(filename):
    with open(filename) as csvDataFile:
        return np.array(list(csv.reader(csvDataFile)))


def convert_encoding_to_utf8(src_path, out_path, src_encoding="ISO-8859-1"):
    import codecs
    BLOCKSIZE = 1048576  # or some other, desired size in bytes
    with codecs.open(src_path, "r", src_encoding) as sourceFile:
        with codecs.open(out_path, "w", "utf-8") as targetFile:
            while True:
                contents = sourceFile.read(BLOCKSIZE)
                if not contents:
                    break
                targetFile.write(contents)
