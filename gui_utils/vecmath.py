import re

import numpy as np

cross = np.cross

def dot(a, b):
    return (a*b).sum(axis=-1)

def length(x):
    return np.sqrt(dot(x, x))

def normalize(x):
    l = length(x)
    assert l>0, x
    return x/l

def ctype_to_dtype(ctype):
    _ctype_to_dtype = {
        'float': np.float32,
        'double': np.float64,
        'char': np.int8,
        'short': np.int16,
        'int': np.int32,
        'longlong': np.int64,
        'uchar': np.uint8,
        'ushort': np.uint16,
        'uint': np.uint32,
        'ulonglong': np.uint64,
    }
    ctype = ctype.replace('long int', 'long')
    ctype = ctype.replace('long long', 'longlong')
    ctype = ctype.replace('unsigned ', 'u')

    if ctype not in _ctype_to_dtype:
        msg = "Cannot determine dtype from ctype '{ctype}'."
        raise ValueError(msg)

    return _ctype_to_dtype[ctype]


def vtype_to_dtype(vtype):
    regexp = re.compile(r'((?:float|double)|u?(?:char|short|int|longlong))(\d*)')

    match = regexp.match(vtype)
    if not match:
        msg = "Cannot extract format from '{pformat}'."
        raise ValueError(msg)

    dtype = ctype_to_dtype(match.group(1))

    count = match.group(2)

    if (count is None):
        return dtype

    count = int(count)

    if count == 0:
        return dtype

    if count <= 4:
        names = tuple('xyzw'[:count])
        formats = [dtype,]*count
        vec_dtype = np.dtype(dict(names=names, formats=formats))
    else:
        vec_dtype = np.dtype( (dtype, (count,)) )

    return vec_dtype
