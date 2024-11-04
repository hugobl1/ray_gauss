import numpy as np

def get_member(varname):

    def getter(self, varname=varname):
        return getattr(self, varname, None)

    return getter


def set_bool(varname, default_value=None):

    def setter(self, value, varname=varname, default_value=default_value):
        if value is None:
            value = default_value
        value = bool(value)
        setattr(self, varname, value)

    return setter


def set_int(varname, default_value=None):

    def setter(self, value, varname=varname, default_value=default_value):
        if value is None:
            value = default_value
        value = np.int32(value)
        setattr(self, varname, value)

    return setter


def set_float(varname, default_value=None):

    def setter(self, value, varname=varname, default_value=default_value):
        if value is None:
            value = default_value
        value = np.float32(value)
        setattr(self, varname, value)

    return setter


def set_float3(varname, default_value=None):

    def setter(self, value, varname=varname, default_value=default_value):
        if value is None:
            value = default_value

        if value is None:
            pass
        elif np.isscalar(value):
            value = np.full(shape=(3,), dtype=np.float32, fill_value=value)
        else:
            value = np.asarray(value, dtype=np.float32)
        setattr(self, varname, value)

    return setter
