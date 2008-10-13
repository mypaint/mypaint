
def save(obj, f):
    for name, type in obj.serialize_members:
        data = getattr(obj, name)
        if type is str:
            assert isinstance(data, str)
            f.write('%d\n' % len(data))
            f.write(data)
        elif type is int:
            f.write('%d\n' % data)
        elif type is float:
            # FIXME: inefficient
            f.write('%s\n' % repr(data))
        else:
            assert False, 'unknown type'

def load(obj, f):
    for name, type in obj.serialize_members:
        if type is str:
            length = int(f.readline())
            data = f.read(length)
            assert len(data) == length
            setattr(obj, name, data)
        elif type is int:
            setattr(obj, name, int(f.readline()))
        elif type is float:
            setattr(obj, name, float(f.readline()))
        else:
            assert False, 'unknown type'
    if hasattr(obj, 'after_unserialize'): # FIXME: inconsistent name
        obj.after_unserialize()
    
