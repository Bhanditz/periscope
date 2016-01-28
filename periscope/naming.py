def class_for_name(fully_qualified_name):
    parts = fully_qualified_name.split('.')
    obj = __import__(".".join(parts[:-1]))
    for component in parts[1:]:
        obj = getattr(obj, component)
    return obj

def class_for_shortname(name):
    if '.' in name:
        full_name = name
    else:
        full_name = 'exp.' + name.lower() + '.' + name.capitalize()
    return class_for_name(full_name)
