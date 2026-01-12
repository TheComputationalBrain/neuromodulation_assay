class AttrDict(dict):
    """A dict that allows dot-notation access: d.key instead of d['key']"""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    @staticmethod
    def convert(obj):
        """Recursively convert nested dicts/lists to AttrDict."""
        if isinstance(obj, dict):
            return AttrDict({k: AttrDict.convert(v) for k, v in obj.items()})
        elif isinstance(obj, list):
            return [AttrDict.convert(x) for x in obj]
        return obj
