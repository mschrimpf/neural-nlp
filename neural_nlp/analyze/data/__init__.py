class prefixdict(dict):
    def __init__(self, default=None, **kwargs):
        super(prefixdict, self).__init__(**kwargs)
        self._default = default

    def __getitem__(self, item):
        subitem = item
        while len(subitem) > 1:
            try:
                return super(prefixdict, self).__getitem__(subitem)
            except KeyError:
                subitem = subitem[:-1]
        return self._default


subject_columns = prefixdict(default='subject_id', 
                             Fedorenko='subject_UID', 
                             Pereira='subject',
                             Blank='subject_UID')
