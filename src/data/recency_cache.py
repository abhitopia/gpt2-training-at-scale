
class RecencyCacheDict:
    def __init__(self, max_size=3):
        self.max_size = max_size
        self.cache = {}
        self._usage_index = -1

    @property
    def usage_counter(self):
        self._usage_index += 1
        return self._usage_index

    def __contains__(self, item):
        return item in self.cache

    def __getitem__(self, item):
        self.cache[item]['last_used'] = self.usage_counter
        return self.cache[item]['value']

    def __setitem__(self, key, value):
        self.cache[key] = {
            'last_used': self.usage_counter,
            'value': value
        }

        if len(self.cache) > self.max_size:
            self.remove_oldest()

    def remove_oldest(self):
        key_to_del = sorted([(key, self.cache[key]['last_used']) for key in self.cache], key=lambda x: x[1])[0][0]
        del self.cache[key_to_del]

    def get(self, key, default=None):
        return self[key] if key in self else default
