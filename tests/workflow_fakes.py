"""Deterministic Redis transport fake with WATCH/MULTI and a controllable clock."""
import copy
import threading

from redis.exceptions import WatchError


class TransactionRedis:
    def __init__(self):
        self.hashes, self.strings, self.expires, self.versions = {}, {}, {}, {}
        self.now = 0
        self.mutex = threading.RLock()
        self.before_exec = None
        self.after_read = None
        self.fail_before_exec = False
        self.fail_hset = self.fail_hdel = False
        self.eval_calls = 0

    def _bump(self, key): self.versions[key] = self.versions.get(key, 0) + 1

    def _expire(self, key):
        if self.expires.get(key, float("inf")) <= self.now:
            self.delete(key)
            self.expires.pop(key, None)

    def hgetall(self, key):
        with self.mutex:
            self._expire(key)
            value = copy.deepcopy(self.hashes.get(key, {}))
            if self.after_read:
                hook, self.after_read = self.after_read, None
                hook()
            return value

    def hget(self, key, field): return self.hgetall(key).get(field)

    def hset(self, key, field=None, value=None, mapping=None):
        with self.mutex:
            if self.fail_hset: raise RuntimeError("MOCK_HSET_UNAVAILABLE")
            self.hashes.setdefault(key, {}).update(mapping or {field: value})
            self._bump(key)
            return 1

    def hdel(self, key, *fields):
        with self.mutex:
            if self.fail_hdel: raise RuntimeError("MOCK_HDEL_UNAVAILABLE")
            count = sum(int(self.hashes.get(key, {}).pop(field, None) is not None) for field in fields)
            if count: self._bump(key)
            return count

    def expire(self, key, seconds):
        self.expires[key] = self.now + seconds
        self._bump(key)
        return True

    def set(self, key, value, nx=False, ex=None):
        with self.mutex:
            self._expire(key)
            if nx and key in self.strings: return False
            self.strings[key] = value
            if ex is not None: self.expires[key] = self.now + ex
            self._bump(key)
            return True

    def get(self, key):
        with self.mutex:
            self._expire(key)
            return self.strings.get(key)

    def delete(self, key):
        with self.mutex:
            removed = key in self.strings or key in self.hashes
            self.strings.pop(key, None); self.hashes.pop(key, None)
            self._bump(key)
            return int(removed)

    def eval(self, script, numkeys, key, token, ttl=None):
        with self.mutex:
            self.eval_calls += 1
            if self.get(key) != token: return 0
            return int(self.expire(key, float(ttl))) if "expire" in script and ttl is not None else self.delete(key)

    def pipeline(self): return _Pipeline(self)


class _Pipeline:
    def __init__(self, redis):
        self.redis, self.watched, self.commands = redis, {}, []
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def watch(self, key):
        self.redis._expire(key)
        self.watched[key] = self.redis.versions.get(key, 0)
    def hgetall(self, key): return self.redis.hgetall(key)
    def multi(self): pass
    def hset(self, *args, **kwargs): self.commands.append(("hset", args, kwargs))
    def hdel(self, *args, **kwargs): self.commands.append(("hdel", args, kwargs))
    def expire(self, *args, **kwargs): self.commands.append(("expire", args, kwargs))
    def execute(self):
        with self.redis.mutex:
            if self.redis.before_exec:
                hook, self.redis.before_exec = self.redis.before_exec, None
                hook()
            for key, version in self.watched.items():
                self.redis._expire(key)
                if self.redis.versions.get(key, 0) != version: raise WatchError()
            if self.redis.fail_before_exec or self.redis.fail_hset:
                raise RuntimeError("MOCK_TRANSACTION_NOT_COMMITTED")
            return [getattr(self.redis, command)(*args, **kwargs) for command, args, kwargs in self.commands]
