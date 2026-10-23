"""In-memory Redis WATCH/MULTI and executor fakes for task tests; no sockets."""
import threading
from concurrent.futures import Future

from redis.exceptions import WatchError


class FakeRedis:
    def __init__(self):
        self.hashes, self.sets, self.expiries, self.versions = {}, {}, {}, {}
        self.lock = threading.RLock()
        self.before_execute = None
        self.execute_error = None

    def _touch(self, key):
        self.versions[key] = self.versions.get(key, 0) + 1

    def pipeline(self, transaction=True):
        return FakePipeline(self)

    def hset(self, key, *args, mapping=None):
        with self.lock:
            self.hashes.setdefault(key, {}).update(mapping or {args[0]: args[1]})
            self._touch(key)
            return 1

    def hget(self, key, field):
        with self.lock:
            return self.hashes.get(key, {}).get(field)

    def sadd(self, key, *values):
        with self.lock:
            target = self.sets.setdefault(key, set())
            before = len(target)
            target.update(values)
            self._touch(key)
            return len(target) - before

    def srem(self, key, *values):
        with self.lock:
            target = self.sets.setdefault(key, set())
            before = len(target)
            target.difference_update(values)
            self._touch(key)
            return before - len(target)

    def smembers(self, key):
        with self.lock:
            return set(self.sets.get(key, set()))

    def expire(self, key, seconds):
        with self.lock:
            self.expiries[key] = seconds
            self._touch(key)
            return True

    def delete(self, *keys):
        with self.lock:
            for key in keys:
                self.hashes.pop(key, None)
                self.sets.pop(key, None)
                self.expiries.pop(key, None)
                self._touch(key)


class FakePipeline:
    def __init__(self, client):
        self.client, self.watched, self.commands = client, {}, []
        self.in_multi = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.reset()

    def reset(self):
        self.watched, self.commands, self.in_multi = {}, [], False

    def watch(self, *keys):
        with self.client.lock:
            for key in keys:
                self.watched.setdefault(key, self.client.versions.get(key, 0))

    def unwatch(self):
        self.reset()

    def multi(self):
        self.in_multi = True

    def __getattr__(self, name):
        operation = getattr(self.client, name)

        def call(*args, **kwargs):
            if self.in_multi:
                self.commands.append((operation, args, kwargs))
                return self
            return operation(*args, **kwargs)
        return call

    def execute(self):
        hook = self.client.before_execute
        if hook is not None:
            hook(self)
        with self.client.lock:
            try:
                if self.client.execute_error is not None:
                    raise self.client.execute_error
                if any(self.client.versions.get(key, 0) != value
                       for key, value in self.watched.items()):
                    raise WatchError("concurrent mutation")
                return [operation(*args, **kwargs)
                        for operation, args, kwargs in self.commands]
            finally:
                self.reset()


class FakePool:
    def __init__(self, error=None):
        self.calls, self.error = [], error

    def submit(self, function, *args):
        if self.error is not None:
            raise self.error
        future = Future()
        self.calls.append((function, args, future))
        return future

    def run_one(self):
        function, args, future = self.calls.pop(0)
        if future.set_running_or_notify_cancel():
            try:
                result = function(*args)
                future.set_result(result)
                return result
            except BaseException as exc:
                future.set_exception(exc)
                raise
