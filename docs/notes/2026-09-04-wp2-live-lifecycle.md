# 2026-09-04 - WP2, live lifecycle and concurrency

WP2 of ADR-0002, in `apps/api`. Four defects, all on the path a closing browser
tab takes.

## Cancelling a task does not stop a thread

`read_video` decoded on a thread and handed frames over with

```python
asyncio.run_coroutine_threadsafe(queue.put(timed), loop).result()
```

into an `asyncio.Queue(maxsize=2)`, cleaning up with `reader.cancel()`. That
cancels the *task* wrapping `to_thread`; the thread itself was sitting inside a
blocking `.result()` waiting for room in a queue whose consumer had gone. It
never came back, and it held the video decoder open. One leaked thread and one
leaked decoder per abandoned clip.

Rewritten so the thread is asked to stop instead of being cancelled: a
`threading.Event` checked between frames, and backpressure moved off the queue
onto a `threading.Semaphore` the producer acquires with a timeout. Nothing
blocks for longer than `_STOP_CHECK_SECONDS` without looking at the flag. The
queue is now unbounded, which is safe because the semaphore is what bounds it.

## The reader's exception went nowhere

Same line: `reader.cancel()` without an await. A decode that failed halfway
raised into a task nobody was watching, so the session ended looking like a
short clip. It is now awaited, and the failure reaches the socket.

Worth noting for the record: I first tried to test this by overwriting the
`.avi` under the open decoder, which hung the suite instead of failing it -
OpenCV's behaviour on a file swapped beneath it is not something to build a test
on. Monkeypatching `video_frames` to raise is deterministic and tests the
contract that actually matters.

## Nothing shut the model worker down

`Estimator.close()` existed and **no caller ever called it**; `create_app` had
no lifespan at all. It also used `shutdown(wait=False)`, which would not have
waited for the frame in flight. There is now a lifespan, `close()` waits and
cancels queued work, and it runs on a thread so the event loop is not the thing
blocking.

## Model work on the wrong thread, again

`_send_new_bodies` called `asyncio.to_thread(live.rest_body, ...)`. That reads
the body model, and the `Estimator` docstring already records that this exact
mistake aborted the process once, because Metal keeps a command buffer per
thread. `estimate_one` had been fixed to go through the executor; this path had
not. It does now.

## Backlog

Sessions serialised on the single model thread, but the pending work behind them
was unbounded and invisible. `MAX_LIVE_SESSIONS = 2` - one on the thread, one
waiting - and a third socket is refused with a message the viewer already knows
how to display, rather than being accepted into a queue.

## Errors on the wire

The catch-all sent `str(error)` to the browser, which is how a local path or a
stack detail reaches a remote client. It now sends a fixed sentence and logs the
detail. The deliberate messages - a request that would not parse, a file outside
the served roots - are unchanged and still say what is wrong.

## Coverage

`apps/api/tests` had no test that opened a socket. There are now ten, over a
fake session, so none of this needs the weights and it all runs in CI.
