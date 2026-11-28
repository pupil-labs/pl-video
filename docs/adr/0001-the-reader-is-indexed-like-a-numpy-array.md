# The reader is indexed like a numpy array

`pupil_labs.video_reader.Reader` takes the one-dimensional keys numpy takes, with numpy's meaning: `reader[i]` is one frame for a Python or numpy integer, negative ones counting from the end. A slice with any step, a Boolean mask as long as the reader, or a list or array of indices returns a `Selection`, which behaves like `[reader[i], reader[k]]` but decodes nothing until a frame is asked for. A selection takes the same keys, so `reader[10:100][::2][3]` is `reader[16]`, and its indices are copied and checked when it is made. A frame's `index` is its position in the video, whatever selection it came from. Every key's result and error match what the key selects from `np.arange(len(reader))`; a lone Boolean and a zero-dimensional array raise, because numpy gives them meanings a sequence of frames cannot have.

pl-video's `Reader` took an integer or a slice, ignored a slice's step, returned a list for slices under 30 frames, and offered access by timestamp through `by_container_timestamps`.

## Considered Options

- **A named method for index arrays and masks, or integer access only.** Proposed in review to keep `__getitem__`'s typing simple. Rejected: a selection that knows all its targets lets the reader plan the decode, several keyframe groups at once, and the typing has a fix: the frame overload takes `int | np.integer`, not anything usable as an index, so `reader[np.int64(3)]` types as a frame and `reader[np.arange(3)]` as a selection.
- **A list for short slices, as pl-video did.** Rejected: one return type is simpler to type and to use.
- **Access by timestamp.** Dropped. Nothing in Neon X or pl-neon-recording called it, a single timestamp had to match a float exactly, and a slice ignored its step. Callers find the index with `np.searchsorted(reader.timestamps, t)` and use `reader[i]`.
- **Indexing into a frame's pixels through the reader** (`reader[i, :, :10]`). Left out: keys are one-dimensional.

## Consequences

- How a reader of several files relabels a frame's `index` and `time` is decided when one is built; adding a way to copy a frame then breaks nothing.
- `reader[True]` is a `TypeError`, although `bool` is an `int` to the type checker.
