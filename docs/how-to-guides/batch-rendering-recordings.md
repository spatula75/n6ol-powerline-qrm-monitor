# Render all of your recordings

## Introduction

After the monitor has been recording events for a while, you end up with a directory full of .wav files and no good way to tell which ones are worth a look.  The filenames only tell you when each event happened, and the file sizes only tell you how long it lasted.

`scripts/batch_render_recordings.py` renders every recording to its own video so you can skim through them and find the interesting ones.  It is the same rendering you would get from [replaying a single recording](replaying-recordings.md) by hand, just done for the whole directory without you sitting there to start each one.

## Activating the Python virtual environment

As with the main module, you need to be running in the Python virtual environment created during [initial setup](../tutorials/getting-started.md).  On Windows, run `.venv\Scripts\activate`, and on macOS, Linux, and FreeBSD use `source .venv/bin/activate`.  Use `deactivate` when you're finished.

## Rendering everything

Run `python scripts/batch_render_recordings.py` with no arguments.  It finds the recordings in the recording directory you configured, and writes the videos to a `renders` subdirectory inside it, each named after the recording it came from.

Be prepared to wait.  Rendering happens in real time, so the batch takes at least as long as your recordings do end to end, plus a little startup time for each one.  Half an hour of recordings is at least half an hour of rendering.  You can leave it running and come back to it.

## Rendering several at once

Add `--jobs` followed by the number of recordings to render simultaneously, for example `--jobs 4`.  This is the one thing that will speed up the rendering process.

Each job is a separate copy of the monitor with its own analysis and its own video encoder, so treat it as roughly one CPU core for each job.  The default is one job, which leaves the machine usable while it works.  Setting it higher than the number of CPU cores you have will not help.

## Choosing which recordings to render

`--max-length` followed by a number of seconds skips anything longer than that, which is useful if you only want to render your shorter recordings.  `--max-length 20` renders only the recordings of 20 seconds or less.  The default is to render every recording, no matter how long it is.

`--limit` followed by a number renders only that many recordings, which is a good way to test your settings before committing to the entire directory.

`--recordings` and `--output-dir` point at a different source directory and a different destination, respectively.  Both are optional, and without them the script uses your configured recording directory and the `renders` subdirectory inside it.

## Stopping and starting again

If you stop the batch part-way through, or it stops on its own, just run it again.  Recordings that already have a video are skipped, so it picks up where it left off rather than starting over.

Any recording that fails to render is reported as it happens and again in the summary at the end, and the batch carries on with the rest rather than stopping.  A failed render does not leave a partial video behind, so running the batch again will retry it.
