"""
Setup and configuration for the powerline QRM monitor.

This package holds everything a new operator needs to get from a fresh clone to a
running monitor: the schema that describes every setting, the loader that merges an
existing `~/.buzz/config.toml` over it, the generator that writes
`config.example.toml`, and, from a later phase, the terminal wizard that edits it all.

`schema.json` is the one source of truth for what a setting is, what it defaults to,
which values are legal, and what it means.  Three things read it instead of repeating
each other.

  * `schema.py` loads it, validates a config against it, and fills a form's starting
    values from an existing config.
  * `example_toml.py` renders `config.example.toml`, so the sample config and the
    wizard can never describe a setting differently.
  * The wizard's screens build their fields from it directly.

The running program still reads the dataclasses in `buzz.config`.  The schema
describes those dataclasses and does not replace them.  `tests/test_setup_schema.py`
pins the two together, so a field added to one and forgotten in the other fails the
suite.
"""
