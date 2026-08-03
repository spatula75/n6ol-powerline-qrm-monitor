"""
Setup and configuration for the powerline QRM monitor.

Everything a new operator needs to get from a fresh clone to a running monitor lives
here: the schema describing every setting, the loader that merges an existing
`~/.buzz/config.toml` over it, the generator that writes `config.example.toml`, and
(from a later phase) the terminal wizard that edits it all.

`schema.json` is the single source of truth for what a setting is, what it defaults
to, what values are legal, and what it means.  Three things read it rather than
repeating each other:

  * `schema.py` - loads it, validates a config against it, and fills a form's
    starting values from an existing config;
  * `example_toml.py` - renders `config.example.toml`, so the sample config and the
    wizard can never describe a setting differently;
  * the wizard's screens, which build their fields from it directly.

The dataclasses in `buzz.config` remain what the running program reads.  The schema
describes them rather than replacing them, and `tests/test_setup_schema.py` pins the
two together so a field added to one and forgotten in the other fails the suite.
"""
