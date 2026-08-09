# Configuration reference

The configuration file is documented in `config.example.toml`, which in turn is auto-generated based on `schema.json` (in `lib/buzz/setup/`).  This is probably the best place to look for detailed explanations of what each of the configuration options does.

This same `schema.json` also drives the sections found in the `buzz.setup` module, keeping everything nice and consistent.

The actual values are ultimately stored in an instance of `buzz.config.BuzzConfig`.
