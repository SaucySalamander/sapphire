Module layout note

The transformer-related implementation files were moved into the `src/transformer/` directory
to follow a module-per-directory layout and to improve encapsulation.

Removed duplicate top-level sources:
- src/activations.c
- src/normalization.c
- src/positional_encoding.c
- src/rope.c
- src/attention.c
- src/attention_strategy.c
- src/test_activations.c
- src/test_normalization.c

If you need to keep the top-level layout for any reason, we can revert these removals. Otherwise
this directory will be the canonical place for transformer components going forward.
