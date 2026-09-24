# Session contract (#48)

`SESSION_SCHEMA_VERSION = 1`

A session is ephemeral REPL state. It stores an ID, generation, status,
message/event history and a **checkpoint name**. It does **not** store model
weights and `reset` does not touch a checkpoint file.

## Commands

`SessionRepl::handle` understands `/reset`, `/save`, `/export`, `/load`,
`/interrupt`, `/tools on|off`, `/planner on|off`, `/tool`, `/plan`, `/salir`.

Tools and planner start **off**. `/tool` records a tool-intent through the
#44 vocabulary without invoking a runtime. `/plan` records a note; it does
not call the planner.

## Persistence

`Session::canonical()` / `export()` / `load()`.

- incompatible `auralis_session` values are rejected;
- event order is preserved;
- secrets are redacted before an event is kept (`token=`, `password=`,
  `sk-live-`, `SECRET`);
- a recorded error does not drop prior events.

## Out of scope

No HTTP server, no multi-user auth, no real network/shell tools.
