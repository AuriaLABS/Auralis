# Adversarial tool suite (#144)

`ADVERSARIAL_SUITE_SCHEMA_VERSION = 1`

Fixtures that must fail closed:

- unknown tool;
- lookalike name (`ref.echoo`);
- malformed arguments;
- malformed / stale-correlation results;
- oversized arguments;
- permission escalation on a mutative tool;
- replayed mutative call without confirmation;
- confused-deputy (`ref.add` while only `ref.echo`/`ref.write` are allowlisted);
- cancel during a mutative stream event.

A deny does **not** increment the reference executor. Every case leaves an
audit/trace reason. No silent partial execution.
