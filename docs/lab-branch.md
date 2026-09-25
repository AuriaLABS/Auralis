# Lab branch contract (#57)

`LAB_BRANCH_SCHEMA_VERSION = 1`

Policy-controlled experimental branches and draft PRs from approved #55 proposals.

- branch names are deterministic (`exp/<proposal>/<baseline8>`);
- generated PRs start as draft and keep hypothesis/scope/baseline/criterion/budget;
- automated paths cannot write `main`;
- scope violations fail closed;
- a negative result can archive the PR without dropping registry IDs.

Does **not** merge, force-push or expand scope without a new approval.
