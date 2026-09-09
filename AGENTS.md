# Repository instructions

## Issue workflow

- Work on every issue in a dedicated Git worktree. Do not implement issue work
  directly in the primary checkout.
- Create all worktrees inside the repository under `.worktree/`, using one
  worktree and one branch per issue.
- When working on multiple issues, handle them sequentially. For each issue:
  implement and validate the change, open a pull request, and merge that pull
  request before starting work on the next issue.
- Do not combine multiple issues into one pull request.

<!-- cluedoc:start -->
## Documentation (Cluedoc)

This repository keeps human-readable, visual "papers" under `.cluedoc/`, one per feature. After finishing a set of code changes, use the **cluedoc** skill to update every affected paper — parent, self, and children — so the docs stay in sync with the code. When answering questions about how the system works, consult these papers and append a short Reading Guide.
<!-- cluedoc:end -->
