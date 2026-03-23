### Housekeeping tasks and requirements
*This is a space for tracking tasks and requirements that help keep the codebase clean and healthy*

## Node20 deprecated
Migrate CI Action(s) to use Node24

From GitHub:

>Node20 will reach end-of-life (EOL) in April of 2026. As a result we have started the deprecation process of Node20 for GitHub Actions. We plan to migrate all actions to run on >Node24 in the fall of 2026.
>
>The newest GitHub runner (v2.328.0) now supports both Node20 and Node24 and uses Node20 as the default version. If you’d like to test Node24 ahead of time, set >FORCE_JAVASCRIPT_ACTIONS_TO_NODE24=true as an env in your workflow or as an environment variable on your runner machine to force the use of Node24.
>
>Beginning on June 2nd, 2026, runners will begin using Node24 by default. To opt out of this and continue using Node20 after this date, set >ACTIONS_ALLOW_USE_UNSECURE_NODE_VERSION=true as an env in your workflow or as an environment variable on your runner machine. This will only work until we upgrade the runner >and remove Node20 later in the fall of 2026.

## Improve pyright and ruff workflow
We are currently running into repeated failed CI Runs due to ruff and pyright failures. Before pushing to remote, we need to make sure ruff and pyright errors are cleared/addressed.

## Out-of-date requirements
* REQ_014 has already been completed.
* REQ_017 was exercised during work to run interventions (REQ_067-068). We are not currently using mutliple families (Interventions were moved to subclass status instead of separate Model Family), but the architecture supports this.
* REQ_019 was implemented under the coarseness work in v0.2.1