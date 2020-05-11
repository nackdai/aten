# Github action to run git command

This action refers to [srt32/git-actions](https://github.com/srt32/git-actions)

## What is Github action

Could you see Github formal document:

https://help.github.com/en/actions/building-actions/about-actions

## What does this action do

This action follows [Creating a Docker container action](https://help.github.com/en/actions/building-actions/creating-a-docker-container-action)

* In `action.yml`, treat git command as inputs `command`, and the outcome as
outputs `result`
* In `Dockerfile`, install git and copy `entrypoint.sh` as ENTRYPOINT
* In `entrypoint.sh`, run specified command and set the outcome to
`outputs.result`
