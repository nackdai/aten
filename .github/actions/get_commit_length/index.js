// https://github.com/GsActions/commit-message-checker

const core = require('@actions/core');
const github = require('@actions/github');

try {
  let commit_length = 0;

  if (
    github.context.payload &&
    github.context.payload.commits &&
    github.context.payload.commits.length) {
    commit_length = github.context.payload.commits.length;
  }

  if (commit_length == 0) {
    throw new Error(`No commits found in the payload.`);
  }

  console.log("commit_length:", commit_length);
  core.setOutput("commit_length", commit_length);
} catch (error) {
  core.error(error);
  core.setFailed(error.message);
}
