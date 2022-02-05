const { context, gitHub } = require('@actions/github');
const core = require('@actions/core');

const commits = context.payload.commits.filter(c => c.distinct);
const repo = context.payload.repository;
const org = repo.organization;
const owner = org || repo.owner;

const FILES = [];
const FILES_MODIFIED = [];
const FILES_ADDED = [];
const FILES_DELETED = [];
const FILES_RENAMED = [];
const FILES_ADDED_MODIFIED = [];

const gh = new github.getOctokit(core.getInput('token'));
const args = { owner: owner.name, repo: repo.name };

function isAdded(file) {
  return 'added' === file.status;
}

function isDeleted(file) {
  return 'deleted' === file.status;
}

function isModified(file) {
  return 'modified' === file.status;
}

function isRenamed(file) {
  return 'renamed' === file.status;
}

async function processCommit(commit) {
  args.ref = commit.id;
  result = await gh.repos.getCommit(args);

  if (result && result.data) {
    const files = result.data.files;

    files.forEach(file => {
      if (isModified(file)) {
        FILES.push(file.filename);
        FILES_MODIFIED.push(file.filename);
        FILES_ADDED_MODIFIED.push(file.filename);
      }
      if (isAdded(file)) {
        FILES.push(file.filename);
        FILES_ADDED.push(file.filename);
        FILES_ADDED_MODIFIED.push(file.filename);
      }
      if (isRenamed(file)) {
        FILES.push(file.filename);
        FILES_RENAMED.push(file.filename);
      }

      isDeleted(file) && FILES_DELETED.push(file.filename);
    });
  }
}

Promise.all(commits.map(processCommit)).then(() => {
  core.setOutput("all", FILES);
  core.setOutput("added", FILES_ADDED);
  core.setOutput("deleted", FILES_DELETED);
  core.setOutput("modified", FILES_MODIFIED);
  core.setOutput("renamed", FILES_RENAMED);
  core.setOutput("added_modified", FILES_ADDED_MODIFIED);

  process.exit(0);
});
