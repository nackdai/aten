# Github action to get commit length

Unfortunately, there is no API to get commit length in workflow. Commit length
means number of commits in `push` or `pull request`.

We can access `payload` in JavaScript and `payload` includes array of commits.
Therefore, we can get commit length in JavaScript.

This action refers to [GsActions/commit-message-checker](https://github.com/GsActions/commit-message-checker)

## What does this action do

This action follows [Creating a JavaScript action](https://help.github.com/en/actions/building-actions/creating-a-javascript-action)

You can know what `inputs` and `outputs` exist from descriptions in
`action.yml`.

## How to create package

1. Install npm modules
```
cd .github/actions/get_commit_length
npm install
```
2. Run package script
```
cd .github/actions/get_commit_length
npm run package
```

Then, we can create `dist/index.js` as package. we need to push `dist/index.js`
instead of `node_modules`.
