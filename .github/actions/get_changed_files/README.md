# Github action to get commit length

Unfortunately, there is no API to get changed files in commits.

This action refers to [lots0logs/gh-action-get-changed-files](https://github.com/lots0logs/gh-action-get-changed-files)

When I used `lots0logs/gh-action-get-changed-files`, some errors happend.
Therefore, I created new one.

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
