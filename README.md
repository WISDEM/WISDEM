# WEIS

This repo contains copies of other codes created by using the `git subtree` commands.
Below are some details about how to add external codes and update them.
Changes to those codes should be added to their original repos, *not* to this WEIS repo.

## How to add a subtree repo
Here's how to add an external code, using OpenFAST as an example:

```
$ git remote add OpenFAST https://github.com/OpenFAST/openfast
$ git fetch OpenFAST
$ git subtree add -P OpenFAST OpenFAST/dev --squash
```

The --squash is important so WEIS doesn't get filled up with commits from the subtree repos.

## How to update a subtree repo
Once a subtree code exists in this repo, we can update it like this:

```
$ git subtree pull --prefix OpenFAST https://github.com/OpenFAST/openfast dev --squash
```
