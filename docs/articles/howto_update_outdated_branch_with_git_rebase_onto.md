# How-To: Update an outdated branch using git rebase --onto

I.e. Merge a feature branch for which the original parent branch was deleted.

## Conventions Used:
* `trunk` branch: Branch conventionally used for collaboration within the repo. This branch is often named `main`, although this is just a convention. For more info see e.g. first paragraph of [this article on trunk based development](https://cloud.google.com/architecture/devops/devops-tech-trunk-based-development).
* `feature` branch: Branch used for developing a feature; branched from `trunk` or another `feature` branch; very similar to the [feature branch used in git flow](https://www.gitkraken.com/learn/git/git-flow#feature-branch), but without a `develop` branch.

## Scenario:

A new documentation-specific `feature` branch was branched off of an enhancement-specific `feature` branch in order to independently iterate on system documentation while finishing the original features (i.e. the branch for enhancement became the parent, or upstream, branch of the branch for documentation).

Then, after passing code review, the enhancement-specific feature branch was deleted upon being squash-merged into the `trunk` branch, thus removing the original [common ancestor](https://git-scm.com/docs/git-merge-base) between the two feature branches.

Now it is time to merge the new feature branch to the trunk branch, which requires some extra steps due to the now-missing commit from the deleted parent branch.

Application: This particular scenario can be a useful way to develop multiple related, but independent, features using separate branches in order to continue feature work while waiting for code review. E.g. feature2 depends on feature1, but feature1 is already in review; instead of waiting for the review to complete, simply start a new branch for feature2 off of the branch for feature1.

## Process: rebase feature branch against trunk branch

To keep the [git history linear](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches#require-linear-history), `rebase` the feature branch against the trunk branch to ensure that a `fast-forward` merge is possible.

**Branches and Commits used in this Example**:
* `report` - `feature` branch for systems documenation
* `c0acfecc7f7cd43fd2bc8117ae7cf78d07514fa3` - original parent commit for `report` feature branch
* (`convert_ipynb`) - deleted parent feature branch, contained original parent for `report`: `c0acfecc7f7cd43fd2bc8117ae7cf78d07514fa3`
* `master` - `trunk` branch

Since the original common ancestor was deleted, a `git rebase` is expected to fail with a merge conflict:

```bash
# prepare for rebase
$ git fetch
$ git checkout report
# expect 0:
$ git diff origin/report; echo $?
# rebase:
$ git rebase origin/master
warning: skipped previously applied commit 2c7f9fa
...
warning: skipped previously applied commit ...
...
warning: skipped previously applied commit c0acfec

Auto-merging Final.ipynb
CONFLICT (content): Merge conflict in Final.ipynb
error: could not apply 8539cae... ipynb: imputing missing speed limits
hint: Resolve all conflicts manually, mark them as resolved with
hint: "git add/rm <conflicted_files>", then run "git rebase --continue".
hint: You can instead skip this commit: run "git rebase --skip".
hint: To abort and get back to the state before "git rebase", run "git rebase --abort".
Could not apply 8539cae... ipynb: imputing missing speed limits

```

Before proceeding, cancel the `rebase` operation with:

```bash
$ git rebase --abort
```

Now we can proceed to investigate how to resolve this merge conflict.

### Investigation:

This error occurs because the original parent-commit for `report` is not on `master`, so git finds an older common ancestor (i.e. the original parent-commit for the deleted branch `convert_ipynb`).

The following steps help understand the commits processed by the `git rebase` command by examining the history of these branches.

We can locate the current common ancestor between `master` and `report` along with historical context by calling `git show` on the commit returned by the [`git merge-base`](https://git-scm.com/docs/git-merge-base) command:

```bash
$ git merge-base origin/master report
b43e68c905329963aad72883658efa0a33f2c0cc

$ git show -s $(git merge-base origin/master report)
commit b43e68c905329963aad72883658efa0a33f2c0cc
Author: yoinkbird <YoinkBird2010@gmail.com>
Date:   Tue May 2 21:44:56 2017 -0500
```

Next, we examine the history of `report` between
the current common-ancestor with `master` `b43e68c905329963aad72883658efa0a33f2c0cc` and
the original parent-commit (`c0acfecc7f7cd43fd2bc8117ae7cf78d07514fa3`) :

```bash
$ git log --oneline b43e68c905329963aad72883658efa0a33f2c0cc^..c0acfecc7f7cd43fd2bc8117ae7cf78d07514fa3
c0acfec ipynb: refactor - removing old headers
8450934 refactor: moving code to own dir
...
90b160b ipynb - xgb: printing features sorted by importance
2c7f9fa adding gitignore
8539cae ipynb: imputing missing speed limits
b43e68c adapting to csv file
```


As we see, the commit
`8539cae ipynb: imputing missing speed limits`
after the common ancestor
`b43e68c adapting to csv file`
corresponds to the commit which `git rebase` could not apply:

```
error: could not apply 8539cae... ipynb: imputing missing speed limits
```

Since this commit is before the parent-commit of the `report` branch, we conclude that this branch isn't introducing this particular merge conflict.

(Note: we can most likely conclude that this is the parent commit for the now-deleted feature branch `convert_ipynb`, and will refer to it as such for the sake of this tutorial).

### Prepare Verify: List files changed on the branch

Prepare a list of files modified on the `report` branch for later use in verifying rebases, merges, etc.

List the files changed on the original `report` branch by diffing against its parent commit and using the `--name-only` flag.

In this particular example, we expect the branch to have only modified files on a few directories, so for simplicity, we filter this list to show only the top-level dirs and files:

```bash
$ git checkout report
$ git diff --name-only c0acfecc7f7cd43fd2bc8117ae7cf78d07514fa3 | perl -nle 'm/^([^\/]+[\/]*)/ && print $1' | uniq
report/
thoughts
tools/
```

Backup and sanity check: Ensure that the local `report` matches remote `report` branch; this is crucial both for verifying the rebase later, as a backup for this branch in case anything goes wrong.

```
$ git fetch
...
$ git checkout report
Switched to branch 'report'
# return code `0` means nothing changed
$ git diff origin/report; echo $?
0
```

At this point, we have a good understanding of the commits shown output of the `git rebase origin/master` command.


## Revised Process: git rebase --onto

Now that the investigation is complete, it's clear that this is a case for `git rebase --onto ...` !

The `git rebase --help` manpage provides several examples,
and is expanded upon with a lovely graphical explanation by the article
https://womanonrails.com/git-rebase-onto .

We want to rebase _onto_ the `origin/master` branch (`origin` to ensure we are getting the most up-to-date code), while _ignoring_ all commits from before the commit from which the branch was forked (i.e. ignore the now-squashed-and-deletd commits from the `convert_ipynb` branch).

This operation is necessary to "remove" the original squashed-and-deleted commits of the `convert_ipynb` (parent-feature branch) from this `report` feature branch so it can successfully rebase against the trunk branch.

Update references to all remotes:

```bash
$ git fetch
remote: Enumerating objects: 5, done.
remote: Counting objects: 100% (5/5), done.
remote: Compressing objects: 100% (3/3), done.
remote: Total 3 (delta 2), reused 0 (delta 0), pack-reused 0
Unpacking objects: 100% (3/3), 300 bytes | 300.00 KiB/s, done.
From github.com:YoinkBird/cyclesafe
   12a8807..ed9a240  master     -> origin/master

```

Now update the feature branch `report` from the remote trunk branch (i.e. `origin/master`) while discarding the initial common-ancestor from the trunk branch (i.e. commit for `convert_ipynb`):


```bash
$ git rebase --onto origin/master c0acfecc7f7cd43fd2bc8117ae7cf78d07514fa3
Successfully rebased and updated refs/heads/report.

```

### Verify

Rebasing one branch against another can sometimes introduce unintentional modifications if a file was modified on both branches, even without a merge conflict.

To ensure that the rebase did not introduce any undesired changes, compare the list of changed files against the list of files from the previous "prepare verify" step.

In this case, this is done by comparing the local `report` branch, which now contains new commits from the `trunk` branch, against the remote `report` branch, which is still at the pre-rebase state. 

This will show all file modifications on `report` introduced by the rebase.

List the file changes introduced by the rebase:

```bash
$ git diff --name-only origin/report
.gitignore
Final.ipynb
code/feature_definitions.py
code/helpers.py
code/model.py
code/txdot_parse.py
output/crashes_300_330.html
run_if_changed.sh
t/route_json/gps_generic.json
t/route_json/gps_generic.py
t/route_json/gps_generic_eerc_to_klane.json
t/route_json/gps_scored_eerc_to_klane.json
test_json_inout.py
```

Or, listing only the top-level using the aforementioned commands:
```bash
$ git diff --name-only origin/report | perl -nle 'm/^([^\/]+[\/]*)/ && print $1' | uniq
.gitignore
CONTRIBUTING.md
Final.ipynb
README.md
code/
output/
run_if_changed.sh
t/
test_json_inout.py
```

None of these files and directories are contained in the previously generated list:
```bash
report/
thoughts
tools/
```

Result: None of the files and directories from the original `report` feature-branch were changed by the rebase; this implies that the branch can be merged to the `trunk` branch without making unintended modifications.

The branch `report` can now be force pushed to `origin/report` and subsequently be fast-forward merged to `master`;
this is a unique situation in which a force-push is appropriate because we have to update the shared history in order to enable a `fast-forward` merge on the remote server.
