
set -euvx
# should not have a breadcrumb
perl ../tools/markdown_toc.pl outline.md > outline_bc.md ; diff outline.md outline_bc.md

# should have a breadcrumb - imperfect, relies on outline.md having an un-updated change
perl ../tools/markdown_toc.pl outline.md bc_toc > outline_bc.md ; diff outline.md outline_bc.md || echo "different is ok!"

# should not do anything
perl ../tools/markdown_toc.pl outline.md undefined > outline_bc.md ; diff outline.md outline_bc.md
