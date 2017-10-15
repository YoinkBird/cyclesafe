# Markdown Rendering
grip outline.md --quiet &

# Generating TOC
Tail the markdown_convert_ymd.pm file for the exact command.

<pre>
echo outline.md | sh -cuv 'read file; perl ../tools/markdown_toc.pl $file > ${file}_out && mv ${file}_out $file' ; git diff

</pre>

# Citations

## Summary

Use the [@identifier] format

pandoc outline.md -t html --filter pandoc-citeproc

## see also:
How to make a scientific looking PDF from markdown (with bibliography)
`- https://gist.github.com/maxogden/97190db73ac19fc6c1d9beee1a6e4fc8

http://www.chriskrycho.com/2015/academic-markdown-and-citations.html


Fancy:
https://pandoc.org/MANUAL.html#citations

Plaintext:
http://pandoc.org/demo/example19/Extension-citations.html

## Generate MD with citations at bottom
Not perfect:

adds citations, but not in markdown format
https://superuser.com/questions/984962/using-pandoc-to-convert-from-markdown-to-markdown-with-references-what-does-r

## Extract Citations from Raw MD
just extract all [@...] 

https://groups.google.com/forum/#!topic/pandoc-discuss/BqoWfP9RM0g
`- https://gitlab.com/egh/zotxt/blob/master/scripts/extractcites.py

cite_re = re.compile(r'(?:@)[\w:\.#$%&_+?<>~/-]+', re.U)


## Convert bibtex to markdown
https://gist.github.com/dsanson/1182008
