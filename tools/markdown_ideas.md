
# Footnotes

## Current Implementation

This project will follow the CRISP-DM data mining process [1]:

[1]: https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining "CRISP DM"

-or for pandoc, use the '@'-

This project will follow the CRISP-DM data mining process [@1]:

[@1]: https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining "CRISP DM"


## Desired Implementation

This project will follow the CRISP-DM data mining process [@crispDmWiki]:

[@crispDmWiki]: https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining "CRISP DM"

-and generate auto-biblio and footnote-numbers, then pandoc can do the rest-

---
references:
- title: crispDmWiki
  id: crispDmWiki
  url: https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining "CRISP DM"
...

This project will follow the CRISP-DM data mining process [20]:

[20]: [@crispDmWiki]

## Idea
<!--
This project will follow the CRISP-DM data mining process [@crispDmWiki]: <!--[@fn:crispDmWiki]-->
[@crispDmWiki]: https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining "CRISP DM"
convert to:
[34]

[@34]: https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining "CRISP DM"
-->

## Idea
<!--
Store all m/[@...]: .../ as name and line number
Find all name m/@[$name]/ , generate footnote numbers from order of appearance
Replace all m/@[$name]/ with footnote number
Should result in notes being [@number] and refs being '[@number]: link'
-->

## Idea
<!-- https://daringfireball.net/projects/markdown/syntax#link -->
<!-- in order to have a biblography built up: part1 - source: -->
<!-- {biblio:label:crispDmWiki} -->
<!-- in order to have a biblography built up: part2, references: -->
<!-- {biblio:label:crispDmWiki;title:"CRISP DM";url:https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining} -->

## Idea
<!-- 
Form hash of all biblio:label entries: labellines = label:firstline
caveat: some lines may have multiple labels
for i keys(labellines):
  

-->
