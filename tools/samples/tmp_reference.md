<!--
pandoc tmp_reference.md -t markdown - -filter pandoc-citeproc
-->
<!--
https://pandoc.org/MANUAL.html#extension-yaml_metadata_block
-->
---
references:
- title: CRISP DM
  id: crispDmWiki
  url: https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining "CRISP DM"
- type: article-journal
  id: WatsonCrick1953
  author:
  - family: Watson
    given: J. D.
  - family: Crick
    given: F. H. C.
  issued:
    date-parts:
    - - 1953
      - 4
      - 25
  title: 'Molecular structure of nucleic acids: a structure for deoxyribose
    nucleic acid'
  title-short: Molecular structure of nucleic acids
  container-title: Nature
  volume: 171
  issue: 4356
  page: 737-738
  DOI: 10.1038/171737a0
  URL: http://www.nature.com/nature/journal/v171/n4356/abs/171737a0.html
  language: en-GB
...

This project will follow the CRISP-DM data mining process [@WatsonCrick1953].

This project will follow the CRISP-DM data mining process [crisp_dm].

[crisp_dm]: https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining "CRISP DM"

This project will follow the CRISP-DM data mining process [@crispDmWiki].

<!--
[@crispDmWiki]: https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining "CRISP DM"
-->

# References

