# Title
<!-- Define a short, significant title which reflects clearly the contents of your report. The title page follows the guidelines of scientific proposals at Department of Earth Sciences (see http://www.erdw.ethz.ch/documents/index). -->

Data Driven Approach towards Improving Road Safety for Cyclists

# Abstract
<!-- Succinct abstract of less than one page. -->

# Table of content
<!-- The table of content lists all chapters (headings/subheadings) including page number. -->

<!--!toc_mini-->
<!--<toc_mini>-->
* Introduction
* Background and results to date
* Goal
* Data Driven Approach towards Improving Road Safety for Cyclists
* * Purpose
* * Goals:
* * * interpret what makes roads safe
* * * interpret how cyclists can ride defensively
* * * design tool to help find safe routes
* Methodology
* Time Plan for Master’s Project Proposal and Master’s Thesis
* Discussion / Conclusion
* Acknowledgements
* Reference & Literature (Bibliography)
* Appendix
<!--</toc_mini>-->


# Introduction
<!-- Explain why this work is important giving a general introduction to the subject, list the basic knowledge needed and outline the purpose of the report. -->

# Background and results to date
<!-- List relevant work by others, or preliminary results you have achieved with a detailed and accurate explanation and interpretation. Include relevant photographs, figures or tables to illustrate the text.  This section should frame the research questions that your subsequent research will address. -->

relevant work - misc traffic studies

preliminary results - class project; outline goal and results; in the next section lead in to the remaining questions

This is an extension of a group project I started during the Spring semester.

My goal is to help cyclists choose safer routes and to identify what makes streets dangerous.

For example, we identified time-of-day as a factor so I created a basic visualization of where crashes happen for the main intervals.
This could then allow cyclists to plan their route based on time of day.
https://nbviewer.jupyter.org/github/YoinkBird/dataMiningFinal/blob/master/Final.ipynb#Maps-of-Crashes 
(caveat: these maps lump together all crashes from 2010-2017 and thereby hide any potential trends)

# Goal
<!-- List the main research question(s) you want to answer. Explain whether your research will provide a definitive answer or simply contribute towards an answer. -->

<hr />

# Data Driven Approach towards Improving Road Safety for Cyclists
**Section Overview:**
<!--!toc_mini-->
<!--<toc_mini>-->
* Purpose
* Goals:
* * interpret what makes roads safe
* * interpret how cyclists can ride defensively
* * design tool to help find safe routes
<!--</toc_mini>-->
## Purpose
Analyse available data to understand how crashes with other vehicles occur.
## Goals:
### interpret what makes roads safe
focus on "external" data features, e.g. weather, bike lane, speed limit
possible break down by intersection and frequency of accidents
### interpret how cyclists can ride defensively
focus on "personal" data features, e.g. wearing helmet, avoiding busy roads
classification into "avoidable" and "avoidable" crashes
e.g. left-turn crash seen as "avoidable" because cyclist can look for vehicles, but crash from rear seen as "unavoidable"  because cyclist has no visibility of vehicles
### design tool to help find safe routes
e.g. assign safety score to routes provided by other tools

<hr />

# Methodology
<!-- Explain the methods and techniques which will be used for your project depending on the subject: field work, laboratory work, modeling technique, interdisciplinary collaboration, data type, data acquisition, infrastructure, software, etc. -->

**Section Overview:**
<!--!toc_mini-->

use available crash-data, augment with other data sources as necessary and as possible
using python data mining libraries and html
# Time Plan for Master’s Project Proposal and Master’s Thesis
<!-- Give a detailed time plan. Show what work needs to be done and when it will be completed. Include other responsibilities or obligations. -->

# Discussion / Conclusion
<!-- Explain what is striking/noteworthy about the results. Summarize the state of knowledge and understanding after the completion of your work. Discuss the results and interpretation in light of the validity and accuracy of the data, methods and theories as well as any connections to other people’s work. Explain where your research methodology could fail and what a negative result implies for your research question. -->

read as "what is this going to change?"

this work will improve understanding of what leads to avoidable crashes, which will enable cyclists to plan better routes and municipal traffic departments to address problem areas

the main limitation will be the unavailability of complete cyclist numbers, e.g. it could be possible that all recorded crashes are outliers and most cyclists ride safely

methodology could fail if:

significant crash data is missing, i.e. crashes which go unreported

models are incorrect

# Acknowledgements
<!-- Thank the people who have helped to successfully complete your project, like project partners, tutors, etc. -->
Source For Outline: 
https://www.ethz.ch/content/dam/ethz/special-interest/erdw/department/dokumente/studium/master/msc-project-proposal-guidelines.pdf 

# Reference & Literature (Bibliography)
<!-- List papers and publication you have already cited in your proposal or which you have collected for further reading. The style of each reference follows that of international scientific journals. -->

# Appendix
<!-- Add pictures, tables or other elements which are relevant, but that might distract from the main flow of the proposal. -->
