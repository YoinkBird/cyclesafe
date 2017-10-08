Formatting HALP:  
https://help.github.com/articles/basic-writing-and-formatting-syntax/  
https://daringfireball.net/projects/markdown/syntax  

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
* * Summary
* * Model
* * * Acquisition
* * * Preprocessing
* * * Analysis
* * * Model
* * * Prediction
* * Application
* * * Technology
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

**Section Overview:**
<!--!toc_mini-->
<!--<toc_mini>-->
* Summary
* Model
* * Acquisition
* * Preprocessing
* * Analysis
* * Model
* * Prediction
* Application
* * Technology
<!--</toc_mini>-->
<!-- Explain the methods and techniques which will be used for your project depending on the subject:
field work, laboratory work, modeling technique, interdisciplinary collaboration, data type, data acquisition, infrastructure, software, etc. -->

## Summary

The final product consists of two parts, the user-facing application, or front-end, and the non-public data processing model, or back-end.  
The core mission of this project is to improve safety for cyclists.  
To this end, a balance between the front-end and back-end needs to struck such that the information is both readily available to the end user while also being accurate.  
This balance is best exemplified by two scenarios: one in which the model is accurate but without a user-facing application, and one in which the application is easy to use but the model is inaccurate.  
<example>
<!-- analogy:
#included: map-based route planning tools are easy to use, but doesn't have any safety prediction at all
#included: the current model can predict the safety of a route, but the end-user would have fit the model to a collection of GPS coordinates representing their route.
#pending, out of place here: The solution is to have a route planning tool which automatically generates a list of GPS coordinates for a route and uses these coordinates to predict the safety factor. The end-user need only plan their route as usual, and the tool does the rest.
-->
The first scenario would be a model which can accurately predict the safety of a route, but requires the end-user to fit the model to an independently generated collection of GPS coordinates representing their route.  
The first scenario could be a model which can accurately predict the safety of a route, but requires the end-user to independently provide the exact data which the model depends upon. Among other things, this would involve generating a collection of GPS coordinates representing the desired route and the various environmental conditions which the model uses for its prediction.  
The second scenario would be existing map-based route planning tools, which are easy to use but have no safety prediction whatsoever.  Although most of these tools offer alternative routes based on various factors, they do not include safety in their calculations.


In scenario one, a model with high accuracy can easily predict the danger of a given route, but would require the end-user to have a detailed understanding of data mining, the language used to create the model, and how to translate their desired route into a format which the model can use to make a prediction.  

In the second scenario, the application with a good UI makes it easy for the end-user to plan their route, but the inaccurate model will mislead the end-user about the actual safety of the route.  
Both scenarios are unfavourable, but fortunately the nature of application design and creating an accurate model help prioritise which component to focus on first.  
Data Mining is an open-ended problem: the model is trying to use existing data to make an accurate prediction about the future. As new data is made available, the accuracy of a given model is constantly changing, which requires the model to constantly change. E.g. new data sources may provide better input, which in turn requires the model to use different parameters or even a different algorithm.  
On the other hand, the application layer is more of a finite problem: the application consumes and presents data in a pre-defined format. The goal of abstracting the model for easier use is accomplished once the interface between the end-user and the and data is created. Of course, the application will change over time to accomodate user feedback, but this is secondary to the primary purpose of allowing users to interact with the data.  
In summary, while the accuracy of the data model can improve over time, the application has no impact on the accuracy of the model.  

With this in mind, the primary focus is on creating a simple UI for interacting with the output from the model.  
The model accuracy is the secondary focus, since it is expected to change over time.  

To this end, the project is done in three phases.  

Phases:  
one: create simple model for the application to consume  
two: create application with minimal functionality  
three: improve model accuracy


## Model
### Acquisition
quicknote: use available crash-data, augment with other data sources as necessary and as possible
primary source: TxDOT data
### Preprocessing
summary: use python, pandas to ensure data is useful
### Analysis
summary: use python, pandas, matplotlib to analyse data
### Model
quicknote: use python data mining libraries to generate the model
start with simple DecisionTree, move to more efficient models later
### Prediction

## Application
### Technology
quicknote: browser-based application using python, html, javascript

# Time Plan for Master’s Project Proposal and Master’s Thesis
<!-- Give a detailed time plan. Show what work needs to be done and when it will be completed. Include other responsibilities or obligations. -->

# Discussion / Conclusion
<!--
* Explain what is striking/noteworthy about the results.
* Summarize the state of knowledge and understanding after the completion of your work.
* Discuss the results and interpretation in light of the validity and accuracy of the data, methods and theories as well as any connections to other people’s work.
* Explain where your research methodology could fail and what a negative result implies for your research question.
* -->

read as "what is this going to change?"

this work will improve understanding of what leads to avoidable crashes, which will enable cyclists to plan better routes and municipal traffic departments to address problem areas

the main limitation will be the unavailability of complete cyclist numbers, e.g. it could be possible that all recorded crashes are outliers and most cyclists ride safely

methodology could fail if:

significant crash data is missing, i.e. crashes which go unreported

models are incorrect

# Future Work
## Data Sources
Use data from strava,mapmyride,etc to find the most common routes (among the users of these apps) and correlate with crash data

# Acknowledgements
<!-- Thank the people who have helped to successfully complete your project, like project partners, tutors, etc. -->
Source For Outline: 
https://www.ethz.ch/content/dam/ethz/special-interest/erdw/department/dokumente/studium/master/msc-project-proposal-guidelines.pdf 

# Reference & Literature (Bibliography)
<!-- List papers and publication you have already cited in your proposal or which you have collected for further reading. The style of each reference follows that of international scientific journals. -->

# Appendix
<!-- Add pictures, tables or other elements which are relevant, but that might distract from the main flow of the proposal. -->


<!--
Markdown Reminders:

# paragraphs
https://daringfireball.net/projects/markdown/syntax#p

## How to have normal text with '\n' turn into line-breaks instead of new paragraphs

### Normal Line Breaks
for line-breaks, either '<br/>' or '  '

item one  \n
item two  \n
item three  \n

### HTML pre-formatted tag
renders as monospaced font, maybe the easiest way to do business
Could, in theory, create a "yoinkbird" tag called "< ! - - < p > - - > " (spaces to avoid this being interpreted as a comment!)
<pre>
item one
item two
item three
</pre>

## Misc
quote-mode, i.e. a '|' in front of each line:
>item one
item two
item three

## no es bueno - standard paragraph
Using the '<p>' tag is idempotent, each line still gets rendered on one line due to missing '  ' or '<br/>'
<p>
Phases:
item one
item two
item three
</p>
-->
