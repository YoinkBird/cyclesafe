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
* Goals
* * Interpret What Makes Roads Safe
* * Interpret How Cyclists Can Ride Defensively
* * Design Tool To Help Find Safe Routes
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
* * * Project Preparation Gantt Chart
* * * Project Execution Gantt Chart
* * * Work-Packages:
* * * WP Deliverables:
* * * Roadmap
* * * * Critical Path
* Time Plan for Master’s Project Proposal and Master’s Thesis
* Discussion / Conclusion
* Future Work
* * Crash Data
* * Data Sources
* Acknowledgements
* Reference & Literature (Bibliography)
* Appendix
<!--</toc_mini>-->


# Introduction
<!-- Explain why this work is important giving a general introduction to the subject, list the basic knowledge needed and outline the purpose of the report. -->

# Background and results to date
<!-- List relevant work by others, or preliminary results you have achieved with a detailed and accurate explanation and interpretation. Include relevant photographs, figures or tables to illustrate the text.  This section should frame the research questions that your subsequent research will address. -->

<hr />

**PROPOSAL DRAFT**  
relevant work - misc traffic studies

preliminary results - class project; outline goal and results; in the next section lead in to the remaining questions

This is an extension of a group project I started during the Spring semester.

My goal is to help cyclists choose safer routes and to identify what makes streets dangerous.

For example, we identified time-of-day as a factor so I created a basic visualization of where crashes happen for the main intervals.
This could then allow cyclists to plan their route based on time of day.
https://nbviewer.jupyter.org/github/YoinkBird/dataMiningFinal/blob/master/Final.ipynb#Maps-of-Crashes 
(caveat: these maps lump together all crashes from 2010-2017 and thereby hide any potential trends)

**/PROPOSAL DRAFT**  

<hr />

# Goals
<!-- List the main research question(s) you want to answer. Explain whether your research will provide a definitive answer or simply contribute towards an answer. -->

<hr />

**PROPOSAL DRAFT**  

Title: Data Driven Approach towards Improving Road Safety for Cyclists  
Intro: Purpose: Analyse available data to understand how crashes with other vehicles occur.  
**Section Overview:**  
<!--!toc_mini-->
<!--<toc_mini>-->
* Interpret What Makes Roads Safe
* Interpret How Cyclists Can Ride Defensively
* Design Tool To Help Find Safe Routes
<!--</toc_mini>-->
## Interpret What Makes Roads Safe
focus on "external" data features, e.g. weather, bike lane, speed limit
possible break down by intersection and frequency of accidents
## Interpret How Cyclists Can Ride Defensively
focus on "personal" data features, e.g. wearing helmet, avoiding busy roads
classification into "avoidable" and "avoidable" crashes
e.g. left-turn crash seen as "avoidable" because cyclist can look for vehicles, but crash from rear seen as "unavoidable"  because cyclist has no visibility of vehicles
## Design Tool To Help Find Safe Routes
e.g. assign safety score to routes provided by other tools

**/PROPOSAL DRAFT**  

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
* * Project Preparation Gantt Chart
* * Project Execution Gantt Chart
* * Work-Packages:
* * WP Deliverables:
* * Roadmap
* * * Critical Path
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


## CRISP-DM

This project will follow the CRISP-DM data mining process [@crispDmWiki]:

[@crispDmWiki]: https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining "CRISP DM"


1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment


## Business Understanding
ABOUT: 'business' not to be taken literally, just means the environment in which the problem exists

<!-- 2 Data to Insights to Decisions -->
<!-- 2.1 Converting Business Problems into Analytics Solutions -->
<!-- 2.2 Assessing Feasibility -->
### Problem Statement
### Feasibility Assessment

## Data Understanding - Analysis
summary: use python, pandas, matplotlib to analyse data
<!--
2.3 Analytics Base Table
This work sits primarily in the Data Understanding phase
2.4 Designing and Implementing Features
2.4.5 Implementing Features
-->
### Analytics Base Table
Choose prediction subject, one-row-per-subject
determine domain concepts for features

### Feature Implementation
ABOUT: i.e. choose or create features for ABT
proxy features
Consider:
data availability
timing of data
longevity of data

Then: how are new features created? e.g. impute mph, create binary categories, etc

availability:
#### Data Acquisition
TODO: this needs to be more along the lines of "which data sources readily exist" or "which were considered"
the later section 'feature implementation' deals with availability, at which point the choice of data-source can be stated
quicknote: use available crash-data, augment with other data sources as necessary and as possible
primary source: TxDOT data


<!-- 3 Data Exploration -->
<!-- 3.1 The Data Quality Report -->
### Data Quality Report
ABOUT: i.e. quality of selected features
<!-- 3.2 Getting to Know the Data -->
<!-- 3.2.1 The Normal Distribution -->

## Data Preparation
summary: use python, pandas to ensure data is useful
<!-- counting "quality issues" as part of crisp-dm preparation" to keep 'quality found' and 'quality fixed' together; may need to change this mentality -->
### Data Quality Issues
<!-- 3.3 Identifying Data Quality Issues -->
#### Identified
<!-- 3.3.1 Missing Values -->
<!-- 3.3.2 Irregular Cardinality -->
<!-- 3.3.3 Outliers -->
<!-- 3.4 Handling Data Quality Issues -->
#### Fixed
<!-- 3.4.1 Handling Missing Values -->
<!-- 3.4.2 Handling Outliers -->
<!--
3.5 Advanced Data Exploration
3.5.1 Visualizing Relationships Between Features
3.5.2 Measuring Covariance and Correlation
-->
<!--
3.6 Data Preparation
3.6.1 Normalization
3.6.2 Binning
3.6.3 Sampling
-->

## Modeling
quicknote: use python data mining libraries to generate the model
start with simple DecisionTree, move to more efficient models later

## Evaluation
ABOUT: Prediction, cv, etc

## Deployment
ABOUT: the application of the 'analytics solution'
aka Application
### Technology
quicknote: browser-based application using python, html, javascript

### Project Preparation Gantt Chart
| status | | | | | | | | | |
|-|-|-|-|-|-|-|-|-|-|
| **prep** | clean project | | | | | | | | |
| **prep** | | generate basic model |


<!--
Basically:
# strip bullets
'<,'>s/\* /  /g
# leading '|'
'<,'>s/  /| /g
# trailing '|'
'<,'>s/$/ |/g
# somehow generate the header
| |...|
|-|...|

-->

### Project Execution Gantt Chart
<!--
* clean project
  * generate basic model
    * route: manual selection of pre-defined GPS coordinates
        * route: manual selection of generic GPS coordinates
          * data: fuzzy-match GPS coordinates
            * route: automatic selection of generic GPS coordinates
              * route: implement map as interface
                * route: overlay score on map
              * impute more mph limits
      * total score
      * partial score
        * mix routes
-->

<!-- NOTE: only have to have leading '|' and one closing. Update the header to add a column -->
| status | | | | | | | |
|--------|-|-|-|-|-|-|-|
| **critPath** | route: manual selection of pre-defined GPS coordinates |
| **critPath** | | | route: manual selection of generic GPS coordinates |
| **critPath** | | | | data: fuzzy-match GPS coordinates |
| **critPath** | | | | | route: automatic selection of generic GPS coordinates |
|              | | | | | | route: implement map as interface |
|              | | | | | | | route: overlay score on map |
|              | | | | | data: impute more mph limits |
| **critPath** | | route: total score |
|              | | | route: recommend best route |
|              | | route: partial score |
|              | | | route: mix routes |


### Work-Packages:
**DRAFT**  
**Staging for explanations down below**
'Work-Packages' and 'WP Deliverables' need to be combined, then fed into 'Roadmap' for details
Work-Packages - outline each WP as a header, use the mini-toc to list them all
WP-Deliverables - merge into 'Roadmap' or 'Work-Packages', TBD
**/DRAFT**

Note: Work packages (WP) need not necessarily be executed in the order of the gantt chart
The current gantt chart reflects the desired order of implementation vs actual dependency.
This needs to be re-worked to properly indicate both the actual inter-dependency and the desired execution timeline

data: fuzzy-match GPS coordinates [GPS-fuzzy-match]  
data: impute more mph limits [impute_mph_limit-noninter]  
route: manual selection of pre-defined GPS coordinates [GPS-manual-predef]  
route: manual selection of generic GPS coordinates [GPS-manual-generic]  
route: automatic selection of generic GPS coordinates [GPS-automatic-generic]  
route: implement map as interface [UI-GPS-generic]  
route: overlay score on map [UI-safety_score]  
route: total score [safety_score-total]  
route: recommend best route [UI-recommend-simple]  
route: partial score [safety_score-partial]  
route: mix routes [UI-recommend-complex]  

### WP Deliverables:
Simple ASCII diagrams for simplicity, see Roadmap for more detail

WP Impact on Functionality of Project

Notation: the WP-names should reflect the scope of the functionality
E.g. "safety_score" implies any WP with the name "safety_score-\*" such as safety_score-total and safety_score-partial

WP: [data:  GPS-fuzzy-match]  
[route: GPS-\*-generic] -> [model: GPS-fuzzy-match] -> [model: safety_score] -> [display score]

WP: [data:  impute_mph_limit-noninter]  
[route: GPS-\*-generic] -> [model: GPS-fuzzy-match] -> [model: impute_mph_limit-noninter] -> [model: safety_score] -> [display score]

WP: [route: GPS-manual-predef]  
[route: GPS-manual-predef]     -> [model: safety_score] -> [display score]  

WP: [route: GPS-manual-generic]  
[route: GPS-manual-generic]    -> [model: safety_score] -> [display score]  

WP: [route: GPS-automatic-generic]  
[route: GPS-automatic-generic] -> [model: safety_score] -> [display score]  

WP: [route: UI-GPS-generic]  
[route: GPS\*] <-> [gui: UI-GPS-generic]

WP: [route: UI-GPS-safety_score]  
[route: GPS\*] -> [gui: UI-GPS-generic] -> [gui: UI-GPS-safety_score]

WP: [route: safety_score-total]  
[route: GPS-\*]     -> [model: safety_score-total] -> [display total score]  

WP: [route: UI-recommend-simple]  
[route,several: GPS-\*]     -> [model: safety_score-total,several] -> [model: safety_score-total] -> [display best total score out of several (i.e. find safest route out of multiple routes)]  

WP: [route: safety_score-partial]  
[route: GPS-\*]     -> [model: safety_score-partial] -> [display partial scores]

WP: [route: UI-recommend-complex]  
[route,several: GPS-\*]     -> [model: safety_score-partial,several] -> [model: safety_score-partial] -> [display best combined scores out of several (i.e. combine safest sections of multiple routes into one route)]   



### Roadmap

**Strategy**: Each stage should result in a usable product while successively improving usabilty

**Terminology**:
* csv-file (csv) : comma-separate values file, such as a spreadsheet, in which data is separated by commas
* model : data-processing script written in python which reads makes a prediction from provided data
* route-mapper: tool with a map-like interface which returns GPS coordinates for manually selected locations on the map
* route-planner: tool with a map-like interface which returns GPS coordinates for an automatically plotted route between manually specified beginning and ending locations.

The following steps make sure to gradually improve the usability by implementing one feature at a time

#### Critical Path
The result is a tool which can score a route generated by a third-party route-planner, but still requires the user to pass in this data.

0. current state: no route creation, no scoring
* route: none, only overlays crash information on a map
* UI: edit python model
* data: csv-file of raw crash data, model-fit-data generated from test-train split
* backend: read-in raw data csv-file, process raw data, fit model on test-train split
* features: importance of features auto-determined from cross-validation
* implements: basic python model, shows feature importance, map generation, 

| inputs | -> [model] -> | output |
| ------ | ----- | ------ | 
| raw crash data | process, test-train split, fit | score |
| features  | x-val to determine important features | |


1. manual route creation from pre-defined coordinates, manual scoring  
* route: user manually creates route from list of known intersections
* UI: user hand-edits csv-file of known intersections, fills in missing data, e.g. weather, street condition, lighting, etc
* data: TBD
* backend: fit model on csv-file
* implements: model fitting on pre-formatted GPS coordinate data, vs. fitting on test-train data
* interfaces: csv-file, python model

<!-- | features: user-input |  | score | -->
| inputs | -> [model] -> | output |
| ------ | ----- | ------ | 
| features: user provided | combine with GPS-coords |  |
| route: pre-defined GPS coords | simple fit | score |


2. manual route creation from arbitrary coordinates, manual scoring  
* route: user manually creates route using a route mapping software. TODO: e.g.?  
* UI: user draws route by hand on a map, manually exports the list of GPS coordinates, then feeds them to tool.  
* backend:
  * fuzzy-match conversion of GPS coordinate input to "model-data" (csv or dataframe)  
fuzzy-match GPS coordinates:  
crash data GPS coordinates will not be exactly same as route-mapper GPS-coordinates. Therefore, imprecisely (fuzzy) compare user-input GPS coords to crash-data GPS coords to find closest match. Initially only perform this fuzzy match on intersection coordinates, as single-location coordinates can be harder to place precisely.  
  * fit model on "model-data"
* implements: model fitting on arbitrary GPS coordinate data, vs selection of pre-defined GPS data
  * interface to third-party route-mapper
  * fuzzy-match GPS coordinates
* interfaces: external route-mapper, csv-file, python model

| inputs | -> [model] -> | output |
| ------ | ----- | ------ | 
| features: user provided | combine with GPS-coords |  |
| route: manually generated arbitrary GPS coords | fuzzy-match against existing data | score |

3. automatic route creation from arbitrary destinations, manual scoring  
* route: user automatically creates route using conventional route planning software.
* UI: third-party tool creates route from user preferences, user manually exports the list of GPS coordinates, then feeds them to tool.  
* backend:
  * fuzzy-match conversion of GPS coordinate input to "model-data" (csv or dataframe)  
  * fit model on "model-data"
* implements: auto-generation of route GPS coordinates, vs hand-generation
  * interface to third-party route-planner
* enables: future automatic interface between tool and route-planner, without user as go-between
* interfaces: external route-planner, csv-file, python model

| inputs | -> [model] -> | output |
| ------ | ----- | ------ | 
| features: user provided | combine with GPS-coords |  |
| route: automatically generated arbitrary GPS coords | fuzzy-match against existing data | score |

About GPS-coordinates for intersections vs non-intersections:
TBD


# Time Plan for Master’s Project Proposal and Master’s Thesis
<!-- Give a detailed time plan. Show what work needs to be done and when it will be completed. Include other responsibilities or obligations. -->

# Discussion / Conclusion
<!--
* Explain what is striking/noteworthy about the results.
* Summarize the state of knowledge and understanding after the completion of your work.
* Discuss the results and interpretation in light of the validity and accuracy of the data, methods and theories as well as any connections to other people’s work.
* Explain where your research methodology could fail and what a negative result implies for your research question.
-->

<hr />

**PROPOSAL DRAFT**  

read as "what is this going to change?"

this work will improve understanding of what leads to avoidable crashes, which will enable cyclists to plan better routes and municipal traffic departments to address problem areas

the main limitation will be the unavailability of complete cyclist numbers, e.g. it could be possible that all recorded crashes are outliers and most cyclists ride safely

methodology could fail if:

significant crash data is missing, i.e. crashes which go unreported

models are incorrect

**/PROPOSAL DRAFT**  

<hr />

# Future Work <!-- very flexible, most of plan will be here -->
## Crash Data

Add data on bike-lane presence

Examine before/after lane reduction: car+car |vv|^^| => |cv|<>|vc| bike+car '|cv|' + turn lane '|<>|'
studies show...

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
