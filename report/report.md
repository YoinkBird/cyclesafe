# Meta
Formatting HALP:  
https://help.github.com/articles/basic-writing-and-formatting-syntax/  
https://daringfireball.net/projects/markdown/syntax  

## DRAFT REVISIONS
Document major changes to roadmap or implementation since last draft  
This is for transparency, so customer is aware of conceptual changes,  
and for docmentation, the iterative nature of the project lifecycle is clear.  
  
<!-- look for the REV<n> keyword throughout the doc -->
REV2 [20171017] roadmap: remove gps fuzzy match from CP  
REV1 [2017xxxx] roadmap: first pass  

## Title
<!-- Define a short, significant title which reflects clearly the contents of your report. The title page follows the guidelines of scientific proposals at Department of Earth Sciences (see http://www.erdw.ethz.ch/documents/index). -->

Data Driven Approach towards Improving Road Safety for Cyclists

## Time Plan for Master’s Project Proposal and Master’s Thesis
<!-- Give a detailed time plan. Show what work needs to be done and when it will be completed. Include other responsibilities or obligations. -->
@STUB: track the completed work-packages from Roadmap


# Abstract
<!-- Succinct abstract of less than one page. -->
<!-- details for abstract courtesy of https://www.honors.umass.edu/abstract-guidelines , review of abstract courtesy of http://www.sfedit.net/abstract.pdf -->

<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [meta](#meta) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [discussion--conclusion](#discussion--conclusion) | [future-work](#future-work) | [acknowledgements](#acknowledgements) | [reference--literature-bibliography](#reference--literature-bibliography) | [appendix](#appendix) | 
<!--</@breadcrumb>-->

<!-- tag for automatically updating the abstract. future work! idea: write abstract, copy-paste to another section. as that other section evolves, the original line from abstract may change. with this tag, the change could be auto-synced to abstract  -->
<!--<abstract_purpose>-->
<!--<abstract_methodology>-->

<!--
goal (purpose), what (methods), why, how (methods),
-----
purpose (goal, why)
methods (what)
results
conclusion
-->

<!-- TODO - state potential compromises in abstract -->
<!-- TODO - state hypothesis in first sentence - in this case, the goal? -->
PURPOSE  
This project aims to help bicycle commuters improve their safety in mixed traffic.  
The resulting increase in safety while cycling will hopefully convince more commuters to use a bicycle.  
If successful, this will reduce the number of severe injuries while also easing traffic congestion.  
@citationNeeded - article: increased cycling increases safety  

METHODOLOGY  
A web-based map interface will help cyclists plan safer routes,  
&nbsp;&nbsp; and an analysis of the data used to plan the route will help cyclists ride more defensively.  
This is accomplished using multiple approaches:  
  The first approach seeks to analyse the potential for severe injury along a generic route.  
A machine learning model will determine the probability of injury for a given route based on cumulative data from all known crashes.  
This is a closed form solution, as it relies entirely on available data.  
  The second approach uses historical crash data to recommend safer routes   
Recommend a route without dangerous intersections, where the danger of an intersection is approximated from crash location data.  
This is an approximation of a closed form solution, as there is insufficient data to determine precisely what leads to a crash.   
<!-- Assign a score to intersections based on crash data (frequency of crashes in intersection, injury severity, etc).  -->
Previous Work:  
This project is based off of a previous project to identify the factors associated with severe injury.  
Caveat:  
This project does not directly focus on factors which cause a crash, as there is not enough data to allow for a closed solution.  

RESULTS:  
These are the results the project hopes to accomplish:  
* top factors leading to severity injury  
* challenges encountered  
* accuracy of model  
* success of the technology used  
* which routes dangerous, and do they have feasible alternatives?  

CONCLUSION:  
The created application will help cyclists ride more defensively, but more effort needs to be made a city level.  
Cycling safely will continue to be inconvenient as long as safe routes are a big detour.  
Recommendation is to increase data collection in areas with most crashes to better analyse the factors leading into it.  


# Table of content
<!-- The table of content lists all chapters (headings/subheadings) including page number. -->
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [meta](#meta) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [discussion--conclusion](#discussion--conclusion) | [future-work](#future-work) | [acknowledgements](#acknowledgements) | [reference--literature-bibliography](#reference--literature-bibliography) | [appendix](#appendix) | 
<!--</@breadcrumb>-->

<!--!toc_mini-->
<!--<toc_mini>-->
* Introduction
  * Structure of this Paper
* Background and Results to Date
  * Relevant Work
  * Previous Work
  * Results to Date
* Goals
* Methodology
  * Summary
  * CRISP-DM
  * Business Understanding
    * Problem Statement
    * Feasibility Assessment
  * Data Understanding - Analysis
    * Analytics Base Table
    * Feature Implementation
    * Data Acquisition
    * Data Quality Report
  * Data Preparation
    * Data Quality Issues
  * Modeling
  * Evaluation
  * Deployment
    * Technology
    * Project Preparation Gantt Chart
    * Project Execution Gantt Chart
    * Work-Packages:
    * Roadmap
* Time Plan for Master’s Project Proposal and Master’s Thesis
* Discussion / Conclusion
* Future Work
  * Crash Data
  * Data Sources
* Acknowledgements
* Reference & Literature (Bibliography)
* Appendix
<!--</toc_mini>-->


# Introduction
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [meta](#meta) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [discussion--conclusion](#discussion--conclusion) | [future-work](#future-work) | [acknowledgements](#acknowledgements) | [reference--literature-bibliography](#reference--literature-bibliography) | [appendix](#appendix) | 
<!--</@breadcrumb>-->
<!-- Explain why this work is important giving a general introduction to the subject, list the basic knowledge needed and outline the purpose of the report. -->

<!--!@section_sentence-->
The goal of this project is to create a continuously updated traffic study using data passively gathered from existing sources and to present the results in a readily interpretable manner in order to improve personal safety for cyclists.  

<!-- general introduction to the subject -->
#### Traffic Studies - Active vs Passive
Current traffic studies typically involve observing specific sections of road for fixed time period.  
This provides a lot of data which can be used to improve traffic safety, but is limited by the frequency and scale at which such studies can be done.  
Staffing is required to run these studies, therefore there are practical constraints on how often and how widespread they can be run.  
The result is very accurate data for a few representative locations, but the data represents a static section in time and is expensive to update.

Machine learning and automated monitoring can help reduce the cost of active studies, but there is still cost involved in obtaining widespread coverage using automated equipment (e.g. installing cameras at every intersection).

A passive traffic study using data from indirect sources can run continuously and monitor as many locations as the data can provide.  
The drawback is that indirect data sources may not provide as much information as actively collected traffic data.  
This alternative application of machine learning has the advantage that it is scalable and requires fewer additional resources.  
In many cities, crash reports are one of the readily available indirect sources. On the one hand, they are continuously updated and widespread - a police report is filed any time police are called to the scene of an accident anywhere in the region. On the other hand, police reports for crashes are meant for establishing facts related to a crash, and as such don't capture all of the data required by an active traffic study. Additionally, not every crash leads to a police report, so these types of report are also limited by statistical significance.  
However, these reports are advantageous due to their frequency and geographic distribution; they can reflect time-based trends and localised trends where active traffic studies can only capture a snapshot.

#### Interpretability in Statistics and Machine Learning
Interpretability describes how readily understandable a process is. For studies based entirely on statistics, it describes how easily the results can be interpretted to correlate cause and effect. This concept extends to machine learning, and describes how easily the inputs to a model can be traced to the model's results.  
Machine learning is heavily based on traditional statistic methods, but is nonetheless often perceived as a black box due to its ability to produce results from large amounts of data which could not be handled by traditional techniques.  
It is important to be able to understand the output of a machine learning model for several reasons. The model exists as part of a larger framework to solve a particlar problem, and as such requires supervision to ensure that the results are relevant. In this context, interpretability is important for adjusting the inputs, or even the problem statement, to achieve better results.  
Interpretability is also important for the end-usage of the model; the model is specific to one particular goal, whereas applications of the model may have various goals and as such need to understand how the model fits in with the overarching application.  
For example, if a model indicates that a certain trait is involved in a certain outcome, it is important to be able to understand the actual significance of this trait.  Another example could be a model which finds that two traits lead to a certain outcome; this result requires being able to understand to what extent these traits interact. 

#### Public Safety and Trafffic Study Interpretability 
To maximise public safety, it is important for traffic studies to be interpretable.  
This report interprets interpretability of a traffic study in two ways: The traditional method of explaining which inputs lead to which outcome, and a more usable interpretation of how to best use the results of the study to improve traffic safety.  
In simpler terms, understanding how the study achieves its results, and how to practically apply the study's results.  
The traditional method involves explaining the methods used and explaining how inputs can be traced to outputs.  
The second method is implemented as a publicly accessible interface to the model used for the study.  

<!-- provide overview of report sections -->

<!-- brainstorming draft
* fundamental goal: improve safety for cyclists
* current approaches seem to be based on conventional methods, e.g. apply various measures after a simple traffic study

  two-pronged approach: determine factors associated with injury severity to improve safety given a crash, and use crash location data to recommend safer routes 

* fundamental goal: improve safety for cyclists
* current approaches seem to be based on conventional methods, e.g. apply various measures after a simple traffic study
* idea: use data to enhance current approach in order to make more effective safety decisions
* possibilities:
  * determine which routes are dangerous
  * determine factors which impact accident severity
  * measure effectiveness of decisions taken by analysing data before/after a change (such as whether installing a bike lane helps, or converting a 4-lane to 2-lane+2-bike-lane has the desired effect)
* narrow down: researching the ideas reveals two %todo%: those which an individual can %todo%, and those which require planning on a municipal level
  * explain: data analysis is common factor for all ideas, big differentiation is time-to-implement once data is analysed. municipal-level decisions can take a long time - idea would have to be presented to %todo%, reviewed, budgeted before implementing. time for implementing decisions at the personal-level depends only on the individual. therefore, narrowing selection down to personal-level ideas, or in other words, solutions which empower individuals to make safer choices.
* choose:
  * determine dangerous routes - presently partially possible on TxDOT website, although not easiest to do. shows overlay of crashes.
    * todo - limitations
  * factors which impact accident severity - this would help inviduals know which factors to avoid or minimise, thus making their route safer
  * => can combine both into one tool: find factors and use them to analyse route danger
* challenge: data availability. analysing which factors lead to a crash is an open-ended problem. one: there is no way to measure whether a crash did not happen. two: there is not sufficient traffic data to establish a baseline for comparison. for example, if exact traffic data were known for one intersection, one could simply calculate the ratio of cyclists to motorised vehicles and state that the number of avoided crashes is the number of total cyclists minus the number of recorded crashes. This is assuming that the data strictly measures only vehicles on the road with a potential of a crash, as it would not matter if a cyclist were e.g. on a sidewalk. However, such precise data could not be found, and most likely does not exist.
  * therefore: limit the question to data which is available. There is no precise data on the number of cyclists, but there is data on crashes which resulted in a police report. This data includes the severity of the crash. This leads to a useful question: given a crash, what factors influence the severity? While this doesn't tell the cyclist how to avoid a crash, it does tell them how they can increase their safety in the case of a crash.
  !extra, don't use:
  third approach, if making assumptions: through lens of defensive riding, can infer crash/no-crash outcomes from dataset. E.g. for each crash, add a category of 'avoidable' based on set of guidelines for defensive riding.
  fourth approach: relative trends - data analysis to find correlations, e.g. what factors are overrepresented (more crashes at higher speeds, or less crashes when bikelanes, etc)
  !/extra, don't use
-->

## Structure of this Paper  

This paper largely follows a traditional structure [@source], but wraps the data-mining-specific process 'CRISP-DM'  
There is some overlap between the two formats, but each have a different focus.  

The following descriptions explain the purpose of sections which seeminly overlap.

| Section | Explanation |
|-|-|
| Abstract |  elevator pitch of entire paper |
| Introduction |  provide context for this paper, introduce reader to concepts required, e.g. DM, CRISP-DM, etc |
| Background and Results to Date |  context within which the research takes place (relevant other research) |
| Goals |  elevator pitch for "main research questions", non-technical |
| Methodology |  methods,techniques => this is where CRISP-DM comes in to play |
| Time Plan | schedule for CRISP-DM phases | 
| Discussion / Conclusion |  |

@TODO: update descriptions

| CRISP-DM Section | Explanation |
|-|-|
| 1. Business Understanding |  how to approach the problem, e.g. how to convert problem into Analytics solution, then assess feasibility of the solution (e.g is relevant data present) |
| 2. Data Understanding |  builds on "assess feasibility". e.g. analyse data (ABT), decide which "compound features" to implement, select data sources) |
| 3. Data Preparation |  process the data, make it useful for a machine-learning model |
| 4. Modeling |  choose,develop the right ML model |
| 5. Evaluation |  improve accuracy of ML model (cross validation, etc) |
| 6. Deployment |  integrating the model into the "business environment", in this case, the application |


# Background and Results to Date
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [meta](#meta) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [discussion--conclusion](#discussion--conclusion) | [future-work](#future-work) | [acknowledgements](#acknowledgements) | [reference--literature-bibliography](#reference--literature-bibliography) | [appendix](#appendix) | 
<!--</@breadcrumb>-->
<!-- List relevant work by others, or preliminary results you have achieved with a detailed and accurate explanation and interpretation. Include relevant photographs, figures or tables to illustrate the text.  This section should frame the research questions that your subsequent research will address. -->

<!-- TODO: brief overview of this section. -->
While improving traffic safety is not a new topic, addressing it with machine learning techniques on an existing dataset seems to be a novel approach.  
This project builds on a previous project which used this approach as a proof-of-concept.  
Conventional approaches using machine learning to improve traffic safety tend to involve traffic studies or long running trials of novel traffic-controls to determine safety factors.  However, this approach tends to require a higher budget and active involvement by a public or private organisation with an interest in traffic safety.  
The advantage of using machine learning techniques on existing data is to provide a low-budget analysis of traffic safety factors with a quick turnaround time, independent of existing policies.  
<hr />

<!-- TODO: include the prior art research -->
## Relevant Work

relevant work - misc traffic studies, links are currently stored in google docs

## Previous Work
preliminary results - class project; outline goal and results; in the next section lead in to the remaining questions

<!-- abstract: previous work -->
[@originalProject]

[@originalProject]: https://nbviewer.jupyter.org/github/YoinkBird/dataMiningFinal/blob/master/Final.ipynb#Maps-of-Crashes 
This project is based off of a previous project to identify the factors associated with severe injury.  
This research produced two tree-based machine learning models, one a complex model for identifying the risk factors, and the other a simplified decision tree to conceptualise safety factors in a human readable format.  
Generating two models addressed a fundamental issue with machine learning: as the model gets more precise, it gets more complex and difficult to understand. The ability to understand how a model works is generally referred to as "interpretability"[@citationNeeded].  
The risk factors identified by the complex model could be used when analysing the safety of an intersection.  
However, the underlying decision tree makes too many decisions for a human to interpret on their own, and therefore would require an extra layer to be human-readable.  
The visualisation generated by the simple model could be used by an individual cyclist trying to determine the effect of environmental factors on their route.  
However, this simplified model is only an approximation of the factors leading to severe injury, and trades accuracy for usability. Essentially, it rephrases the problem as "what are the safety factors, given the constraint of a limited number of possible choices". This tradeoff is acceptable, as manual interpretation of complex models is likely to result in error anyway @citationNeeded. In summary, for model interpretability, the error introduced by simplifying the model needs to be balanced with the error a human would make while reading a complex model.  
Furthermore, a complex model is inconvenient to read and would likely not find much usage @citationNeeded.  
<!-- todo - did the human-readable model use the XGB feats? I thought so -->
<!--
maybe an abstract on 'why we can't interpret complex models? 
could be useful for explaining the choices made. 
point out how the trees ask the same question at multiple levels of different paths 
e.g., for the format (question1:answer -> q2:answer)
one path could be be: '(A>0.50)->(B<0.50)->(A>0.75)->(C<0.50)'
and another could be: '(A>0.50)->(B>0.50)->(A<0.75)->(C<0.50)'
and another could be: '(A<0.50)->(C>0.50)->(B<0.50)->(A>0.25)'
yet another could be: '(A<0.50)->(B>0.50)->(A<0.25)->(C>0.50)'
because each of these features and their values have a different impact on the decisions.

Reducing this to a binary choice removes the duplicate questions,
but note that the tree make take on a different route since the data has changed.

one path could be be: '(A:1)->(B:0)->(D:0)'
and another could be: '(A:1)->(B:1)->(D:1)'
and another could be: '(A:0)->(C:1)->(B:0)'
yet another could be: '(A:0)->(B:1)->->(C:1)'

-->


<!-- TODO - analyse other project to create a proper summary of the results -->
## Results to Date  
@STUB  
For example, time-of-day was identified as a strong factor, so a visualisation was created to display where crashes happened during different time intervals [@originalProject].  
This visualisation could then allow cyclists to consider the time of day when planning their route.  
However, this is a raw visualisation as it does no further data processing to interpret the data.  
One major issue is that it displays all crashes from 2010-2017, and thereby hides any potential trends.  
For example, this visualisation does not portray whether the safety of an intersection changed over time.  

<!-- seque into current project -->
<!-- please, please, please reword this whole section. so much cringe -->
**Past Project's Future, Presently**  
The previous project layed a good foundation, and in doing so opened up many possibilities for new projects.  

This project continues the work of the previous project and will combine the accurate but non-interpretable model with the inaccurate but interpretable model into one tool which will both be accurate and interpretable. It will also make sure that this sentence gets changed to be less poindexter.  

<hr />

# Goals
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [meta](#meta) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [discussion--conclusion](#discussion--conclusion) | [future-work](#future-work) | [acknowledgements](#acknowledgements) | [reference--literature-bibliography](#reference--literature-bibliography) | [appendix](#appendix) | 
<!--</@breadcrumb>-->
<!-- List the main research question(s) you want to answer. Explain whether your research will provide a definitive answer or simply contribute towards an answer. -->
<!-- note: each goal should also be present in the abstract --> 

<!--
exclude for now, 'Goals' has no subsections and therefore toc gen keeps inserting too much
**Section Overview:**  
-->
<!--!toc_mini-->
<!--
fill in from abstract:
seems the first two goals need to be re-synced with abstract.
make one of them correspond to 'generic route anlysis'. "explanation of factors" is an open ended problem consider removing this and adding it later in the text
make the other correspond to the 'historical crash data'
-->

@STUB  
Create tool to help cyclists plan relatively-safe routes.  
In this context, relatively-safe is defined as "given a crash, will the cyclist be incapacitated or worse".  
<!--
(May want to redefine according to less severe injuries, e.g. "safe" as "able to ride away")  
-->

<!--
Todo: terminology: use the phrase "severity", or something, in lieu of "safety". Keeps the mission clear, and is accurate without interruption. Later, introduce the safety score.  
-->
[@terminology]: relatively-safe - placeholder term. see relative-safety  
[@terminology]: relative-safety - placeholder term. Assuming a cyclist is involved in a crash, how likely are they to be severely inured. ML: This is the target feature. @TODO: use featdef to list features and definitions  

<!-- tags: studies
Todo: should research whether accident severity correlates with responsibility. If studies show that light injury is usually cyclists fault, would be able to draw a line and cite the study.  
-->

This Involves:

Model to find relationship between environmental factors and safety/severity  


Interface to model which interprets the results in a manner which is meaningful to a human user.  
I.e. something that lets cyclists conveniently factor in relative-safety when planning their route  

Model depends on GIS-enabled crash data which lists locations of past crashes and environmental factors  
This data interpreted in two ways:  
Generic location-unaware analysis in order to calculate relative-safety of any generic route  
i.e. user provides a route, but exact coordinates are not used in the analysis  
Specific location-aware analysis in order to improve pertinance/relevance/argh of relative-safety score  
i.e. user provides a route, and analysis factors in known coordinates

Caveat: the model itself will be location unaware, whereas the interface to the model will enable the location-aware features.  
Model is location unaware because it would essentially end up scoring the more popular roots with a higher safety-factor. location-aware modelling could be useful for identifying where more crashes happen, but in a practical route-planning scenario this would lead to cyclists having to take large detours.   


<hr />

# Methodology
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [meta](#meta) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [discussion--conclusion](#discussion--conclusion) | [future-work](#future-work) | [acknowledgements](#acknowledgements) | [reference--literature-bibliography](#reference--literature-bibliography) | [appendix](#appendix) | 
<!--</@breadcrumb>-->

**Section Overview:**
<!--!toc_mini-->
<!--<toc_mini>-->
* Summary
* CRISP-DM
* Business Understanding
  * Problem Statement
  * Feasibility Assessment
* Data Understanding - Analysis
  * Analytics Base Table
  * Feature Implementation
  * Data Acquisition
  * Data Quality Report
* Data Preparation
  * Data Quality Issues
* Modeling
* Evaluation
* Deployment
  * Technology
  * Project Preparation Gantt Chart
  * Project Execution Gantt Chart
  * Work-Packages:
  * Roadmap
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

<!-- map to CRISP-DM, i.e. 'three' is the continued deployment -->
To this end, the project is done in three phases:  
one: create simple model for the application to consume  
two: create application with minimal functionality  
three: improve model accuracy  

This phased incremental approach is an application of a process commonly used in industry, which is covered in the next section.  


## CRISP-DM

The cyclical product development resulting from the trade-off between performance and usability fits perfectly within the CRISP-DM structure.

This section provides an abstract overview of CRISP-DM followed by its implementation for this project.  
At certain points, comparisons to other common project management frameworks are made.  

CRISP-DM (CRoss Industry Standard Porcess for Data Mining) [@crispDmWiki] is a cyclical framework designed to structure a data mining project from first conception to its deployment.

[@crispDmWiki]: https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining "CRISP DM"

<!-- original pdf available as well, this is a more convenient webpage -->
[@crispDmStages]: http://www.sv-europe.com/crisp-dm-methodology/

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

@TODO: update summary based on description in 'Business Understanding' - structure of waterfall with planned flexibility of agile.  
The framework consists of 6 stages which flow into each other and can loop back into a previous stage TODO at any time.  
This is similar to the traditional waterfall framework, with the major exception that the stages can be updated as new information becomes available.  

The stages are:  

1. Business Understanding  
This stage is the project-planning portion of the project and as such combines both waterfall and agile planning strategies.  
The project is planned out in terms of phases and resources as it would be with waterfall, but is meant to be updated under certain conditions as with agile project planning.  
This is particularly well adapted for data mining, as the goal must be well understood in advance, but the model has to be able to change based on its inputs.  
In essence, this stage lays the groundwork for a successful project.  

* determine the desired outputs of the project.  
* Feasibility Assessment - assess the feasibility of the project.  
* determine the data mining goals.  
* create project plan.  

2. Data Understanding  
This stage is for acquiring and exploring data in order to evaluate the quality and usefulness of the data sources provided in stage 1.  
@STUB:  
* Data Collection Report
* Data Description Report
* Data Exploration Report
* ABT
* Data Quality Report

3. Data Preparation  
@STUB:  
* Select
* Clean
* Construct required data (derived features, imputed values, etc)
* Integrate (merge/collate together different sources, aggregate multiple records into one)

4. Modeling  
@STUB:  
* choose technique, list assumptions
* create test train-test-eval plan
* create model (set parameters, build model, report about process involved)
* assess model - interpret according to domain knowledge and test/train. revise parameters and start over as needed.
5. Evaluation  
@STUB:  
Note: not same as error evaluation  
* evaluate results
  * assess results in terms of project/business success criteria
  * list approved models, i.e. which models to be used for the project
* review process: summarise process until this point, determine what needs to be repeated or still be done
* determine next steps: decide whether to proceed to next stage or loop back to a previous stage based on the current results.
  * list out possible actions, reasons, pro/con ; then describe decision
6. Deployment  
@STUB:  
* deployment plan - how will this be used for the business?
* monitoring + maintenance plan - how will model be updated and supervised?
* final report
* review project - "experience documentation" to summarise the knowledge gained during this project

# CRISP-DM Report
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [meta](#meta) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [discussion--conclusion](#discussion--conclusion) | [future-work](#future-work) | [acknowledgements](#acknowledgements) | [reference--literature-bibliography](#reference--literature-bibliography) | [appendix](#appendix) | 
<!--</@breadcrumb>-->

The report will summarise the cycles of the CRISP-DM as one pass, i.e. without revising sections.    
Where applicable, the first passes will be described.  

| CRISP-DM: |                                                      <!-- "tag anchors" for navigation in text-editor -->
[business-understanding](#business-understanding) |                <!--business_understanding-->
[data-understanding---analysis](#data-understanding---analysis) |  <!--data_understanding-->
[data-preparation](#data-preparation) |                            <!--data_preparation-->
[modeling](#modeling) |                                            <!--modeling-->
[evaluation](#evaluation) |                                        <!--evaluation-->
[deployment](#deployment) |                                        <!--deployment-->


<!-- 2 Data to Insights to Decisions -->
<!-- 2.1 Converting Business Problems into Analytics Solutions -->
<!-- 2.2 Assessing Feasibility -->

## Business Understanding
<!--business_understanding-->
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [meta](#meta) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [discussion--conclusion](#discussion--conclusion) | [future-work](#future-work) | [acknowledgements](#acknowledgements) | [reference--literature-bibliography](#reference--literature-bibliography) | [appendix](#appendix) | 
<!--</@breadcrumb>-->
| CRISP-DM: |                                                      <!-- "tag anchors" for navigation in text-editor -->
[business-understanding](#business-understanding) |                <!--business_understanding-->
[data-understanding---analysis](#data-understanding---analysis) |  <!--data_understanding-->
[data-preparation](#data-preparation) |                            <!--data_preparation-->
[modeling](#modeling) |                                            <!--modeling-->
[evaluation](#evaluation) |                                        <!--evaluation-->
[deployment](#deployment) |                                        <!--deployment-->

The "business" can be understood as any entity concerned with public transportation safety. The "business needs" for this project are oriented around increasing traffic safety for cyclists.  

* determine the desired outputs of the project.  

Desired Outputs:  
Objective: The objective of this project is to help cyclists avoid crashes which lead to severe injury.  
Objective: The objective of this project is to help cyclists severe injury when involved in a crash.
Project Plan: This will be achieved by creating a data mining model to analyse crash data, and then integrating the model into a route-planning tool.  
Success Criteria: The project will be considered a success if the resulting product can be used by cyclists to evaluate any arbitrary route for the possibility of severe injury given a crash.  
In simpler terms, a successful project will provide a product which cyclists use to make informed decisions about which routes to choose.  

Constraints on the Objective:  
The objective of this project is not to help cyclist avoid crashes altogether, as the main data source is post-crash data.  

Original Objective: The objective was originally to help cyclists avoid crashes in general. However, during the feasibility assessment supporting data was found to be insufficient. 
In particular, there is no data on the number of cyclists for a given road segment, and the crash data lists only reported crashes isntead of a cross-sample of all crashes.  

@STUB: round1: Use crash data to help reduce number of accidents  -> no available data  
@STUB: round2: Use crash data to help reduce number of accidents with severe injury ->  available data  


#### Feasibility Assessment
<!--feasibility_assessment-->
* assess the feasibility of the project.  

@STUB: round1: data on crashes-vs-non-crashes doesn't exist (no tracking of avoided accidents). loop back to problem statement fo round2 (as expected in the CRISP-DM lifecycle)  
@STUB: round2: accident data not available txdot data on crashes is available.  there is enough data to train at least a basic model  


Inventory of resources:  
Data on Crashes can be obtained from Texas Department of Transportation (TXDOT), National Highway and Safety Administration (NHSA), and the City of Austin Police Department (APD).  
Data on actual ridership is very sparse in comparisson to data on crashes, which prevents significant correlation and therefore will not be considered.  
Data on traffic-counts is available for certain road segments and is the total count for a 24 hour period.  
The software necessary for data processing and modelling is available as free python libraries (pandas, sklearn, scikit learn, other ML libraries as needed).  
The scale of this project is appropriate for any modern hardware as it does not require intense computing resources.  

Requirements, assumptions, and constraints:  
Requirements: all data and tools are free for use  
Assumptions: The crash data is assumed to be accurately reported as it is sourced directly from police crash reports @TODO:terminology.  
Constraints: The crash data only represents reported crashes, which may be biased towards severe injury. Therefore, it is possible that the model will be biased towards predicting more severe injuries than would happen in reality.  
@citationNeeded:[sources on how many crashes get reported]  
This overestimation cannot be assumed to be evenly distributed accross the dataset without understanding the factors which lead to a police report being filed, which  is beyond the scope of this project.  

@TODO: fill in more assumptions,requirements while processing next sections


Risks and Contingencies:  

Terminology:  
@TODO: fill in from elsewhere

Costs and Benefits:  


#### determine the data mining goals.  
Business Success Criteria: tool which can display any route and its associated risk of severe injury in a crash  
Data Mining Success Criteria: model which predicts on "accident severity", i.e. the crash severity


### Project Preparation Gantt Chart
project plan:
@TODO: create overview of model creation plan; this is not the same as the work-packages for deploymennt

| status | | | | | | | | | |
|-|-|-|-|-|-|-|-|-|-|
| **prep** | clean project | | | | | | | | |
| **prep** | | stub_model |
| **analysis** | | | interpretable_model |
| **analysis** | | | interpretable_model2 |
<!-- | **modeling** | | | | optimised_model1 | -->

#### WP: stub_model
Simple model using unoptimised decision tree and 3 binary features to predict 1 binary target  
Purpose: Gathering requirements and enabling all dependent work-packages  
Creating an intermediate simple model allows for the external interfaces to be defined and enables the rest of the technology stack.  
This strategy was found to be very useful for quickly iterating through the CRISP-DM process to "flush out" hidden requirements and dependencies.  
The stub model also allows for work on deployment to begin before the model is finalised. This approach works if the business requirements include deploying the project. Otherwise, this strategy bypasses the evaluation stage, as deployment is contingent upon meeting the business and data-mining success criteria.  
Dependency: @TODO  

#### WP: interpretable_model  
Model using decision tree with as many features as possible to predict the crash severity.  
Purpose: data analysis to decide which model to choose ; allows dependent interfaces to work with real data.  
Dependency: @TODO

@TODO: summarise changes

tools and techniques:
@TODO: create overview of terminology for model creation; may have overlap with the work-packages for deploymennt
Note: mention updates

<!--
#### WP: optimised_model1  
Model using type of algorithm most suited to the data  
Purpose: increase prediction accuracy  
see 'modeling' for specifics
keep in sync with 'modeling':
Likely RandomForest (bagged) or XGB (boosted), depends on data available
Dimension <4k, so boosted tree likely best solution (src: [@caruana_et_al_2008]) 

[@caruana_et_al_2008]: src: http://icml2008.cs.helsinki.fi/papers/632.pdf
-->


## Data Understanding - Analysis
<!--data_understanding-->
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [meta](#meta) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [discussion--conclusion](#discussion--conclusion) | [future-work](#future-work) | [acknowledgements](#acknowledgements) | [reference--literature-bibliography](#reference--literature-bibliography) | [appendix](#appendix) | 
<!--</@breadcrumb>-->
| CRISP-DM: |                                                      <!-- "tag anchors" for navigation in text-editor -->
[business-understanding](#business-understanding) |                <!--business_understanding-->
[data-understanding---analysis](#data-understanding---analysis) |  <!--data_understanding-->
[data-preparation](#data-preparation) |                            <!--data_preparation-->
[modeling](#modeling) |                                            <!--modeling-->
[evaluation](#evaluation) |                                        <!--evaluation-->
[deployment](#deployment) |                                        <!--deployment-->

* Data Collection Report
* Data Description Report
* Data Exploration Report
* ABT
* Data Quality Report
@STUB: Use the pandas python library to create the ABT, add functionality as needed.  
@TODO: this can be added to report later in project  
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

@STUB: Then: how are new features created? e.g. impute mph, create binary categories, etc  

availability:  
### Data Acquisition  
TODO: this needs to be more along the lines of "which data sources readily exist" or "which were considered"  
the later section 'feature implementation' deals with availability, at which point the choice of data-source can be stated  
@STUB:
quicknote: use available crash-data, augment with other data sources as necessary and as possible  
primary source: TxDOT data  
 practical assessment: using txdot limited feature csv format is simpler than other more complicated formats, e.g. NIST or txdot csv with all features.  

@BEGIN: the crash data was obtained from txdot website

<!-- 3 Data Exploration -->
<!-- 3.1 The Data Quality Report -->
### Data Quality Report
ABOUT: i.e. quality of selected features
<!-- 3.2 Getting to Know the Data -->
<!-- 3.2.1 The Normal Distribution -->
@STUB: use pandas libs and custom functions to generate report. current implementation only a draft.

## Data Preparation
<!--data_preparation-->
| CRISP-DM: |                                                      <!-- "tag anchors" for navigation in text-editor -->
[business-understanding](#business-understanding) |                <!--business_understanding-->
[data-understanding---analysis](#data-understanding---analysis) |  <!--data_understanding-->
[data-preparation](#data-preparation) |                            <!--data_preparation-->
[modeling](#modeling) |                                            <!--modeling-->
[evaluation](#evaluation) |                                        <!--evaluation-->
[deployment](#deployment) |                                        <!--deployment-->

* Select
* Clean
* Construct required data (derived features, imputed values, etc)
* Integrate (merge/collate together different sources, aggregate multiple records into one)
@STUB: explain txdot_parse.py works to prepare data, create features, etc  
@STUB: introduce featdef.py in it's role of feature creation, but make sure to mention it under modelling as well for its feature management capabilities  
@STUB: 
summary: use python, pandas to ensure data is useful

@BEGIN:
The data preparation steps were summarised as python code in order to apply the same steps to new data from the same source in the future.  

This section will mention the python functions when describing the steps taken.  
Python module: txdot_parse.py


**Select**  
'clean_data' loads the crash data into a pandas (pd) dataframe (df) from a csv file, which requires ignoring the non-machine-readable header.  

**Clean**  
The function 'clear-csv' ensures that the data is machine readable to avoid parsing errors as well as to facilitate the coding process.  
List:  
* replace punctuation with underscores
* lowercase feature names 
* encode strings containing numbers as a numeric   datatype
* encode strings containing dates   as a date-time datatype
* encode strings representing missing data as a null datatype
  * convert human-readable descriptions for missing data to machine-readable datatype
    * This includes converting '0' for "missing" to 'np.nan' to prevent the modelling algorithms from evaluating it as a real value. This also improves runtime performance, as a proper null value is evaluated quicker than an integer representation (numpy knows not to evaluate np.nan, but can't know in advance whether an 'int' is '0' and therefore spends extra time to evaluate it)
    Caveat: This does not apply to data which is actually '0', only to cases when '0' represents a missing value.

#### Construct required data (derived features, imputed values, etc) (@TODO: merge 'feature implementation' further above into this portion
**Construct Required Data**  
**Imputed Values**
imput speed limits - of the set of intersections with multiple entries, for any intersections missing the speed limit, impute speed limit from identical rows. If speed limit changed throughout time, use first available value either from the future speed limit or the past speed limit. As most speed limits were observed to increase over time, this induces a bias towards associating crash severity with higher-than-actual speed limits. This is accepted as the difference in speed limit is typically only 5mph.
Note that this will bias the overall model towards a higher speed limit associated with injury severity. This has a few consequences, but can be considered as a relatively safe assumption as the correlation between crash-severity and speed-of-impact is well understood. Furthermore, the speed limit itself is not the same as the impact speed.  

**Derived Features**
* decimal crash time - encoded 24h time to decimal as model does not understand 24h time and processes it as an integer value  

* 30min-rounded crash time - round date-time to within 30minute intervals to clarify visualisations

binary categories: 
This was done for a few reasons:  
Some of the data provided was more granular than required, as it encapsulaetd a degree of information unobtainable in production.

Visualisation: when visualising the data, more categories can become difficult to interpret. 

Some values were encoded as categorical data, but some algorithms only operate on numeric data or yes/no outcomes.  

surface condition:  
Factorize 'Wet' 'Dry' to '1' '0'

crash severity:  
convert 'non_incapacitating_injury','possible_injury','not_injured' to 1  
convert 'incapacitating_injury','killed' to 0  
This is how TxDot groups crash severity in their visualisations. @citationNeeded
Other splits were tried in an attempt to create a model to simply predict injury, but most reported crashes result in injury. @citationNeeded:[refer to visualisations and data report]  

intersection related:  

encoded as 1    :  'intersection_related', 'intersection'  
encoded as 0    :  'non_intersection', 'driveway_access'  
encoded as null :  'not_reported'  

@ASSUMPTION  
Two assumptions: 
Routing data will not distinguish at this level of granularity, especially for driveway_access, but does provide information as to when to turn.  
intersection-related crashes assumed to be avoidable by cyclist (not always true) and non-intersection crashes assumed to be unavoidable by cyclist (not always true).

light condition:  
Simple breakdown between "good visibility" and "bad visibility".  

encoded as 0 : 'dark_lighted', 'dark_not_lighted', 'dusk', 'dark_unknown_lighting', 'dawn'
encoded as 1 : daylight

manner of collision: 
This feature is complicated for binarisation as it requires several assumptions.  
In the end, it was not used.  
Explanation: Manner of Collision - direction of Units involved  
motorist fault likelihood higher for "non changing" situations, e.g. if going straight  

encoded as 1: 
* 'one_motor_vehicle_going_straight',
* 'angle_both_going_straight',
* 'one_motor_vehicle_other',
* 'opposite_direction_one_straight_one_left_turn',
* 'same_direction_one_straight_one_stopped'
encoded as 0: 
* 'one_motor_vehicle_backing',
* 'same_direction_both_going_straight_rear_end',
* 'opposite_direction_both_going_straight',
* 'one_motor_vehicle_turning_left',
* 'one_motor_vehicle_turning_right',


#### Integrate (merge/collate together different sources, aggregate multiple records into one)

<!-- counting "quality issues" as part of crisp-dm preparation" to keep 'quality found' and 'quality fixed' together; may need to change this mentality -->
### Data Quality Issues
@STUB: add report of overall quality issues, e.g. before/after fixing  
@STUB: find the references to dropna or features without enough values (e.g. average daily traffic amount)  
Note: combine 'Identified' and 'fixed' per-feature, i.e. 'qual issue for feat abc, fixed yes/no'  

@BEGIN:
speed_limit was sometimes 0, sometimes -1

@ASSUMPTION;
impact-speed not available, assume speed limit
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
<!--modeling-->
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [meta](#meta) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [discussion--conclusion](#discussion--conclusion) | [future-work](#future-work) | [acknowledgements](#acknowledgements) | [reference--literature-bibliography](#reference--literature-bibliography) | [appendix](#appendix) | 
<!--</@breadcrumb>-->
| CRISP-DM: |                                                      <!-- "tag anchors" for navigation in text-editor -->
[business-understanding](#business-understanding) |                <!--business_understanding-->
[data-understanding---analysis](#data-understanding---analysis) |  <!--data_understanding-->
[data-preparation](#data-preparation) |                            <!--data_preparation-->
[modeling](#modeling) |                                            <!--modeling-->
[evaluation](#evaluation) |                                        <!--evaluation-->
[deployment](#deployment) |                                        <!--deployment-->

* choose technique, list assumptions
* create test train-test-eval plan
* create model (set parameters, build model, report about process involved)
* assess model - interpret according to domain knowledge and test/train. revise parameters and start over as needed.  

The modeling phase was revisited several times as part of the rapid-implementation strategy.  
In essence, the first iteration started with a simple placeholder model and by the final iteration had been transformed into an application-specific model suited for real-world implementation.  

As determined during the data understanding phase, the dataset is mostly categorical.  
Therefore, each model is based on a decision tree algorithm, which are well suited for analysing categorical data.  

@DATADESC: Most of the features in the dataset are categorical, and most of the continuous features have discrete values and can be used as categorical features.  

NOTE: model had to be "dialed back" to accomodate what users can provide, @FUTUREWORK: segmentation  

#### Overview of Modeling Stage:  

The stub_model was quickly implemented in order to develop the framework for modeling and deployment as well as for data exploration.  
It was then replaced by the interpretable_model, which was quickly found to rely on features which were not available in deployment.  
The interpretable_model2 is the re-implemented interpretable_model containing only features whch could be obtained once deployed.  
Further models were not implemented at this time.  
@FUTUREWORK: Improve precision and accuracy  
Data Segmentation - improve precision - use models created from subsets of the data according to which features can be expected.  
Improved accuracy - The next step to improve accuracy would be to create a boosted tree model (optimised_model1). Boosted trees are an improved implementation of a decision tree and are well suited for smaller datasets [@caruana_et_al_2008].   
Such a model was created for this project's predecessor, and could be re-implemented for this project without the unavailable features.  

As such, only interpretable_model2 will be evaluated in this section as it includes stub_model and interpretable_model.  
stub_model and interpretable_model will be mentioned in context of their role in the overall lifecycle.  


@STUB: describe the x-val functions in model.py  

#### interpretable_model2:  
Technique:  Decision Tree  
Assumptions: ignores location, ignores intersection, only focuses on ... TODO: featdef    
Test Design: xval, roc score, see model.py  
Build Model: parameters - see model.py  
Assessment:  
put Evaluation in next section as this is the final model  


Explanation of Assumptions:  
Location unaware : @TODO: refactor this sentence, is it a stream-of-thought:  In essence, the model is deliberately location unaware, with the caveat and assumption that end-users can't simply avoid parts of town. If location 'b' has more crashes, and the route is 'A'->'B'->'C', it isn't helpful to tell end-user that they have to avoid 'B' by routing through a potential 'D','E','F'. HOWEVER - it could be useful to inform users of this factor, but would require a different model. For now, the goal is to make safe transit more convenient; it is already known that one can ride on the sidewalk for the whole commute at 15mph to increase relative-safety , so adding in "avoid these entire areas" won't increase the convenience.  
@TODO: find the term for intentionally biasing a model to ignore a feature; it's not the same as avoiding overfitting, but it's in the same conceptual category  

#### stub_model:  
Technique: DecisionTree  
Evaluation: Adequate enough for enabling deployment  
Deployment: very useful for finalising architecture and enabling the technologies involved, e.g. able to quickly see how route data needed to be converted for use with model, e.g. confronted with architecture challenges immediately  

<!-- compare interpretable_model perf with interpretable_model2 perf -->
#### interpretable_model: 
Technique: DecisionTree  
Evaluation: better than stub_model 
Deployment: dataset for crash data contains data not easily obtainable in the field. E.g. whether a particular GPS coordinate has an intersection.  
The decision was made to build a new model without this field-unobtainable data. @FUTUREWORK: segmentation  
The alternative would have been to follow the CRISP-DM flowchart and "loop back" to the business-understanding phase to reasses the feasibility of this project. For a future instance of this project, one potential solution could be to build a database of GPS-coordinates and intersections based on the existing crash data. However, this would require having to maintain two separate models, one with the extra information and one without, since not every route will be represented.  

@FUTUREWORK:  
#### segmentation_models: 
Technique: Multiple models operating on Segmented Dataset  
Assumption: #@TODO:notsure# output of different models with more/less features is statistically significant @citationNeeded  
Test Design: xval, roc score, etc; need to test the combined score.  
Build Model: determine avail feats, choose model according to avail feats, resulting in more/less accurate score.  
Caveat: This is not stacking, it is dataset segmentation, i.e. choosing the most appropriate model for the dataset prediction.  

Purpose: real-world sometimes has missing data. naive approach is to create model with "lowest common denominator" of missing data, as in the interpretable_model\*, but the resulting model lacks features unique to each route. Better approach is to create multiple models based on anticipated data avalability, i.e. one model based on a "common denominator" dataset, one model based on "common dataset" + "feature set A", one model based on "common dataset" + "feature set B", etc. This allows the most optimal model to be used based on the data available, providing a more accurate score based on increased availability of data. In general, training models on different slices of the dataset is referrred to as segmentation. @citationNeeded  
(e.g. lighting condition, weather is always available, but can be assumed to be identical or indistinguishable for any route. E.g. user can only input conditions at start, would require advanced knowledge to know whether lighting conditions will change further along in the route )  



## Evaluation
<!--evaluation-->
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [meta](#meta) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [discussion--conclusion](#discussion--conclusion) | [future-work](#future-work) | [acknowledgements](#acknowledgements) | [reference--literature-bibliography](#reference--literature-bibliography) | [appendix](#appendix) | 
<!--</@breadcrumb>-->
| CRISP-DM: |                                                      <!-- "tag anchors" for navigation in text-editor -->
[business-understanding](#business-understanding) |                <!--business_understanding-->
[data-understanding---analysis](#data-understanding---analysis) |  <!--data_understanding-->
[data-preparation](#data-preparation) |                            <!--data_preparation-->
[modeling](#modeling) |                                            <!--modeling-->
[evaluation](#evaluation) |                                        <!--evaluation-->
[deployment](#deployment) |                                        <!--deployment-->

* evaluate results
  * assess results in terms of project/business success criteria
  * list approved models, i.e. which models to be used for the project
* review process: summarise process until this point, determine what needs to be repeated or still be done
* determine next steps: decide whether to proceed to next stage or loop back to a previous stage based on the current results.
  * list out possible actions, reasons, pro/con ; then describe decision

Pending: evaluation comparison between interpretable_model2, interpretable_model, stub_model , using the previous project's xgb model as a benchmark.   This is low-priority as the focus for this project is on interpretability and deployment.  

## Deployment
<!--crispdm_deployment-->
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [meta](#meta) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [discussion--conclusion](#discussion--conclusion) | [future-work](#future-work) | [acknowledgements](#acknowledgements) | [reference--literature-bibliography](#reference--literature-bibliography) | [appendix](#appendix) | 
<!--</@breadcrumb>-->
| CRISP-DM: |                                                      <!-- "tag anchors" for navigation in text-editor -->
[business-understanding](#business-understanding) |                <!--business_understanding-->
[data-understanding---analysis](#data-understanding---analysis) |  <!--data_understanding-->
[data-preparation](#data-preparation) |                            <!--data_preparation-->
[modeling](#modeling) |                                            <!--modeling-->
[evaluation](#evaluation) |                                        <!--evaluation-->
[deployment](#deployment) |                                        <!--deployment-->

ABOUT: the application of the 'analytics solution'
aka Application
* deployment plan - how will this be used for the business?
* monitoring + maintenance plan - how will model be updated and supervised?
* final report
* review project - "experience documentation" to summarise the knowledge gained during this project
@STUB: currently this is the mapgen.py  
@STUB: introduce the final concept, but leave details to the defined work-packages to define how the overall solution will work together.  

Deployment Plan / Application Overview:  
For this report, the model itself is already "deployed", as the model is already integrated into an application.  
Therefore, at this point the paper will describe the application architecture and further plans for deployment.  

Currently, the model is integrated into a user-facing application which can be run on a local machine.  

monitoring and maintenance plan:  
Crash data can be obtained in an automated report, which can be used to continuously update the model.  
This will require further work to parse, as it uses a different format than the manual query used to build this model.  

For ongoing maintenance, the model parameters will need to be updated as new data becomes available.  
The schedule for this will be based on the amount of new data received, for which a threshold needs to be set.  

Project Review: 

### Technology
@STUB: brief overview of the stack used, WPs describe the rest
quicknote: browser-based application using python, html, javascript


### Project Execution Gantt Chart
<!--
REV2 [20171017] roadmap: remove gps fuzzy match from CP
* simplest proof-of-concept doesn't require real-world gps coords at all
* scoring doesn't depend on gps coordinates
REV1 [2017xxxx] roadmap: first pass
-->
<!--
deprecating the use of this bullet-point chart, no use in maintaing this and the table
* clean project
  * generate basic model
    * generate more complex model
      * etc
-->

<!-- careful - this is a slight duplication of the work-package descriptions -->

Legend:
This table describes the phases of the project, as well as abbreviations used within the table

| phrase | description |
|-|-|
| c-p , CP , critPath | critical path i.e. core requirements for project or subproject (i.e. component of a project) |
| poc | proof of concept, implementaiton of a CP . e.g. code implemented such that its state conforms with a critical path. |

| phase | description | importance | 
|-|-|-|
| poc1 | each tech implemented (first proof-of-concept) <br/> model, simple csv-based user interface, non-interactive map display of route |  

<!-- NOTE: only have to have leading '|' and one closing. Update the header to add a column -->
Note: non-obvious dependencies marked with [DEP: <paraphrased description of dependency>]  
note: only a project-phase chart, not a gantt chart with work-packages  

Minimal Description of phases (makes it easier to manange the table)  
<!-- todo: keep this tied-in to the work-packages. see comments below for more ideas, this just a placeholder -->
poc1: csv-ui, encode route using csv, model reads csv, gets gps coords, html+js display route on map  
<!-- TODO: add more descriptions of deliverables -->
<!-- TODO: correlate this table with the WP names, already getting out of sync. 
ideally, use the WP-name tags in the table, then have a script find-replace them.
| **poc1** | [GPS-manual-predef] |
should render:
| **poc1** | route: manual selection of pre-defined GPS coordinates |
while at it, maybe go back to bullet points:
[status][phase]
* [**poc1**][GPS-manual-predef]
  * [**crit**][GPS-manual-generic]
should render:
| status | | | | | | | |
| **poc1** | route: manual selection of pre-defined GPS coordinates |
| **crit** | | route: manual selection of generic GPS coordinates |
-->
<!-- todo: add the tags to this table, probably as: [$tag]<br/>$description  -->
<!-- TODO: make the proj-planning terminology consistent and explain:
keep in mind that this mixing WPs and deliverables, where deliverable contains WPs.
agile:thisproject
epic : critpath, deliverable
story,task : WP

-->
<!-- TODO: move 'dep' down into the dependencies, -->
<!-- TODO: convert this into a deliverables chart, and have a separate one for WPs? or just put the critpaths at top and have the WPs below?-->
<!-- leaving one extra col to be sure not cutting any off by mistake!-->
| status | | | | | |
|--------|-|-|-|-|-|
| **poc1** | [GPS-manual-predef]<br/>route: manual selection of pre-defined GPS coordinates |
| **crit** | | [GPS-manual-generic]<br/>route: manual selection of generic GPS coordinates |
|          | | | [GPS-fuzzy-match]<br/>data: fuzzy-match GPS coordinates |
|          | | | | [impute_mph_limit-noninter]<br/>data: impute more mph limits |
| **poc1** | | [UI-nointer-GPS-generic]<br/>route: implement map as output interface (non-interactive) |
|          | | | [UI-inter-GPS-generic]<br/>route: implement map as input  interface (interactive) <br/>[DEP:fuzzy-match]<br/>[DEP:auto-select generic GPS] |
| **crit** | | | | [GPS-automatic-generic]<br/>route: automatic selection of generic GPS coordinates |
|          | | | [UI-map-safety_score-total]<br/>[UI-map-safety_score-partial]<br/>route: overlay score on map |
| **poc2** | | [safety_score-total]<br/>route: total score |
|          | | | [UI-recommend-simple]<br/>route: recommend best route |
|          | | [safety_score-partial]<br/>route: partial score |
|          | | | [UI-recommend-complex]<br/>route: mix routes |


### Work-Packages:
**DRAFT**  
**Staging for explanations down below**  
[x] WP-Deliverables - merge into 'Work-Packages' (not into Roadmap)  
[x] 'Work-Packages' and 'WP Deliverables' need to be combined. Action Taken: convert "WPs" into headers, move "WP Deliverables" under the headers  
[x] Work-Packages - outline each WP as a header, use the mini-toc to list them all  
[ ] reference WP names from 'Roadmap', move explanations up into WP description  
**/DRAFT**

Note: Work packages (WP) need not necessarily be executed in the order of the gantt chart  
The current gantt chart reflects the desired order of implementation vs actual dependency.  
This needs to be re-worked to properly indicate both the actual inter-dependency and the desired execution timeline  

<!-- TODO: auto-list these, like a TOC. reason: have work-packages be a summary, then 'wp deliverables' the explanation, which re-uses the eact same titles. -->
<!--!toc_mini-->
Simple ASCII diagrams for simplicity, see Roadmap for more detail  

WP Impact on Functionality of Project  

Notation: the WP-names should reflect the scope of the functionality  
E.g. "safety_score" implies any WP with the name "safety_score-\*" such as safety_score-total and safety_score-partial  

Each WP lists the a critical path (i.e. simplest functioning product ) it can be integrated into.  


#### WP: data: fuzzy-match GPS coordinates [GPS-fuzzy-match]  
WP: [data:  GPS-fuzzy-match]  
Dependency: TODO  
[route: GPS-\*-generic] -> [model: GPS-fuzzy-match] -> [model: safety_score] -> [display score]  
**Description:**   
crash data GPS coordinates will not be exactly same as route-mapper GPS-coordinates. Therefore, imprecisely (fuzzy) compare user-input GPS coords to crash-data GPS coords to find closest match. Initially only perform this fuzzy match on intersection coordinates, as single-location coordinates can be harder to place precisely.  

#### WP: data: impute more mph limits [impute_mph_limit-noninter]  
WP: [data:  impute_mph_limit-noninter]  
Dependency: TODO  
[route: GPS-\*-generic] -> [model: GPS-fuzzy-match] -> [model: impute_mph_limit-noninter] -> [model: safety_score] -> [display score]  
<!-- TODO: 1. auto-create glossary 2. grep for terminology tags, make sure explained before used. could potentially examine "git log -p" in reverse to find terminology introductions, othewise this requires user to be self-aware and add the when they use the term. -->

[@terminology]: segment - a part of a road  

[@terminology]: segment data - crash-data entry for a segment. can be anywhere on a road, including at an intersection  

**Description:**   
Impute speed limits (mph limit) for segment data [@term:segment-data] which does not correspond to an intersection.  
@originalProject already imputes speed limiits for intersections. TODO: <!-- this is definitely explained somewhere, just copy-paste it -->

#### WP: route: manual selection of pre-defined GPS coordinates [GPS-manual-predef]  
WP: [route: GPS-manual-predef]  
Dependency: TODO  
[route: GPS-manual-predef]     -> [model: safety_score] -> [display score]  
**Description:**   
TODO: fill in from roadmap, critical path  

#### WP: route: manual selection of generic GPS coordinates [GPS-manual-generic]  
WP: [route: GPS-manual-generic]  
Dependency: TODO  
[route: GPS-manual-generic]    -> [model: safety_score] -> [display score]  
**Description:**   
TODO: fill in from roadmap, critical path  

#### WP: route: automatic selection of generic GPS coordinates [GPS-automatic-generic]  
WP: [route: GPS-automatic-generic]  
Dependency: TODO  
[route: GPS-automatic-generic] -> [model: safety_score] -> [display score]  
**Description:**   
TODO: fill in from roadmap, critical path  

#### WP: route: implement map as output interface [UI-nointer-GPS-generic]  
WP: [route: UI-nointer-GPS-generic]  
Dependency: TODO  
[route: GPS\*] --> [gui: UI-nointer-GPS-generic]  
**Description:**   
html+js display route on map  
current state: html+js display GPS coordinates on map  

#### WP: route: implement map as input interface [UI-inter-GPS-generic]  
WP: [route: UI-inter-GPS-generic]  
Dependency: [UI-nointer-GPS-generic]  
[route: GPS\*] <-> [gui: UI-inter-GPS-generic]  
**Description:**   
html+js let user plan route using map in addition to displaying route  


#### WP: route: overlay score on map [UI-map-safety_score-partial]  
WP: [route: UI-map-safety_score-partial]  
Dependency: [UI-nointer-GPS-generic] + TODO  
[route: GPS\*] -> [gui: UI-\*-GPS-generic] -> [gui: UI-map-safety_score-partial]  
**Description:**   
Show the safety score for partial route on the map.

#### WP: route: overlay score on map [UI-map-safety_score-total]  
WP: [route: UI-map-safety_score-total]  
Dependency: [UI-nointer-GPS-generic] + TODO  
[route: GPS\*] -> [gui: UI-\*-GPS-generic] -> [gui: UI-map-safety_score-total]  
**Description:**   
Show the safety score for entire route on the map.  

#### WP: route: total score [safety_score-total]  
WP: [route: safety_score-total]  
Dependency: TODO  
[route: GPS-\*]     -> [model: safety_score-total] -> [display total score]  
**Description:**   
calculate safety score for entire route  

#### WP: route: recommend best route [UI-recommend-simple]  
WP: [route: UI-recommend-simple]  
Dependency: TODO  
[route,several: GPS-\*]     -> [model: safety_score-total,several] -> [model: safety_score-total] -> [display best total score out of several (i.e. find safest route out of multiple routes)]  
**Description:**   
retrieve multiple routes from third-party mapping service, calculate total score (safety_score-total) for each one, recommend the safest  

#### WP: route: partial score [safety_score-partial]  
WP: [route: safety_score-partial]  
Dependency: TODO  
[route: GPS-\*]     -> [model: safety_score-partial] -> [display partial scores]  
**Description:**   
calculate safety score for each route segment

#### WP: route: mix routes [UI-recommend-complex]  
WP: [route: UI-recommend-complex]  
Dependency: TODO  
[route,several: GPS-\*]     -> [model: safety_score-partial,several] -> [model: safety_score-partial] -> [display best combined scores out of several (i.e. combine safest sections of multiple routes into one route)]   
**Description:**   
retrieve multiple routes from third-party mapping service, calculate segment scores (safety_score-partial) for each one, combine lowest scores to create a safest route (i.e. combine safest sections of multiple routes into one route)]  



### Roadmap

**Strategy**: Each stage should result in a usable product while successively improving usabilty

**Terminology**:
* csv-file (csv) : comma-separate values file, such as a spreadsheet, in which data is separated by commas
* user interface (UI, interface) : method for user interaction with the tool. Can be graphical, command-line, text file, etc.
* graphical user interface (GUI) : UI which relies on a visual (i.e. non-textual) interface
* csv-UI : csv-file based interface. e.g. user manipulates a CSV  to interact with the tool, or tool manipulates a CSV to relay information to the user
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


WP: [GPS-manual-predef]
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

WP: [GPS-manual-generic]  
2. manual route creation from arbitrary coordinates, manual scoring  
* route: user manually creates route using a route mapping software.
  * e.g: GPX standard and http://www.gpsvisualizer.com/draw/
  * parsing: https://github.com/tkrajina/gpxpy
  * see related project: https://github.com/yoinkbird/chirp/tree/master/Messenger/test/gps
* UI: user draws route by hand on a map, manually exports the list of GPS coordinates, then feeds them to tool.  
* backend:
  * fuzzy-match conversion of GPS coordinate input to "model-data" (csv or dataframe)  
  * fit model on "model-data"
* implements: model fitting on arbitrary GPS coordinate data, vs selection of pre-defined GPS data
  * interface to third-party route-mapper
  * fuzzy-match GPS coordinates
* interfaces: external route-mapper, csv-file, python model

| inputs | -> [model] -> | output |
| ------ | ----- | ------ | 
| features: user provided | combine with GPS-coords |  |
| route: manually generated arbitrary GPS coords | fuzzy-match against existing data | score |

WP: [GPS-automatic-generic]  
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
TBD - TODO: combine with WP descriptions

### Architecture
@STUB: describe the 'model.py' in its current implementation  
quicknote: use python data mining libraries to generate the model,  

<!-- github doesn't know about 'pre' tags, I guess. either do '< -' or '&lt;-' -->
#### Legend:

<pre>
[ end-point ]
{ data transfer }

</pre>

#### User-client Interaction: 

<pre>
[user] ----{Map-UI: route start,end}--&gt; [ client ]  
[    ] &lt;---{Map-UI: route + scores }--- [        ]  
</pre>

End-user uses client application as a conventional routing tool.  
Client application displays available routes and their score.  


#### Client-Model Interaction:

<pre>
[ client ] ----{Map-API: start,end         }--&gt; [External Routing Service]
[        ] &lt;---{Map-API: multi route coords}--- [External Routing Service]
    ^
    |
{rest-json: vvv route vvv |rest-json: ^^^scores^^^}
    |
    V
[ server ] ----{json-file: route coords}--&gt; [ Modeling Application ]
[        ] &lt;---{json-file: route scores}--- [                      ]
</pre>

Client Application sends route start, end to external routing service.  
External routing service returns geo-json containing routing information, including GPS coordinates representing route.  

Client submits geo-json to server  
Server relays geo-json to modelling application, which itself is not a server  

Modeling Application processes relevant information from geo-json to score route, stores in geo-json

Modeling Application sends route-score geo-json to server
Server relays route-score geo-json to client

Client displays original route information plus route-scoring information to client


#### Modeling Application:

<pre>
Data Preparation and Feature Implementation:  
/data sources/ ---&gt; [Preprocessor per source, feature] ---&gt; [df: dataset | df: feature definitions ]

Model Creation:   
[dataset,featdef] ---{query: desired features}---&gt;---{slice: dataset}---&gt;[model]

</pre>

Build models using feature definitions and optimise using python ML libraries

Modules:

| Filename | Purpose |
|---|---|
| model.py        | Model Build, Optimise, Predict Route-Score |
| txdot_parse.py  | Prepare data as outlined under Data Preparation  |
| feature_defs.py | Track features and their purpose  |
| mapgen.py       | Generate maps for static heatmap visualisation  |
| helpers.py      | Useful functions |


**Data Parser**  
Convert input data format to pandas dataframe, handle quality issues and feature generation.  
Updates feature definitions.  

**Feature Definitions**  
aka featdef - pandas dataframe to track features and their attributes, meant to be queried when creating a model.  
The attributes defined in the feature df can be queried using pandas syntax, 
thus making it trivial to maintain multiple models and their attributes.

A few examples:  
Building a model requires a set of descriptive features and target features.  
Typically, this is done using individual arrays of feature names, which are then used to query the pandas dataframe containing all features.  
With featdef, the features can be queried dynamically as 'target' or 'non-target' instead of maintaining individual lists.  
One implication is that as the project evolves, any new features are automatically picked up by the models instead of the maintainer having to update each model's individual list of descriptive and target features.  

featdef is used to track the origin of newly implemented features, so if a model needs to exclude them it can easily do so.

featdef tracks the type of feature as well to identify which features are meant to be used in the model and which aren't, such as the case-id.   


# Discussion / Conclusion
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [meta](#meta) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [discussion--conclusion](#discussion--conclusion) | [future-work](#future-work) | [acknowledgements](#acknowledgements) | [reference--literature-bibliography](#reference--literature-bibliography) | [appendix](#appendix) | 
<!--</@breadcrumb>-->
<!--
* Explain what is striking/noteworthy about the results.
* Summarize the state of knowledge and understanding after the completion of your work.
* Discuss the results and interpretation in light of the validity and accuracy of the data, methods and theories as well as any connections to other people’s work.
* Explain where your research methodology could fail and what a negative result implies for your research question.
-->

<hr />

@STUB: update from conclusion in "abstract" 

**PROPOSAL DRAFT**  

read as "what is this going to change?"

this work will improve understanding of what leads to avoidable crashes, which will enable cyclists to plan better routes and municipal traffic departments to address problem areas

the main limitation will be the unavailability of complete cyclist numbers, e.g. it could be possible that all recorded crashes are outliers and most cyclists ride safely

methodology could fail if:

significant crash data is missing, i.e. crashes which go unreported

models are incorrect

**/PROPOSAL DRAFT**  

<hr />

# Future Work
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [meta](#meta) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [discussion--conclusion](#discussion--conclusion) | [future-work](#future-work) | [acknowledgements](#acknowledgements) | [reference--literature-bibliography](#reference--literature-bibliography) | [appendix](#appendix) | 
<!--</@breadcrumb>-->
<!-- very flexible, most of plan will be here -->
## Crash Data

Add data on bike-lane presence

---

Examine before/after lane reduction: car+car |vv|^^| => |cv|<>|vc| bike+car '|cv|' + turn lane '|<>|'
support with studies


---

Interpret How Cyclists Can Ride Defensively
additional requirement: data source update, new model, but reuse application layer, re-analysis of data (i.e. start a new CRISP-DM lifecycle)
concept:
focus on "personal" data features, e.g. wearing helmet, avoiding busy roads
classification into "avoidable" and "avoidable" crashes
e.g. left-turn crash seen as "avoidable" because cyclist can look for vehicles, but crash from rear seen as "unavoidable"  because cyclist has no visibility of vehicles

---

interpret limited data more creatively  
e.g. analyse frequency of crashes to determine which locations tend to have more reported crashes.  
This could be loosely correlated with the probability of a crash, although it could also just mean that certain locations tend to be over-reported vs others. However, since there's not much data, this is also not a bad idea in a pragmatic sense. Can't avoid an unknown, but can avoid a known - work with the data which is available.

---

traffic: use general traffic data (instead of cyclist data) and find 'reported crashes'/'street segment traffic'  
src: traffic count : https://data.austintexas.gov/Transportation-and-Mobility/Traffic-Count-Study-Area/cqdh-farx

---

## Data Sources

Use data from strava,mapmyride,etc to find the most common routes (among the users of these apps) and correlate with crash data

# Acknowledgements
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [meta](#meta) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [discussion--conclusion](#discussion--conclusion) | [future-work](#future-work) | [acknowledgements](#acknowledgements) | [reference--literature-bibliography](#reference--literature-bibliography) | [appendix](#appendix) | 
<!--</@breadcrumb>-->
<!-- Thank the people who have helped to successfully complete your project, like project partners, tutors, etc. -->
Source For Outline: 
https://www.ethz.ch/content/dam/ethz/special-interest/erdw/department/dokumente/studium/master/msc-project-proposal-guidelines.pdf 

Source For Abstract:  
https://www.honors.umass.edu/abstract-guidelines  
http://www.sfedit.net/abstract.pdf  

# Reference & Literature (Bibliography)
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [meta](#meta) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [discussion--conclusion](#discussion--conclusion) | [future-work](#future-work) | [acknowledgements](#acknowledgements) | [reference--literature-bibliography](#reference--literature-bibliography) | [appendix](#appendix) | 
<!--</@breadcrumb>-->
<!-- List papers and publication you have already cited in your proposal or which you have collected for further reading. The style of each reference follows that of international scientific journals. -->

# Appendix
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [meta](#meta) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [discussion--conclusion](#discussion--conclusion) | [future-work](#future-work) | [acknowledgements](#acknowledgements) | [reference--literature-bibliography](#reference--literature-bibliography) | [appendix](#appendix) | 
<!--</@breadcrumb>-->
<!-- Add pictures, tables or other elements which are relevant, but that might distract from the main flow of the proposal. -->

**CRISP-DM**  

DRAFT - expanded description of CRISP-DM process

1. Business Understanding  
This stage is the project-planning portion of the project and as such combines both waterfall and agile methods.
The project scope is defined, project feasibility is determined, and data-mining goals are set.  
In contrast to waterfall, this stage is more flexible as it can be revised as-needed based on the subsequent "data understanding" phase.
In essence, this stage lays the groundwork for a successful project.  

First, determine the desired outputs of the project.  
Set the objectives - determine the primary goal and which goals are secondary.  
Create a project plan for how to set up the data mining goals, and how these results will be implemented to achieve the business goals.  
Then define measurable criteria to determine success.  
This will set expectations for everyone involved while also setting clear completion criteria.  

Second, assess the feasibility of the project.  
This involves determining which resources are required and which still need to be obtained, such as data sources or domain experts.  
The full project requirements are determined in order to obtain the best possible a priori understanding of project risks, constraints, and results.  
At this point, terminology relevant to the project should be compiled into a glossary. Keep in mind that this can be updated from the next stage as needed.  
The feasibility asessment concludes with a cost-benefit analysis of project cost vs benefit to the business.

Third, determine the data mining goals.  
This step maps the business goals to the data mining goals, ensuring that there is not a mismatch in objectives.  
This is done by defining the success criteria for the business and data mining objectives.  
I.e. first define project success in terms of the business, then define project success in technical terms.
This helps set clear goals, while also ensuring that the technical goals are aligned with the business objectives.  

Fourth, and finally, produce the project plan.  
At this point, the goals of the project is clear and the project can be planned out.  
The project plan lists a timeline of project stages and the resources they require.  
At the end of each phase this plan is then updated and adjusted as needed.  
This strategy is most similar to agile project planning.

In summary, the "business understanding" phase is a combination of waterfall and agile project planning.  
The project is planned out in terms of phases and resources as it would be with waterfall, but is meant to be updated under certain conditions as with agile project planning.  
This is particularly well adapted for data mining, as the goal must be well understood in advance, but the model has to be able to change based on its inputs.  


**Machine Learning Notes:**  

interpretable models
https://www.google.com/search?q=machine+learning+understandability+of+model&oq=machine+learning+understandability+of+model&aqs=chrome..69i57.4599j0j1&client=ubuntu&sourceid=chrome&ie=UTF-8

**Misc**  
@FUTUREWORK: change parameters such as 'lighting condition' according to local time, i.e. if route time estimate starts during daylight, ends in twilight, change the values according to the time estimates per segment.  

<!--
Table Generation from Bullets
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
