<!--
# Meta
Formatting HALP:  
https://help.github.com/articles/basic-writing-and-formatting-syntax/  
https://daringfireball.net/projects/markdown/syntax  

## DRAFT REVISIONS
Document major changes to roadmap or implementation since last draft  
This is for transparency, so customer is aware of conceptual changes,  
and for docmentation, the iterative nature of the project lifecycle is clear.  
-->
  
<!-- look for the REV<n> keyword throughout the doc -->
<!--
REV2 [20171017] roadmap: remove GPS fuzzy match from critPath
REV1 [2017xxxx] roadmap: first pass  
-->

<!-- Define a short, significant title which reflects clearly the contents of your report. The title page follows the guidelines of scientific proposals at Department of Earth Sciences (see http://www.erdw.ethz.ch/documents/index). -->

<!-- old title: Data Driven Approach towards Improving Road Safety for Cyclists -->

# Probabilistic Routing-Based Injury Avoidance Navigation Framework for Pedalcyclists


<!--
Probabilistic  : machine-learning model
Routing-Based  : input is route data
Bicycle-Injury : prediction is for crash severity given a crash
Avoidance      : multiple scored route choices 
Navigation     : user-selected routes
Framework      : the tech stack implementation (not just one app) 
Pedalcyclists  : official term for cyclist
-->


<!--
## Time Plan for Master’s Project Proposal and Master’s Thesis
@STUB: track the completed work-packages from Roadmap
-->
<!-- Give a detailed time plan. Show what work needs to be done and when it will be completed. Include other responsibilities or obligations. -->

# Abstract
<!-- Succinct abstract of less than one page. -->
<!-- details for abstract courtesy of https://www.honors.umass.edu/abstract-guidelines , review of abstract courtesy of http://www.sfedit.net/abstract.pdf -->

<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists](#probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [roadway-safety-analysis](#roadway-safety-analysis) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [technical-implementation](#technical-implementation) | [conclusion](#conclusion) | [acknowledgements](#acknowledgements) | [references](#references) | [appendix](#appendix) | 
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

<!-- introduction of problem -->
<!--
allowing pedalcyclists to choose routes based on the results of the traffic studies 

reactively and therefore only in areas for which the 

conventional safety improvements are reactive, can take a while, difficult to measure [@safetyImpactBikelanePolitiFact] 

municipalities freely make data available, invite community to help solve problems @citationNeeded:openAustin

citizen contribution to problem affecting everyone

problem affecting everyone: traffic congestion, can be alleviated with more alternative transportation, cycling one of them
increase cycling by increasing safety of cycling
increased cycling as a factor in increasing safety, i.e. small start can accumulate (tell cyclists where to ride safely, motorvehicles see more cyclists -> more aware -> drive more carefully, overall safety now improved) 

how to make this change?
-->
<!-- introduction of paper and goals -->
<!-- injury severity used as metric of route safety, only metric which uses measurable data -->
<!-- develop risk model using crash data to estimate severity of injury given a crash -->
<!-- routing data gathered and collated with environmental data to describe various routes -->
<!-- framework uses model to assign score to each possible route -->
<!-- goal is to provide end-user with both recommendation and information, can see routes and choose best one  -->
<!-- GO: -->
Easing traffic congestion in urban areas is a continuous effort of optimising many factors. 
Encouraging commuters to use alternative transportation such as bicycles is a simple way to reduce traffic congestion by reducing the number of motor vehicles in traffic. 
However, commuters involved in bicycle crashes sustain much more severe injuries than commuters involved in similar motor vehicle crashes. 
This risk of injury presents a hurdle towards convincing commuters to use bicycles and can be reduced by improving roadway safety. 
Conventional roadway safety improvements are made at a municipality level based on recommendations from studies analysing crash and traffic data. 
While changes made using this approach can increase safety in the long term, the cost-benefit analysis of implementing these changes leads to these improvements being made in limited areas for maximum impact. 
This reactive approach has several drawbacks as it only improves certain roadways, takes a long time to implement, and doesn't focus on safety benefits for bicycle commuters. 
As an alternative, the concerns with the conventional reactive approach can be addressed by adapting it for proactive use. 
The traffic and crash data used to create the studies is freely available to the public in many municipalities, and advancements in data mining have greatly reduced the effort required to analyse data for a traffic study. 
In this paper, a navigation framework is developed which analyses crash data to recommend safe routes for bicycle commuters. 
The injury severity present in the crash data is used as as the metric for route safety. 
Machine learning techniques are applied to the crash data to develop a risk model for estimating route safety. 
A framework is created to obtain multiple routes and generate their safety scores by applying the risk model. 
An end-user navigation application was implemented to present multiple routes along with their safety scores. 
This proactive approach enables commuters to switch to bicycling
by providing personalised safety recommendations 
without the limitations of the conventional approach. 

<!--
other approaches: 
this addresses flaws with other approaches ...
pedalcyclist traffic volume data too sparse to make meaningful prediction 
-->

<!-- OLD: -->
<!-- TODO - state potential compromises in abstract -->
<!-- TODO - state hypothesis in first sentence - in this case, the goal? -->
<!--
PURPOSE  
This project aims to help bicycle commuters improve their safety in mixed traffic. 
The resulting increase in safety while cycling will hopefully convince more commuters to use a bicycle. 
If successful, this will reduce the number of severe injuries while also easing traffic congestion. 
@moreCyclingMoreSafetyNacto - article: increased cycling increases safety  
-->

<!--
This project introduces a web-based map interface to help cyclists plan safer routes. 
and an analysis of the data used to plan the route will help cyclists ride more defensively.  
-->
<!--
This project will help cyclists plan safer routes through a web-based map interface which uses multiple approaches to analyse route safety.  
  The first approach seeks to analyse the potential for severe injury along a generic route. 
A machine learning model will determine the probability of injury for a given route based on cumulative data from all known crashes. 
This approach relies entirely on available data and therefore is a closed-form, or complete, solution.  
  The second approach uses historical crash data to recommend safer routes. 
Routes with fewer dangerous intersections are considered safer, where the danger of an intersection is approximated from crash location data. 
This is an incomplete solution as there is insufficient data to determine precisely what leads to a crash, and therefore only an approximation of a closed form solution. 
-->
<!-- Assign a score to intersections based on crash data (frequency of crashes in intersection, injury severity, etc).  -->
<!--
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
The created application will help cyclists ride more defensively, but more effort needs to be made on a municipal level.  
Cycling safely will continue to be inconvenient as long as safe routes are a big detour.  
Recommendation is to increase data collection in areas with most crashes to better analyse the factors leading into it.  
-->


# Table of content
<!-- The table of content lists all chapters (headings/subheadings) including page number. -->
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists](#probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [roadway-safety-analysis](#roadway-safety-analysis) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [technical-implementation](#technical-implementation) | [conclusion](#conclusion) | [acknowledgements](#acknowledgements) | [references](#references) | [appendix](#appendix) | 
<!--</@breadcrumb>-->

<!--!toc_mini-->
<!--<toc_mini>-->
* Introduction
  * Structure of this Paper
* Background and Results to Date
  * Previous Work
  * Related Work
* Goals
* Roadway Safety Analysis
* Methodology
  * Introduction to Data Mining
  * Navigation Framwork Overview
  * CRISP-DM
* CRISP-DM Report
  * Business Understanding
    * Problem Statement
    * Feasibility Assessment
    * Project Plan
  * Data Understanding - Analysis
    * Data Collection Report
    * Data Description Report
    * Data Exploration Report
    * Data Quality Report
    * Analytics Base Table
  * Data Preparation
    * Feature Implementation
  * Modeling
  * Evaluation
  * Deployment
* Technical Implementation
  * Framework for CRISP-DM
  * Predicting using Route Data
    * Mapping Routing Features to Crash Report Features
    * Predicting using Mapped Routing Features
    * Impact on the Route Score
  * User Application
    * Architecture
* Conclusion
* Acknowledgements
* References
* Appendix
  * Appendix: Data Description Report
    * TXDOT data
  * Appendix: Featdef Values
  * Appendix: Modeling - Feature Selection for Feature Reduction Model
  * Appendix: Application Implementation Gantt Chart
    * Work-Packages:
    * Roadmap
<!--</toc_mini>-->


# Introduction
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists](#probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [roadway-safety-analysis](#roadway-safety-analysis) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [technical-implementation](#technical-implementation) | [conclusion](#conclusion) | [acknowledgements](#acknowledgements) | [references](#references) | [appendix](#appendix) | 
<!--</@breadcrumb>-->
<!-- Explain why this work is important giving a general introduction to the subject, list the basic knowledge needed and outline the purpose of the report. -->

<!--!@section_sentence-->
The goal of this project is to create a continuously updated traffic study using data passively gathered from existing sources and to present the results in a readily interpretable manner in order to improve personal safety for cyclists.  

<!-- general introduction to the subject -->
#### Traffic Studies - Active vs Passive
Current traffic studies typically involve observing specific sections of road for a fixed time period.  
This provides a lot of data which can be used to improve traffic safety, but is limited by the frequency and scale at which such studies can be done.  
Staffing is required to run these studies, therefore there are practical constraints on how often and how widespread they can be run.  
The result is very accurate data for a few representative locations, but the data represents a static section in time and is expensive to update.

Machine learning and automated monitoring can help reduce the cost of active studies, but there is still cost involved in obtaining widespread coverage using automated equipment (e.g. installing cameras at every intersection).

A passive traffic study using data from indirect sources and can run continuously while monitoring as many locations as the data can provide.  
The drawback is that indirect data sources may not provide as much information as actively collected traffic data.  
This alternative application of machine learning has the advantage that it is scalable and requires fewer additional resources.  
In many cities, crash reports are one of the readily available indirect sources. On the one hand, they are continuously updated and widespread - a police report is filed any time police are called to the scene of an accident anywhere in the region. On the other hand, police reports for crashes are meant for establishing facts related to a crash, and as such don't capture all of the data required by an active traffic study. Additionally, not every crash leads to a police report, so these types of report are also limited by statistical significance.  
However, these reports are advantageous due to their frequency and geographic distribution; they can reflect both time-based trends and localised trends where active traffic studies can only capture a snapshot.

#### Interpretability in Statistics and Machine Learning
Interpretability describes how readily understandable a process is. For studies based entirely on statistics, it describes how easily the results can be interpreted to correlate cause and effect. This concept extends to machine learning, and describes how easily the inputs to a model can be traced to the model's results.  
Machine learning is heavily based on traditional statistic methods, but is nonetheless often perceived as a black box due to its ability to produce results from large amounts of data which could not be handled by traditional techniques.  
It is important to be able to understand the output of a machine learning model for several reasons. The model exists as part of a larger framework to solve a particular problem, and as such requires supervision to ensure that the results are relevant. In this context, interpretability is important for adjusting the inputs, or even the problem statement, to achieve better results.  
Interpretability is also important for the end-usage of the model; the model is specific to one particular goal, whereas applications of the model may have various goals and as such need to understand how the model fits in with the overarching application.  
For example, if a model indicates that a certain trait is involved in a certain outcome, it is important to be able to understand the actual significance of this trait.  Another example could be a model which finds that two traits lead to a certain outcome; this result requires being able to understand to what extent these traits interact. 

#### Public Safety and Traffic Study Interpretability 
To maximise public safety, it is important for traffic studies to be interpretable.  
This report defines interpretability of a traffic study in two ways: The traditional method of explaining which inputs lead to which outcome, and a more usable interpretation of how to best use the results of the study to improve traffic safety.  
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
  * determine dangerous routes - presently partially possible on TXDOT website, although not easiest to do. shows overlay of crashes.
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

This paper is organised as follows: One section presents the probability-based adaptation of existing roadway safety analysis. The next section introduces the industry standard CRISP-DM process for creating a data analytics solution, followed by a section on using this process to create a machine learning model to analyse traffic data. The subsequent section describes the implementation of a user-interface to the created analytics solution. 

<!--
This paper largely follows a traditional structure [@citationNeeded], but wraps the data-mining-specific process 'CRISP-DM'.  
There is some overlap between the two formats, but each have a different focus.  

The following descriptions explain the purpose of sections which seemingly overlap.

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
-->


# Background and Results to Date
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists](#probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [roadway-safety-analysis](#roadway-safety-analysis) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [technical-implementation](#technical-implementation) | [conclusion](#conclusion) | [acknowledgements](#acknowledgements) | [references](#references) | [appendix](#appendix) | 
<!--</@breadcrumb>-->
<!-- List relevant work by others, or preliminary results you have achieved with a detailed and accurate explanation and interpretation. Include relevant photographs, figures or tables to illustrate the text.  This section should frame the research questions that your subsequent research will address. -->

<!-- TODO: brief overview of this section. -->
While improving traffic safety is not a new topic, addressing it with machine learning techniques soley based on an existing dataset seems to be a novel approach. 
Conventional applications of using machine learning to improve traffic safety tend to involve traffic studies or long running trials of novel traffic-controls to determine safety factors. 
However, this method tends to require a higher budget and active involvement by an organisation with an interest in traffic safety. 
The advantage of using machine learning techniques on existing data is to provide a low-budget analysis of traffic safety factors with a quick turnaround time, independent of existing policies. 

## Previous Work

<!-- abstract: previous work -->
This project is a successor of a previous research project to identify the factors associated with severe injury for crashes involving cyclists [@originalProject]. 
This research project produced two machine learning models, a complex model using a boosted trees classifier and a simplified model using a decision tree classifier. 
The complex model was optimised for identifying the most important features leading to severe injury. 
The simplified model reduced the features identified by the complex model into a simplified decision tree to conceptualise safety factors in a human readable format. 
However, neither of these models can be used for safety-based navigation. 
The complex model relies on features only available in the crash dataset, and the simplified decision tree would have to be manually interpretted for each section of each possible route. 

<!--
The second model was generated for interpretability, which is considered a fundamental issue with applied machine learning: as the model gets more precise, it gets more complex and therefore difficult to understand or apply to real-world problems. 
The risk factors identified by the complex model could be used when analysing the safety of an intersection. 
However, the underlying decision tree makes too many decisions for a human to interpret on their own, and therefore would require an extra layer to be human-readable. 
The visualisation generated by the simple model could be used by an individual cyclist trying to determine the effect of environmental factors on their route. 
However, this simplified model is only an approximation of the factors leading to severe injury, and trades accuracy for usability. 
Essentially, it rephrases the problem as "what are the safety factors, given the constraint of a limited number of possible choices". 
This trade-off was considered as acceptable, as manual interpretation of complex models is likely to result in error anyway. 
In summary, when creating an interpretable model, the error introduced by simplifying the model needs to be balanced with the error a human would make while reading a complex model. 
Furthermore, a complex model is inconvenient to read and would likely not find much usage.  
-->
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
<!--
## Results to Date  
@STUB  
For example, time-of-day was identified as a strong factor, so a visualisation was created to display where crashes happened during different time intervals [@originalProject].  
This visualisation could then allow cyclists to consider the time of day when planning their route.  
However, this is a raw visualisation as it does no further data processing to interpret the data.  
One major issue is that it displays all crashes from 2010-2017, and thereby hides any potential trends.  
For example, this visualisation does not portray whether the safety of an intersection changed over time.  
-->

<!-- segue into current project -->
<!--
**Impact on Current Project**  
The previous project layed a good foundation, and in doing so opened up many possibilities for new projects.  

This project continues the work of the previous project and will combine the accurate but non-interpretable model with the inaccurate but interpretable model into one tool which will both be accurate and interpretable.
-->

## Related Work

Injury-avoidance based navigation using a predictive model primarily relates to research on safety-based navigation and event prediction models. 

**Safety-Based Navigation**:
Navigation based on avoiding unsafe areas is an active research topic. 
Work has been done on avoiding high-crime areas for pedestrian navigation [@relatedWorkSafeNavUrbanEnv], 
in which routes are calculated in order to minimise exposure to high crime areas. 
This project estimates foot-traffic volume from the locations of reported crimes in order to normalise the predictions. 
This approach would not work for this injury-avoidance navigation project as it would mean assuming a correlation between reported crashes and cyclist traffic volume. 
This correlation cannot be proven, especially since crash data only records crashes which lead to injury and therefore is not a heterogeneous description of all crashes. 

**Event Prediction Models**: 
Using data to predict events has many applications, most relevant to this paper is research into using crime and crash data for predictions. 
In relation to this paper, work has been done on predicting crashes by analysing various data sources [@relatedWorkPredCrashUrbanData]. 
This project combined data for motor vehicle crashes, weather, demographics, and road networks to create a model predicting the probility of a crash for a given road section. 
However, this project does not provide a navigation interface and makes several assumptions which would not work for injury avoidance navigation. 
In particular, this project seeks to predict occurence of a crash and uses negative sampling to generate data to simulate crashes which did not occur. 
This is because crash data is only recorded for a crash and does not measure avoided crashes. 
The resulting model is heavily dependent on the quality of the negative sampling algorithm. 
The injury-avoidance navigation avoids this issue by predicting the injury severity given a crash, which relies entirely on existing data. 

Research has also been done on using crime data to predict future crimes [@relatedWorkCriminalityPred]. 
This project analysed reported crimes to predict which types of areas may have an increase in criminality. 
The approach uses spatial information to identify urban areas with a high propensity for crime. 
This approach would not work for injury-avoidance navigation, as each road section has very different properties which lead to an entirely different injury severity. 



# Goals
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists](#probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [roadway-safety-analysis](#roadway-safety-analysis) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [technical-implementation](#technical-implementation) | [conclusion](#conclusion) | [acknowledgements](#acknowledgements) | [references](#references) | [appendix](#appendix) | 
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

Create navigation application for scoring routes in real time. 
This will provide a familiar interface to allow end-users to make use of safety scores while also taking advantage of the highly optimised services offered by third-party route planners.  
This is developed as an interface to the route scoring model which allows the end-user to conveniently factor in safety scores while planning their route.  
Develop framework for managing crash data. 
This includes facilitating the use of data from different sources and the management of features throughout the various stages of the modelling process.  
Produce machine learning model. 
Follow the CRISP-DM process to produce a thorough machine learning product developed according to widespread industry standards. 
This will be implemented as a machine learning model which finds a relationship between environmental factors and safety of a route.  

<!--
(May want to redefine according to less severe injuries, e.g. "safe" as "able to ride away")  
-->

<!--
Todo: terminology: use the phrase "severity", or something, in lieu of "safety". Keeps the mission clear, and is accurate without interruption. Later, introduce the safety score.  

[@terminology]: relatively-safe - placeholder term. see relative-safety  
[@terminology]: relative-safety - placeholder term. Assuming a cyclist is involved in a crash, how likely are they to be severely inured. ML: This is the target feature. @TODO: use featdef to list features and definitions  
-->

<!-- tags: studies
Todo: should research whether accident severity correlates with responsibility. If studies show that light injury is usually cyclists fault, would be able to draw a line and cite the study.  
-->

# Roadway Safety Analysis
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists](#probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [roadway-safety-analysis](#roadway-safety-analysis) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [technical-implementation](#technical-implementation) | [conclusion](#conclusion) | [acknowledgements](#acknowledgements) | [references](#references) | [appendix](#appendix) | 
<!--</@breadcrumb>-->

This paper introduces a novel method for analysing a route and assigning a safety score. 
State of the art descriptive analytics are adapted for use with predictive analysis. 
This section introduces and analyses the commonly used crash rate formulas, 
introduces the challenges faced with applying these formulas predictively, 
and concludes with a description of the predictive crash severity formula used for this project. 

**Road Safety Analysis based on Traffic Data**  
The Federal Highway Administration (FHWA) preforms a descriptive analysis of existing data for a given road section to measure the crash rate for that portion of road [@fhwa3DataAnalysis]. 
This crash rate is used to determine which roadway sections need to be improved, and this analysis is performed before and after the improvement to measure the effectiveness.  
This analysis is not performed for segments without crash data.  
The crash rate is calculated for one specific portion of a roadway, which can be either a road segment or an intersection. 
Two formulas are used when calculating this rate, one for road segments and one for intersections.  
<!-- !@TERMINOLOGY -->
Terminology:  
An intersection is defined as the confluence of two roads at grade, i.e. at the same level such that the roads physically interact. 
A road segment, or segment, is defined as a continuous stretch of road between intersections. 
This paper will use the term "road section" or "roadway section" as a general descriptor of either a segment or an intersection.
Traffic count, traffic volume, or traffic flow each refer to the recorded number of vehicles entering a given road section. 
In the City of Austin, where this study is performed, the traffic count reflects a 24h period and is collected on an "as needed basis" [@coagovTrafficCount].

The crash rate generally reflects the number of crashes per vehicles entering the intersection, and is expressed differently for intersections and road segments. 
For intersections the crash rate unit is crashes per 1 million entering vehicles (MEV), where MEV expresses the number of vehicles entering the intersection. 
For segments the crash rate unit is crashes per 100 million-vehicle-miles of travel (VMT), where VMT expresses the number of vehicles travelling per mile of road. 


Each formula uses the following variables [@fhwa3DataAnalysisCrashRate]:  
<!-- formula variables -->
R = Crash rate for the road section, i.e. intersection or segment  
C = Total number of crashes in the study period.  
N = Number of years of data.  
V = Number of vehicles per day  
L = Length of the roadway segment in miles [only for road segment crash rate].  

The crash frequency is defined as the traffic count normalised by the traffic volume data, then normalised by a section-specific constant:  
<!--
$$
\\ \text{crash frequency} = \frac{  C * \text{<section-constant>} }{ 365 \cdot N \cdot V }
$$  
-->

![formulaCrashFrequency]

[formulaCrashFrequency]: res/formulae/formulaCrashFrequency.png 

The intersection crash rate scales the frequency by a constant of 10^6 [@fhwa3DataAnalysisCrashRateIntersection] : 
<!--
rate = 1 * E^6 * "Recorded Crashes" / ( 365 * "years of data" * "daily traffic volume" )
-->
<!--
$$
\\ \text{intersection crash rate} = 1 \cdot 10^6 \times \frac{  C }{ 365 \cdot N \cdot V } \ MEV
$$
-->

![formulaIntersectionCrashRate]

[formulaIntersectionCrashRate]: res/formulae/formulaIntersectionCrashRate.png 

i.e.:
<!--
$$
\\ rate = \frac{ 1 \cdot 10^6 \cdot "C: Crashes" }{ 365 \frac da \cdot "N: Years Of Data" a \cdot "V: Traffic Volume" \frac 1d }
$$
-->

![formulaIntersectionCrashRateVerbose]

[formulaIntersectionCrashRateVerbose]: res/formulae/formulaIntersectionCrashRateVerbose.png 

The road segment crash rate scales the frequency by a constant of 100 * 10^6 and divides the result by the segment length [@fhwa3DataAnalysisCrashRateSegment]: 
<!--
rate = 100 * E^6 * "Recorded Crashes" / ( 365 * "years of data" * "daily traffic volume" * "length of segment" )
-->
<!--
$$
\\ \text{road segment crash rate} = 100 \cdot 10^6 \times \frac{  C }{ 365 \cdot N \cdot V } \times \frac 1L  \ VMT
$$
-->

![formulaSegmentCrashRate]

[formulaSegmentCrashRate]: res/formulae/formulaSegmentCrashRate.png 

<!-- with units:
$$
\\ rate = 100 \cdot \mathrm{E}\,6 \times \frac{ C }{ 365 \cdot N \cdot V} \frac{vehicles}{\frac {d \cdot a}{a \cdot d}} \times \frac 1L \frac {1}{\text{mile}} \ VMT
$$
-->
i.e.:
<!--
$$
\\ rate = \frac{ 100 \cdot 10^6 \cdot "C: Crashes" }{ 365 \frac da \cdot "N: Years Of Data" a \cdot "V: Traffic Volume" \frac 1d \cdot "L: Segment Length"  }
$$
-->

![formulaSegmentCrashRateVerbose]

[formulaSegmentCrashRateVerbose]: res/formulae/formulaSegmentCrashRateVerbose.png 


<!-- explanation -->
<!-- normalise by days: crashes / 365 1/d * N 1/a -->
The formulas initially normalise the crash data to match the timescale of the traffic volume data. As the traffic volume is measured daily, the resulting ratio expresses crashes per day. 
Typically, these evaluations only consider yearly crash data, and this normalisation can be adjusted for other timescales. 
<!-- normalise with volume,length C/V and C/V*L-->
The crashes per day are then normalised against the daily traffic volume to obtain the relative frequency of crashes per traffic volume; note that this normalisation removes the timescale from the crash rate.  
The formula for the road segment rate further normalises the crash-count by dividing it with the segment length, resulting in the relative frequency of crashes per traffic volume and unit of measurement, e.g. miles or kilometers. 
<!-- upscale -->
The resulting crash frequency as per-volume or per-volume-distance is then scaled up to millions of vehicles using a section-dependent constant.
This results in a crash frequency expressed in per millions of vehicles travelling on a given road section. 
For segments, the resulting crash rate is expressed as crashes per 100 million vehicle miles of travel.
For intersections, the resulting crash rate is expressed as crashes per 1 million vehicles entering the intersection. 

<!--
**Properties of Segment Crash Rate**
The segment crash rate is correlated with the length of a given road segment; a longer segment results in a lower ratio, a shorter segment results in a higher ratio. 
In this sense, the crash-severity risk can be correlated with the segment length in the same way to normalise the crash-severity risk with the segment length. 
In essence, since the VMT ratio is correlated with the segment length, it would also make sense to correlate the crash-severity risk with the segment length in a similar fashion. 
This would preserve the property of normalising the score against the length of the segment, which avoids misrepresenting crash-severity risk. 
For example, two road segments of different lengths may have the same number of recorded crashes, which would lead to a higher ratio for the shorter segment and a lower ratio for the longer segment. 
This reflects that the shorter segment has more crashes per unit-distance. 
In the same manner, two road segments of different lengths may have the same crash-severity risk based on the model's predictions. 
However, the model does not use segment distance in its calculation as it is based on point measurements. 
As a result, the segment length is not reflected in the calculated crash-severity risk as it would be in the crash rate. 
Since the segment crash rate is calculated as the normalised crash-ratio divided by the segment length, and the crash-severity risk was previously defined as the replacement for the normalised crash-ratio, the segment crash-severity risk can be calculated as the crash-severity risk divided by the segment length. 
-->

**Relationship between Intersection Crash Rate and Segment Crash Rate**
When examining the crash rate formulas, it becomes apparent that each of them express the crash frequency distributed over a given distance: 
the segment crash rate is calculated for the length of the road segment, 
and the intersection crash rate is calculated for one point, i.e. the intersection. 

The formula for segment crash rate is proportional to the intersection crash rate divided by the segment length. 
Conversely, the formula for the intersection crash rate is proportional to the segment crash rate if the segment length is set to a fixed value of '1'. 
This implies that there could be a linear relationship between the segment rate VMT calculation and the intersection rate MEV calculation. 
This relationship is explored in the following formulas: 

<!--
$$
\\ \text{Intersection Rate} = 1 \cdot 10^6 \times \frac{ C }{ 365 \cdot N \cdot V }
$$
-->

![formulaRelInterCrashIntersectionCR]  

[formulaRelInterCrashIntersectionCR]: res/formulae/formulaRelInterCrashIntersectionCR.png 
<!--
$$
\\ \text{Segment Rate} = 100 \cdot 10^6 \times \frac{ C }{ 365 \cdot N \cdot V }  \times \frac 1L  
$$
-->

![formulaRelInterCrashSegmentCR]  

[formulaRelInterCrashSegmentCR]: res/formulae/formulaRelInterCrashSegmentCR.png 
<!--
$$
\\ \text{let L = 1: } 
$$
-->

![formulaRelInterCrashL1]  

[formulaRelInterCrashL1]: res/formulae/formulaRelInterCrashL1.png 
<!--
$$
\\ \text{Segment Rate(L=1)} = 100 \cdot 10^6 \times \frac{ C }{ 365 \cdot N \cdot V } \times \frac 11
$$
-->

![formulaRelInterCrashSegmentCRL1Expand]  

[formulaRelInterCrashSegmentCRL1Expand]: res/formulae/formulaRelInterCrashSegmentCRL1Expand.png 
<!--
$$
\\ \text{Segment Rate(L=1)} = 100 \cdot 10^6 \times \frac{ C }{ 365 \cdot N \cdot V }
$$
-->

![formulaRelInterCrashSegmentCRL1Reduce]  

[formulaRelInterCrashSegmentCRL1Reduce]: res/formulae/formulaRelInterCrashSegmentCRL1Reduce.png 
<!--
$$
\\ \text{Segment Rate(L=1)} == 100 \cdot Intersection Rate
$$
-->

![formulaRelInterCrashSegmentCRL1equInterCr]  

[formulaRelInterCrashSegmentCRL1equInterCr]: res/formulae/formulaRelInterCrashSegmentCRL1equInterCr.png 

This demonstrates that the intersection crash rate formula can be expressed using the segment rate formula for a distance of 1 unit and a fixed constant of 1/100 . 

<!--
$$
\\ \text{Intersection Rate} = \frac{1}{100} \times \text{Segment Rate(L = 1) } = \frac{1}{100} \times \frac{ C }{ 365 \cdot N \cdot V }
$$
-->

![formulaRelInterCrashInterCRrelSegCR]  

[formulaRelInterCrashInterCRrelSegCR]: res/formulae/formulaRelInterCrashInterCRrelSegCR.png 
This demonstrates that the intersection crash rate formula can be expressed in terms of the segment crash rate. 

This supports the initial claim that these formulae correlate risk with the crash frequency and length of the road section. 

**Predictive Road Safety Analysis**  
The formulas currently used to evaluate road segment safety rely on the presence of data for the specific segment being analysed. 
As a result, any crashes which are not reported will not be factored in the analysis. 
For example, a segment which leads to minor collisions which require no police involvement would not be fixed if no report is filed. 
The NHTSA does have some techniques for predictive modelling [@fhwaHSIPproblemIdentificaton] but these involve complex formulas which in turn rely on a plethora of measurements. 

These formulas are meant for posterior analysis in order to determine which roadways to improve. 
This is a good approach for continuously improving safety, but requires crashes to happen before action can be taken. 
This is a particularly problematic approach for cyclists as they tend to sustain more severe injury than they would in a motor-vehicle collistions. 

This problem could be addressed by applying a predictive approach towards road segment safety analysis. 
The existing formulas and body of knowledge based on crash rates can be adapted for crash probability with a few changes.  
The formulas need to predict future crashes instead of analysing existing crashes, 
and the crash rate formula needs to be altered to use predicted crashes instead of measured crashes. 

The concept of crash prediction involves identifying trends in the road segments for reported crashes and using these trends to analyse a road segment to make an educated guess about the number of crashes it should have. 
This estimated number of crashes for a segment can then be used in the existing formulas to calculate that segment's safety score. 
In essence, this changes the formula from measuring the number of reported crashes normalised with traffic volume to 
measuring the number of predicted crashes normalised with traffic volume. 

This approach is still limited by the availability of traffic volume data for a given road segment. 
However, as it is a predictive model, its accuracy can be improved by ensuring that traffic volume data is continuously updated. 
The exact reasons for re-measuring traffic volume on a segment can vary, but may be combination of age of the data, predicted crashes for the segment, and actually measured crashes for the segment. 
This maintenance is to be expected, as predictive models are meant to be used with the descriptive techniques to ensure maximum coverage for the best overall solution. 


**Predictive Road Safety Analysis using Crash Severity**  
The predictive approach relies on traffic volume for a given segment, and is therefore limited by age and availability of the data. Older data will not reflect current trends, and segments with unavailable data cannot be evaluated.  

Traffic flow cannot be measured for every segment using current techniques. 
The methods for measuring traffic flow are only applied on demand [@coagovTrafficCount] and are therefore not available for every road segment. 
Furthermore, the methods for measuring traffic flow do not measure cyclists as they cannot distinguish between types of vehicle [@trafficCountBicycleVsCar]. 

One solution could be to create a separate predictive model for traffic volume, which would represent an entirely different problem than predicting crashes. 
Traffic volume itself depends on several factors not measured in crash data, for example, connectivity of the road segment (thoroughfare, neighbourhood road, dead end), lane width, population density along the segment, and several other factors. 
While there is some research into predicting traffic volume [@trafficForecastingCrowdFlows], this falls beyond the scope of this project.  

Therefore, in order to create a safety measurement applicable to any generic road section, a different measurement must be used which does not rely on the availability of traffic flow data. 
Crash severity measures the severity of injury sustained in a crash, and is a self-contained measurement in that it is measured per individual reported crash instead of per total reported crashes for a segment. 
This imposes the limitation that it can not be used to avoid a crash, but instead to avoid severe injury given a crash. 
Therefore, any prediction based on crash severity will be constrained by the assumption that a crash occurred. 
This removes the reliance on unavailable traffic volume data, but imposes the bias that any prediction assumes that a crash has already occurred. 

The formula introduced for assessing crash-risk for road sections cannot be adapted for crash-severity. 
The crash-risk measures the frequency of crashes (measured or predicted) per section of road, whereas crash-severity measures the impact of a single crash. It is therefore not a measure of the frequency of an event, and cannot be measured as a simple ratio. 
The crash-severity risk will be taken directly from the confidence of the predictive model.  
This preserves the probability property of risk as a number between 0 and 1.

The relationship between crash-frequency and crash severity:

<!--
$$
\\ \text{Crash Freqency} 
\mapsto
\\ \text{Crash-Severity Risk}
$$
-->

![formulaMapFreqToSeverText]  

[formulaMapFreqToSeverText]: res/formulae/formulaMapFreqToSeverText.png 
<!--
$$
\\ \frac{ C }{ 365 \cdot N \cdot V}
\mapsto
\\ \text{Predicted Crash Severity}
$$
-->

![formulaMapFreqToSeverEquation]  

[formulaMapFreqToSeverEquation]: res/formulae/formulaMapFreqToSeverEquation.png 

**Mapping Crash Rate to Predictive Crash Severity**  
The relative safety of a road section can be calculated using the crash-severity risk by applying the principles underlying the crash rate formulas. 
The crash rate formulas correlate crash frequency with road section length, which can be mapped to crash-severity by correlating crash-severity risk with road section length. 
The relative safety of a road section can be calculated by applying the mapping between crash frequency and crash-severity risk to the descriptive crash rate formulas. 
The descriptive formulas scale the crash frequency by a constant based on the section type, then divide that number by the road section length, which is '1' for intersections. 
The crash-severity risk is independent of traffic volume and therefore does not need to implement the constant used to scale the crash frequency to a certain number of vehicles. 

Intersection Crash-Severity Risk:  

<!--
$$
\\ \text{Intersection Crash Rate} 
\mapsto
\text{Intersection Crash-Severity} 
$$
-->

![formulaMapCRtoCSevInterText]  

[formulaMapCRtoCSevInterText]: res/formulae/formulaMapCRtoCSevInterText.png 
<!--
$$
1 \cdot 10^6 \times \frac{ C }{ 365 \cdot N \cdot V}
\mapsto
\\ \text{Predicted Crash Severity}
$$
-->

![formulaMapCRtoCSevInterEquation]  

[formulaMapCRtoCSevInterEquation]: res/formulae/formulaMapCRtoCSevInterEquation.png 

Segment Crash-Severity Risk:  

<!--
$$
\\ \text{Segment Crash Rate} 
\mapsto
\text{Segment Crash-Severity} 
$$
-->

![formulaMapCRtoCSevSegText]  

[formulaMapCRtoCSevSegText]: res/formulae/formulaMapCRtoCSevSegText.png 
<!--
$$
\\ 100 \cdot 10^6 \times \frac{ C }{ 365 \cdot N \cdot V} \times \frac 1L
\mapsto
\\ \text{Predicted Crash Severity} \times \frac 1L
$$
-->

![formulaMapCRtoCSevSegEquation]  

[formulaMapCRtoCSevSegEquation]: res/formulae/formulaMapCRtoCSevSegEquation.png 


**Applying Safety Score to a Route**  
The descriptive crash rate formulas are used to compare road sections and determine which ones need to be improved. While this ratio can be used predictively, [@fhwa3DataAnalysisPotentialCrashes], it is mainly used for road section improvement and not applied to multiple consecutive sections as it would need to be for analysing an entire route.  

<!-- just use the product of segments, product of intersections.
Each of these scores is the cumulative sum of their respective section types for the route. 
-->

<!--
The previously derived risk-based scores can be adapted for risk-aware route planning.  
-->

This is addressed by adapting the previously derived risk-based section scores for risk-aware route planning.
Route planning generates paths consisting of sections, therefore the risk score for each section type can be combined to a total risk score for each path which then allows them to be scored.  
The crash-severity risk score for the entire route comprises two numbers; the route intersections score and the route segments score. 
This distinction between the section types preserves the distinction made in the official FHWA crash rate formulas for intersection and segment. 

Measuring the risk of a route is calculated as:

<!--
<pre>
intersections risk(Route Intersections) = 1 - PRODUCT[intersection element Route] ( 1 - risk(intersection) )
segments  risk(Route Segments) = 1 - PRODUCT[segment element Route] ( 1 - risk(segment) )
</pre>
-->

<!-- "in set" : \n : src https://proofwiki.org/wiki/Symbols:Set_Operations_and_Relations#Is_an_Element_Of -->
<!-- 
intersection \ risk = 1 \ - \prod_{intersection \ \in \ route}  (1 - risk(intersection) )
segment \ risk = 1 \ - \prod_{segment \ \in \ route}  (1 - risk(segment) )
-->

![formulaRouteRiskScoreInter]

[formulaRouteRiskScoreInter]: res/formulae/formulaRouteRiskScoreInter.png 

![formulaRouteRiskScoreSeg]

[formulaRouteRiskScoreSeg]: res/formulae/formulaRouteRiskScoreSeg.png 

**Route Planning**  
Route planning will be done by third party route planning services for several reasons.  
Third party services already provide several alternative routes, which eliminates the need to implement routing algorithms.  
Modern route planning tools consider several factors when optimising routes, whereas this project only aims to optimise one factor. By using an external routing service, the risk optimisation can be performed on routes pre-optimised for other factors, thereby providing a more optimal service.  

<!--
@TODO: use this when discussing the data limitations @WRONGPLACE
* no problem if not enough data to distinguish segments, simply compare two routes and choose safest => no problem - the "segment score" encapsulates this concept. "risk score" calculated based on whatever data the model consumes, serves as abstraction for "route score" . if "segment score" gets more accurate, "route score" automatically reflects this.  
-->

# Methodology
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists](#probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [roadway-safety-analysis](#roadway-safety-analysis) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [technical-implementation](#technical-implementation) | [conclusion](#conclusion) | [acknowledgements](#acknowledgements) | [references](#references) | [appendix](#appendix) | 
<!--</@breadcrumb>-->

This section introduces the data mining process, followed by an overview of the framework implementation, and concludes with the CRISP-DM process used to create the predictive models. 

<!-- Explain the methods and techniques which will be used for your project depending on the subject:
field work, laboratory work, modeling technique, interdisciplinary collaboration, data type, data acquisition, infrastructure, software, etc. -->

## Introduction to Data Mining
This section provides an overview of the supervised machine learning process. 
A model tries to find a trend in data in order to make accurate predictions on future unknown data. 
This is done by analysing existing an dataset consisting of inputs with known outputs in order to find a correlation. 
Training refers to the process of providing the existing dataset inputs and outputs to the model. 
Predicting refers to the process of providing new inputs to the model for it to estimate the outputs. 
The data inputs are often referred to as features or predictors, the output is referred to as the "target". 
Dimensionality refers to the number of available data points, and refers either to for a specific feature or the entire dataset.
The python sklearn models refer to 'training' as 'fit' and prediction as "predict". 

Improving model accuracy is done using various techniques. 
Train-Test Split converts a training dataset with known outputs into two separate datasets, where "Train" is the one used for training and "Test" the one used for predicting. 
This simulates the usage of the model on unknown data, with the difference that the correct outcomes are known for the test dataset. 
The model is trained using the training data and then predicts using the test data. 
The predicted outputs for the test-data are then compared to their actual outputs to measure the model accuracy. 
Cross Validation refers to the process of comparing predictions against known outcomes. 
Cross Validation is often run multiple times while adjusting model parameters in order to find the maximum score against the test dataset. 

There are several algorithms for measuring model accuracy and depend on the type of model and data. 
The Cross Validation Score refers to the model accuracy determined from the test-dataset prediction. 

ROC-AUC (Receiver Operating Characteristic) 
is often used to measure model performance for a categorical dataset. 
It measures the true-positive and against the false-positive rate of the predictions, thereby forming a curve. 
AUC (Receiver Operating Characteristic Area Under Curve) 
measures the area under the curve formed by ROC and used as as the ROC score. 
The goal is to maximise the area, with an AUC of 1 meaning a perfect model. 

As a side note, the receiver operating characteristic (ROC) owes its name to its invention for use by radio receiver operators during World War 2. 
The metric was originally used to evaluate the prediction accuracy of radar systems when detecting aircraft and later found use in machine learning. [@wikiROChistory]

**Feature Selection and Elimination**  
Feature selection generally refers to the process of choosing which features to include in a model.  
By default, a model can be trained on all features and left to decide which features are most important.  
However, there are several reasons that this approach may not be appropriate for finding the optimal solution.  
Some features can be strong but induce a lot of variance in the model results, or some features are omitted based on domain expertise.  

Feature elimination is used to improve model accuracy scores in case certain features induce too much variability.  
In particular, it can be used to increase the prediction reliability for a decision tree classifier by removing features which cause high variance in model scores.  
Decision Trees work by recursively partitioning the dataset between features based on how well they split the dataset.  
Due to this criteria, when determining each partition there can be several eligible features.  
Therefore, many implementations choose this feature at random with the expectation that the overall best features will be chosen by creating several decision trees together in what is known as a Random Forest.  

Some features are ill-suited for a decision tree model because their data leads to several splits without increasing the prediction score, thus "hiding" more useful features.  

**Recursive Feature Elimination**  
Recursive feature elimination (RFE) works by creating models with a decreasing number of features until a desired number of features is reached [@sklearn_feat_sel_rfe] .  
This technique is used to reduce the dimensionality of a feature set to a fixed magnitude.  

**Recursive Feature Elimination with Cross Validation**   
Recursive Feature Elimination with Cross Validation (RFECV) runs RFE in a cross-validation loop in order to find the optimal number of features to remove.   
This technique is used to reduce the dimensionality of a feature set without an a priori constraint on the magnitude.  
The cross-validation loop correlates an accuracy score with the number of features, which is used to determine which features can be removed with minimal impact to model accuracy.  

However, this is not an automatic process as judgement is required to evaluate the resulting RFECV scores.  
For example, simply using the max score to determine the optimal number of features could ignore similar scores with more features,
or choose an outlier score while ignoring a more stable local maximum of scores.  
For this reason, judgement needs to be exercised when deciding which of the optimal feature numbers to choose.  
Sometimes the variability between scores is too high to make a meaningful decision, at which point the initial feature list needs to be re-evaluated or other techniques may need to be used.  


**Low-Data Feature Selection**  
Some features may be strong predictors for a decision tree but not have enough data-points to make a reliable model.  
In many datasets there can be features with unknown values which cannot be easily imputed.  
In such cases, a trade-off must be made between the number of features and the amount of available data.  
This trade-off can be evaluated by iteratively removing features with a low number of data-points in combination with RFECV.  
The results of this iterative process can be compared to determine which low-data features to remove.  


## Navigation Framwork Overview

The safety navigation application consists of two components, the injury severity prediction model and the safety navigation framework which uses the model.  
The core mission of this project is to improve safety for cyclists, which means that a balance must be made between accurate predictions and the ability to use them for safe navigation. 
This balance is best exemplified by two scenarios: one in which the model is accurate but without a navigation application, and one in which the navigation application is easy to use but the model is inaccurate. 
The first scenario is exemplified as a model which can accurately predict the safety of a route, but requires the end-user to manually create their own route from GPS coordinates. 
The second scenario is exemplified by existing navigation tools, which are easy to use but do not include safety predictions. 

In scenario one, the high-accuracy model can predict the danger of a given route, 
but would require the end-user to have a detailed understanding of data mining, 
and how to convert their route such that the model can use it for making a prediction. 
In the second scenario, the navigation application makes it easy for the end-user to plan their route, 
but an inaccurate model will mislead the end-user about the actual safety of the route.  

Both scenarios are unfavourable, but the nature of data mining and the process of application design informs which scenario to address first. 
Data mining is an open-ended problem as the model requires constant improvement to remain accurate as new data is made available. 
On the other hand, designing an application to consume the model is a finite problem as the interface between application and model is static. 
Focussing on the interface first also accomplishes the goal of making the model easier to use, as it provides an interface for the end-user. 

In summary, while the accuracy of the prediction model can improve over time, the application has no impact on the accuracy of the model. 
Therefore, the primary focus of this project is on creating a UI for interacting with the output from the model. 
As the model itself is expected to change over time, its optimisation is a secondary focus.  

This approach correlates with the industry standard CRISP-DM process, which foresees the need for ongoing maintenance of the model. 

## CRISP-DM

The cyclical product development resulting from the trade-off between performance and usability fits perfectly within the CRISP-DM structure. 
CRISP-DM (CRoss Industry Standard Process for Data Mining) [@wikiCrispDM] is a cyclical framework designed to structure a data mining project from first conception to its deployment. 

This section provides an abstract overview of CRISP-DM followed by its implementation for this project. 
At certain points, comparisons to other common project management frameworks are made. 

The CRISP-DM framework consists of 6 stages which each provide input into the next stage and allow for stages to be revised at certain points in the process.
This is similar to the traditional waterfall framework, with the major exception that it is designed to flexibly accommodate new requirements as new information becomes available.  

![imageCrispDm]

[imageCrispDm]: https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/CRISP-DM_Process_Diagram.png/220px-CRISP-DM_Process_Diagram.png

**Overview of the CRISP-DM Stages**

<!--
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment
-->

1\. **Business Understanding**: 
This stage is the project-planning portion of the project and as such combines both waterfall and agile planning strategies.  
The project is planned out in terms of phases and resources as it would be with waterfall, but is meant to be updated under certain conditions as with agile project planning.  
This is particularly well adapted for data mining, as the goal must be well understood in advance, but the model has to be able to change based on its inputs.  
In essence, this stage lays the groundwork for a successful project.  

<!--
Overview:  
* determine the desired outputs of the project.  
* Feasibility Assessment - assess the feasibility of the project.  
* determine the data mining goals.  
* create project plan.  
-->

2\. **Data Understanding**: 
This stage is for acquiring and exploring data in order to evaluate the quality and usefulness of the data sources provided in stage 1. 
Data reports generated by this stage are referred to throughout the remaining stages. 

<!--
Overview:  
* Data Collection Report
* Data Description Report
* Data Exploration Report
* Data Quality Report
* ABT - Analytics Base Table
-->

3\. **Data Preparation**: 
This stage is for processing data to select features for use by the modeling stage. 
Features relevant to the problem being solved are selected from the data and any data quality issues are fixed. 
New features can be constructed, for example existing features can be combined into a new feature to satisfy a domain-specific requirement. 
Missing data can also by constructed by imputing missing values where applicable. 
Finally, data from various sources is combined into one resource for consumption by the model. 

<!--
Overview:  
* Select
* Clean
* Construct required data (derived features, imputed values, etc)
* Integrate (merge/collate together different sources, aggregate multiple records into one)
-->

4\. **Modeling**: 
This stage is for creating a model which satisfies the data mining objectives outlined in the business understanding stage. 
The technique for creating the model is defined and the train-test-evaluation plan is created. 
The scores resulting from the test-train-evaluation plan are documented for each model, and this stage is revised as needed. 

<!--
Overview:  
* choose technique, list assumptions
* create test train-test-eval plan
* create model (set parameters, build model, report about process involved)
* assess model - interpret according to domain knowledge and test/train. revise parameters and start over as needed.
-->

5\. **Evaluation**: 
This stage is for determining which models achieved the business success criteriea.
The process developed through the previous stages is documented and analysed. 
Finally, the decision is made to procede to deployment, to revisit and improve a previous stages results, or to terminate the project due to inadequate results. 

<!--
Overview:  
Note: not same as error evaluation  
* evaluate results
  * assess results in terms of project/business success criteria
  * list approved models, i.e. which models to be used for the project
* review process: summarise process until this point, determine what needs to be repeated or still be done
* determine next steps: decide whether to proceed to next stage or loop back to a previous stage based on the current results.
  * list out possible actions, reasons, pro/con ; then describe decision
-->

6\. **Deployment**: 
The final stage is for defining how the model will be used in order to achieve the business objectives defined in the business understanding stage. 
The deployment plan outlines how to integrate the model within the business processes. 
The monitoring and maintenance plan documents the process for monitoring model performance over time and updating it with new data. 

<!--
Overview:  
* deployment plan - how will this be used for the business?
* monitoring + maintenance plan - how will model be updated and supervised?
* final report
* review project - "experience documentation" to summarise the knowledge gained during this project
-->

# CRISP-DM Report
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists](#probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [roadway-safety-analysis](#roadway-safety-analysis) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [technical-implementation](#technical-implementation) | [conclusion](#conclusion) | [acknowledgements](#acknowledgements) | [references](#references) | [appendix](#appendix) | 
<!--</@breadcrumb>-->

This section will present the results of each CRISP-DM stage, and will outline each revision where applicable. 

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
| [table-of-content](#table-of-content) | [probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists](#probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [roadway-safety-analysis](#roadway-safety-analysis) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [technical-implementation](#technical-implementation) | [conclusion](#conclusion) | [acknowledgements](#acknowledgements) | [references](#references) | [appendix](#appendix) | 
<!--</@breadcrumb>-->
| CRISP-DM: |                                                      <!-- "tag anchors" for navigation in text-editor -->
[business-understanding](#business-understanding) |                <!--business_understanding-->
[data-understanding---analysis](#data-understanding---analysis) |  <!--data_understanding-->
[data-preparation](#data-preparation) |                            <!--data_preparation-->
[modeling](#modeling) |                                            <!--modeling-->
[evaluation](#evaluation) |                                        <!--evaluation-->
[deployment](#deployment) |                                        <!--deployment-->

<!-- probably not needed now
The "business" can be understood as any entity concerned with public transportation safety. The "business needs" for this project are oriented around increasing traffic safety for cyclists.  
-->

### Problem Statement  
<!--
* determine the desired outputs of the project.  
-->

The objective of this project is to help cyclists avoid severe injury when involved in a crash. 
Project Plan: This will be achieved by creating a data mining model to analyse crash data, and then integrating the model into a route-planning tool. 
Success Criteria: The project will be considered a success if the resulting product can be used by cyclists to evaluate any arbitrary route for the possibility of severe injury given a crash. 
A successful project will provide a product which cyclists use to make informed decisions about which routes to choose.  
Constraints on the Objective: 
The objective of this project is not to help cyclist avoid crashes altogether, as the main data source is post-crash data. 

<!--
Original Objective: The objective was originally to help cyclists avoid crashes in general. However, during the feasibility assessment supporting data was found to be insufficient. 
In particular, there is no data on the number of cyclists for a given road segment, and the crash data lists only reported crashes instead of a cross-sample of all crashes. 
-->


### Feasibility Assessment
<!--feasibility_assessment-->
<!--
* assess the feasibility of the project.  
-->

Inventory of resources:  
Data on Crashes can be obtained from Texas Department of Transportation (TXDOT), National Highway Traffic Safety Administration (NHTSA), and the City of Austin Police Department (APD).  
Data on cyclist ridership is very sparse in comparison to data on crashes, which prevents significant correlation and therefore will not be considered.  
Data on traffic-counts is available for certain road segments and is the total count for a 24 hour period.  
The software necessary for data processing and modelling is available as free python libraries (pandas, sklearn, scikit learn, other ML libraries as needed).  
The scale of this project is appropriate for any modern hardware as it does not require intense computing resources.  

Requirements, assumptions, and constraints:  
Requirements: all data and tools are free for use  
Assumptions: The crash data is assumed to be accurately reported as it is sourced directly from law enforcement officers [@txdot_crash_report_source].  
Constraints: The crash data only represents reported crashes, which are biased towards crashes involving injury. 
Therefore, the model will be biased towards predicting a higher injury severity. 

Feasibility Conclusion:  
The available crash data on the TXDOT website contains information about incidents involving bicyclists and contains enough data to create a predictive model.  

Risks and Contingencies:  
The TXDOT crash dataset is based on reported crashes, of which only crashes which lead to injury are required to be reported. 
As a result, the dataset does not encapsulate unreported crashes, and may exclude reported crashes which did not lead to injury.  
Therefore, the resulting model will be biased towards crashes which lead to injury while being unaware of less severe crashes. 
This bias is mitigated by a few factors. The end-usage of the model is for comparing different routes and as such the relative accuracy between routes is most important. Since the prediction bias results from the entire dataset, each route's prediction will be equally biased. The predictions will be relatively accurate instead of absolutely accurate. 
For the end goal of reducing severe injury, the tendency towards over-predicting injury is more desirable than under-predicting injury. The predictions are meant for use by individual cyclists, who may want to exercise as much caution as possible and therefore would benefit from an overly cautious recommendation. The risk with this assumption is that overly cautious safety recommendations could also lead to commuters perceiving cycling as unsafe and favouring other modes of transportation.  
These risks are acceptable within the constraints of this project, as they will not prevent the project from being completed and can be addressed in future iterations using additional data sources. 

<!-- TODO: 1. auto-create glossary 2. grep for terminology tags, make sure explained before used. could potentially examine "git log -p" in reverse to find terminology introductions, otherwise this requires user to be self-aware and add the when they use the term. 
Terminology:  


[@terminology]: segment - a section of a road  

[@terminology]: segment data - crash-data entry for a segment. can be anywhere on a road, including at an intersection  

[@terminology]: crash report
-->

<!-- N/A, but could be "project will benefit society by reducing cycling crashes and improving the transit situation overall at a relatively low cost of maintaining computing resources and expertise to update the resulting tool.
Ideally would base benefit on some numbers on cost of cycling crashes or potential reduction in traffic if cycling increased; base cost in having an agency maintain the tool and having experts keep model updated
Costs and Benefits:  
-->


#### Data Mining Goals
The project success criteria are defined as a tool which can display any route and its associated risk of severe injury in a crash.   
The data mining success criteria are defined as model which can create injury severity predictions using environmental and navigation data. 
The model accuracy is of secondary importance, as the primary requirement is to correlate navigation and environmental data with a model created from crash data. 

<!--
### Project Preparation Gantt Chart
project plan:

| status | | | | | | | | | |
|-|-|-|-|-|-|-|-|-|-|
| **prep** | clean project | | | | | | | | |
| **prep** | | stub_model |
| **analysis** | | | interpretable_model |
| **analysis** | | | interpretable_model2 |

Names:
clean_project -> Migrate Previous Project
stub_model -> enablement_model
interpretable_model -> Feature Reduction Model
interpretable_model2 -> Routing Model

-->
<!-- | **modeling** | | | | optimised_model1 | -->

### Project Plan
The plan for creating the final route scoring model involves several intermediate stages in order to simultaneously enable the model deployment. 

**Migrate Previous Project**: 
The codebase inherited from the previous project focused on model optimisation and feature selection, and therefore needs to be rewritten as a generic model generator for this project.   

<!-- WP: stub_model -->
**Create Enablement Model**: 
Create a simple model with basic optimisation and a limited feature set.   
Purpose: Gathering requirements and enabling all dependent work-packages  
Creating an intermediate simple model allows for the external interfaces to be defined and enables the rest of the technology stack. 
The enablement model allows the deployment stage to start in parallel with model optimisation. 
This approach works if the business requirements include deploying the project. 
Otherwise, this strategy bypasses the evaluation stage, as deployment is contingent upon meeting the business and data-mining success criteria.  
Dependency: Data Preparation

<!-- interpretable_model  -->
**Create Feature Reduction Model**: 
Create a simple model with advanced optimisation and as many features as possible to predict the crash severity.  
Purpose: Data analysis, implementation and definition of interface between application and model
Data Analysis: This model helps with processing the dataset through a process called feature selection, which can also help increase the amount of usable data.  
Implementation and Definition of interface between application and model: The process of creating a machine learning model is separate from the process for using the model with new data, as is needed for the final application.  This model helps explore and define a process for allowing the final application and the model to send data back and forth.  Once this process is defined, the model also allows the application to work with the model's predictions and discover any potential flaws in the communication process. Until then, the application would be using static predictions.  
Dependency: Data Preparation , Routing Application


**Tools and Techniques**:
For this project, the tools and techniques are created from readily available python machine learning libraries and will be described in further sections.  No third-party services or machine learning software packages will be used to process the data, as may be expected in other contexts.  Note that using python machine learning libraries is not considered as using a machine learning software package, the usage of which requires no software programming.  
<!--
@TODO: create overview of terminology for model creation; may have overlap with the work-packages for deployment
-->

<!--
#### WP: optimised_model1  
Model using type of algorithm most suited to the data  
Purpose: increase prediction accuracy  
see 'modeling' for specifics
keep in sync with 'modeling':
Likely RandomForest (bagged) or XGB (boosted), depends on data available
Dimension <4k, so boosted tree likely best solution (src: [@caruana_et_al_2008]) 

-->


## Data Understanding - Analysis
<!--data_understanding-->
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists](#probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [roadway-safety-analysis](#roadway-safety-analysis) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [technical-implementation](#technical-implementation) | [conclusion](#conclusion) | [acknowledgements](#acknowledgements) | [references](#references) | [appendix](#appendix) | 
<!--</@breadcrumb>-->
| CRISP-DM: |                                                      <!-- "tag anchors" for navigation in text-editor -->
[business-understanding](#business-understanding) |                <!--business_understanding-->
[data-understanding---analysis](#data-understanding---analysis) |  <!--data_understanding-->
[data-preparation](#data-preparation) |                            <!--data_preparation-->
[modeling](#modeling) |                                            <!--modeling-->
[evaluation](#evaluation) |                                        <!--evaluation-->
[deployment](#deployment) |                                        <!--deployment-->

<!--
**Overview**  
* Data Collection Report
* Data Description Report
* Data Exploration Report
* Data Quality Report  
* ABT
-->

<!-- ### Data Acquisition  -->
<!-- list data sources,locations,methods, problems encountered, solutions -->
**Data Collection Report**:  
**Crash Data**: 
TXDOT crash data will be used in csv format obtained via an interactive web-based query. 
This requires a one-time effort to manually formulate the query each time the dataset is to be updated.    
Several other sources were considered as well. 
NHTSA and TXDOT both offer a complete dataset which is not in a simple csv format and requires significant processing in order to be queried.  
APD (Austin Police Department) offers a limited dataset for certain years, which excludes it from consideration as a source of data.  
For the purpose of this project, it is more efficient to focus the available resources on the easily obtained yet complete dataset offered by the interactive web-query. 
However, for ongoing maintenance, the full-featured dataset should be used as it can be obtained automatically.  

**Routing Data**: The deployed model will be predicting on routing data obtained from a third party.  
Google Maps Routing API will be used for this as it is well supported and free of charge. 
OpenStreetMaps is another widely used mapping API, but requires a separate service to be used for routing. This additional complexity makes OpenStretMaps a non-optimal choice for this project.  

**Data Description Report**:  
**Crash Data**: The TxDOT crash data is in CSV format with a header describing the query parameters used to obtain the data  
The data contains 2233 records with 25 fields, some of which contain mostly null values. 
The features corresponding to these fields will be processed by the models using recursive feature elimination. 
The meaning of each feature is described in the TxDOT Highway Safety Improvement Manual [@txdotHSIManualMannerCollision] . 
The "crash severity" correlates best with "injury severity", and takes on a range of 5 values indicating the severity of injury sustained in a crash. 
This range includes "Not Injured", "Possible Injury", "Non Incapacitating Injury", "Incapacitating Injury", and "Killed". 
"Incapacitating" is understood to mean that medical treatment was required. 
This feature should be explored for use as the injury severity target predictor. 

<!-- too much info, would have to look this up for each feature!
The manner of collision can take on 46 values [@txdotHSIManualMannerCollision], but the crash data for pedalcyclists is limited to 10 distinct values:  
  'one motor vehicle going straight',  
  'angle both going straight',  
  'one motor vehicle other',  
  'same direction one straight one stopped'  
  'same direction both going straight rear end',  
  'one motor vehicle backing',  
  'opposite direction both going straight',  
  'opposite direction one straight one left turn', # just "one" straight, not sure if motorist or cyclist  
  'one motor vehicle turning left',  
  'one motor vehicle turning right',  
-->

<!--
@TODO: fill in from featdef.py in [data description report appendix](#appendix-data-description-report), add excerpt with relevant data here  
-->

**Routing Data**: The Google Maps routing data is in json format and contains several fields meant for consumption by the Google Maps display API. A subset of the routing data corresponds to a subset of the crash data.


<!--
3.5 Advanced Data Exploration
3.5.1 Visualizing Relationships Between Features
3.5.2 Measuring Covariance and Correlation
-->
<!-- not enough time.
**Data Exploration Report**:
Most of the features in the dataset are categorical, and most of the continuous features have discrete values and can be converted to categorical features.  

* ABT  
* Data Quality Report
  * mph incomplete, encoded either as -1 or 0
  * "average daily traffic amount" and "average daily traffic year" only present for major roads   
-->

<!-- 3 Data Exploration -->
<!-- 3.1 The Data Quality Report -->
**Data Quality Report**: See Appendix for the TxDOT Data Quality Report and TxDOT Data Undefined Values Report
<!-- 3.2 Getting to Know the Data -->
<!-- 3.2.1 The Normal Distribution -->


<!--
2.3 Analytics Base Table
This work sits primarily in the Data Understanding phase
2.4 Designing and Implementing Features
2.4.5 Implementing Features
-->
**Analytics Base Table**: 
The analytics base table is implemented as a pandas dataframe and meant to be modified as needed by each model. 
The final ABT comprises 61 features and 2232 entries, and as such will not be reprinted. 

<!--
Choose prediction subject, one-row-per-subject  
determine domain concepts for features  
first findings, hypothesis  
-->

## Data Preparation
<!--data_preparation-->
| CRISP-DM: |                                                      <!-- "tag anchors" for navigation in text-editor -->
[business-understanding](#business-understanding) |                <!--business_understanding-->
[data-understanding---analysis](#data-understanding---analysis) |  <!--data_understanding-->
[data-preparation](#data-preparation) |                            <!--data_preparation-->
[modeling](#modeling) |                                            <!--modeling-->
[evaluation](#evaluation) |                                        <!--evaluation-->
[deployment](#deployment) |                                        <!--deployment-->

<!--
**Overview**  
* Select
* Clean
* Construct required data (derived features, imputed values, etc)
* Integrate (merge/collate together different sources, aggregate multiple records into one)  


**Select**  
-->

<!--
@TODO: mention the results of the framework, then mention in one line that a framework was used to achieve these outputs.
E.g. for "data removed" mention which list is used for what, such as: average-traffic-count is not used for models due to low number of samples
-->

**Clean Data**  
<!-- 3.3 Identifying Data Quality Issues -->
<!-- 3.3.1 Missing Values -->
<!-- 3.3.2 Irregular Cardinality -->
<!-- 3.3.3 Outliers -->
<!-- 3.4 Handling Data Quality Issues -->
<!-- 3.4.1 Handling Missing Values -->
<!-- 3.4.2 Handling Outliers -->
<!--
3.6 Data Preparation
3.6.1 Normalization
3.6.2 Binning
3.6.3 Sampling
-->
The data was processed using the following rules:    
Replace punctuation with underscores.  
Lowercase feature names.    
Encode strings containing numbers as a numeric   datatype.  
Encode strings containing dates   as a date-time datatype.  
Encode strings representing missing data as a null datatype.  
Convert human-readable descriptions for missing data to machine-readable datatype. 
This includes converting '0' for "missing" to 'np.nan' to prevent the modelling algorithms from evaluating it as a real value. This also improves runtime performance, as a proper null value is evaluated quicker than an integer representation (numpy knows not to evaluate np.nan, but can't know in advance whether an 'int' is '0' and therefore spends extra time to evaluate it)
    Caveat: This does not apply to data which is actually '0', only to cases when '0' represents a missing value.

Note: Missing values were not removed, as different models and visualisations require different sets of features. 
As such, the missing data is handled at the time of usage.  

Certain features have very few data-points and therefore only used for specific visualisations.  
"average daily traffic amount" and "average daily traffic year" are features to describe the traffic volume on a road and only have data for large roads.  


GPS coordinates will be omitted from the model creation, as geographic location alone cannot be used to predict traffic safety. 
Furthermore, GPS coordinates in the dataset are directly correlated with the crash report and crash id, and therefore would bias the model towards "remembering" locations of crashes. 
This is counter to the goal of determining generic crash-factors which can be used to analyse any road segment.  
If GPS coordinates were used to create a location-aware model, the resulting predictions would be contingent upon number of reported crashes.
This would need to be combined with total ridership data or traffic volume data for a specific road segment. 
This requires additional data and is beyond the scope of this project.  

### Feature Implementation
<!--
#### Construct required data (derived features, imputed values, etc) ([x]@TODO: merge 'feature implementation' further above into this portion
-->
**Construct Required Data**  
**Imputed Values**
impute speed limits - of the set of intersections with multiple entries, for any intersections missing the speed limit, impute speed limit from identical rows. If speed limit changed throughout time, use first available value either from the future speed limit or the past speed limit. As most speed limits were observed to increase over time, this induces a bias towards associating crash severity with higher-than-actual speed limits. This is accepted as the difference in speed limit is typically only 5mph.
Note that this will bias the overall model towards a higher speed limit associated with injury severity. This has a few consequences, but can be considered as a relatively safe assumption as the correlation between crash-severity and speed-of-impact is well understood. Furthermore, the speed limit itself is not the same as the impact speed.  

**Derived Features**
decimal crash time: encoded 24h time to decimal as model does not understand 24h time and processes it as an integer value  
30min-rounded crash time round date-time to within 30minute intervals to clarify visualisations  

Binary categories were created for certain features. 
Some of the data provided was more granular than required, as it encapsulated a degree of information unobtainable in production.
When visualising the data, categories with several values can become difficult to interpret. 
Some values were encoded as categorical data, but some algorithms only operate on numeric data or yes/no outcomes.  

surface condition:  
Factorize 'Wet' 'Dry' to '1' '0'

crash severity:  
convert 'non_incapacitating_injury','possible_injury','not_injured' to 1  
convert 'incapacitating_injury','killed' to 0  
This is in accordance with the crash severity groupring used in TXDOT visualisations [@txdotCrashSeverityGrouping].  

intersection related:  
encoded as 1    :  'intersection_related', 'intersection'  
encoded as 0    :  'non_intersection', 'driveway_access'  
encoded as null :  'not_reported'  

This feature is not intended for use in the safey score model, and is only for models used for visualisation. 
Two assumptions were made for the "intersection related" category: 
Routing data will not distinguish at this level of granularity, especially for driveway_access. 
However, it does provide information as to when to turn.  

light condition:  
encoded as 0 : 'dark_lighted', 'dark_not_lighted', 'dusk', 'dark_unknown_lighting', 'dawn'
encoded as 1 : daylight

This feature is not intended for use in the safey score model, and is only for models used for visualisation. 
Assumption: This is a simple conversion to "good visibility" and "bad visibility".  

<!--
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
-->

<!--
#### Integrate (merge/collate together different sources, aggregate multiple records into one)
none.  
-->

## Modeling
<!--modeling-->
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists](#probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [roadway-safety-analysis](#roadway-safety-analysis) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [technical-implementation](#technical-implementation) | [conclusion](#conclusion) | [acknowledgements](#acknowledgements) | [references](#references) | [appendix](#appendix) | 
<!--</@breadcrumb>-->
| CRISP-DM: |                                                      <!-- "tag anchors" for navigation in text-editor -->
[business-understanding](#business-understanding) |                <!--business_understanding-->
[data-understanding---analysis](#data-understanding---analysis) |  <!--data_understanding-->
[data-preparation](#data-preparation) |                            <!--data_preparation-->
[modeling](#modeling) |                                            <!--modeling-->
[evaluation](#evaluation) |                                        <!--evaluation-->
[deployment](#deployment) |                                        <!--deployment-->

<!-- steps:
* choose technique, list assumptions
* create test train-test-eval plan
* create model (set parameters, build model, report about process involved)
* assess model - interpret according to domain knowledge and test/train. revise parameters and start over as needed.  
-->

The modeling phase was revisited several times as part of the rapid-implementation strategy.  
The first iteration started with a simple placeholder model and the final iteration consisted of an application-specific model suited for real-world implementation.  

As determined during the data understanding phase, the dataset is mostly categorical.  
Therefore, each model is based on a decision tree algorithm, which are well suited for analysing categorical data.  


#### Overview of Modeling Stage:  

The Enablement Model was used for data exploration as well as for developing the framework for modeling and deployment. 
The Feature Reduction Model selects the optimal number of features to maximise the prediction accuracy using the crash information dataset. 
The Routing Model re-implements the Feature Reduction Model without features unavailable during deployment. 

<!--
@FUTUREWORK: Improve precision and accuracy  
@STUB: accuracy not the focus, usability is -and- accuracy doesn't matter as much when comparing -and- available data makes all routes look the same anyway

The created application will help cyclists ride more defensively, but more effort needs to be made on a municipal level.  
-->

<!--
Data Segmentation - improve precision - use models created from subsets of the data according to which features can be expected.  
Improved accuracy - The next step to improve accuracy would be to create a boosted tree model (optimised_model1). Boosted trees are an improved implementation of a decision tree and are well suited for smaller datasets [@caruana_et_al_2008].   
Such a model was created for this project's predecessor, and could be re-implemented for this project without the unavailable features.  

As such, only Routing Model will be evaluated in this section as it includes Enablement Model and Feature Reduction Model.  
The Enablement Model and Feature Reduction Model will be mentioned in context of their role in the overall lifecycle.  
-->


#### Dataset Feature Selection
The processed dataset contains features with a dimensionality (number of available data points) an order of magnitude lower than most of the other features. 
RFECV was used to determine the importance of these features during the model creation process. 
The importance of features with low dimensionality was evaluated by running RFECV before and after temporarily removing the feature from the dataset. Features with low importance were permanently removed. 
Additionally, an average ROC-AUC score was calculated by running k-fold cross-validation on the dataset before and after feature removal. 
This ROC-AUC score was used to measure the impact of feature removal prior to running RFECV. 
If the ROC-AUC score increased, the feature was assumed to cause underfitting. 
If the ROC-AUC score decreased, the feature was assumed to be important for the model. 
None of the ROC-AUC scores were high enough to assume a given feature was causing overfitting. 

<!--
Explanation of Assumptions:  
Location unaware - model does not use GPS coordinates for prediction as it needs to analyse road segment types regardless of location. This was explained during the data preparation phase.  
-->
<!--
Location unaware : @TODO: refactor this sentence, is it a stream-of-thought:  In essence, the model is deliberately location unaware, with the caveat and assumption that end-users can't simply avoid parts of town. If location 'b' has more crashes, and the route is 'A'->'B'->'C', it isn't helpful to tell end-user that they have to avoid 'B' by routing through a potential 'D','E','F'. HOWEVER - it could be useful to inform users of this factor, but would require a different model. For now, the goal is to make safe transit more convenient; it is already known that one can ride on the sidewalk for the whole commute at 15mph to increase relative-safety , so adding in "avoid these entire areas" won't increase the convenience.  
@TODO: find the term for intentionally biasing a model to ignore a feature; it's not the same as avoiding overfitting, but it's in the same conceptual category  
-->

#### Enablement Model  
<!-- stub_model: rename to Enablement Model -->
Purpose: Used only during enablement of navigation framework, not for actual predictions.  
Technique: Decision Tree Classifier  
Feature Selection: This model was run with a minimum set of features in order to optimise its performance as a dummy interface. 
Evaluation: Optimisation and Evaluation was not performed for this stub model  

<!-- compare interpretable_model performance with interpretable_model2 performance -->
#### Feature Reduction Model
<!-- interpretable_model: rename to: Feature Reduction Model  -->
Purpose: Enable application architecture, choose optimal predictors from features in crash dataset  
Technique: Decision Tree Classifier  
Assumptions: Uses feature selection to maximise the dimensionality of usable data, which reduces model accuracy in favour of precision. The resulting model will not be able to use as much data as possible for predicting, but is able to use more data overall to make a better informed prediction.  
Test Design: cross validation using ROC score
Feature Selection:  
The removal of the following features increased the available data and optimised the average ROC-AUC score of the resulting dataset: 
'average_daily_traffic_amount', 'average_daily_traffic_year', 'crash_year' 
.  
The usable data was increased to 1644 usable samples and 47 features with a mean ROC of (AUC= 0.58 +/- 0.05) 
from an initial dataset of 233 usable samples and 50 features with a mean ROC of (AUC= 0.55 +/- 0.07).  
Evaluation: Model evaluation was performed in combination with feature selection. 
The model parameters were not optimised from their defaults as this model is meant to enable the overall application infrastructure. 

Create Model: 
The following features were omitted from the feature reduction model:  
'average_daily_traffic_amount'  
'average_daily_traffic_year'  
'crash_year'  

![imageFeatRedModROCcurve]

[imageFeatRedModROCcurve]: modeling_evaluation_files/qt_img78102309535481860.png

![imageFeatRedModImpFeats]  

[imageFeatRedModImpFeats]: modeling_evaluation_files/qt_img78142583443816452.png

#### Routing Model  
<!-- interpretable_model2: rename to: Routing Model  -->
Technique: Decision Tree Classifier  
Purpose: Predict using data obtained from routing information and environmental data  
Assumptions: Only uses data obtainable from routing information, which results in loss of important features.    
Test Design: cross validation using ROC score
Feature Selection:  
The removed features were chosen based on their unavailability during deployment, and therefore were not chosen based on average ROC-AUC score. 
The usable data was increased to 2213 usable samples and 41 features with a mean ROC of (AUC= 0.63 +/- 0.10)
from the feature-selection dataset of 1655 usable samples and 47 features with a mean roc of (AUC= 0.58 +/- 0.05)  
Evaluation: Model evaluation was performed in combination with feature selection. 
The model parameters were not optimised from their defaults as this model is meant to enable the overall application infrastructure. 

Create Model:
The following features were omitted from the routing model:
'speed_limit'  
'surface_condition'  
'intersection_related: 'driveway_access'  
'road_base_type: concrete, flex_base_granular_', 'stabilized_earth_or_flex_granular_'  
'average_daily_traffic_amount'  
'average_daily_traffic_year'  
'crash_year'  

![imageRouteModelROCcurve]  

[imageRouteModelROCcurve]: modeling_evaluation_files/qt_img78410443374198788.png

![imageRouteModelImpFeats]  

[imageRouteModelImpFeats]: modeling_evaluation_files/qt_img78449476036984836.png


<!-- @FUTUREWORK:  -->
<!--
#### segmentation_models 
Technique: Multiple models operating on Segmented Dataset  
Assumption: #@TODO:notsure# output of different models with more/less features is statistically significant @citationNeeded  
Test Design: xval, roc score, etc; need to test the combined score.  
Build Model: determine avail feats, choose model according to avail feats, resulting in more/less accurate score.  
Caveat: This is not stacking, it is dataset segmentation, i.e. choosing the most appropriate model for the dataset prediction.  

Purpose: real-world sometimes has missing data. naive approach is to create model with "lowest common denominator" of missing data, as in the Feature Reduction Model\*, but the resulting model lacks features unique to each route. Better approach is to create multiple models based on anticipated data availability, i.e. one model based on a "common denominator" dataset, one model based on "common dataset" + "feature set A", one model based on "common dataset" + "feature set B", etc. This allows the most optimal model to be used based on the data available, providing a more accurate score based on increased availability of data. In general, training models on different slices of the dataset is referred to as segmentation. @citationNeeded  
(e.g. lighting condition, weather is always available, but can be assumed to be identical or indistinguishable for any route. E.g. user can only input conditions at start, would require advanced knowledge to know whether lighting conditions will change further along in the route )  
-->

<!-- integrated into each model description
#### Analysis Evaluation

Cross-Validation Strategy: StratifiedKFold, score: roc_auc_score  
Feature-Elimination: RFECV with cvFold(2), scoring =  roc_auc  
-->


## Evaluation
<!--evaluation-->
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists](#probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [roadway-safety-analysis](#roadway-safety-analysis) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [technical-implementation](#technical-implementation) | [conclusion](#conclusion) | [acknowledgements](#acknowledgements) | [references](#references) | [appendix](#appendix) | 
<!--</@breadcrumb>-->
| CRISP-DM: |                                                      <!-- "tag anchors" for navigation in text-editor -->
[business-understanding](#business-understanding) |                <!--business_understanding-->
[data-understanding---analysis](#data-understanding---analysis) |  <!--data_understanding-->
[data-preparation](#data-preparation) |                            <!--data_preparation-->
[modeling](#modeling) |                                            <!--modeling-->
[evaluation](#evaluation) |                                        <!--evaluation-->
[deployment](#deployment) |                                        <!--deployment-->

<!--
* evaluate results
  * assess results in terms of project/business success criteria
  * list approved models, i.e. which models to be used for the project
* review process: summarise process until this point, determine what needs to be repeated or still be done
* determine next steps: decide whether to proceed to next stage or loop back to a previous stage based on the current results.
  * list out possible actions, reasons, pro/con ; then describe decision
-->

<!--
Pending: evaluation comparison between interpretable_model2, interpretable_model, stub_model , using the previous project's XGB model as a benchmark.   This is low-priority as the focus for this project is on interpretability and deployment.  
-->

Enablement Model: 
The enablement model worked as desired for finalising the architecture and enabling the technologies involved.  

Feature Reduction Model: 
The feature reduction model optimised the available data from the crash dataset. 
The model will not be used in deployment, as it relies on features beyond the scope of the navigation framework used for model deployment. 
The poor model accuracy is to be expected from a simple decision tree classifier. 
Model performance could be improved by using a boosted tree model, which would be more appropriate for a dataset of this size [@caruana_et_al_2008].  

<!-- @FUTUREWORK : 
For a future instance of this project, one potential solution could be to build a database of GPS-coordinates and intersections based on the existing crash data. However, this would require having to maintain two separate models, one with the extra information and one without, since not every route will be represented. 
-->
<!-- @FUTUREWORK: segmentation  -->

Routing Model: 
The routing model was created only with the subset of the crash data features obtainable from the navigation framework. 
This meant removing the most important feature "speed limit", which had a relative importance of 27% in the feature reduction model, followed by the second most important feature "intersection related: non-intersection" with only 9% relative importance. 
While the ROC-AUC score did improve, removal of the speed limit means this model will be much less accurate at predicting crash severity than the Feature Reduction Model. 
However, the model does preserve the order of several important features from the Feature Reduction Model. 
"intersection related: non intersection" is now the most important feature at 17%, as would be expected after removing the previously most important feature. 
"intersection related: intersection" is the third most important feature at 12%, and "manner of collision: one motor vehicle turnin right" is 9% important. 
The importance of these particular features is desired, as the navigation framework relies on both "intersection related" and "manner of collision" to map the routing data to the crash data. 
Furthermore, the various features for the categories "intersection related" and "manner of collision" have a wide range of importance within the model, implying that predictions for different values of these features will lead to much different scores. 
The distributed importance of these features within the routing model leads to the expectation that the deployed model will provide meaningful distinction between roadway section types. 
Therefore, this model satisfies the data mining success criteria as it creates injury severity predictions from navigation data, 
and should be able to satisfy the project success criteria as it will differentiate between different types of routes. 

The routing data model will be used to continue on to the deployment stage.  

<!--
* review process: summarise process until this point, determine what needs to be repeated or still be done
At this point in the process, the original dataset has been successfully converted for use with a machine learning model and an initial "enablement model" has been created.  
The most important features have been determined via a combination of recursive feature elimination, cross validation, and manual feature reduction.  
The resulting decision tree model has been optimised using cross validation, thereby demonstrating that the developed framework adequately implements the technical requirements of the CRISP framework.  

* determine next steps: decide whether to proceed to next stage or loop back to a previous stage based on the current results.
  * list out possible actions, reasons, pro/con ; then describe decision

The interpretable_model/interpretable_model2 is functional and therefore satisfies the requirements for enabling the rest of the required technology.  
However, the model will need to be re-implemented using more robust techniques as part of ongoing deployment.  
-->

## Deployment
<!--crispdm_deployment-->
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists](#probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [roadway-safety-analysis](#roadway-safety-analysis) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [technical-implementation](#technical-implementation) | [conclusion](#conclusion) | [acknowledgements](#acknowledgements) | [references](#references) | [appendix](#appendix) | 
<!--</@breadcrumb>-->
| CRISP-DM: |                                                      <!-- "tag anchors" for navigation in text-editor -->
[business-understanding](#business-understanding) |                <!--business_understanding-->
[data-understanding---analysis](#data-understanding---analysis) |  <!--data_understanding-->
[data-preparation](#data-preparation) |                            <!--data_preparation-->
[modeling](#modeling) |                                            <!--modeling-->
[evaluation](#evaluation) |                                        <!--evaluation-->
[deployment](#deployment) |                                        <!--deployment-->

<!-- * deployment plan - how will this be used for the business?  -->
#### Deployment Plan  
The models created for the dataset will be used within a navigation framework to create prediction scores for routes.  

<!-- * monitoring + maintenance plan - how will model be updated and supervised?  -->
#### Monitoring and Maintenance Plan  
Crash data can be obtained in an automated report, which can be used to continuously update the model.  
This will require further work to parse, as it uses a different format than the manual query used to build this model.  

For ongoing maintenance, the model parameters will need to be updated as new data becomes available.  
The schedule for this will be based on the amount of new data received, for which a threshold needs to be set.  

<!-- "drift" -->
Commonly, this is done by comparing the model scores for the original dataset against scores for new incoming data.  
If the scores start to decline too much, the model needs to be re-evaluated.  
On the one hand, it could be that the underlying data has changed, i.e. there are simply more severe crashes.  
If this is determined to be the case, the model parameters do not need to be re-tuned.  
Otherwise, the model needs to be re-recreated with different parameters to improve the score.  

<!-- * final report -->
<!-- * review project - "experience documentation" to summarise the knowledge gained during this project  -->
<!-- @TODO: add the deployment plan from the book -->


# Technical Implementation
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists](#probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [roadway-safety-analysis](#roadway-safety-analysis) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [technical-implementation](#technical-implementation) | [conclusion](#conclusion) | [acknowledgements](#acknowledgements) | [references](#references) | [appendix](#appendix) | 
<!--</@breadcrumb>-->

## Framework for CRISP-DM

The CRISP-DM process uses different features from the data set during each of its stages. 
Business Understanding and Data Understanding use the original set of features from the dataset while evaluating the problem and examining the data. 
Data Preparation uses a subset of the original features and often additionally generates new features. 
Modeling uses the list of features generated during data preparation for creating the final model. 

To manage this changing list of features, a framework was created to centralise and abstract the feature management. 
Without this framework, the list of features would have to be maintained separately at various points throughout the different stages, which becomes difficult to manage as the project evolves and features change. 

The framework ensures that each feature from the original dataset is categorised based on its purpose within the application. For example, features can be categorised based on the type of visualisation they can be used with, whether they are a predictive feature or a target feature, or what type of data the feature represents. 
This allows features to be selected by category through a query instead of being individually listed each time they are used.

This framework is used during data understanding to plot the data, and is updated with any generated features during data preparation. 
The modeling phase queries the framework to retrieve the prepared features and divide them between predictive and target features for model generation and cross-validation. 
This reduces much of the boilerplate typically involved when generating several models by eliminating the need to maintain lists of features throughout the code. 

<!--
<pre>
code_snippet_featdef_and_model_gen
code_snippet_no_featdef_and_model_gen
</pre>
-->


The strategy of querying the required features by category also facilitates the addition of new features and data sources throughout the CRISP-DM lifecycle. Once a feature is added to the framework with the correct categories it automatically gets included in all queries for that category. 
This means that everything from visualisations to models automatically get updated with new features, which greatly reduces the overhead and potential error of manually maintaining the various feature lists. 

New categories can be added at any time, which can help manage the various ways in which the data is used. 
For example, a category for "feature importance" was added in order to manually mark which features needed to be excluded from modelling once they had been found to be insignificant. A conventional approach without the framework would have involved manually removing these features wherever they were referenced throughout the code. 


**Implementation**  
The framework is implemented in the "feature_definitions" ("featdef") module as a pandas dataframe. 
A dataframe is a structure to store and query data, which lends itself well to data manipulation. 
The data is stored in a row-column format, forming a matrix of feature names in the rows and corresponding categories in the columns. 
The value for a feature's category is stored at the intersection of that feature's row and the category's column.
This abstraction for managing features centralises the feature management and eliminates the need to maintain several lists of features as previously mentioned.  

This framework is further explained in the [Appendix on Featdef Values](#appendix-featdef-values).  

<!--
<pre>
code_snippet_featdef_grid_example
</pre>
-->

## Predicting using Route Data
<!--
<pre>
* [X] route features map to model features
  * [-] model features also posterior, unavailable by default from route data
    * [X] manner of collision
    * [X] intersection related
* [X][Roadway Safety Analysis] can't simply predict route, need to have a score => adapt NHTSA rates
  * [X] CANNOT USE: road-segment rate calc: ratio of #crashes to count * length i.e.  1M * #crashes / (days x numyears) * #trafficCount * length 
    since no trafficcount, and going for prob not count: ratio of seg-prob to count, i.e.  : (1 day) * length (no trafficcount in routing data) 
  * [X] CANNOT USE: intersection: same as road-segment, without length
  * [X] crash rate mileage - i.e. dividing by 'L' to distribute crash rate blah blah blah
  * [X] conclusion:
    * [X] segment: crash-prob per mile
    * [X] intersection: crash-prob per million entering vehicles
    * [X] add all together? => two scores, each is product of inverse risks
  * [X] intersection related : only applies to intersections, not the portion between. I.e. can't assign "not intersection" to each GPS point between intersections
  * [X] -> score separation - needs something to reflect number of intersections, perhaps count intersections and 

[x] Score: inter+seg
[x] intersection: model prob
[x] segment: model prob/ length

OMITTING FOR NOW IN INTEREST OF TIME
[-] segment detail:
routing hides intersections, "collapses" straight route into one, trade-off required because of no data
traditionally with crash rates would be a problem because, let 'a','b' crashes, 'x','y' lengths : (a+b)/(x+y) != (a/x) + (b/y)
[-] however, using probs is not a rate, therefore let 'k','l' be probs INDEPENDENT length, model doesn't consider segment length for prob because crash data doesn't contain, therefore model would predict same 'k' for 'x','y','x+y' and therefore 'k' == 'l' for identical input features and therefore (k+l)/(x+y) = (2k) / (x+y) => still a problem, just different form


[X] The routing data is used by the machine learning model to predict the crash severity ...
[X] The data provided by the routing service contains only a subset of the features available from the crash data. 
</pre>
-->

### Mapping Routing Features to Crash Report Features  
The crash severity prediction model was developed using features from crash data, but will not predict on crash data during deployment. 
The features used for model prediction are defined by the features present in the crash data used to build the model. 
For deployment, the model predictions will be used to evaluate crash severity for routes instead of analysing crash data. 
Therefore the crash-related input features for the model need to be mapped to features available from routing data and other sources. 
Most of the crash-related features are a priori environmental measurements, such as time of day and weather condition, but some of the features are for data which can only be measured after a crash. 
The features "manner of collision" and "intersection related" both measure attributes of the collision, and therefore require the crash to have already occurred. 
The model could be generated without these features, but this would reduce the model accuracy. 
Instead, these a posteriori measurements can be causally mapped to influencing factors which can be obtained from environmental or routing data. 

<!-- TODO: mention remaining factors, e.g. time of day, weather, etc -->

#### Manner of Collision
The manner of collision measures the direction of movement for each participant in a collision. 
This direction of movement during the collision results from the direction each participant was travelling immediately prior to the collision. 
Therefore, the direction each participant is travelling immediately prior to a crash has a causal correlation with the posterior manner of collision. 

For the route analysis, this travel direction can be obtained directly from the routing data being analysed by the model. 
The travel direction does not have a direct mapping to the manner of collision, as the manner of collision dos not make a distinction between the units. 
Therefore, domain specific knowledge must be applied in order to create a mapping between the routing-data travel direction and the crash-data manner of collision. 
To create this mapping, all possible permutations of travel direction were created for each recorded manner of collision. 
This results in a one-to-many mapping between travel direction and manner of collision. 

The routing travel direction reflects the type of turn needed to be made, and can take on 'left', 'right', and 'straight'. 
The subset of possible 'manner of collision' values present in the model is mapped to routing travel direction as follows:  
<!-- in order of decreasing granularity, or increasing ambiguity -->
|S|L|R| Manner of Collision |
|-|-|-|-|
|x| | |'one motor vehicle going straight',|
|x| | |'angle both going straight',|
|x| | |'one motor vehicle other',|
|x| | |'same direction one straight one stopped'|
|x| | |'same direction both going straight rear end',|
|x| | |'one motor vehicle backing',|
|x| | |'opposite direction both going straight',|
|x|x| |'opposite direction one straight one left turn', # just "one" straight, not sure if motorist or cyclist|
|x|x| |'one motor vehicle turning left',|
|x| |x|'one motor vehicle turning right',|

Base assumptions: 
During travel, both units move predictably and according to the traffic laws. For example, cyclists are expected to turn only at intersections and otherwise stay in their lane.  
Manner of Collision is recorded accurately and correctly, e.g. the 'opposite direction both going straight' precludes the possibility of either unit making an unpredictable turn, which would be recorded as 'opposite direction on straight one left turn'. 

#### Intersection Related  
The crash report feature "intersection related" measures what type of road section the crash occurred on. 
Crashes occurring in or around an intersection are recorded as "intersection" or "intersection related", crashes on a segment as "non intersection", and crashes on a segment with a driveway or parking lot access as "driveway access". 
Mapping routing data to the values 'intersection' or 'intersection related' is possible if the routing information mentions every intersection. 

However, routing data is typically meant as an instruction to a human end-user, and as such usually only explicitly mentions intersections for which the end-user needs to change their travel direction. 
This is because routing software typically requires the end-user to assume a default straight travel direction along segments or through intersections, and therefore omits straight travel directions. 
<!--
If you're thinking about "continue straight", this is to resolve ambiguity when road sections converge or a road section name changes. 
-->

This is because routing software typically omits directions for a straight travel direction along segments or through intersections as it requires the end-user to assume a default straight travel direction. 
The google directions routing information only mentions intersections when the travel direction needs to be changed, and therefore does not make explicit mention of intersections or segments. 
This is because it requires the user to assume a straight travel direction by default. 
Therefore, only routing intersections which alter the travel direction can be mapped to the crash data. 

Mapping routing data to the values 'non intersection', i.e. road segment, and 'driveway access' presents a different challenge. 
The routing data does not contain references to driveways, which precludes any mapping to the 'driveway access' category. 
The routing data does not contain explicit references to segments, however segments can be partially inferred from the intersections. 
For simplicity, segments can be assumed to be between intersections, and as such a segment can be inserted between each intersection in the routing data. 
However, this assumption fails for routing data containing two consecutive intersections. 
In order to process two consecutive intersections, a definition is required to determine the minimum length required for a road section to be considered as a segment. 
The distance between two intersections could then be evaluated to determine whether to interpolate a segment between the two or whether to consider them as consecutive intersections. 
However, TxDOT does not explicitly define the minimum length required to define a road section as a segment [@txdotClassificationCR102]. 
Instead, for certain consecutive intersection types, TxDOT defines a minimum distance required to consider them as separate intersections [@txdotClassificationCR102]. 
Using this approximate guideline of 30 feet, segments could be inserted between any intersection for which the GPS coordinates are at least 30 feet apart. 
However, this guideline is not meant for all intersections and further research would be required to verify its applicability. 
Therefore, the interpolation of segments between intersections was not implemented for this project. 


### Predicting using Mapped Routing Features  
The "manner of collision" feature is one-hot encoded as several distinct features, i.e. the model considers each type of "manner of collision" as a separate feature. 
This facilitates the one-to-many mapping of the travel direction to the manner of collision. 
The travel direction from the incoming routing data is converted to the corresponding manner of collision and added to the remaining transmitted a priori environmental measurements. 
These measurements are then passed to the model, which will in effect be predicting the crash severity using the all possible values for manner of collision obtained from the travel direction. 

The "intersection related" feature is one-hot encoded as several distinct features, i.e. the model considers each value of "intersection related" as a separate feature. 
Each intersection from the routing data is mapped as 'intersection', but no other mapping is made as this data is unavailable. 
The model will therefore only predict on routing directions which indicate a required change in direction, and will not consider any straight intersections. 

### Impact on the Route Score  
The lack of routing data for segments, driveways, and straight-turns impacts both the "manner of collision" and "intersection related" features. 
Without these road section types, each feature can only be used to predict on intersections which require a change in travel direction. 
Ultimately, this means the safety score will not be evaluated for "straight sections, i.e. road sections for which the routing data travel direction is straight. 
This hides variation between routes with different numbers of intersections, as a route with several "straight intersections" will receive the same score as a route with few straight intersections, where "straight intersection" is understood to be a straight section which is an intersection. 
The data-mapping for the feature "manner of collision" does not change based on road section type; as long as the travel direction is straight, all "manner of collision" are considered possible. 
Therefore, the lack of straight sections only leads to a lack of accuracy for the feature "manner of collision". 
The data-mapping for the feature "intersection related" changes based on road section type, as it takes on different values for intersections and road segments. 
Therefore, the lack of straight sections leads to a lack of precision for the feature "intersection related". 

These limitations cannot be resolved using only routing data, but for scoring purposes still allow for a distinction between routes to be made with the caveat that straight sections are not evaluated. 

<!-- TODO: mention correlation between "manner of collision" and "crash severity"; intuitively the manner in which a car collides with a cyclist should have a large impact on the severity of injury -->

<!--
## Comparing Multiple Route Scores
When comparing routes, several environmental factors remain the same for each route, e.g. weather condition and time of day. 
These values do not change between the routes, and therefore the model assigns each of them the same score. 
There are a few approaches to address this issue. 
-->

## User Application
The user-facing interface is a browser-based application using html and javascript to interface with a server written in python to generate safety predictions for routes generated by a third-party routing service. 

<!--
Section Glossary:  
[@terminology]: geo-json
-->


**User Interface**  
The user interface is browser-based and was designed to be similar in feel to many popular routing services. 
The user visits a webpage and types in a source address and a destination address, upon which the possible routes are displayed with the safety score overlayed at each intersection. 
The user can then choose the route with the lowest score, which corresponds to the lowest injury risk.  
For routing, the user can either type in the exact address or use autocomplete by typing a partial match and selecting the correct result from a list of possible matches. This autocompletion is implemented using the Google Places API.

![img_of_routing_ui_autocomplete]

[img_of_routing_ui_autocomplete]: res/img/img_of_routing_ui_autocomplete.png

![img_of_routing_ui_evaluated]

[img_of_routing_ui_evaluated]: res/img/img_of_routing_ui_evaluated.png

**Route Scoring Server**  
The route scoring server is designed to process third-party routing geo-json by extracting the intersections and processing them using the scoring model. 
The server itself receives routing geo-json and returns a custom format geo-json containing routing scores. 
The routing score geo-json contains scores for each intersection of each route along with a final score for the route, stored at both the beginning and end for the route.  

**Technology**  
The routing information is retrieved from google maps as geo-json, and then sent to the server hosting the model.  
The route scoring information is retrieved from scoring server as geo-json, and then displayed on the map.

### Architecture

<!-- github doesn't know about 'pre' tags, I guess. either do '< -' or '&lt;-' -->

#### User-Client Interaction: 

<!--
#### Legend:
<pre>
[user] ----{Map-UI: route start,end}--&gt; [ client ]  
[    ] &lt;---{Map-UI: route + scores }--- [        ]  
</pre>
-->

The end-user uses the client application as a conventional routing tool.  
Client application displays available routes and their score.  

From the perspective of the end user, the client application is used to plan the route and display the safety scores. 
The only action required by the user is the route planning, the scoring happens automatically. 
This simple interface makes the application as easy to use as conventional routing services, which achieves the original goal of making the tool easy to use.  

The routing and map display are implemented using the javascript Google Maps API. The scoring display is implemented using google maps markers with the intersection scores retrieved from the scoring server. 

![arch_client_server]

#### Client-Model Interaction:

<!--
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
-->

[arch_client_server]: 
http://gravizo.com/svg?@startuml;/%27;%20%20%20%20package;%20%20%20%20node;%20%20%20%20folder;%20%20%20%20frame;%20%20%20%20cloud;%20%20%20%20database;%27/;actor%20User;package%20"Client"%20{;%20%20[GUI]%20<->%20[httpClient];};/%27;node%20"Other%20Groups"%20{;%20%20FTP%20-%20[Second%20Component];%20%20[First%20Component]%20-->%20FTP;}%20;%27/;%27skinparam%20linetype%20ortho;%27skinparam%20linetype%20polyline;cloud%20"Third-Party%20Map%20API"%20{;%20%20[AutoCompleteService];/%27;};cloud%20{;%27/;%20%20[Routing%20Service];};package%20"Scoring%20Server"%20{;%20%20%27%20database%20"FileSystem";%20%20[Scoring%20Application];%20%20[Json%20Server];};[User]%20-->%20[GUI]%20:%20route%20origin,%20destination;[User]%20<--%20[GUI]%20:%20display%20route%28s%29%20+%20scores;[httpClient]%20-->%20[AutoCompleteService]%20:%20partial\nroute\norig,%20dest;[httpClient]%20<--%20[AutoCompleteService]%20:%20resolved\nroute\norig,%20dest;[httpClient]%20-->%20[Routing%20Service]%20:%20route\norig,%20dest;[httpClient]%20<--%20[Routing%20Service]%20:%20route\ngeo-json;[httpClient]%20-->%20[Json%20Server]%20:%20rest:\nroute%20geo-json;[httpClient]%20<--%20[Json%20Server]%20:%20rest:\ncore%20geo-json;%27%20scoring%20server;/%27;[Json%20Server]%20-->%20[FileSystem]%20:%20route%20geo-json;[Json%20Server]%20<--%20[FileSystem]%20:%20score%20geo-json;FileSystem%20-->%20[Scoring%20Application]%20:%20route%20geo-json;FileSystem%20<--%20[Scoring%20Application]%20:%20score%20geo-json;%27/;[Json%20Server]%20-->%20[Scoring%20Application]%20:%20route%20geo-json;[Json%20Server]%20<--%20[Scoring%20Application]%20:%20score%20geo-json;@enduml;

[arch_scoring_app]: http://gravizo.com/svg?@startuml;skinparam%20componentStyle%20uml2;/%27;PURPOSE:%20architecture%20of%20the%20scoring%20application,%20which%20uses%20scoring%20model%20on%20gps+env%20data;%27/;/%27;%20%20%20%20package;%20%20%20%20node;%20%20%20%20folder;%20%20%20%20frame;%20%20%20%20cloud;%20%20%20%20database;%27/;%27%20"feature%20data";%27%20route%20data;%27cloud%20{;%20%20package%20extInterface{;%20%20%20%20interface%20routeInfo;%20%20%20%20interface%20scoreInfo;%20%20};%27};package%20ScoringServer%20{;%20%20frame%20dataSources;%20%20dataSources%20-->%20preprocessor;%20%20package%20ScoringApplication{;%20%20%20%20node%20preprocessor%20{;%20%20%20%20%20%20node%20featdef;%20%20%20%20%20%20%27database%20dataframe%20as%20dataset;%20%20%20%20};%20%20%20%20%28%29%20"featureSelection\n%28query%20featdef%29"%20as%20featureSelection;%20%20%20%20node%20procRouteInfo%20{;%20%20%20%20%20%20node%20extractEnvData;%20%20%20%20%20%20node%20extractGPSData;%20%20%20%20};%20%20%20%20routeInfo%20-->%20procRouteInfo;%20%20%20%20node%20modelUse%20as%20modelBuild%20{;%20%20%20%20%20%20node%20model;%20%20%20%20%20%20%27%20"fit"%20is%20sklearn-specific;%20%20%20%20%20%20%27no%20fit%20during%20deploy%27%20%28%29%20"train"%20as%20fit;%20%20%20%20%20%20%27%20during%20model%20creation,%20predict%20is%20for%20the%20x-val%20%27test%27;%20%20%20%20%20%20%27%20during%20model%20usage,%20%20%20%20predict%20is%20to%20get%20the%20prediction%20scores;%20%20%20%20%20%20%28%29%20"predict"%20%20as%20predict;%20%20%20%20};%20%20%20%20frame%20scores;%20%20%20%20predict%20-->%20scores;%20%20%20%20predict%20<-left-%20model;%20%20%20%20node%20generateScoreInfo%20{;%20%20%20%20%20%20node%20combineScoreAndLocation;%20%20%20%20%20%20scores%20-->%20combineScoreAndLocation;%20%20%20%20%20%20extractGPSData%20-->%20combineScoreAndLocation;%20%20%20%20%20%20combineScoreAndLocation%20-->%20scoreInfo;%20%20%20%20};%20%20%20%20database%20localStorage%20{;%20%20%20%20%20%20node%20scoringModel;%20%20%20%20};%20%20%20%20model%20<-%20scoringModel%20:%20retrieve%20from%20disk;%20%20};%20%20extractEnvData%20-->%20predict;};@enduml;


The client application sends the user-input route start and end to the external routing service.  
The external routing service returns geo-json containing routing information, which contains GPS coordinates representing the routes.  

<!-- environmental requires diagram update -->
The client submits the third party routing data <!-- and environmental information --> to the server as geo-json. 
The json server passes the data to the modelling application, which returns the route scores to the json-server after processing the input data. 

<!-- environmental requires diagram update -->
The modeling application uses relevant features from the routing <!-- and environmental --> geo-json in order to score each intersection along the route and calculate a total score for the route.  These scores are stored in a geo-json format. 

The modeling application then sends route-score geo-json to server, which relays this data back to the client. 

The client then displays the unmodified third-party routing information along with the route-scoring information to the user. 

For this implementation, the Google Maps Directions Service was used as the third-party routing provider, and the Google Places API was used to enable autocompletion for the starting and end location names. 

![arch_scoring_app]

#### Data Formats
**google maps geo-json**

The google maps geo-json compises a list of scores for each calculated route. 
Each route contains at least one leg, defined as the distance between two markers along a route. 
For this project, custom markers were not implemented and therefore each route comprises exactly one leg. 
Each leg of a route contains several steps, each of which contain a set of instructions intended for consumption by the end user. 
This project maps the "maneuver", or instruction for alteration of travel direction, to the "manner of collision". 
<!-- TODO -->

    'legs': [{'distance': {'text': '0.5 mi', 'value': 736},  
            'duration': {'text': '10 mins', 'value': 584},  
            'start_address': '2501 speedway, austin, tx 78712, usa',  
            'start_location': {'lat': 30.2882269, 'lng': -97.73692160000002},  
            'end_address': '2606 guadalupe st, austin, tx 78705, usa',  
            'end_location': {'lat': 30.2912016, 'lng': -97.7412483},  
            'steps': [{'distance': {'text': '453 ft', 'value': 138},  
                       'duration': {'text': '2 mins', 'value': 101},  
                       'encoded_lat_lngs': 'mtzwdvfpsqwagqagmai',  
                       'end_location': {'lat': 30.2894657,  
                                        'lng': -97.73679190000001},  
                       'end_point': {'lat': 30.2894657,  
                                     'lng': -97.73679190000001},  
                       'instructions': 'head <b>north</b> on <b>speedway</b> '  
                                       'toward <b>e dean keeton st</b>',  
                       'lat_lngs': [{'lat': 30.288230000000002,  
                                     'lng': -97.73692000000001},  
                                    {'lat': 30.288670000000003,  
                                     'lng': -97.73688000000001},  
                                    {'lat': 30.289080000000002,  
                                     'lng': -97.73684000000002},  
                                    {'lat': 30.28947,  
                                     'lng': -97.73679000000001}],  
                       'maneuver': '',  
                       'path': [{'lat': 30.288230000000002,  
                                 'lng': -97.73692000000001},  
                                {'lat': 30.288670000000003,  
                                 'lng': -97.73688000000001},  
                                {'lat': 30.289080000000002,  
                                 'lng': -97.73684000000002},  
                                {'lat': 30.28947, 'lng': -97.73679000000001}],  
                       'polyline': {'points': 'mtzwdvfpsqwagqagmai'},  
                       'start_location': {'lat': 30.2882269,  
                                          'lng': -97.73692160000002},  
                       'start_point': {'lat': 30.2882269,  
                                       'lng': -97.73692160000002},  
                       'travel_mode': 'bicycling'},  

**route score geo-json**

The route score geo json comprises a list of scores for each route. 
Each route comprises a list of scores for each road section. 

{'routes': [[{'lat': 30.2894657,  
              'lng': -97.73679190000001,  
              'score': 0.9385474860335196},  
             {'lat': 30.28981659999999,  
              'lng': -97.7413871,  
              'score': 0.9088541666666666},  
             {'lat': 30.2912016,  
              'lng': -97.7412483,  
              'score': 0.9088541666666666}]],  
 'totalScores': [0.918751939788951]}  


#### Modeling Application:

<!--
<pre>
Data Preparation and Feature Implementation:  
/data sources/ ---&gt; [Preprocessor per source, feature] ---&gt; [df: dataset | df: feature definitions ]

Model Creation:   
[dataset,featdef] ---{query: desired features}---&gt;---{slice: dataset}---&gt;[model]

</pre>
-->

The Modeling application uses the feature definition framework to create the different models required for scoring the route.  
The application is written in python, and uses python machine learning libraries to create the models. 

Modules:

| Filename | Purpose |
|---|---|
| model.py        | Model Build, Optimise, Predict Route-Score |
| txdot_parse.py  | Prepare data as outlined under Data Preparation  |
| feature_defs.py | Track features and their purpose  |
| mapgen.py       | Generate maps for static heatmap visualisation  |
| helpers.py      | Useful functions |


**Data Parser**  
Convert input data format to pandas dataframe, handles the data preparation stage and updates the feature definition framework. 

**Feature Definitions**  
The feature definition framework implemented as a pandas dataframe, which provides an abstraction for feature management throughout the application. 

![arch_model_gen]( http://gravizo.com/svg?@startuml;skinparam%20componentStyle%20uml2;/%27;PURPOSE:%20architecture%20of%20generating%20scoring%20model;%27/;/%27;%20%20%20%20package;%20%20%20%20node;%20%20%20%20folder;%20%20%20%20frame;%20%20%20%20cloud;%20%20%20%20database;%27/;%27%20"feature%20data";cloud%20{;%20%20database%20txdot;%20%20%27database%20trafficFlow;};package%20ScoringServer%20{;%20%20frame%20dataSources;%20%20package%20ScoringApplication{;%20%20%20%20node%20preprocessor%20{;%20%20%20%20%20%20node%20featdef;%20%20%20%20%20%20database%20dataframe%20as%20dataset;%20%20%20%20};%20%20%20%20%28%29%20"featureSelection\n%28query%20featdef%29"%20as%20featureSelection;%20%20%20%20node%20crossValidation%20{;%20%20%20%20%20%20frame%20testData;%20%20%20%20%20%20frame%20trainData;%20%20%20%20};%20%20%20%20node%20modelBuild%20as%20modelBuild%20{;%20%20%20%20%20%20node%20model;%20%20%20%20%20%20%27%20"fit"%20is%20sklearn-specific;%20%20%20%20%20%20%28%29%20"train"%20as%20fit;%20%20%20%20%20%20%27%20during%20model%20creation,%20predict%20is%20for%20the%20x-val%20%27test%27;%20%20%20%20%20%20%27%20during%20model%20usage,%20%20%20%20predict%20is%20to%20get%20the%20prediction%20scores;%20%20%20%20%20%20%28%29%20"test"%20%20as%20predict;%20%20%20%20};%20%20%20%20frame%20scores;%20%20%20%20database%20localStorage%20{;%20%20%20%20%20%20node%20scoringModel;%20%20%20%20};%20%20%20%20model%20->%20scoringModel%20:%20save%20to%20disk;%20%20};};txdot%20-->%20dataSources;%27trafficFlow%20-->%20dataSources;dataSources%20-->%20preprocessor;preprocessor%20-->%20dataset;preprocessor%20-->%20featdef;crossValidation%20-->%20testData;crossValidation%20-->%20trainData;crossValidation%20-->%20modelBuild%20:%20adjust\nparams;crossValidation%20<--%20scores%20:%20feedback;dataset%20-->%20featureSelection;featdef%20-->%20featureSelection;%27modelBuild%20<-L-%20featureSelection;crossValidation%20<-L-%20featureSelection;%27%20train%20model;%27%20trainData%20-->%20modelBuild%20:%20train;trainData%20-->%20fit;fit%20-->%20model;%27%20predict;predict%20<--%20model;testData%20-->%20predict;predict%20-->%20scores;@enduml;)


# Conclusion
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists](#probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [roadway-safety-analysis](#roadway-safety-analysis) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [technical-implementation](#technical-implementation) | [conclusion](#conclusion) | [acknowledgements](#acknowledgements) | [references](#references) | [appendix](#appendix) | 
<!--</@breadcrumb>-->
<!--
* Explain what is striking/noteworthy about the results.
* Summarize the state of knowledge and understanding after the completion of your work.
* Discuss the results and interpretation in light of the validity and accuracy of the data, methods and theories as well as any connections to other people’s work.
* Explain where your research methodology could fail and what a negative result implies for your research question.
-->

<hr />

This paper explored the process of applying machine learning to solve a real world problem by providing a user-friendly interface to the model's predictions.  

<!-- Goal: Develop framework for managing crash data.  -->
A framework was created for managing the data throughout the CRISP-DM process, which proved useful when creating new models and adapting to changing requirements.
Dataframes, in particular python's Pandas library, are very useful at every stage of the CRISP-DM process, as they can be used to manage the input data as well as the internal state of the program.  


<!-- Goal: Create application for scoring routes in real time.  -->
The user-facing application for scoring routes in real time was implemented as a proof-of-concept and demonstrated the ability to score different routes as well as ease-of-use for the end-user. 
The application architecture was designed to allow for additional data sources to be easily integrated in order to improve the prediction accuracy. 

<!-- Goal: Produce machine learning model.  -->
The machine learning model for calculating route scores proved to be the most challenging aspect. 
Deploying it for use with data available proved to be challenging, and its accuracy had to be reduced to accommodate the available data. 
However, acquiring the necessary data is a solved problem simply beyond the scope of this project. 
A future iteration could use other datasources and techniques to accumulate the missing data. 



Ultimately, the success of such projects depends on the available data more than the underlying prediction techniques.  
To improve the prediction accuracy, more data on cyclist ridership and crashes needs to be made available. 
This is both the responsibility of the municipality, which can improve crash data collection and accessibility, and the cyclists, who can improve ridership data by self-reporting their riding habits.  

Increased ridership data collection would additionally allow for traffic flow analysis, something this project was not able to take advantage of. 
Such analysis would allow crash-frequency to be calculated to identify problem areas which lead to more crashes. 
This would allow the municipality to address these areas through improved traffic control, and would allow cyclists to avoid these areas until they are improved.

The end result would be an increase in traffic safety for cyclists, and more cyclists would lead to reduced traffic congestion for motorised vehicles.  




<!--
# Future Work
< !--!@breadcrumb-- >
## Crash Data

Add data on bike-lane presence

Examine before/after lane reduction: car+car |vv|^^| => |cv|<>|vc| bike+car '|cv|' + turn lane '|<>|'
support with studies

Interpret How Cyclists Can Ride Defensively
additional requirement: data source update, new model, but reuse application layer, re-analysis of data (i.e. start a new CRISP-DM lifecycle)
concept:
focus on "personal" data features, e.g. wearing helmet, avoiding busy roads
classification into "avoidable" and "avoidable" crashes
e.g. left-turn crash seen as "avoidable" because cyclist can look for vehicles, but crash from rear seen as "unavoidable"  because cyclist has no visibility of vehicles

interpret limited data more creatively  
e.g. analyse frequency of crashes to determine which locations tend to have more reported crashes.  
This could be loosely correlated with the probability of a crash, although it could also just mean that certain locations tend to be over-reported vs others. However, since there's not much data, this is also not a bad idea in a pragmatic sense. Can't avoid an unknown, but can avoid a known - work with the data which is available.

traffic: use general traffic data (instead of cyclist data) and find 'reported crashes'/'street segment traffic'  
src: traffic count : https://data.austintexas.gov/Transportation-and-Mobility/Traffic-Count-Study-Area/cqdh-farx

## Data Sources

Use data from strava,mapmyride,etc to find the most common routes (among the users of these apps) and correlate with crash data
-->

# Acknowledgements
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists](#probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [roadway-safety-analysis](#roadway-safety-analysis) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [technical-implementation](#technical-implementation) | [conclusion](#conclusion) | [acknowledgements](#acknowledgements) | [references](#references) | [appendix](#appendix) | 
<!--</@breadcrumb>-->
<!-- Thank the people who have helped to successfully complete your project, like project partners, tutors, etc. -->
Source For Outline: 
https://www.ethz.ch/content/dam/ethz/special-interest/erdw/department/dokumente/studium/master/msc-project-proposal-guidelines.pdf 

Source For Abstract:  
https://www.honors.umass.edu/abstract-guidelines  
http://www.sfedit.net/abstract.pdf  

# References
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists](#probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [roadway-safety-analysis](#roadway-safety-analysis) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [technical-implementation](#technical-implementation) | [conclusion](#conclusion) | [acknowledgements](#acknowledgements) | [references](#references) | [appendix](#appendix) | 
<!--</@breadcrumb>-->
<!-- List papers and publication you have already cited in your proposal or which you have collected for further reading. The style of each reference follows that of international scientific journals. -->

<!-- background on crashes -->
<!--
[@safetyImpactBikelanePolitiFact]: http://www.politifact.com/texas/statements/2016/jul/29/bike-austin/bike-austin-says-bike-lanes-sidewalks-reduce-austi/
@citationNeeded:openAustin
@citationNeeded - article: increased cycling increases safety  
[@moreCyclingMoreSafetyNacto]: https://nacto.org/2016/07/20/high-quality-bike-facilities-increase-ridership-make-biking-safer/
[@moreBikeLaneMoreRiders]: https://www.citylab.com/transportation/2014/06/protected-bike-lanes-arent-just-safer-they-can-also-increase-cycling/371958/
[@moreBikeLaneMoreSafety]: https://www.citylab.com/transportation/2012/10/dedicated-bike-lanes-can-cut-cycling-injuries-half/3654/

@citationNeeded: segmentation - training models on different slices of the dataset is referred to as segmentation.
-->

[@originalProject]: https://nbviewer.jupyter.org/github/YoinkBird/dataMiningFinal/blob/master/Final.ipynb#Maps-of-Crashes 

[@coagovTrafficCount]: https://data.austintexas.gov/Transportation-and-Mobility/Traffic-Count-Study-Area/cqdh-farx

<!-- data mining -->
[@wikiROChistory]: https://en.wikipedia.org/wiki/Receiver_operating_characteristic#History

<!-- crisp dm -->
[@wikiCrispDM]: https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining

<!-- general data mining topic -->
[@caruana_et_al_2008]: http://icml2008.cs.helsinki.fi/papers/632.pdf

[@fhwaBeforeAfterAnalysis]: https://www.fhwa.dot.gov/publications/research/safety/02089/05.cfm#b
<!-- "The goal is simple: to prevent the severe types of crashes that can change lives forever." -->
[@fhwaReduceCrashSeverity]: https://safety.fhwa.dot.gov/intersection/ 
[@txdot_crash_report_source]: http://www.txdot.gov/driver/laws/crash-reports.html
[@txdotCrashSeverityGrouping]: http://ftp.dot.state.tx.us/pub/txdot-info/aus/bicycle-master-plan/crashes.pdf
[@fhwa3DataAnalysis]: https://safety.fhwa.dot.gov/local_rural/training/fhwasaxx1210/s3.cfm  
[@fhwa3DataAnalysisCrashRate]: https://safety.fhwa.dot.gov/local_rural/training/fhwasaxx1210/s3.cfm#s32  
[@fhwa3DataAnalysisPotentialCrashes]: https://safety.fhwa.dot.gov/local_rural/training/fhwasaxx1210/s3.cfm#s34  
[@fhwa3DataAnalysisCrashRateIntersection]: https://safety.fhwa.dot.gov/local_rural/training/fhwasaxx1210/s3.cfm#s322  
[@fhwa3DataAnalysisCrashRateSegment]: https://safety.fhwa.dot.gov/local_rural/training/fhwasaxx1210/s3.cfm#s321  
[@fhwaHSIPproblemIdentificaton]: https://safety.fhwa.dot.gov/hsip/resources/fhwasa09029/sec2.cfm#24
[@trafficForecastingCrowdFlows]: https://dl.acm.org/citation.cfm?doid=2996913.2996934
<!-- txdotHSIManualMannerCollision : page 1-18 -->
[@txdotHSIManualMannerCollision]: http://onlinemanuals.txdot.gov/txdotmanuals/hsi/hsi.pdf
[@txdotClassificationCR102]: ftp://ftp.dot.state.tx.us/pub/txdot-info/library/forms/cit/crash102_final_10_08.pdf <!-- page 14 -->
[@sklearn_feat_sel_rfe]: http://scikit-learn.org/stable/modules/feature_selection.html#recursive-feature-elimination
[@trafficCountBicycleVsCar]: http://waycount.com/classic/faq
<!-- related work -->
[@relatedWorkSafeNavUrbanEnv]: http://www2.cs.uic.edu/~urbcomp2013/urbcomp2014/papers/Galbrun_Safe%20Navigation.pdf
[@relatedWorkPredCrashUrbanData]: http://urbcomp.ist.psu.edu/2017/papers/Predicting.pdf
[@relatedWorkCriminalityPred]: https://dspace.cvut.cz/handle/10467/69704  


# Appendix
<!--!@breadcrumb-->
<!--<@breadcrumb>-->
| [table-of-content](#table-of-content) | [probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists](#probabilistic-routing-based-injury-avoidance-navigation-framework-for-pedalcyclists) | [abstract](#abstract) | [introduction](#introduction) | [background-and-results-to-date](#background-and-results-to-date) | [goals](#goals) | [roadway-safety-analysis](#roadway-safety-analysis) | [methodology](#methodology) | [crisp-dm-report](#crisp-dm-report) | [technical-implementation](#technical-implementation) | [conclusion](#conclusion) | [acknowledgements](#acknowledgements) | [references](#references) | [appendix](#appendix) | 
<!--</@breadcrumb>-->
<!-- Add pictures, tables or other elements which are relevant, but that might distract from the main flow of the proposal. -->

<!--
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
The feasibility assessment concludes with a cost-benefit analysis of project cost vs benefit to the business.

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
-->

## Appendix: Data Description Report  
### TXDOT data  

Crash ID
Average Daily Traffic Amount
Average Daily Traffic Year
Crash Death Count
Crash Incapacitating Injury Count
Crash Non-incapacitating Injury Count
Crash Not Injured Count
Crash Possible Injury Count
Crash Severity
Crash Time
Crash Year
Day of Week
Intersecting Street Name
Intersection Related
Latitude
Light Condition
Longitude
Manner of Collision
Medical Advisory Flag
Object Struck
Road Base Type
Speed Limit
Street Name
Surface Condition
Weather Condition

## Appendix: TxDOT Data Undefined Values Report
| feature                      |no.*data |unknown |not.*reported |not.*applicable |
|------------------------------|---------|--------|--------------|----------------|
| average_daily_traffic_amount |  1904.0 |    0.0 |          0.0 |            0.0 |
| average_daily_traffic_year   |  1936.0 |    0.0 |          0.0 |            0.0 |
| crash_severity               |     0.0 |    4.0 |          0.0 |            0.0 |
| day_of_week                  |     0.0 |    0.0 |          0.0 |            0.0 |
| intersecting_street_name     |     0.0 |  194.0 |          8.0 |            0.0 |
| intersection_related         |     0.0 |    0.0 |          1.0 |            0.0 |
| latitude                     |   235.0 |    0.0 |          0.0 |            0.0 |
| light_condition              |     0.0 |   37.0 |          0.0 |            0.0 |
| longitude                    |   235.0 |    0.0 |          0.0 |            0.0 |
| manner_of_collision          |     0.0 |    0.0 |          0.0 |            0.0 |
| medical_advisory_flag        |     0.0 |    0.0 |          0.0 |            0.0 |
| object_struck                |     0.0 |    0.0 |          0.0 |         2221.0 |
| road_base_type               |  1904.0 |    0.0 |          0.0 |            0.0 |
| street_name                  |     0.0 |    0.0 |          0.0 |            0.0 |
| surface_condition            |     0.0 |   10.0 |          0.0 |            0.0 |
| weather_condition            |     0.0 |   17.0 |          0.0 |            0.0 |


## Appendix: TxDOT Data Quality Report

Report in res/rpt/data_quality_report_txdot.html

## Appendix: Featdef Values  

| attribute | description | values | 
|----------|--------------------------------|---------------|
|'target'  | indicate whether feature is a predictor or target | [False, True] |
|'regtype' | type of values w.r.t. modelling | string - continuous, categorical, bin_cat (binary) , onehot (dummy-encoded) |
|'input'   | importance of feature to make a prediction | integer - ascending from 0 |
|'dummies' | whether feature can be dummy-encoded | [False, True] |
|'type'    | datatype, used to filter features, e.g. models need to ignore data encoded as 24h  | string - ['int', 'str', '24h', 'street', 'gps', 'datetime', 'float'] |
|'pairplot'| whether data can be plotted in a pairplot | [False, True] | 
|'jsmap'   | whether data can be plotted on a map | [False, True] |
|'origin'  | name of contributing feature for derived features | string - ['crash_datetime', 'crash_severity', 'intersection_related', 'light_condition', 'manner_of_collision', 'day_of_week', 'road_base_type'] |

<!--
|---|---|
| data preparation | register generated features |
| Data Preparation - select | 'clean_data' loads the crash data into a pandas (pd) dataframe (df) from a csv file, which requires ignoring the meta-header. |
| Data Preparation - clean | The function 'clean_data' ensures that the data is machine readable to avoid parsing errors as well as to facilitate the coding process.  |


A few examples:  
Building a model requires a set of descriptive features and target features.  
Typically, this is done using individual arrays of feature names, which are then used to query the pandas dataframe containing all features.  
With featdef, the features can be queried dynamically as 'target' or 'non-target' instead of maintaining individual lists.  
One implication is that as the project evolves, any new features are automatically picked up by the models instead of the maintainer having to update each model's individual list of descriptive and target features.  

featdef is used to track the origin of newly implemented features, so if a model needs to exclude them it can easily do so.

featdef tracks the type of feature as well to identify which features are meant to be used in the model and which aren't, such as the case-id.   
-->

## Appendix: Modeling - Feature Selection for Feature Reduction Model

RFECV on the full set of features with 233 data points lead to high variance between RFECV scores.  
This indicated that the dataset for this number of features could not lead to a reliable model.  
<pre>
-I-: First Run
-I-: creating new dataset without []
NaN handling: Samples: NaN data 1999 / 2232 fullset => 233 newset
NaN handling: no  feature reduction after dropna(): pre 54 , post 54
</pre>


For the second run the dataset was increased to reduce the variance in RFECV scores.  
The 'average_daily_traffic_amount' and 'average_daily_traffic_year' were removed from the feature list as they were strong features with the least amount of data-points.
This lead to a dataset with 1644 data-points instead of only 233, but there was still high variance between the RFECV scores although it had settled.  
The most strongest features were crash_year and speed_limit.  
<pre>
-I-: Second Run
-I-: creating new dataset without ['average_daily_traffic_amount', 'average_daily_traffic_year']
NaN handling: Samples: NaN data 588 / 2232 fullset => 1644 newset
NaN handling: no  feature reduction after dropna(): pre 52 , post 52
</pre>


crash_year was removed for the next run, as it is a posterior feature, i.e. it can be assumed that the current year will not be predictive of a crash.  
The other strong feature, 'speed_limit', was not removed.  
This lead to a local maximum stabilisation of the RFECV scores starting at 16 features.  
The strongest features were speed_limit and surface_condition.  
<pre>
-I-: Third Run
-I-: creating new dataset without ['average_daily_traffic_amount', 'average_daily_traffic_year', 'crash_year']
NaN handling: Samples: NaN data 588 / 2232 fullset => 1644 newset
NaN handling: no  feature reduction after dropna(): pre 51 , post 51
</pre>


The next run was performed without any of the previous strongest features.
This lead to a RFECV scores with a sharp peak at 5 features which then gradually descended.  
The most important features were 5 days of the week.
This was the final run, as the resulting scores were much lower than the previous run. This indicated that this and any further feature elimination would only weaken the model.

<pre>
-I-: Fourth Run
-I-: creating new dataset without ['average_daily_traffic_amount', 'average_daily_traffic_year', 'crash_year', 'speed_limit', 'surface_condition']
NaN handling: Samples: NaN data 19 / 2232 fullset => 2213 newset
NaN handling: no  feature reduction after dropna(): pre 49 , post 49
</pre>

**Final Features**  
The features from the third run were chosen for the model due to the stable RFECV scores.  
The fourth run resulted in features with scores an order of magnitude lower than the previous scores of the manually excluded features.
Therefore, the feature list from the fourth run was determined to be the one to use for optimal model prediction.  

## Appendix: Application Implementation Gantt Chart
<!--
REV2 [20171017] roadmap: remove GPS fuzzy match from critPath
* simplest proof-of-concept doesn't require real-world GPS coords at all
* scoring doesn't depend on GPS coordinates
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
|-----------------|---------------------------------------------|
| critPath , crit | critical path i.e. core requirements for project or subproject (i.e. component of a project) |
| poc | proof of concept, implementation of a critPath . e.g. code implemented such that its state conforms with a critical path. |

| phase | description | importance | 
|-|-|-|
| poc1 | each tech implemented (first proof-of-concept) <br/> model, simple csv-based user interface, non-interactive map display of route |  

<!-- NOTE: only have to have leading '|' and one closing. Update the header to add a column -->
Note: non-obvious dependencies marked with [DEP: <paraphrased description of dependency>]  
note: only a project-phase chart, not a gantt chart with work-packages  

Minimal Description of phases (makes it easier to manange the table)  
<!-- todo: keep this tied-in to the work-packages. see comments below for more ideas, this just a placeholder -->
poc1: csv-ui, encode route using csv, model reads csv, gets GPS coords, html+js display route on map  
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
|----------|-|-|-|-|-|
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
<!--
**DRAFT**  
**Staging for explanations down below**  
[x] WP-Deliverables - merge into 'Work-Packages' (not into Roadmap)  
[x] 'Work-Packages' and 'WP Deliverables' need to be combined. Action Taken: convert "WPs" into headers, move "WP Deliverables" under the headers  
[x] Work-Packages - outline each WP as a header, use the mini-toc to list them all  
[ ] reference WP names from 'Roadmap', move explanations up into WP description  
**/DRAFT**
-->

<!-- TODO: auto-list these, like a TOC. reason: have work-packages be a summary, then 'wp deliverables' the explanation, which re-uses the eact same titles. -->
<!--!toc_mini-->

Notation: the WP-names should reflect the scope of the functionality  
E.g. "safety_score" implies any WP with the name "safety_score-\*" such as safety_score-total and safety_score-partial  

Each WP lists the a critical path (i.e. simplest functioning product ) it can be integrated into.  


#### WP: data: fuzzy-match GPS coordinates [GPS-fuzzy-match]  
WP: [data:  GPS-fuzzy-match]  
Dependency: GPS-coordinates
[route: GPS-\*-generic] -> [model: GPS-fuzzy-match] -> [model: safety_score] -> [display score]  
**Description:**   
crash data GPS coordinates will not be exactly same as route-mapper GPS-coordinates. Therefore, imprecisely (fuzzy) compare user-input GPS coords to crash-data GPS coords to find closest match. Initially only perform this fuzzy match on intersection coordinates, as single-location coordinates can be harder to place precisely.  

#### WP: data: impute more mph limits [impute_mph_limit-noninter]  
WP: [data:  impute_mph_limit-noninter]  
Dependency: GPS-fuzzy-match
[route: GPS-\*-generic] -> [model: GPS-fuzzy-match] -> [model: impute_mph_limit-noninter] -> [model: safety_score] -> [display score]  

**Description:**   
Impute speed limits (mph limit) for segment data [@term:segment-data] which does not correspond to an intersection.  
<!--
@originalProject already imputes speed limits for intersections.
-->

#### WP: route: manual selection of pre-defined GPS coordinates [GPS-manual-predef]  
WP: [route: GPS-manual-predef]  
Dependency: None
[route: GPS-manual-predef]     -> [model: safety_score] -> [display score]  
**Description:**   
<!--
TODO: fill in from roadmap, critical path  
-->

#### WP: route: manual selection of generic GPS coordinates [GPS-manual-generic]  
WP: [route: GPS-manual-generic]  
Dependency: GPS-manual-predef
[route: GPS-manual-generic]    -> [model: safety_score] -> [display score]  
<!--
**Description:**   
TODO: fill in from roadmap, critical path  
-->

#### WP: route: automatic selection of generic GPS coordinates [GPS-automatic-generic]  
WP: [route: GPS-automatic-generic]  
Dependency: GPS-manual-predef
[route: GPS-automatic-generic] -> [model: safety_score] -> [display score]  
<!--
**Description:**   
TODO: fill in from roadmap, critical path  
-->

#### WP: route: implement map as output interface [UI-nointer-GPS-generic]  
WP: [route: UI-nointer-GPS-generic]  
Dependency: GPS-manual-predef
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
Dependency: [UI-nointer-GPS-generic]
[route: GPS\*] -> [gui: UI-\*-GPS-generic] -> [gui: UI-map-safety_score-partial]  
**Description:**   
Show the safety score for partial route on the map.

#### WP: route: overlay score on map [UI-map-safety_score-total]  
WP: [route: UI-map-safety_score-total]  
Dependency: [UI-nointer-GPS-generic]
[route: GPS\*] -> [gui: UI-\*-GPS-generic] -> [gui: UI-map-safety_score-total]  
**Description:**   
Show the safety score for entire route on the map.  

#### WP: route: total score [safety_score-total]  
WP: [route: safety_score-total]  
Dependency: GPS-manual-predef
[route: GPS-\*]     -> [model: safety_score-total] -> [display total score]  
**Description:**   
calculate safety score for entire route  

#### WP: route: recommend best route [UI-recommend-simple]  
WP: [route: UI-recommend-simple]  
Dependency: safety_score-total
[route,several: GPS-\*]     -> [model: safety_score-total,several] -> [model: safety_score-total] -> [display best total score out of several (i.e. find safest route out of multiple routes)]  
**Description:**   
retrieve multiple routes from third-party mapping service, calculate total score (safety_score-total) for each one, recommend the safest  

#### WP: route: partial score [safety_score-partial]  
WP: [route: safety_score-partial]  
Dependency: GPS-manual-predef
[route: GPS-\*]     -> [model: safety_score-partial] -> [display partial scores]  
**Description:**   
calculate safety score for each route segment

#### WP: route: mix routes [UI-recommend-complex]  
WP: [route: UI-recommend-complex]  
Dependency: safety_score-partial
[route,several: GPS-\*]     -> [model: safety_score-partial,several] -> [model: safety_score-partial] -> [display best combined scores out of several (i.e. combine safest sections of multiple routes into one route)]   
**Description:**   
retrieve multiple routes from third-party mapping service, calculate segment scores (safety_score-partial) for each one, combine lowest scores to create a safest route (i.e. combine safest sections of multiple routes into one route)]  



### Roadmap

**Strategy**: Each stage should result in a usable product while successively improving usability

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

<!-- this syntax works when rendering the pdf with
pandoc -s --data-dir=2007_and_2010/Thesis/Master_Document_Template.dotx report.md -o report.pdf --mathjax -t context
Mathjax:  
$$
\\( ax^2 + \sqrt{bx} + c = 0 \\)
$$
-->

<!--
Markdown Reminders:

# citation links for pandoc
citation inline:      [@linkName]
citation definition:  [@linkName]: link_url

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
