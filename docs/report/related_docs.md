<!-- ################################################################################ -->
<hr/>  
crisp-dm:
https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=9&cad=rja&uact=8&ved=0ahUKEwjqwtuUiY3XAhUm44MKHTpKAjsQFghVMAg&url=http%3A%2F%2Fwww-staff.it.uts.edu.au%2F~paulk%2Fteaching%2Fdmkdd%2Fass2%2Freadings%2Fmethodology%2FCRISPWP-0800.pdf&usg=AOvVaw2kGc_WEje8Ejz8Cj-h9j4M 

web version: http://www.sv-europe.com/crisp-dm-methodology/

<!-- ################################################################################ -->
<hr/>  
https://dspace.cvut.cz/handle/10467/69704  
https://dspace.cvut.cz/bitstream/handle/10467/69704/F8-DP-2017-Maurerova-Veronika-thesis.pdf?sequence=-1&isAllowed=y  

Methods for <subject>  
* map of crashes, simply avoid crashes
  * list issues with this approach
* ML
  * statistical model, list biases etc


data preparation  
feature space  

4. 
provide overview of ML methods
e.g. trees

data prep
no reduction other than dropna, dataset already very limited

sparse

base models

libraries used

6 . implementation  
current application is flexible enough to constantly update without supervision (in worst case)  
as more data sources obtained, should revisit 
keep the baseline in mind  
continuously obtain csv  
find other data sources  

<!-- ################################################################################ -->
<hr/>  
https://arxiv.org/abs/1710.08464

focus on interpretability

concepts for future research:  
model understanding (gulf of evaluation)  
model interactions (gulf of execution)  

framework:  
ML component and explanation component => similar to  model + map

see 'inference component implementation' 

see also section 4 'evaluation ...' and 'presenting ...'

<!-- ################################################################################ -->
<hr/>  
source for breakdown of severity:
http://ftp.dot.state.tx.us/pub/txdot-info/aus/bicycle-master-plan/crashes.pdf

<!-- ################################################################################ -->
<hr/>
<!-- ################################################################################ -->
<hr/>
<!-- ################################################################################ -->
<hr/>


misc papers:  

**which tree to use** 
src: http://icml2008.cs.helsinki.fi/papers/632.pdf
boosted decision trees perform exceptionally well when dimensionality is low.  In this study boosted trees are the method of choice for up to about 4000 dimensions. Above that, random forests have the best overall performance.

https://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml06.pdf

There is less than a 20% chance that a method other than boosted trees, random forests, and bagged trees would rank in the top three, and no chance (0.0%) that another method would rank 1st — it appears to be a clean sweep for ensembles of trees.
