# Contributing

## Caveat

This is a quick-and-dirty prototype; speed of implementation was chosen over best practices. Some of the required improvements are tracked via the GH issues for each project; many are still in the author's head, waiting to get documented in their spare time :-)

In particular, proper testing was put off in favour of quickly iterating on a framework with allowed for both model training, validation, and delivery; some of which require manual intervention and are difficult to test, and others which are "good enough" covered by integration testing (and testing models is still a hot topic to this day).

When this project was created back in 2017, XGBoost was all the rage, [tensorflow was in early stages](https://en.wikipedia.org/wiki/TensorFlow), and [kubeflow did not yet exist](https://en.wikipedia.org/wiki/Kubeflow), hyperparameter tuning was a new scikit-learn feature (instead of manually writing for-loops :-) ), etc, so this codebase became quite outdated even within 2 years.

Nevertheless, as a bare-bones no-frills framework, it touches on most of the things needed to manage the model tuning and delivery lifecycle, which makes it a great learning tool.

## Developing

The main code is under the [./code/](./code) directory:

* [model.py](https://github.com/YoinkBird/cyclesafe/blob/3890efa32538505fcadbbba2c4ad238599944856/code/model.py#L1506) can be run standalone or imported as a module.
* The remaining modules are explained in the section ["Modeling Application" in the system documentation](https://github.com/YoinkBird/cyclesafe/blob/report/report/report.md#modeling-application).

Data formats are defined in the section ["Data Formats" of the system documentation](https://github.com/YoinkBird/cyclesafe/blob/report/report/report.md#data-formats); for now, this unfortunately also serves as the API documenation.

## Testing

During development, [model.py](https://github.com/YoinkBird/cyclesafe/blob/3890efa32538505fcadbbba2c4ad238599944856/code/model.py#L1506) can be run standalone to manually verify functionality of functions. Unit tests to follow; see rationale at end of this section.

During verification, interface testing is run via the 
[cyclesafe server orchestration script](https://github.com/YoinkBird/cyclesafe_server/blob/60c8ffaea646c9f680458f03c5ddef7f055a65df/setup.sh#L187); this is implemented using `curl` and json files defining requests and expected responses (similar to [expect testing](https://en.wikipedia.org/wiki/Expect); this is a quick and dirty way to verify system functionality at a high level; see rationale at the end of this section.

Rationale: During development, the trade-off was made to rely on integration testing to prevent backsliding, with unit tests to be added later where possible. This is both due to the nature of testing ML models (results aren't always predictable) and the prototype nature of the project (i.e. most of the code is expected to be refactored anyway).

## Related Projects

Hard fork of https://github.com/YoinkBird/dataMiningFinal , which was the initial project to explore bicycle safety and spawned this entire project.
