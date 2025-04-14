ARISA MLOps Final Graded Assignment
Termin: 10 maja 2025 23:59

Instrukcje

Assignment overview
In this assignment you will take all of the theory and architecture developments we have worked on from lessons 1 through 5 and apply them to a new dataset. 

Instructions
Review the course content of lessons 1 to 5, as well as the code and README.md in https://github.com/clausagerskov/ARISA-MLOps and the architectural diagrams.
Create a new, public repository on Github (do not fork the course repository, but do use it as a guide for constructing your own).
Find a new dataset on Kaggle (https://www.kaggle.com/datasets) and implement the same architecture on that new dataset, while referring back to the first four levels of the MLOps Maturity Assessment (https://marvelousmlops.substack.com/p/mlops-maturity-assessment). 
The existing preproc and train code will not work on your new dataset, so refactoring is needed.
Remember the clean and reproducible code guidelines of the first lesson, e.g. working in appropriately named branches, protecting the main branch, code linting workflow, making pull requests, etc.
It is highly recommended to choose a small tabular dataset, either classification or regression, and no more than a couple of megabytes to both keep costs low and reduce time needed to debug pipelines.
Regarding MLFlow hosting, it is also recommended to host the metadata database locally on the Codespaces instance to reduce costs. 
For the artifact store, any cloud storage can be used. We have used S3 in this course, but any is okay, and there are guides on the internet as well for setting up MLFlow to connect to google drive.
You are allowing to change any part of the infrastructure, but if you decide to do so, please make an updated diagram and specify what you changed and why.

After implementing the infrastructure and successfully running the pipelines on the new dataset, write a report where you describe, for each point in the first four levels of the MLOps Maturity Assessment, how that part of is implemented in your code and motivation for that solution.

Remember, this assignment is graded.

Full marks will be given for a link to a new repository with successful pipeline runs on a new data (i.e. not Titanic), and a report with descriptions and motivations.

Finally, remember the last slide of the first presentation:
LMGTFY: https://letmegooglethat.com/ and 
RTFM: https://medium.com/@shivam2003/rtfm-a-guide-to-not-just-surviving-but-thriving-as-a-developer-42fa1d3ff546
and if you can't figure things out by yourself: https://medium.com/@katiebrouwers/why-rubber-ducking-is-one-of-your-greatest-resources-as-a-developer-99ac0ee5b70a