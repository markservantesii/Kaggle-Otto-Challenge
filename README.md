# Summary
The files contained in this repo were used to generate the classification models and predictions while participating in the Otto Group Product Classification Challenge on Kaggle. The code is messy, poorly organized, and lacks proper documentation, but it managed to land me a finishing spot in the top 10% (286/3514). Looking back, I wish I would have spent some time early on to plan and organize my approach. Unfortunately, I was still learning Python and  various scikit-learn tools during the competition. As the deadline drew near, I scrambled to improve my models; I added to existing code as a means to save time rather than rewriting code from scratch. By the end, I had accumulated a hodge-podge of franken-code. I'm keeping the files as they are as a reminder  to myself that organization and clean code is important, but also to remember what I was able to achieve with limited Python knowledge and experience and how far I've come since then.

This write-up is still a work in progress. I hope to update this in the future to fully document my work. In the meantime, I provide a brief description:

# What I used
I performed all data cleaning and visualziation using R, and built all models in Python. The final model was an ensemble of Extra Trees, XGBoost, and Neural Networks. I used scikit-learn for Extra Trees, xgb with a wrapper (following scikit-learn API), and keras for Neural Networks. To summarize:

- **Python Modules:** sklearn, xgb, keras
- **Models:** ExtraTreesClassifier, xgboost, neural networks

I initially was using the lasagne package for building neural networks. There was a bit of a learning curve for me using lasagne, but eventually I got the hang of it. Eventually I discovered keras which was incredibly easy to use and more intuitive. They performed approximately the same, so I decided to stick to keras for the ease-of-use.

Random Forests, Gradient Boosted Claffiers, Logistic Regression, and Topic Models (lda module) were also tested, but performed poorly.

## The Files
- **keras_otto.py:** A script that generated the neural networks.
- **rf_otto.py:** A script that originally generated Random Forest models. Currently used to generate Extra Trees and XGBoost models.
- **otto_file_combine.py:** A script that averages the predictions from multiple files
- **otto_funs.py:** A file containing various utility functions I used while building models.
- **XGBoostClassifier2.py:** A wrapper written by [Henning Sperr](https://www.github.com/hsperr) for the xgb module. It follows the scikit-learn API

# Model Tuning
I struggled early on with tuning my models. I didn't understand the function of each parameter, and I was still learning the theory behind the models. My approach was to iteratively build many different models and the best performing ones as a starting point. I manually tweaked these models with trial and error, using some of the knowledge I had. I figured that this approach would lead to solution faster than if I started by learning the theory. My mindset was: "*What can I accomplish with limited knowledge?*"

# The Ensemble
Learning and implementing model stacking/ensembling techniques was one of the more interesting parts of this competition. I find this to be an extremely fascinating topic, and I wish I had time during the competition to explore this in more detail. I experimented with a variety of ensemble techniques including simple Averaging, Weighted Averaging, and training second-level classifiers (Random Forests, Logistic Regression, and Neural Networks) to find the best way to combine models. In the end, I simply averaged the predictions produced from the models. This performed well enough given the time constraints, but I'm sure I could have squeezed out extra improvement by taking the time to properly build a second-level classifier.
