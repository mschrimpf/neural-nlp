
# The neural architecture of language: Integrative modeling converges on predictive processing 

Code accompanying the paper 
[The neural architecture of language: Integrative modeling converges on predictive processing](https://www.pnas.org/content/118/45/e2105646118) by Schrimpf, Blank, Tuckute, Kauf, Hosseini, Kanwisher, Tenenbaum, and Fedorenko.

Large-scale evaluation of neural network language models 
as predictive models of human language processing.
This pipeline compares dozens of state-of-the-art models and 4 human datasets (3 neural, 1 behavioral).
It builds on the [Brain-Score](www.Brain-Score.org) framework and can easily be extended with new models and datasets.

## Installation
```bash
git clone https://github.com/mschrimpf/neural-nlp.git
cd neural-nlp
pip install -e .
```
You might have to install nltk by hand / with conda.

## Run
To score gpt2-xl on the Blank2014fROI-encoding benchmark:

```bash
python neural_nlp run --model gpt2-xl --benchmark Blank2014fROI-encoding --log_level DEBUG
```

Other available benchmarks are e.g. Pereira2018-encoding (takes a while to compute), and Fedorenko2016v3-encoding.

You can also specify different models to run -- 
note that some of them require additional download of weights (run `ressources/setup.sh` for automated download).

## Precomputed scores
Scores for models run on the neural, behavioral, and computational-task benchmarks are also available, see the [`precomputed-scores.csv`](precomputed-scores.csv) file.
You can re-create the figures in the paper using the [`analyze`](neural_nlp/analyze/__main__.py) scripts.

## Citation
If you use this work, please cite
```
@article{Schrimpf2021,
	author = {Schrimpf, Martin and Blank, Idan and Tuckute, Greta and Kauf, Carina and Hosseini, Eghbal A. and Kanwisher, Nancy and Tenenbaum, Joshua and Fedorenko, Evelina},
	title = {The neural architecture of language: Integrative modeling converges on predictive processing},
	year = {2021},
	journal = {Proceedings of the National Academy of Sciences},
	url = {https://www.pnas.org/content/118/45/e2105646118}
}

```
