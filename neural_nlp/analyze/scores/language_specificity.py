import numpy as np

from neural_nlp.analyze.scores import collect_scores, models, average_adjacent, choose_best_scores, align_scores


def Fedorenko2016(best_layer=True):
    scores_lang = collect_scores(benchmark='Fedorenko2016v3-encoding', models=models, normalize=True)
    scores_nonlang = collect_scores(benchmark='Fedorenko2016v3nonlang-encoding', models=models, normalize=True)
    scores_lang, scores_nonlang = average_adjacent(scores_lang).dropna(), average_adjacent(scores_nonlang).dropna()
    if best_layer:
        scores_lang, scores_nonlang = choose_best_scores(scores_lang), choose_best_scores(scores_nonlang)
    scores_lang, scores_nonlang = align_scores(scores_lang, scores_nonlang,
                                               identifier_set=['model'] if best_layer else ['model', 'layer'])
    diffs = scores_lang['score'] - scores_nonlang['score']
    print(f"median drop {np.nanmedian(diffs)}+-{np.std(diffs)}")
    mults = scores_lang['score'] / scores_nonlang['score']
    print(f"median multiplicative drop {np.nanmedian(mults)}+-{np.std(mults)}")


if __name__ == '__main__':
    Fedorenko2016()
