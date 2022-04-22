import json
from sentiment_analysis import sentiment_analysis_func

if __name__ == '__main__':
    model_names = ["torch_RNN", "torch_GRU", "torch_LSTM", "GRU", "LSTM"]
    results = {}

    imdb_results_acc = {}
    imdb_results_time = {}
    for model in model_names:
        acc, run_time = sentiment_analysis_func(model)
        imdb_results_acc[model] = acc
        imdb_results_time[model] = run_time

    results["acc"] = imdb_results_acc
    results["run_time"] = imdb_results_time
    with open('results.json', 'w') as fp:
        json.dump(results, fp)
