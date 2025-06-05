import matplotlib.pyplot as plt
import numpy as np
import copy

def barplot_with_errors_test_elements(heatmaps, offset=[14,162,79,42], n_windows = 8, color='steelblue', set_limits_y = True,
                                      title='Test all elements', show=True, save=True, path='test_elements.png', limit_min='', limit_max=''):
    distributions = []
    if set_limits_y:
        if limit_max == '':
            limit_max = np.max(heatmaps)
        if limit_min == '':
            limit_min = np.min(heatmaps)
    for hm in heatmaps:
        single_d = []
        for i in range(n_windows):
            single_d.append(hm[i*offset[i]])
        single_d = np.array(single_d)
        distributions.append(single_d)
    dist_mean = np.mean(distributions, axis=0)
    dist_std = np.std(distributions, axis=0)
    x = np.arange(1,n_windows+1)
    fig, ax = plt.subplots()
    ax.errorbar(x, dist_mean, yerr=dist_std, fmt="o", color=color)
    if set_limits_y:
        ax.set_ylim(limit_min, limit_max)
    ax.set_xlabel("Shap Window")
    ax.set_ylabel("Shap Value")
    ax.set_title(title)
    ax.set_xticks([1,2,3,4])
    ax.set_xticklabels(['Profilo Finanziario', 'Stato Patrimoniale', 'Conto Economico', 'Indici'])
    #labels = [item.get_text() for item in plt.get_xticklabels()]
    #plt.xticks(['Profilo Finanziario', 'Stato Patrimoniale', 'Conto Economico', 'Indici'])
    #plt.xticks(labels)
    if save:
        fig.savefig(path)
    if show:
        fig.show()

def barplot_test_elements(heatmaps, offset=[14,162,79,42], n_windows = 8, color='steelblue', set_limits_y = True,
                                      title='Test all elements', show=True, save=True, path='test_elements.png', limit_min='', limit_max=''):
    distributions = []
    if set_limits_y:
        if limit_max == '':
            limit_max = np.max(heatmaps)
        if limit_min == '':
            limit_min = np.min(heatmaps)
    for hm in heatmaps:
        single_d = []
        for i in range(n_windows):
            single_d.append(hm[i*offset[i]])
        single_d = np.array(single_d)
        distributions.append(single_d)
    dist_mean = np.mean(distributions, axis=0)
    #x = np.arange(1,9)
    x = np.arange(1, n_windows+1)
    fig, ax = plt.subplots()
    ax.bar(x, dist_mean, color=color, width=0.3)
    if set_limits_y:
        ax.set_ylim(limit_min, limit_max)
        #plt.ylim(limit_min, limit_max)
    ax.set_xlabel("Shap Window")
    ax.set_ylabel("Shap Value")
    ax.set_title(title)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['Profilo Finanziario', 'Stato Patrimoniale', 'Conto Economico', 'Indici'])
    if save:
        fig.savefig(path)
    if show:
        fig.show()

def barplot_classes_comparison(heatmaps, real_labels, offset=[14,162,79,42], n_windows=8, colors=['seagreen', 'tomato', 'steelblue'],
                               set_limits_y=True, title='Classes comparison', show=True, save=True, path='test_elements.png',
                               n_classes=3, width=0.25, class_labels = ['Decrese', 'Increase'], limit_min='', limit_max=''):
    distributions = []
    for i in range(n_classes):
        distribution_int = copy.deepcopy([])
        distributions.append(distribution_int)
    if set_limits_y:
        if limit_max == '':
            limit_max = np.max(heatmaps)
        if limit_min == '':
            limit_min = np.min(heatmaps)
    for i in range(len(heatmaps)):
        hm = heatmaps[i]
        rl = real_labels[i]
        single_d = []
        for i in range(n_windows):
            single_d.append(hm[i * offset[i]])
        single_d = np.array(single_d)
        distributions[rl].append(single_d)
    array_means = []
    for i in range(n_classes):
        dist_mean = np.mean(distributions[i], axis=0)
        array_means.append(dist_mean)
    x = np.arange(1, n_windows+1)
    fig, ax = plt.subplots()
    for i in range(n_classes):
        offset_multibar = width*i
        ax.bar(x+offset_multibar, array_means[i], width=width, color=colors[i], alpha=0.5, edgecolor='black', label = class_labels[i])
    if set_limits_y:
        ax.set_ylim(limit_min, limit_max)
    ax.set_xlabel("Shap Window")
    ax.set_ylabel("Shap Value")
    ax.legend()
    ax.set_title(title)
    ax.set_xticks([1,2,3,4])
    ax.set_xticklabels(['Financial Profile', 'Balance Sheet', 'Income Statement', 'Ratio Analysis'])
    if save:
        fig.savefig(path)
    if show:
        fig.show()

def barplot_corrects_errors_comparison(heatmaps, real_labels, predictions, offset=[14,162,79,42], n_windows=8, colors=['seagreen', 'tomato', 'steelblue'],
                               set_limits_y=True, title='Classes comparison', show=True, save=True, path='test_elements.png',
                               n_classes=3, width=0.25, class_labels = ['Decrese', 'Increase'], limit_min='', limit_max=''):
    distributions = []
    for i in range(n_classes):
        distribution_int = copy.deepcopy([])
        distributions.append(distribution_int)
    if set_limits_y:
        if limit_max == '':
            limit_max = np.max(heatmaps)
        if limit_min == '':
            limit_min = np.min(heatmaps)
    for i in range(len(heatmaps)):
        p = predictions[i]
        hm = heatmaps[i]
        rl = real_labels[i]
        single_d = []
        for i in range(n_windows):
            single_d.append(hm[i * offset[i]])
        single_d = np.array(single_d)
        if rl == p:
            distributions[0].append(single_d)
        else:
            distributions[1].append(single_d)
    array_means = []
    for i in range(n_classes):
        dist_mean = np.mean(distributions[i], axis=0)
        array_means.append(dist_mean)
    x = np.arange(1, n_windows+1)
    fig, ax = plt.subplots()
    for i in range(n_classes):
        offset_multibar = width*i
        ax.bar(x+offset_multibar, array_means[i], width=width, color=colors[i], label = class_labels[i])
    if set_limits_y:
        ax.set_ylim(limit_min, limit_max)
    ax.set_xlabel("Shap Window")
    ax.set_ylabel("Shap Value")
    ax.legend()
    ax.set_title(title)
    ax.set_xticks([1,2,3,4])
    ax.set_xticklabels(['Profilo Finanziario', 'Stato Patrimoniale', 'Conto Economico', 'Indici'])
    if save:
        fig.savefig(path)
    if show:
        fig.show()

def barplot_for_financial_statements(array_plot, p_colors, normalize=False, normalized_values=[14, 162, 79, 42],
                                     save=False, show=True, filename='plot.png',
                                     ylabels=['Financial Profile', 'Balance Sheet', 'Income Statement', 'Ratio Analysis']):
    fig, ax = plt.subplots()
    width = 0.3
    if normalize:
        ax.bar([0], array_plot[0] / normalized_values[0], width=width, color=p_colors[0], edgecolor='black')
        ax.bar([1], array_plot[1] / normalized_values[1], width=width, color=p_colors[1], edgecolor='black')
        ax.bar([2], array_plot[2] / normalized_values[2], width=width, color=p_colors[2], edgecolor='black')
        ax.bar([3], array_plot[3] / normalized_values[3], width=width, color=p_colors[3], edgecolor='black')
    else:
        ax.bar([0], array_plot[0], width=width, color=p_colors[0], edgecolor='black')
        ax.bar([1], array_plot[1], width=width, color=p_colors[1], edgecolor='black')
        ax.bar([2], array_plot[2], width=width, color=p_colors[2], edgecolor='black')
        ax.bar([3], array_plot[3], width=width, color=p_colors[3], edgecolor='black')
    ax.set_xlabel("Part of the finance statements")
    ax.set_ylabel("Frequency")
    ax.set_title("Frequency in top 10 features")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(ylabels)
    if save:
        fig.savefig(filename)
    if show:
        fig.show()