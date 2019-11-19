import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import median_absolute_deviation, kruskal

dt_scores = [0.96920738, 0.96653356, 0.96292071, 0.97071687,0.96691386, 0.96995627, 0.96957597, 0.96482221, 0.96709776, 0.96442838]
kn_scores = [0.98403345, 0.98573873, 0.98554858, 0.98725994, 0.98478798, 0.98630918, 0.98573873, 0.98668948, 0.98611639, 0.98516264]
rf_scores = [0.97138001, 0.97772432, 0.96500273, 0.97610468, 0.96657374, 0.97375539, 0.97134617, 0.96987115, 0.97408799, 0.96811699]
ml_scores = [0.98213267, 0.98497813, 0.98326678, 0.98459783, 0.98155543, 0.98364708, 0.98345693, 0.98326678, 0.98212248, 0.98345064]
he_scores = [0.97052671, 0.97569889, 0.97373481, 0.97966792, 0.97529367, 0.97607608, 0.97530183, 0.97336566, 0.97286408, 0.97207677]
me_scores = [0.98308306, 0.98630918, 0.98307663, 0.98745009, 0.98440768, 0.98478798, 0.98402738, 0.98478798, 0.98402434, 0.98440175]
full_scores = dt_scores + kn_scores + rf_scores + ml_scores + he_scores + me_scores

box_plot_data = [dt_scores, kn_scores, rf_scores, ml_scores, he_scores, me_scores]
labels = ['DecTree','KNN','RandForest','MLP', 'HetEns', 'MLPEns']
plt.boxplot(box_plot_data, patch_artist=True, labels=labels)
plt.show()

mean = np.mean(full_scores)
std = np.std(full_scores)
mad = median_absolute_deviation(full_scores)
print(f"Mean: {mean}\nDeviation: {std}\nMedian Deviation: {mad}\n")
    
stat, p = kruskal(dt_scores, kn_scores, rf_scores, ml_scores, he_scores, me_scores)
print('Kruskal-Wallis Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')
