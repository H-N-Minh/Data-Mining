import numpy as np
import matplotlib.pyplot as plt


def graphPlotter(normal_points, anomalies, title, filename):
    plt.figure(figsize=(8, 8))
    plt.scatter(normal_points[:, 0], normal_points[:, 1], c='green', label='Normal Points', alpha=0.6)
    plt.scatter(anomalies[:, 0], anomalies[:, 1], c='red', label='Anomalies', alpha=0.8)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(np.arange(0, 101, 10))
    plt.yticks(np.arange(0, 101, 10))
    plt.minorticks_on()
    plt.gca().set_xticks(np.arange(0, 101, 5), minor=True)
    plt.gca().set_yticks(np.arange(0, 101, 5), minor=True)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.savefig(filename)
    plt.close()

# Generate dataset for Type 1: 3 clusters and 3 anomalies points
cluster1 = np.random.normal(loc=[20, 30], scale=6, size=(100, 2))
cluster2 = np.random.normal(loc=[50, 80], scale=5, size=(100, 2))
cluster3 = np.random.normal(loc=[80, 50], scale=7, size=(130, 2))
normal_points_type1 = np.vstack([cluster1, cluster2, cluster3])
anomalies_type1 = np.array([[10, 90], [90, 10], [50, 10]])

# Generate dataset for Type 2: Regional Anomalies
normal_points_type2 = np.random.normal(loc=[50, 50], scale=7, size=(500, 2))
region1 = np.random.normal(loc=[20, 80], scale=1, size=(5, 2))
region2 = np.random.normal(loc=[90, 10], scale=2, size=(5, 2))
region3 = np.random.normal(loc=[10, 20], scale=2, size=(5, 2))
anomalies_type2 = np.vstack([region1, region2, region3])


graphPlotter(normal_points_type1, anomalies_type1, 'Type 1: Point Anomalies', 'point_anomalies.png')
graphPlotter(normal_points_type2, anomalies_type2, 'Type 2: Regional Anomalies', 'regional_anomalies.png')