import matplotlib.pyplot as plt
import pandas as pd



df = pd.read_csv('full_grain_data.csv')
feature_cols = [f'band_{i+1}' for i in range(256)]
for grain_type in df['grain_type'].unique():
    sample = df[df['grain_type'] == grain_type].iloc[0]
    plt.plot(sample[feature_cols], label=grain_type.split('/')[-1])
plt.xlabel('Band')
plt.ylabel('Reflectance')
plt.legend()
plt.show()