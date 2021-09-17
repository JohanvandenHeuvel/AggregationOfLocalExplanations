import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('score_table.csv')

fig, axs = plt.subplots(3, 1, figsize=(10, 10))
for i, score in enumerate(['insert', 'delete', 'irof']):
    ax = axs[i]
   
    x = np.linspace(0, 1, 100)
    ax.errorbar(df['method'],
                df['{} mean'.format(score)],
                df['{} std'.format(score)],
    )
    ax.set_ylabel(score)
    ax.set_xticklabels(df['method'], rotation=45)

plt.tight_layout()
plt.show()
fig.savefig('{}.png'.format("plot"), dpi=fig.dpi)
                
    
    
    
