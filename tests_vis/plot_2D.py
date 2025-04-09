import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import register_cmap

plt.rcParams.update({'font.size': 16})

# use custom 2-color palette 
# https://xkcd.com/color/rgb/
pos = [0.0,1.0]
colors=['#363737', '#8c000f']
cmap = LinearSegmentedColormap.from_list("", list(zip(pos, colors)), 2)
register_cmap("my_cmap", cmap)
my_palette= sns.color_palette("my_cmap",n_colors=2)
sns.set_palette(my_palette)

colnames = [ "en", "x", "y" ]
dold=pd.read_csv('old_2d.csv', names=colnames, header=None) 
dold=dold.drop(columns=['en'])
dold = dold.assign(signal=pd.Series(1., index=dold.index, dtype='float32').values)

dnew=pd.read_csv('new_2d.csv', names=colnames, header=None) 
dnew=dnew.drop(columns=['en'])
dnew = dnew.assign(signal=pd.Series(0., index=dnew.index, dtype='float32').values)

df = pd.concat([dold,dnew], ignore_index=True)

g = sns.PairGrid(df, hue='signal', corner=False)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.scatterplot, alpha=0.2)
#g.map_lower(sns.scatterplot, alpha=0.2)
plt.show()
plt.savefig("plot_2D.pdf", bbox_inches='tight')
plt.savefig("plot_2D.png", bbox_inches='tight', dpi=200)
