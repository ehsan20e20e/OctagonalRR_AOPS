"""
The code for the research presented in the paper titled "Inverse design of plasmon-based nonlinear octagonal resonators enabled by deep neural networks"


This code is corresponding to the Forward Deep Neural Network (DNN) section of the article.
This code regenerates the Fig S6a-n of the Supplementary Information of the paper.
Please cite the paper in any publication using this code.
"""
import matplotlib.pyplot as plt 
import pandas as pd 
from matplotlib.font_manager import FontProperties

Loss = pd.read_csv("OctagonalRR_AOPS_Fig_S06.csv")
Loss=Loss.to_numpy()

# fig S6m
plt.plot(Loss[:,0],Loss[:,6], linewidth=2, color=((255/255, 127/255, 14/255))) 
plt.plot(Loss[:,0],Loss[:,8], linewidth=2, color=((0/255, 114/255, 189/255)), linestyle='--') 
plt.title('Juxtaposition of 13 hidden layers NN and FDTD', fontname='Times New Roman', fontsize=16, loc='center')
plt.xlabel('Wavelength(nm)', fontname='Times New Roman', fontsize=18)
plt.ylabel('Transmission', fontname='Times New Roman', fontsize=18)
plt.xticks(fontfamily='Times New Roman', fontsize=16)
plt.yticks(fontfamily='Times New Roman', fontsize=16)
font_prop = FontProperties(family="Times New Roman", size=16)
plt.legend(['FDTD', 'DL'], prop=font_prop)
plt.savefig("OctagonalRR_AOPS/P3_FigureS6m.png", format="png", dpi=600, bbox_inches="tight")
plt.show()

# fig S6k
plt.plot(Loss[:,0],Loss[:,6], linewidth=2, color=((255/255, 127/255, 14/255))) 
plt.plot(Loss[:,0],Loss[:,10], linewidth=2, color=((0/255, 114/255, 189/255)), linestyle='--') 
plt.title('Juxtaposition of 11 hidden layers NN and FDTD', fontname='Times New Roman', fontsize=16, loc='center')
plt.xlabel('Wavelength(nm)', fontname='Times New Roman', fontsize=18)
plt.ylabel('Transmission', fontname='Times New Roman', fontsize=18)
plt.xticks(fontfamily='Times New Roman', fontsize=16)
plt.yticks(fontfamily='Times New Roman', fontsize=16)
font_prop = FontProperties(family="Times New Roman", size=16)
plt.legend(['FDTD', 'DL'], prop=font_prop)
plt.savefig("OctagonalRR_AOPS/P3_FigureS6k.png", format="png", dpi=600, bbox_inches="tight")
plt.show()

# fig S6i
plt.plot(Loss[:,0],Loss[:,6], linewidth=2, color=((255/255, 127/255, 14/255))) 
plt.plot(Loss[:,0],Loss[:,12], linewidth=2, color=((0/255, 114/255, 189/255)), linestyle='--') 
plt.title('Juxtaposition of 9 hidden layers NN and FDTD', fontname='Times New Roman', fontsize=16, loc='center')
plt.xlabel('Wavelength(nm)', fontname='Times New Roman', fontsize=18)
plt.ylabel('Transmission', fontname='Times New Roman', fontsize=18)
plt.xticks(fontfamily='Times New Roman', fontsize=16)
plt.yticks(fontfamily='Times New Roman', fontsize=16)
font_prop = FontProperties(family="Times New Roman", size=16)
plt.legend(['FDTD', 'DL'], prop=font_prop)
plt.savefig("OctagonalRR_AOPS/P3_FigureS6i.png", format="png", dpi=600, bbox_inches="tight")
plt.show()

# fig S6g
plt.plot(Loss[:,0],Loss[:,6], linewidth=2, color=((255/255, 127/255, 14/255))) 
plt.plot(Loss[:,0],Loss[:,14], linewidth=2, color=((0/255, 114/255, 189/255)), linestyle='--') 
plt.title('Juxtaposition of 7 hidden layers NN and FDTD', fontname='Times New Roman', fontsize=16, loc='center')
plt.xlabel('Wavelength(nm)', fontname='Times New Roman', fontsize=18)
plt.ylabel('Transmission', fontname='Times New Roman', fontsize=18)
plt.xticks(fontfamily='Times New Roman', fontsize=16)
plt.yticks(fontfamily='Times New Roman', fontsize=16)
font_prop = FontProperties(family="Times New Roman", size=16)
plt.legend(['FDTD', 'DL'], prop=font_prop)
plt.savefig("OctagonalRR_AOPS/P3_FigureS6g.png", format="png", dpi=600, bbox_inches="tight")
plt.show()

# fig S6e
plt.plot(Loss[:,0],Loss[:,6], linewidth=2, color=((255/255, 127/255, 14/255))) 
plt.plot(Loss[:,0],Loss[:,16], linewidth=2, color=((0/255, 114/255, 189/255)), linestyle='--') 
plt.title('Juxtaposition of 5 hidden layers NN and FDTD', fontname='Times New Roman', fontsize=16, loc='center')
plt.xlabel('Wavelength(nm)', fontname='Times New Roman', fontsize=18)
plt.ylabel('Transmission', fontname='Times New Roman', fontsize=18)
plt.xticks(fontfamily='Times New Roman', fontsize=16)
plt.yticks(fontfamily='Times New Roman', fontsize=16)
font_prop = FontProperties(family="Times New Roman", size=16)
plt.legend(['FDTD', 'DL'], prop=font_prop)
plt.savefig("OctagonalRR_AOPS/P3_FigureS6e.png", format="png", dpi=600, bbox_inches="tight")
plt.show()

# fig S6c
plt.plot(Loss[:,0],Loss[:,6], linewidth=2, color=((255/255, 127/255, 14/255))) 
plt.plot(Loss[:,0],Loss[:,18], linewidth=2, color=((0/255, 114/255, 189/255)), linestyle='--') 
plt.title('Juxtaposition of 3 hidden layers NN and FDTD', fontname='Times New Roman', fontsize=16, loc='center')
plt.xlabel('Wavelength(nm)', fontname='Times New Roman', fontsize=18)
plt.ylabel('Transmission', fontname='Times New Roman', fontsize=18)
plt.xticks(fontfamily='Times New Roman', fontsize=16)
plt.yticks(fontfamily='Times New Roman', fontsize=16)
font_prop = FontProperties(family="Times New Roman", size=16)
plt.legend(['FDTD', 'DL'], prop=font_prop)
plt.savefig("OctagonalRR_AOPS/P3_FigureS6c.png", format="png", dpi=600, bbox_inches="tight")
plt.show()

# fig S6a
plt.plot(Loss[:,0],Loss[:,6], linewidth=2, color=((255/255, 127/255, 14/255))) 
plt.plot(Loss[:,0],Loss[:,20], linewidth=2, color=((0/255, 114/255, 189/255)), linestyle='--') 
plt.title('Juxtaposition of 1 hidden layer NN and FDTD', fontname='Times New Roman', fontsize=16, loc='center')
plt.xlabel('Wavelength(nm)', fontname='Times New Roman', fontsize=18)
plt.ylabel('Transmission', fontname='Times New Roman', fontsize=18)
plt.xticks(fontfamily='Times New Roman', fontsize=16)
plt.yticks(fontfamily='Times New Roman', fontsize=16)
font_prop = FontProperties(family="Times New Roman", size=16)
plt.legend(['FDTD', 'DL'], prop=font_prop)
plt.savefig("OctagonalRR_AOPS/P3_FigureS6a.png", format="png", dpi=600, bbox_inches="tight")
plt.show()

# fig S6n
plt.plot(Loss[:,0],Loss[:,7], linewidth=2, color=((255/255, 127/255, 14/255))) 
plt.plot(Loss[:,0],Loss[:,9], linewidth=2, color=((0/255, 114/255, 189/255)), linestyle='--') 
plt.title('Juxtaposition of 13 hidden layers NN and FDTD', fontname='Times New Roman', fontsize=16, loc='center')
plt.xlabel('Wavelength(nm)', fontname='Times New Roman', fontsize=18)
plt.ylabel('Transmission', fontname='Times New Roman', fontsize=18)
plt.xticks(fontfamily='Times New Roman', fontsize=16)
plt.yticks(fontfamily='Times New Roman', fontsize=16)
font_prop = FontProperties(family="Times New Roman", size=16)
plt.legend(['FDTD', 'DL'], prop=font_prop)
plt.savefig("OctagonalRR_AOPS/P3_FigureS6n.png", format="png", dpi=600, bbox_inches="tight")
plt.show()

# fig S6l
plt.plot(Loss[:,0],Loss[:,7], linewidth=2, color=((255/255, 127/255, 14/255))) 
plt.plot(Loss[:,0],Loss[:,11], linewidth=2, color=((0/255, 114/255, 189/255)), linestyle='--') 
plt.title('Juxtaposition of 11 hidden layers NN and FDTD', fontname='Times New Roman', fontsize=16, loc='center')
plt.xlabel('Wavelength(nm)', fontname='Times New Roman', fontsize=18)
plt.ylabel('Transmission', fontname='Times New Roman', fontsize=18)
plt.xticks(fontfamily='Times New Roman', fontsize=16)
plt.yticks(fontfamily='Times New Roman', fontsize=16)
font_prop = FontProperties(family="Times New Roman", size=16)
plt.legend(['FDTD', 'DL'], prop=font_prop)
plt.savefig("OctagonalRR_AOPS/P3_FigureS6l.png", format="png", dpi=600, bbox_inches="tight")
plt.show()

# fig S6j
plt.plot(Loss[:,0],Loss[:,7], linewidth=2, color=((255/255, 127/255, 14/255))) 
plt.plot(Loss[:,0],Loss[:,13], linewidth=2, color=((0/255, 114/255, 189/255)), linestyle='--') 
plt.title('Juxtaposition of 9 hidden layers NN and FDTD', fontname='Times New Roman', fontsize=16, loc='center')
plt.xlabel('Wavelength(nm)', fontname='Times New Roman', fontsize=18)
plt.ylabel('Transmission', fontname='Times New Roman', fontsize=18)
plt.xticks(fontfamily='Times New Roman', fontsize=16)
plt.yticks(fontfamily='Times New Roman', fontsize=16)
font_prop = FontProperties(family="Times New Roman", size=16)
plt.legend(['FDTD', 'DL'], prop=font_prop)
plt.savefig("OctagonalRR_AOPS/P3_FigureS6j.png", format="png", dpi=600, bbox_inches="tight")
plt.show()

# fig Sh
plt.plot(Loss[:,0],Loss[:,7], linewidth=2, color=((255/255, 127/255, 14/255))) 
plt.plot(Loss[:,0],Loss[:,15], linewidth=2, color=((0/255, 114/255, 189/255)), linestyle='--') 
plt.title('Juxtaposition of 7 hidden layers NN and FDTD', fontname='Times New Roman', fontsize=16, loc='center')
plt.xlabel('Wavelength(nm)', fontname='Times New Roman', fontsize=18)
plt.ylabel('Transmission', fontname='Times New Roman', fontsize=18)
plt.xticks(fontfamily='Times New Roman', fontsize=16)
plt.yticks(fontfamily='Times New Roman', fontsize=16)
font_prop = FontProperties(family="Times New Roman", size=16)
plt.legend(['FDTD', 'DL'], prop=font_prop)
plt.savefig("OctagonalRR_AOPS/P3_FigureS6h.png", format="png", dpi=600, bbox_inches="tight")
plt.show()

# fig S6f
plt.plot(Loss[:,0],Loss[:,7], linewidth=2, color=((255/255, 127/255, 14/255))) 
plt.plot(Loss[:,0],Loss[:,17], linewidth=2, color=((0/255, 114/255, 189/255)), linestyle='--') 
plt.title('Juxtaposition of 5 hidden layers NN and FDTD', fontname='Times New Roman', fontsize=16, loc='center')
plt.xlabel('Wavelength(nm)', fontname='Times New Roman', fontsize=18)
plt.ylabel('Transmission', fontname='Times New Roman', fontsize=18)
plt.xticks(fontfamily='Times New Roman', fontsize=16)
plt.yticks(fontfamily='Times New Roman', fontsize=16)
font_prop = FontProperties(family="Times New Roman", size=16)
plt.legend(['FDTD', 'DL'], prop=font_prop)
plt.savefig("OctagonalRR_AOPS/P3_FigureS6f.png", format="png", dpi=600, bbox_inches="tight")
plt.show()

# fig S6d
plt.plot(Loss[:,0],Loss[:,7], linewidth=2, color=((255/255, 127/255, 14/255))) 
plt.plot(Loss[:,0],Loss[:,19], linewidth=2, color=((0/255, 114/255, 189/255)), linestyle='--') 
plt.title('Juxtaposition of 3 hidden layers NN and FDTD', fontname='Times New Roman', fontsize=16, loc='center')
plt.xlabel('Wavelength(nm)', fontname='Times New Roman', fontsize=18)
plt.ylabel('Transmission', fontname='Times New Roman', fontsize=18)
plt.xticks(fontfamily='Times New Roman', fontsize=16)
plt.yticks(fontfamily='Times New Roman', fontsize=16)
font_prop = FontProperties(family="Times New Roman", size=16)
plt.legend(['FDTD', 'DL'], prop=font_prop)
plt.savefig("OctagonalRR_AOPS/P3_FigureS6d.png", format="png", dpi=600, bbox_inches="tight")
plt.show()

# fig S6b
plt.plot(Loss[:,0],Loss[:,7], linewidth=2, color=((255/255, 127/255, 14/255))) 
plt.plot(Loss[:,0],Loss[:,21], linewidth=2, color=((0/255, 114/255, 189/255)), linestyle='--') 
plt.title('Juxtaposition of 1 hidden layer NN and FDTD', fontname='Times New Roman', fontsize=16, loc='center')
plt.xlabel('Wavelength(nm)', fontname='Times New Roman', fontsize=18)
plt.ylabel('Transmission', fontname='Times New Roman', fontsize=18)
plt.xticks(fontfamily='Times New Roman', fontsize=16)
plt.yticks(fontfamily='Times New Roman', fontsize=16)
font_prop = FontProperties(family="Times New Roman", size=16)
plt.legend(['FDTD', 'DL'], prop=font_prop)
plt.savefig("OctagonalRR_AOPS/P3_FigureS6b.png", format="png", dpi=600, bbox_inches="tight")
plt.show()
