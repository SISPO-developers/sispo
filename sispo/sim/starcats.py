import logging
import matplotlib.pyplot as plt
import starcat as sc

from pathlib import Path

import numpy as np


logger = logging.getLogger("starcats")
starcat_dir = Path(".").resolve() / "data" / "UCAC4"
starcat = sc.StarCatalog(Path(".").resolve(), logger)
starcat2 = sc.StarCatalog(Path(".").resolve(), logger, starcat_dir)

star_cat_data = starcat2.get_stardata(50.0, 50.0, 0.5, 0.5)
vizier_data = starcat.get_stardata(50.0, 50.0, 0.5, 0.5)

x1 = []
y1 = []
for x,y,_ in star_cat_data:
    x1.append(x)
    y1.append(y)

x2 = []
y2 = []
for x,y,_ in vizier_data:
    x2.append(x)
    y2.append(y)

print(len(star_cat_data))
print(np.min(x1), np.max(x1), np.min(y1), np.max(y1))
#print(star_cat_data)
print(len(vizier_data))
print(np.min(x2), np.max(x2), np.min(y2), np.max(y2))
#print(vizier_data)
plt.scatter(x1,y1)
plt.scatter(x2,y2,alpha=0.2)
plt.show()