from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('multipage.pdf')

import matplotlib.pyplot as plt

plt.plot([0,1,2], [5,6,7])
pp.savefig()
pp.close()

