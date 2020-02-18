import glob

import imageio

filenames = glob.glob("Inst_*")
images = []
with imageio.get_writer('flyby.gif', mode='I') as writer:
    for filename in filenames:
		print(f"Processing file {filename}")
        image = imageio.imread(filename)
        writer.append_data(image)