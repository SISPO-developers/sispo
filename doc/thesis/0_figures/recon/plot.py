import csv
import sys

import matplotlib.pyplot as plt
import numpy as np

def main(file_name):
	labels = []
	label_size_p = []
	points = []
	vertices = []
	faces = []
	with open(file_name, "r") as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=";")
		for i, row in enumerate(csv_reader):
			if i == 0:
				continue
			labels.append(str(row[8]).upper().replace("_", " "))
			label_size_p.append(str(row[9]) + "\n" + labels[-1])
			points.append(float(row[10]))
			vertices.append(float(row[11]))
			faces.append(float(row[12]))
	
	x = np.arange(len(labels))  # the label locations
	width = 0.25  # the width of the bars
	
	fig, ax = plt.subplots()
	
	rects1 = ax.bar(x - width, points, width, label='Points')
	rects2 = ax.bar(x, vertices, width, label='Vertices')
	rects2 = ax.bar(x + width, faces, width, label='Faces')
	
	ax.set_ylabel('Value normalised to PNG [-]')
	ax.set_xlabel('Size of data set compared to PNG [%]')
	ax.set_title(file_name[:-4].replace("_", " "))
	ax.set_xticks(x)
	ax.set_xticklabels(label_size_p)
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	
	fig.tight_layout()
	plt.savefig(file_name[:-4] + ".png")
	#plt.show()
	
def heatmap():
	data_1k = [["Seq1", "Seq2", "Seq2", "Seq2"],
			   ["Seq1", "Seq1", "Seq2", "Seq2"],
			   ["Seq1", "Seq1", "Seq2", "Seq2"],
			   ["Seq1", "Seq1", "Seq2", "Seq2"],
			   ["Seq1", "Seq2", "Seq2", "-"]]
	data_10k = [["Seq2", "-", "Seq2", "Seq1"],
				["Seq2", "-", "Seq2", "Glob"],
				["Seq2", "-", "Seq2", "Seq2"],
				["Seq2", "-", "Seq2", "Seq2"],
				["Seq2", "-", "Seq2", "Seq1"]]
				
	labels = ["PNG", "JP2 1000", "JP2 100", "JP2 10", "JP2 1"]
	dists = ["50", "100", "200", "400"]
	
	fig, ax = plt.subplots()
	im = ax.imshow(data_1k)
				
	
			
if __name__ == "__main__":
	if len(sys.argv) > 1:
		main(sys.argv[1])
	else:
		#heatmap()
		for elem in ["120i_50km_1k.csv", "120i_50km_10k.csv",
					 "120i_100km_1k.csv", "120i_100km_10k.csv",
					 "120i_200km_1k.csv", "120i_200km_10k.csv",
					 "120i_400km_1k.csv", "120i_400km_10k.csv"]:
			main(elem)