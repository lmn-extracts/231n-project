PREPROCESS_DATA
Copies images with alphabetic labels to the <target_directory> with the ground truth labels stored in 'gt.txt' in the following format:
<img_name> <label>

Example: If the image name is 123.jpg with label 'convolution', then the corresponding record in 'gt.txt' would be:
123 convolution

--------------------------

Usage: python preprocess_data.py -ds <dataset> -d <data_directory> -t <target_directory>

<dataset> currently supports 'ICDAR03', 'ICDAR13' and 'mjsynth'.

Prior to The data directoy should be organized in the following format:
- parent_to_data_directory
	- gt.txt (for ICDAR13, mjsynth) / word.xml (ICDAR03)
	- <data_directory>

Note: <target directory> can be located anywhere outside the <data_directory>