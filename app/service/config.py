# Image Size
IMAGE_SIZE = int(256)

# Weight for content
ALPHA = float(1e-1)

# Weight for style
BETA = float(1e-1)

# Weight for mrf
GAMMA = float(1e-6)

# Weight for distance loss
DELTA = float(1e2)

# Patch size
PATCH_SIZE = int(5)

# Path to content image
CONTENT_PATH = str('./service/inputs/contents/Swallow-Silhouette.jpg')

# Path to style image
STYLE_PATH = str('./service/inputs/styles/delicate.jpg')

# Path to save output files
OUTPUT_DIR = str('./service/sample_outputs/')

# USE CUDA 0 for GPU -1 for cpu
CUDA = int(-1)

# Number of iterations to run
EPOCH = int(5000)