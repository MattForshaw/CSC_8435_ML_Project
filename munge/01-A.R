# Example preprocessing script.

hmnist_28_RGB = read.csv("data/hmnist_28_28_RGB.csv")
cache('hmnist_28_RGB')

hmnist_28_L = read.csv("data/hmnist_28_28_L.csv")
cache('hmnist_28_L')

hmnist_8_RGB = read.csv("data/hmnist_8_8_RGB.csv")
cache('hmnist_8_RGB')

hmnist_8_L = read.csv("data/hmnist_8_8_L.csv")
cache('hmnist_8_L')

HAM10000_metadata = read.csv("data/HAM10000_metadata.csv")
cache('HAM10000_metadata')
