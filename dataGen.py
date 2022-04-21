import imageAugmentation


imageAugmentation.GeneratorFunction("Dataset/single_person", "testSet/single_person", 1, 80)

print("Single Person done")

imageAugmentation.GeneratorFunction("Dataset/double_person", "testSet/double_person", 1, 78)