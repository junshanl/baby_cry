from ubm import adapt_ubm, load_ubm, preprocess


path = './TIMIT/TRAIN/DR1/*/*.WAV'

weight, means, covars = load_ubm('./ubm_112.yml')

X = preprocess(path)

train_X = X
test_X = X[2:]

weight, means, covars = adapt_ubm(train_X, weight, means, covars)
print weight
