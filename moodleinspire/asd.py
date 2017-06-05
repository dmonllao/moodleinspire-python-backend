from moodleinspire import inputs

filepath = '/home/davidm/Desktop/pip-moodleinspire/asd.txt'
#filepath = '/home/davidm/Desktop/predict.test.txt'

data_provider = inputs.DataProvider(filepath, [0, 1], training=True, test_data=0.2, batch_size=1000)

print('get data!!!!!!!!')
it = data_provider.get_data()
for x, y in it:
    print(x)
    print(y)
    pass

print('get test data!!!!!!!!')
it = data_provider.get_test_data()
for x, y in it:
    print(x)
    print(y)
    pass
