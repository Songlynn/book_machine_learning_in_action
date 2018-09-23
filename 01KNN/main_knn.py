import knn

print('dating classify:\n')

grades, labels = knn.createData()

print('small test:')
test1 = [1, 2]
result1 = knn.classify0(test1, grades, labels, 3)
print(test1, ':', result1)

print('\ntest classify:')
knn.datingLabelTest('01KNN_dating.txt',0.1 , 3, False)
knn.datingLabelTest('01KNN_dating.txt',0.1 , 5, False)
knn.datingLabelTest('01KNN_dating.txt',0.2 , 5, False)

print('\npredict:')
#knn.classifyPerson()

print('-------------------------------------')

print('digit classify:\n')
print('test classify:')
#knn.classifyDigits()# 错误率特别大
