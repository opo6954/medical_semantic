import os

additionalParent = 'largeData'
cifar10Prefix = 'cifar10'
asanPrefix = 'data_png'


def cifar10labelModifier(parentPath):
    allFiles = os.listdir(parentPath)

    for eachFile in allFiles:
        eachFileAbsPath = os.path.join(parentPath, eachFile)
        eachLinesModified = []

        print('for file ' + eachFile)



        with open(eachFileAbsPath) as readFile:
            allLines = readFile.readlines()

        #     change ../exp_cifar10/resizedData/test/4590.jpg,3 to ../largeData/cifar10/exp_cifar10/resizedData/test/4590.jpg,3
        for eachLine in allLines:
            remain = eachLine.split('/')[0:]
            remain.insert(1, cifar10Prefix)
            remain.insert(1, additionalParent)
            remainModified = '/'.join(remain)

            eachLinesModified.append(remainModified)

        with open(eachFileAbsPath, 'w') as writeFile:
            for each in eachLinesModified:
                writeFile.write(each)



def ASANlabelModifier(parentPath):
    allFiles = os.listdir(parentPath)

    for eachFile in allFiles:
        if(eachFile.split('.')[-1] != 'label'):
            continue

        eachFileAbsPath = os.path.join(parentPath, eachFile)
        eachLinesModified = []

        print('for file ' + eachFile)

        with open(eachFileAbsPath) as readFile:
            allLines = readFile.readlines()

        # change ../wholeData_bbox/activated/0112800791_01Nodule_0.png,0 to ../largeData/data_png/wholeData_bbox/activated/0112800791_01Nodule_0.png,0
        for eachLine in allLines:
            remain = eachLine.split('/')[0:]
            remain.insert(1, asanPrefix)
            remain.insert(1, additionalParent)
            remainModified = '/'.join(remain)

            eachLinesModified.append(remainModified)

        with open(eachFileAbsPath, 'w') as writeFile:
            for each in eachLinesModified:
                writeFile.write(each)

ASANlabelModifier('./largeData/data/exp_jh/')

# cifar10labelModifier('./largeData/cifar10/exp_cifar10/labelfile')




