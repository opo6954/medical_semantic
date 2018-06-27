# test train model with args

from testTrainModel import train_SPHash

import argparse


# train_SPHash(inputType="Origin", bitSize=16, num_epochs=50)


def main(args):
    inputType = args.inputType
    bitSize = args.bitSize
    num_epochs = args.num_epochs
    isJH = args.isJH
    weightSM = args.weightSM
    weightApp = args.weightApp
    weightFire = args.weightFire



    print('//////////////////////////////////////////////////////////////////////////////')
    print('train condition for ' + inputType + '_' + str(bitSize) + '_' + str(num_epochs) + '+' + str(isJH))
    print('//////////////////////////////////////////////////////////////////////////////')

    train_SPHash(modelSavedPath='../modelStore/ASAN/180530', inputType=inputType, bitSize=bitSize, num_epochs=num_epochs, isJH=isJH, weightSM=weightSM, weightApp=weightApp, weightFire=weightFire)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--inputType',
      type=str,
      help='Input type for training(Origin, Activated, Cropped)',
      required=True)
  parser.add_argument(
      '--bitSize',
      type=int,
      help='Bit Size(16, 32, 48)',
      required=True)
  parser.add_argument(
      '--num_epochs',
      type=int,
      help='Number of epochs',
      required = True)
  parser.add_argument(
      '--isJH',
      type=bool,
      help='is JH dataset',
      required=True)

  parser.add_argument(
      '--weightSM',
      type=float,
      help='weight of SM',
      required=True)

  parser.add_argument(
      '--weightApp',
      type=float,
      help='weight of App',
      required=True)

  parser.add_argument(
      '--weightFire',
      type=float,
      help='weight of Fire',
      required=True)

  FLAGS, unparsed = parser.parse_known_args()

  main(FLAGS)




# train_SPHash(inputType="Origin", bitSize=16, num_epochs=50)