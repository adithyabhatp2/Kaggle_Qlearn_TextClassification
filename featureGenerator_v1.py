import sys, os, os.path
import csv



def generateBinaryFeatureOnPresense(inputFilePath, tokenList, isCaseSensitive, addFeatureHeaders, keepLabels):

    if not os.path.exists(inputFilePath):
        print 'Input File ', inputFilePath, 'does not exist'
        sys.exit(1)

    header = []

    if keepLabels:
        header.append("id")

    for token in tokenList:
        header.append('p_'+str.replace(token, ' ','_'))

    output = []
    if addFeatureHeaders:
        output.append(header)

    label_colInfile = 0
    data_colInFile = 1


    with open(inputFilePath, 'r') as inputFile:
        # skip header line
        next(inputFile)

        for instance in inputFile:
            featureVector = []
            lineParts = instance.split(',')

            if keepLabels:
                label = int(lineParts[label_colInfile])
                featureVector.append(label)

            data = lineParts[data_colInFile]
            if not isCaseSensitive:
                data = data.lower()

            for token in tokenList:
                if token in data:
                    featureVector.append(1)
                else:
                    featureVector.append(0)
            output.append(featureVector)

    return output




def main():

    type = "train"
    inputFilePath = "./"+type+"_questions.txt"
    outFilePath = "./v1/"+type+"Data.csv"
    questionTokens = {"who", "what", "when", "where", "why", "which", "how", "like", "many", "how many", "does", "is"}

    abbrClassTokens = {"full form", "expansion", "stand for", "mean", "what does", "meaning"}
    humanClassTokens = {"who was", "father", "mother", "person", "whom", "with"}
    locationClassTokens = {"location", "where", "city", "place"}
    descriptionClassTokens = {"describe", "doing"}
    entityClassTokens = {"what", "called"}
    numberClassTokens = {"much", "how much", "many","tall", "height", "width", "weight", "age", "what is the mean", "average", "distance", "many", "big"}


    presenceTokenList = list(questionTokens | abbrClassTokens | humanClassTokens | locationClassTokens | descriptionClassTokens | entityClassTokens | numberClassTokens)

    featureHeaders = "Label"

    presenceFeatures = generateBinaryFeatureOnPresense(inputFilePath, presenceTokenList, False, True, True)


    numberRegex = '[0-9]+'
    abbrRegex = '[A-Z][A-Z][A-Z]+'

    citiesList = ""

    featureVectors = presenceFeatures

    with open(outFilePath, 'w') as outFile:
        writer = csv.writer(outFile)
        writer.writerows(featureVectors)



if __name__ == '__main__':
    main()