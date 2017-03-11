import sys, os, os.path
import csv
import numpy as np
import re



def generateFeatureOnTextStart(inputFilePath, tokenList):
    if not os.path.exists(inputFilePath):
        print 'Input File ', inputFilePath, 'does not exist'
        sys.exit(1)

    output = []
    header = []
    data_colInFile = 1

    for token in tokenList:
        header.append('s_'+str.replace(token, ' ','_'))
    output.append(header)

    with open(inputFilePath, 'r') as inputFile:
        # skip header line
        next(inputFile)

        for instance in inputFile:
            featureVector = []
            lineParts = instance.split(',')
            data = lineParts[data_colInFile]

            data = data.lower()

            for token in tokenList:
                if data.startswith(token):
                    featureVector.append(1)
                else:
                    featureVector.append(0)
            output.append(featureVector)
    return output


def generateRegexFeatureOnPresence(inputFilePath):
    if not os.path.exists(inputFilePath):
        print 'Input File ', inputFilePath, 'does not exist'
        sys.exit(1)

    header = ["r_num", "r_abbr"]

    numberRegex = '[0-9]+'
    abbrRegex = '[A-Z][A-Z][A-Z]+'
    locationRegex = '(in|of)( the)? [A-Z][A-Z,a-z]*'

    numberPattern = re.compile(numberRegex)
    abbrPattern = re.compile(abbrRegex)
    locationPattern = re.compile(locationRegex)

    output = []
    output.append(header)
    data_colInFile = 1

    with open(inputFilePath, 'r') as inputFile:
        # skip header line
        next(inputFile)
        for instance in inputFile:
            featureVector = []
            lineParts = instance.split(',')
            data = lineParts[data_colInFile]

            if numberPattern.search(data):
                featureVector.append(1)
            else:
                featureVector.append(0)

            if abbrPattern.search(data):
                featureVector.append(1)
            else:
                featureVector.append(0)

            if locationPattern.search(data):
                featureVector.append(1)
            else:
                featureVector.append(0)

            output.append(featureVector)

    return output



def generateBinaryFeatureOnPresence(inputFilePath, tokenList, isCaseSensitive, addFeatureHeaders, keepLabels):

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

    id_colInfile = 0
    data_colInFile = 1

    with open(inputFilePath, 'r') as inputFile:
        # skip header line
        next(inputFile)

        for instance in inputFile:
            featureVector = []
            lineParts = instance.split(',')

            if keepLabels:
                label = int(lineParts[id_colInfile])
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
    outFilePath = "./v3/"+type+"Data.csv"
    questionTokens = {"who", "what", "when", "where", "why", "which", "how", "like", "many", "how many", "does", "is", "isn't", "doesnt", "doesn't"}

    abbrClassTokens = {"full form", "expansion", "stand for", "what does", "meaning", "abbr"}
    humanClassTokens = {"who was", "father", "mother", "person", "whom", "with", "man", "woman", "he", "she", "is that", "do"}
    locationClassTokens = {"location", "where", "city", "place", "from", "lie in", "visit", "capital", "state", "country", "continent", "ocean", "sea", "nation", "island", "constellation", "river"}
    descriptionClassTokens = {"describe", "doing", "what should", "how can", "how did", "how would", "is there a", "way", "way to", "for", "like", "mean ", "entail", "origin of", "name of", "diff", "cause", "chance"}
    entityClassTokens = {"what", "called", "has", "have", "is there", "name", "some of"}
    numberClassTokens = {"much", "how much", "many","tall", "height", "width", "weight", "age", "what is the mean", "average", "distance", "many", "big", "how long", "how many", "year", "amount", "top", "date", "time", "number", "numeral", "sum", "percent", "date", "day", "time", "old", "popul", "cost", "rate"}

    prepositionTokens = {"in", "on", "among", "upon", "of", "by", "some", "like", "into", "some of"}
    adjectivetokens = {"best", "most", "major", "minor", "good", "bad", "famous", "famed", "largest", "least", "lowest", "max", "minimum"}
    start_with_tokens = {"how to", "how do", "is there", "what is", "what was", "are we", "what kind", "how much", "how many", "how long", "how can", "what was", "what were"}

    presenceTokenList = list(questionTokens | abbrClassTokens | humanClassTokens | locationClassTokens |
                             descriptionClassTokens | entityClassTokens | numberClassTokens | prepositionTokens | adjectivetokens | start_with_tokens)

    featureHeaders = "Label"

    presenceFeatures = generateBinaryFeatureOnPresence(inputFilePath, presenceTokenList, False, True, True)

    regexFeatures = generateRegexFeatureOnPresence(inputFilePath)

    startsWithFeatures = generateFeatureOnTextStart(inputFilePath, start_with_tokens)

    citiesList = ""

    featureVectors = np.concatenate((presenceFeatures,regexFeatures, startsWithFeatures), axis=1)

    with open(outFilePath, 'w') as outFile:
        writer = csv.writer(outFile)
        writer.writerows(featureVectors)



if __name__ == '__main__':
    main()