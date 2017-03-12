import sys, os, os.path
import csv
import numpy as np
import re



def otherFeatureGeneration(inputFilePath, tokenListOfLists):
    if not os.path.exists(inputFilePath):
        print 'Input File ', inputFilePath, 'does not exist'
        sys.exit(1)

    output = []
    header = []
    data_colInFile = 1

    for tokenList in tokenListOfLists:
        header.append('any_'+str.replace(tokenList[0], ' ','_'))

    header.append('length')
    output.append(header)

    with open(inputFilePath, 'r') as inputFile:
        # skip header line
        next(inputFile)

        for instance in inputFile:
            featureVector = []
            lineParts = instance.split(',')
            data = lineParts[data_colInFile]

            data = data.lower().lstrip(' ')
            data = data.replace('what \'s', 'what is')
            data = data.replace('who \'s', 'who is')
            data = data.replace('when \'s', 'when is')



            for tokenList in tokenListOfLists:
                found = False
                for token in tokenList:
                    if token in data:
                        featureVector.append(1)
                        found = True
                        break
                if found==False:
                    featureVector.append(0)

            featureVector.append(len(data.split(' ')))
            output.append(featureVector)

    return output




def generateFeatureOnTextStart(inputFilePath, tokenList):
    if not os.path.exists(inputFilePath):
        print 'Input File ', inputFilePath, 'does not exist'
        sys.exit(1)

    output = []
    header = []
    data_colInFile = 1

    for token in tokenList:
        header.append('sw_'+str.replace(token, ' ','_'))

    header.append('sw_isShe')
    output.append(header)

    with open(inputFilePath, 'r') as inputFile:
        # skip header line
        next(inputFile)

        for instance in inputFile:
            featureVector = []
            lineParts = instance.split(',')
            data = lineParts[data_colInFile]

            data = data.lower().lstrip(' ')
            data = data.replace('what \'s', 'what is')
            data = data.replace('who \'s', 'who is')
            data = data.replace('when \'s', 'when is')

            for token in tokenList:
                if data.startswith(token):
                    featureVector.append(1)
                else:
                    featureVector.append(0)

            if data.lower().startswith("is he") or data.lower().startswith("is she"):
                featureVector.append(1)
            else:
                featureVector.append(0)

            output.append(featureVector)
    return output


def generateRegexFeatureOnPresence(inputFilePath):
    if not os.path.exists(inputFilePath):
        print 'Input File ', inputFilePath, 'does not exist'
        sys.exit(1)

    numberRegex = '[0-9]+'
    abbrRegex = '[A-Z][A-Z][A-Z]+'
    locationRegex = ' (in|of)( the)? [A-Z][A-Z,a-z]*'
    isHumanRegex = 'is (s)?he'

    header = ["r_num", "r_abbr", "r_loc", "r_isHum"]

    numberPattern = re.compile(numberRegex)
    abbrPattern = re.compile(abbrRegex)
    locationPattern = re.compile(locationRegex)
    isHumanPattern = re.compile(isHumanRegex)

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

            if isHumanPattern.search(data):
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
    outFilePath = "./v5/"+type+"Data.csv"
    questionTokens = {"who", "what", "when", "where", "why", "which", "how", "like", "many", "how many", "does", "is", "isn't", "doesnt", "doesn't"}

    abbrClassTokens = {"full form", "expansion", "stand for", "what does", "meaning", "abbr"}
    humanClassTokens = {"who was", "father", "mother", "person", "whom", "with", "man ", " he ", "she", "is that", "do", "name of"}
    locationClassTokens = {"location", "where", "city", "place", "from", "lie in", "visit", "capital", "state", "country", "continent", "ocean", "sea", "nation", "island", "constellation", "river", "canal"}
    descriptionClassTokens = {"describe", "doing", "what should", "how can", "how did", "how would", "is there a", "way", "way to", "for", "like", "mean ", "entail", "origin of", "name of", " diff", "cause", "chance", " if ", "between"}
    entityClassTokens = {"what", "called", "has", "have", "is there", "name", "some of", "name a"}
    numberClassTokens = {"much", "how much", "many","tall", "height", "width", "weight", " age", "what is the mean", "average", "distance", "many", "big", "how long", "how many", "year", "amount", "top", "date", "time", "number", "numeral", "sum", "percent", "old", "cost", "rate", "day", "populat", "cost", "rate", "max", "minimum", "largest", "least", "lowest"}

    prepositionTokens = {"in", "on", "among", "upon", "of", "by", "some", "like", "into", "some of"}
    adjectivetokens = {"best", "most", "major", "minor", "good", "bad", "famous", "famed", "worst"}
    start_with_tokens = {"how to", "how do", "is there", "what is", "what was", "are we", "what kind", "how much", "how many", "how long", "how can", "what was", "what were", "what do you", "what do"} | questionTokens

    presenceTokenList = list(questionTokens | abbrClassTokens | humanClassTokens | locationClassTokens |
                             descriptionClassTokens | entityClassTokens | numberClassTokens | prepositionTokens | adjectivetokens | start_with_tokens)

    featureHeaders = "Label"

    presenceFeatures = generateBinaryFeatureOnPresence(inputFilePath, presenceTokenList, False, True, True)

    regexFeatures = generateRegexFeatureOnPresence(inputFilePath)

    startsWithFeatures = generateFeatureOnTextStart(inputFilePath, start_with_tokens)

    locSynList = ['city', 'country', 'river', 'ocean', 'canal', 'town', 'village', 'place', 'solar system']
    personSynList = ['president', 'player', 'painter', 'artist', 'wife', 'actor', 'actress', 'banker', 'writer', 'author', 'doctor']
    desc_syn_list = ['difference', 'similarit']
    date_syn_list = ['date', 'day', 'time', 'month']

    lol = []
    lol.append(locSynList)
    lol.append(personSynList)
    lol.append(desc_syn_list)
    lol.append(date_syn_list)

    synFeatures = otherFeatureGeneration(inputFilePath, lol)

    citiesList = ""

    featureVectors = np.concatenate((presenceFeatures,regexFeatures, startsWithFeatures, synFeatures), axis=1)

    with open(outFilePath, 'w') as outFile:
        writer = csv.writer(outFile)
        writer.writerows(featureVectors)



if __name__ == '__main__':
    main()