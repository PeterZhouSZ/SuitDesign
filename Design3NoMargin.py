import cv2
import numpy as np
from copy import copy
import itertools
import random

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import json

# import ComponentExtractor

codeDictOld = ['1', '2', '3', '4', '5', '6', '7', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'J', 'K', 'L',
                               'M', 'P', 'Q', 'R', 'T', 'U', 'V', 'Y']

codeDict1 = ['1', '2', '3', '4', '5', '6', '7', 'A', 'B', 'C', 'D', 'E', 'F', 'J', 'K', 'L',
                               'M', 'P', 'R', 'T', 'U', 'V', 'Y', 'a', 'e', 'f', 'g', 'h', 'i', 'r', 't', '#', '$', '&']

codeDict2 = [c for c in '1234567ABCDEFJKLMPRTUVYaefhirt#&!?ΓΔλΨΩΣΠ¥']

class Config:
    def __init__(self):
        self.dilationSize = 2.5
        self.drawMargin = True

        self.drawContourLine = True
        self.contourLineSize = 0.2

        self.squareSize = 3.2
        self.startPoint = [25, 0]
        self.startWithWhite = True

        self.codeOnBlackSquare = False
        self.pixelsPerCm = 46
        self.fontName = "consola.ttf"
        self.codeDict = codeDictOld
        self.fontSize = 120
        self.repermuteCodes = False
        self.randomSeed = 12345
        #'black', 'crop', 'whiteTriangle'
        self.marginType = 'black'
        self.triangleEdgeLength = 0.3
        self.textureAugment = 1
        self.useTextureBackground = False
        self.smallWhiteComponentsRemovement = True
        self.whiteComponentsSizeLowerBound = 350

        self.addNoCodeCorners = False

class UnitardDesignGenerator:
    def __init__(self):
        self._config = Config()
        self._texture = None

    def setConfig(self, config):
        self._config = config
        self.squareSizeInPixels = int(self._config.pixelsPerCm * self._config.squareSize)

    def generateCodeSuit(self, templateFile ):
        random.seed(self._config.randomSeed)

        dilationSize = self._config.dilationSize

        startPoint = self._config.startPoint
        startWithWhite = self._config.startWithWhite

        pixelsPerCm = self._config.pixelsPerCm

        self.imgT = cv2.imread(templateFile)
        self.imgT_gray = cv2.cvtColor(self.imgT, cv2.COLOR_BGRA2GRAY)

        self.suitMask = self.imgT_gray != 255
        # print(suitMask)
        imgBlank = np.ones(self.suitMask.shape) * 255
        suitBlank = copy(imgBlank)

        suitBlank[self.suitMask] = 0

        # cv2.imwrite("suitBlank.png", suitBlank)

        kernel = np.ones((int(dilationSize * pixelsPerCm), int(dilationSize * pixelsPerCm)), np.uint8)

        suitBlankDilation = cv2.dilate(suitBlank, kernel, iterations=1)

        # cv2.imwrite("suitBlankDilation.png", suitBlankDilation)

        suitMaskkDilation = suitBlankDilation != 255

        self.contourMask = self.suitMask != suitMaskkDilation

        self.suitMaskkDilation = suitMaskkDilation

        self.contour = copy(imgBlank)

        self.contour[self.contourMask] = 0
        # cv2.imwrite("contour.png", contour)

        imgW = self.imgT.shape[1]
        imgH = self.imgT.shape[0]

        self.numSquaresHorizonal = int((imgW - 1 - startPoint[0]) / (self.squareSizeInPixels))
        self.numSquaresVertical = int((imgH - 1 - startPoint[1]) / (self.squareSizeInPixels))

        self.pts = np.zeros((self.numSquaresVertical + 1, self.numSquaresHorizonal + 1, 2), dtype=np.float64)
        self.ptsMask = np.zeros((self.numSquaresVertical + 1, self.numSquaresHorizonal + 1), dtype=np.bool)

        for x, y in itertools.product(range(self.numSquaresHorizonal + 1), range(self.numSquaresVertical + 1)):
            self.pts[y, x] = [startPoint[0] + x * self.squareSizeInPixels, startPoint[1] + y * self.squareSizeInPixels]

        self.squares = []
        self.allSquares = []

        coords = [(x, y) for x in range(self.numSquaresHorizonal) for y in range(self.numSquaresVertical)]

        numWhites = 0
        numBlacks = 0

        for x, y in coords:
            s = Square([x, y], startPoint, self.squareSizeInPixels)
            self.allSquares.append(s)
            s.white = (x % 2) != (y % 2)

            if not startWithWhite:
                s.white = not s.white

            s.avalible = checkSquare(suitMaskkDilation, s)

            if s.white and s.avalible:
                numWhites = numWhites + 1
                self.ptsMask[y, x] = True
                self.ptsMask[y, x + 1] = True
                self.ptsMask[y + 1, x + 1] = True
                self.ptsMask[y + 1, x] = True
            if (not s.white) and s.avalible:
                numBlacks = numBlacks + 1

            self.squares.append(s)

        self.suitWithChb = copy(self.contour)

        #
        for s in self.squares:
            if (not s.white) or (not s.avalible):
                cv2.rectangle(self.suitWithChb, s.cornersInt[0], s.cornersInt[2], (0,), thickness=-1)

        if (self._config.useTextureBackground):
            self.putTextureOnSquares(self.suitWithChb)
        self.suitInverseMask = np.logical_not(self.suitMask)
        self.suitWithChb[self.suitInverseMask] = 255

        # cv2.imwrite("suitWithChb.png", self.suitWithChb)

        #
        self.suitWithChbMarginWithWhiteSquare = copy(self.contour)
        for s in self.squares:
            if (not s.white):
                cv2.rectangle(self.suitWithChbMarginWithWhiteSquare, s.cornersInt[0], s.cornersInt[2], (0,), thickness=-1)

        if (self._config.useTextureBackground):
            self.putTextureOnSquares(self.suitWithChbMarginWithWhiteSquare)

        self.suitInverseMask = np.logical_not(self.suitMask)
        self.suitWithChbMarginWithWhiteSquare[self.suitInverseMask] = 255
        self.suitWithChbMarginWithWhiteSquare[self.contourMask] = 0

        # cv2.imwrite("suitWithChbMarginWithWhiteSquare.png", suitWithChbMarginWithWhiteSquare)

        #
        self.suitWithChbMarginWithWhiteTriangle = copy(self.contour)
        for s in self.squares:
            if (not s.white):
                cv2.rectangle(self.suitWithChbMarginWithWhiteTriangle, s.cornersInt[0], s.cornersInt[2], (0,), thickness=-1)
            elif not s.avalible:
                drawSquare4Triangles(self.suitWithChbMarginWithWhiteTriangle, s, self.ptsMask, config)

        if (self._config.useTextureBackground):
            self.putTextureOnSquares(self.suitWithChbMarginWithWhiteTriangle)

        self.suitWithChbMarginWithWhiteTriangle[self.suitInverseMask] = 255
        self.suitWithChbMarginWithWhiteTriangle[self.contourMask] = 0

        # cv2.imwrite("suitWithChbMarginWithWhiteTriangle.png", self.uitWithChbMarginWithWhiteTriangle)

        self.suitWithChbNoMargin = copy(imgBlank)
        for s in self.allSquares:
            if (not s.white):
                cv2.rectangle(self.suitWithChbNoMargin, s.cornersInt[0], s.cornersInt[2], (0,),
                              thickness=-1)
        self.suitWithChbNoMargin[self.suitInverseMask] = 255


        # suitWithChbPIL = suitWithChb cv2.cvtColor(suitWithChb, cv2.COLOR_BGR2RGB)
        if self._config.drawMargin:
            if self._config.marginType == 'crop':
                suitFinal = self.suitWithChbMarginWithWhiteSquare
            elif self._config.marginType == 'black':
                suitFinal = self.suitWithChb
            elif self._config.marginType == 'whiteTriangle':
                suitFinal = self.suitWithChbMarginWithWhiteTriangle
        else:
            suitFinal = self.suitWithChbNoMargin

        if self._config.drawContourLine:
            EKernel = np.ones((int(self._config.contourLineSize * pixelsPerCm), int(self._config.contourLineSize * pixelsPerCm)), np.uint8)
            suitErosion = cv2.erode(suitBlank, EKernel, iterations=1)

            # cv2.imwrite("suitBlankDilation.png", suitBlankDilation)

            suitErosionMask = suitErosion != 255

            self.contourLineMask = self.suitMask != suitErosionMask
            suitFinal[self.contourLineMask] = 0

        if self._config.smallWhiteComponentsRemovement:
            self.removeSmallWhiteComponents(suitFinal, 255)

        suitWithChbPIL = Image.fromarray(suitFinal)

        font = ImageFont.truetype(self._config.fontName, self._config.fontSize)

        codes = []

        codeId = 0
        for c1, c2 in itertools.product(self._config.codeDict, self._config.codeDict):
            codes.append(c1 + c2)

        if self._config.repermuteCodes:
            random.shuffle(codes)

        for s in self.squares:
            if s.white and s.avalible:
                sCenter = (s.corners[1, :] + s.corners[3, :]) / 2
                drawText(suitWithChbPIL, codes[codeId], sCenter, font)
                s.code = codes[codeId]
                codeId = codeId + 1

        if self._config.codeOnBlackSquare:
            codeIdBlack = 0
            codesBlack = codes
            random.shuffle(codesBlack)
            for s in self.squares:
                if (not s.white) and s.avalible:
                    sCenter = (s.corners[1, :] + s.corners[3, :]) / 2
                    drawText(suitWithChbPIL, codesBlack[codeIdBlack], sCenter, font, 255)
                    s.code = codesBlack[codeIdBlack]
                    codeIdBlack = codeIdBlack + 1

        self.suitWithChbPIL = suitWithChbPIL.convert('RGB')
        # suitWithChbPIL.save("suitWithChbPIL.png")
        self.numWhites = numWhites
        self.numBlacks = numBlacks

        print("number of white squares: ", numWhites)
        print("number of Black squares: ", numBlacks)

        self.cornersKeysWhite = [["" for x in range(self.numSquaresHorizonal + 1) ] for y in range(self.numSquaresVertical + 1)]
        self.cornersKeysBlack = [["" for x in range(self.numSquaresHorizonal + 1) ] for y in range(self.numSquaresVertical + 1)]


        for s in self.squares:
            if s.white and s.avalible and s.code != "":
                for cI, i in zip(s.cornersIds, range(4)):
                    self.cornersKeysWhite[cI[1]][cI[0]] = self.cornersKeysWhite[cI[1]][cI[0]] + s.code + str(i)

            if (not s.white) and s.avalible and s.code != "":
                for cI, i in zip(s.cornersIds, range(4)):
                    self.cornersKeysBlack[cI[1]][cI[0]] = self.cornersKeysBlack[cI[1]][cI[0]] + s.code + str(i)

    def setTexture(self, texture):
        self._texture = cv2.resize(texture, (self.squareSizeInPixels, self.squareSizeInPixels))
        self._texture = cv2.cvtColor(self._texture, cv2.COLOR_BGRA2GRAY)
        # self._texture = (self._texture * self._config.textureAugment).astype(np.uint8)

    def putTextureOnSquares(self, img):
        for s in self.squares:
            if s.avalible:
                if s.white:
                    img[s.cornersInt[0][1]:(s.cornersInt[0][1]+self.squareSizeInPixels), s.cornersInt[0][0]:(s.cornersInt[0][0]+self.squareSizeInPixels)] = self._texture
                else:
                    img[s.cornersInt[0][1]:(s.cornersInt[0][1] + self.squareSizeInPixels), s.cornersInt[0][0]:(s.cornersInt[0][0] + self.squareSizeInPixels)] \
                    = 255*np.ones(self._texture.shape, dtype=self._texture.dtype) - self._texture

    def writeCornerKeysSet(self, fileName):
        f = open(fileName, 'w', encoding='utf-8')
        for x, y in itertools.product(range(self.numSquaresHorizonal + 1), range(self.numSquaresVertical + 1)):
            if self.cornersKeysWhite[y][x] != "" or self.cornersKeysBlack[y][x] != "":
                f.write(self.cornersKeysWhite[y][x] + "-" + self.cornersKeysBlack[y][x])

                f.write(" " + str(int(self.pts[y,x,0])) + " " + str(int(self.pts[y,x,1])))
                f.write('\n')
        f.close()

    def writeCornerKeysSetJson(self, fileName):
        SuitInfo = {'numWhite':self.numWhites, 'numBlack':self.numBlacks,'corners':[], 'squares':[]}
        for x, y in itertools.product(range(self.numSquaresHorizonal + 1), range(self.numSquaresVertical + 1)):
            if self.cornersKeysWhite[y][x] != "" or self.cornersKeysBlack[y][x] != "":
                cornerInfo = {
                    'pts': [int(self.pts[y, x, 0]), int(self.pts[y, x, 1])],
                    'codeWhite': self.cornersKeysWhite[y][x],
                    'codeBlack': self.cornersKeysBlack[y][x]
                }
                SuitInfo['corners'].append(cornerInfo)

            elif self.suitMask[int(self.pts[y, x, 1]), int(self.pts[y, x, 0])] and config.addNoCodeCorners:
                cornerInfo = {
                    'pts': [int(self.pts[y, x, 0]), int(self.pts[y, x, 1])],
                    'codeWhite': "",
                    'codeBlack': "",
                }
                SuitInfo['corners'].append(cornerInfo)

        for s in self.squares:
            squareInfo = {
                'color':'white' if s.white else 'black',
                'code': s.code,
                'pts': s.corners.tolist()
            }
            SuitInfo['squares'].append(squareInfo)

        f = open(fileName, 'w', encoding='utf-8')
        json.dump(SuitInfo, f, indent=4)
        f.close()

    def removeSmallWhiteComponents(self, img, color = 255):
        pass
        # whiteComponents = ComponentExtractor.componentsExtractor(img, color, self._config.whiteComponentsSizeLowerBound)
        # for component in whiteComponents:
        #     if len(component) < self._config.whiteComponentsSizeLowerBound:
        #         component = [(c[1], c[0]) for c in component]
        #         coords = tuple(np.array(component).T)
        #         print("Remove components with size: ", len(component))
        #         img[coords] = 0


class Square:
    def __init__(self):
        self.corners = []
        self.white = True
        self.avalible = False
        self.code = ""

    def __init__(self, x, y, size):
        self.corners = np.array([
            (x,y),
            (x+size, y),
            (x + size, y + size),
            (x, y+ size),
        ])

        self.cornersInt = [
            (int(x), int(y)),
            (int(x + size - 1), int(y)),
            (int(x + size - 1), int(y + size - 1)),
            (int(x), int(y + size - 1)),
        ]
        self.cornersId = []

        self.white = True
        self.avalible = False
        self.code = ""


    def __init__(self, ids, startPoint, size):
        x = startPoint[0] + ids[0] * size
        y = startPoint[1] + ids[1] * size

        self.corners = np.array([
            (x,y),
            (x+size, y),
            (x + size, y + size),
            (x, y+ size),
        ])

        self.cornersInt = [
            (int(x), int(y)),
            (int(x + size - 1), int(y)),
            (int(x + size - 1), int(y + size - 1)),
            (int(x), int(y + size - 1)),
        ]

        self.cornersIds = [
            (ids[0], ids[1]),
            (ids[0] + 1, ids[1]),
            (ids[0] + 1, ids[1] + 1),
            (ids[0], ids[1] + 1),
        ]

        self.white = True
        self.avalible = False
        self.code = ""


def checkSquare(mask, s):
    valid = True
    s.cornerValidMask = []

    for c in s.corners:
        if not mask[int(c[1]), int(c[0])]:
            valid = False
            s.cornerValidMask.append(False)
        else:
            s.cornerValidMask.append(True)
    return valid

#def drawSquare4Triangles(img, s, config):
#    midPts = np.empty((0,2), np.float64)
#    cv2.rectangle(img, s.cornersInt[0], s.cornersInt[2], (0,), thickness=-1)
#    for i in range(s.corners.shape[0]):
#        nextI = (i + 1) % (s.corners.shape[0])
#        # print(s.corners[i,:])
#        newPt = np.array([config.triangleEdgeLength * s.corners[i,:] + (1 - triangleEdgeLength) * s.corners[nextI, :]])
#        midPts = np.append(midPts, newPt, axis=0)
#    # print(midPts)
#    for i in range(s.corners.shape[0]):
#        if s.cornerValidMask[i]:
#            t = np.array([s.corners[i,:], midPts[i,:], midPts[(i+3)%4, :]])
#            t = t.reshape((-1,1,2)).astype(np.int32)
#            cv2.drawContours(img, [t], 0, (255,), -1)
#    # midPts = midPts.reshape((-1,1,2)).astype(np.int32)
#    # cv2.drawContours(img, [midPts], 0, (0,), -1)

def drawSquare4Triangles(img, s, ptsMask, config):
    midPts = np.empty((0,2), np.float64)
    cv2.rectangle(img, s.cornersInt[0], s.cornersInt[2], (0,), thickness=-1)
    # for i in range(s.corners.shape[0]):
    #     nextI = (i + 1) % (s.corners.shape[0])
    #     # print(s.corners[i,:])
    #     newPt = np.array([0.5 * s.corners[i,:] + 0.5 * s.corners[nextI, :]])
    #     midPts = np.append(midPts, newPt, axis=0)
    # print(midPts)
    for i in range(s.corners.shape[0]):
        if s.cornerValidMask[i] and ptsMask[s.cornersIds[i][1], s.cornersIds[i][0]]:
            nextI = (i + 1) % (s.corners.shape[0])
            lastI = (i - 1) if i - 1 >= 0 else s.corners.shape[0] - 1
            pNext = np.array((1 - config.triangleEdgeLength) * s.corners[i,:] + config.triangleEdgeLength * s.corners[nextI, :])
            pLast = np.array((1 - config.triangleEdgeLength) * s.corners[i,:] + config.triangleEdgeLength * s.corners[lastI, :])
            t = np.array([s.corners[i,:], pNext , pLast])
            t = t.reshape((-1,1,2)).astype(np.int32)
            cv2.drawContours(img, [t], 0, (255,), -1)

def drawText(imgPIL, str, centerPt, font, color = 0):
    draw = ImageDraw.Draw(imgPIL)
    strSize = font.getsize(str)
    pt = (int(centerPt[0] - strSize[0] / 2), int(centerPt[1] - strSize[1] / 2))
    draw.text(pt, str, font=font, fill=color)

if __name__ == "__main__":
    # In centimeters
    # dilationSize = 2.5
    # squareSize = 3.2
    #
    # startPoint = [25, 0]
    # startWithWhite = True
    #
    # pixelsPerCm = 45

    inTemplateFile = r'Template.png'

    uGenerator = UnitardDesignGenerator()

    # uGenerator.config.repermuteCodes = True
    # uGenerator.config.marginType = 'black'
    # uGenerator.config.fontName = 'consolab.ttf'
    # uGenerator.config.squareSize = 3.15
    # outFile = "suitRandomizedCodes.png"
    # outCornerKeysFile = "suitRandomizedCodes_codeset.txt"

    # uGenerator.config.repermuteCodes = True
    # uGenerator.config.marginType = 'black'
    # uGenerator.config.fontName = 'consolab.ttf'
    # uGenerator.config.codeDict = codeDict2
    # uGenerator.config.squareSize = 2.5
    # uGenerator.config.fontSize = 100
    # outFile = "suitRandomizedNewCodes.png"

    # uGenerator.config.repermuteCodes = True
    # uGenerator.config.marginType = 'black'
    # uGenerator.config.fontName = 'cour.ttf'
    # uGenerator.config.codeDict = codeDict2
    # uGenerator.config.squareSize = 2.5
    # uGenerator.config.fontSize = 95
    # outFile = "suitRandomizedNewCodesCourierbd.png"

    # uGenerator.config.repermuteCodes = True
    # uGenerator.config.marginType = 'crop'
    # uGenerator.config.codeOnBlackSquare = True
    # uGenerator.config.fontSize = 95
    # uGenerator.config.fontName = 'consolab.ttf'
    # outFile = "suitRandomizedCodesBlackAndWhite.png"

    # textureFile = r"Texture/WrinkledPapaer2Zoomed.png"
    # config = Config()
    # config.repermuteCodes = True
    # config.marginType = 'crop'
    # config.codeOnBlackSquare = True
    # config.fontSize = 95
    # config.squareSize = 3.15
    # config.useTextureBackground = True
    # uGenerator.setConfig(config)
    # texture = cv2.imread(textureFile)
    # config.fontName = 'consolab.ttf'
    #
    # outFile = "suitRandomizedCodesBlackAndWhiteWithTexture.png"
    # outCornerKeysFile = "suitRandomizedCodesBlackAndWhiteWithTexture_codeset.txt"

    # uGenerator.config.repermuteCodes = True
    # uGenerator.config.marginType = 'black'
    # uGenerator.config.fontName = 'consolab.ttf'
    # uGenerator.config.codeDict = codeDict2
    # uGenerator.config.squareSize = 2.5
    # uGenerator.config.fontSize = 95
    # outFile = "suitRandomizedNewCodes.png"
    # outCornerKeysFile = "suitRandomizedNewCodes_codeset.txt"

    #config = Config()
    #config.repermuteCodes = True
    #config.marginType = 'black'
    #config.fontName = 'consola.ttf'
    #config.squareSize = 3.15
    #config.useTextureBackground = False

    #uGenerator.setConfig(config)

    #outFile = "suitRandomizedCodesNonbold.png"
    #outCornerKeysFile = "suitRandomizedCodesNonbold_codeset.txt"

    # config = Config()
    # config.repermuteCodes = True
    # config.marginType = 'whiteTriangle'
    # config.fontName = 'consolab.ttf'
    # config.fontSize = 100
    # config.squareSize = 3.15
    # config.useTextureBackground = False
    #
    # uGenerator.setConfig(config)
    #
    # outFile = "Production2/suitRandomizedCodesNonbold.png"
    # outCornerKeysFile = "Production2/suitRandomizedCodesNonbold_codeset.txt"

    config = Config()
    inTemplateFile = r'Template3.png'
    config.dilationSize = 2
    config.drawMargin = False
    config.repermuteCodes = False
    config.startPoint = [100, 50]
    config.marginType = 'crop'
    config.fontName = 'consolab.ttf'
    config.fontSize = 100
    config.squareSize = 3.2
    config.useTextureBackground = False
    config.addNoCodeCorners = True
    uGenerator.setConfig(config)

    outFile = "Production3/suitNoMarginDalition1_5_withNoCodeCorner.png"
    outCornerKeysFile = "Production3/suitNoMarginDalition1_5_withNoCodeCorner.json"


    # config = Config()
    # config.dilationSize = 2.5
    # config.startPoint = [60,50]
    # config.pixelsPerCm = 46
    # config.drawMargin = False
    # config.repermuteCodes = False
    # config.marginType = 'crop'
    # config.fontName = 'consolab.ttf'
    # config.fontSize = 100
    # config.squareSize = 3.2
    # config.useTextureBackground = False
    #
    # uGenerator.setConfig(config)
    #
    # inTemplateFile = r'Template3.png'
    # outFile = "Production3/suitNoMargin.png"
    # outCornerKeysFile = "Production3/suitNoMargin.txt"



    #uGenerator.setTexture(texture)
    uGenerator.generateCodeSuit(inTemplateFile)
    uGenerator.suitWithChbPIL.save(outFile)
    uGenerator.writeCornerKeysSetJson(outCornerKeysFile)
