from matplotlib import pyplot as plt
import cv2, json


if __name__ == '__main__':
    img = cv2.imread(r'Production3\suitNoMarginDalition1_5_withNoCodeCorner.png')
    cornerData = json.load(open(r'Production3\suitNoMarginDalition1_5_withNoCodeCorner.json'))

    fig,ax = plt.subplots()
    ax.imshow(img)

    # fig.set_size_inches(20, 10)

    for corner in cornerData['corners']:
        if corner["codeWhite"] == '':
            ax.plot(corner["pts"][0], corner["pts"][1], 'x', color='red')
    ax.axis('off')
    fig.savefig('Production3\suitNoMarginDalition1_5_withNoCodeCorner_new.pdf', dpi=2000, transparent=True, bbox_inches='tight', pad_inches=0)