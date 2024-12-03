import PIL.Image

def isInside(point, polygon):
    # for every two consecutive points inside the polygon
    inside = False
    p1 = polygon[0]
    vertices = len(polygon)

    x = point[0]
    y = point[1]

    for i in range(1, vertices+1):
        # check if the x coordinate of the point is between the x coordinates of polygon[i], polygon[i+1]
        p2 = polygon[i % vertices]

        # bs magic from https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/
        if y > min(p1[1], p2[1]):
            if y <= max(p1[1], p2[1]):
                if x < max(p1[0], p2[0]):
                    intersection_x = (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
                    if x <= intersection_x or p1[0] == p2[0]:
                        inside = not inside

        p1 = p2

    return inside


def generateImage(output, size, polygons): # polygons is a list of polygons where each polygon is a list of (x, y)-tuples
    im = PIL.Image.new(mode='RGB', size=(size, size))
    pixels = im.load()
    for i in range(im.size[0]):
        for j in range(im.size[1]):
            # for larger lists of polygons this could be improved by clustering the polygons first
            for polygon in polygons:
                if isInside((i, j), polygon):
                    pixels[i, j] = (255, 255, 255)
    im.show()
    im.save(output)

if __name__ == "main":
    generateImage("data/img/test.png", 500, [[(10, 10), (300, 250), (250, 300), (100, 100), (10, 490)], [(490, 10), (470, 100), (300, 50)]])