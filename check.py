import csv


class image:
    filename: str
    width_height: int
    TL: list
    BR: list
    points: list

    def __init__(self, fn, tlx, tly, brx, bry, points):
        # print(f'fucking {fn}')
        self.filename = fn
        self.TL = [tlx, tly]
        self.BR = [brx, bry]
        self.points = points
        self.width_height = self.BR[0] - self.TL[0]

    def to_list(self):
        return [self.filename, self.TL, self.width_height, self.points]


file = open('data\\data\\relativeCoordsSV.csv')
csvreader = csv.reader(file)
all = []
i = 0
for row in csvreader:
    if i == 0:
        i += 1
        continue
    ye = True
    for value in row:
        if value.startswith('-'):
            ye = False
            break
    if ye:
        all.append(image(row[0], int(row[1]), int(
            row[2]), int(row[3]), int(row[5]), row[5:]).to_list())
file.close()
print(all)
print(len(all))
# with open('GFG.csv', 'w') as f:

#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#     write.writerows(all)
