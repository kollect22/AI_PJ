class animal:
    def __init__(self, features, type_val):
        # features là một list chứa các giá trị int
        self.hair = features[0]
        self.feathers = features[1]
        self.eggs = features[2]
        self.milk = features[3]
        self.airborne = features[4]
        self.aquatic = features[5]
        self.predator = features[6]
        self.toothed = features[7]
        self.backbone = features[8]
        self.breathes = features[9]
        self.venomous = features[10]
        self.fins = features[11]
        self.legs = features[12]
        self.tail = features[13]
        self.domestic = features[14]
        self.catsize = features[15]
        self.type = type_val

    def get_features(self):
        return [
            float(self.hair), float(self.feathers), float(self.eggs), float(self.milk),
            float(self.airborne), float(self.aquatic), float(self.predator), float(self.toothed),
            float(self.backbone), float(self.breathes), float(self.venomous), float(self.fins),
            float(self.legs), float(self.tail), float(self.domestic), float(self.catsize)
        ]

    def get_type(self):
        return self.type