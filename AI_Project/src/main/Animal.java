package src.main;

public class Animal {
    // Các đặc trưng động vật
    private int hair, feathers, eggs, milk;
    private int airborne, aquatic, predator, toothed;
    private int backbone, breathes, venomous, fins;
    private int legs, tail, domestic, catsize;
    private int type; // Loại động vật (1-7)

    // Constructor đơn giản
    public Animal(int[] features, int type) {
        this.hair = features[0];
        this.feathers = features[1];
        this.eggs = features[2];
        this.milk = features[3];
        this.airborne = features[4];
        this.aquatic = features[5];
        this.predator = features[6];
        this.toothed = features[7];
        this.backbone = features[8];
        this.breathes = features[9];
        this.venomous = features[10];
        this.fins = features[11];
        this.legs = features[12];
        this.tail = features[13];
        this.domestic = features[14];
        this.catsize = features[15];
        this.type = type;
    }

    // Lấy đặc trưng dưới dạng mảng
    public double[] getFeatures() {
        return new double[] {
                hair, feathers, eggs, milk,
                airborne, aquatic, predator, toothed,
                backbone, breathes, venomous, fins,
                legs, tail, domestic, catsize
        };
    }

    public int getType() { return type; }
}