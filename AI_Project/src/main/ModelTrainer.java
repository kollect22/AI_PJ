package src.main;

import java.io.*;
import java.util.*;

public class ModelTrainer {
    private int k = 3; // Mặc định 3 neighbors
    private double[][] trainData;
    private int[] trainLabels;


    public void train(double[][] trainData, int[] trainLabels) {
        this.trainData = trainData;
        this.trainLabels = trainLabels;
        System.out.println("Đã lưu " + trainData.length + " mẫu huấn luyện");
    }


    public int predict(double[] sample) {
        // Tính khoảng cách đến tất cả mẫu huấn luyện
        List<Neighbor> neighbors = new ArrayList<>();

        for (int i = 0; i < trainData.length; i++) {
            double distance = 0;
            for (int j = 0; j < sample.length; j++) {
                double diff = sample[j] - trainData[i][j];
                distance += diff * diff;
            }
            distance = Math.sqrt(distance);
            neighbors.add(new Neighbor(distance, trainLabels[i]));
        }



        // Đếm phiếu của k neighbors gần nhất
        Map<Integer, Integer> votes = new HashMap<>();
        for (int i = 0; i < Math.min(k, neighbors.size()); i++) {
            int label = neighbors.get(i).label;
            votes.put(label, votes.getOrDefault(label, 0) + 1);
        }





    private static class Neighbor {
        double distance;
        int label;

        Neighbor(double distance, int label) {
            this.distance = distance;
            this.label = label;
        }
    }

    // Getter/Setter
    public void setK(int k) { this.k = k; }
    public int getK() { return k; }


    public void save(String filePath) throws IOException {
        try (ObjectOutputStream out = new ObjectOutputStream(
                new FileOutputStream(filePath))) {
            out.writeObject(this);
        }
    }


}