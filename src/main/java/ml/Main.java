package ml;

import ml.model.LinearRegression;

public class Main {

    public static void main(String[] args) {

        double[] x = {1,2,3,4,5};
        double[] y = {10,30,50,70,90};

        LinearRegression model = new LinearRegression();

        model.setWeight(20);
        model.setLearningRate(0.01);

        model.train(x, y, 1000);

        System.out.println("Prediction : " + model.predict(6));
    }
}