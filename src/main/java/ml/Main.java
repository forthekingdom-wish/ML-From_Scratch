package ml;

import ml.model.LinearRegression;
import ml.model.LogisticRegression;


public class Main {

    public static void main(String[] args) {

        double[] x = {1, 2, 3, 4, 5, 6};
        double[] y = {0, 0, 0, 1, 1, 1};

        LogisticRegression model = new LogisticRegression(0, 0, 0.1, 0.5);

        model.train(x, y, 10000);

        System.out.println("Probability : " + model.probability(2) + ", Predict : " + model.predict(2));

        System.out.println("Probability : " + model.probability(5) + ", Predict : " + model.predict(5));

    }
}