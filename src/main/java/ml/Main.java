package ml;

import ml.model.LinearRegression;
import ml.model.LogisticRegression;
import ml.model.SoftmaxRegression;

import java.util.Arrays;

public class Main {

    public static void main(String[] args) {

        testSoftmax();
    }

    private static void testLogisticRegression(){

        double[] x = {1, 2, 3, 4, 5, 6};
        double[] y = {0, 0, 0, 1, 1, 1};

        LogisticRegression model = new LogisticRegression(0, 0, 0.1, 0.5);

        model.train(x, y, 10000);

        System.out.println("Probability : " + model.probability(2) + ", Predict : " + model.predict(2));

        System.out.println("Probability : " + model.probability(5) + ", Predict : " + model.predict(5));

    }

    private static void testSoftmax() {
        // feature = 2 예) 공부시간, 수면시간
        // class = 3 예) 성적 0 1 2

        double[][] x = {
                {1, 1},
                {2, 1},
                {1, 2},

                {8, 8},
                {9, 8},
                {8, 9},

                {15, 15},
                {16, 15},
                {15, 16}
        };

        int[] y = {
                0, 0, 0,
                1, 1, 1,
                2, 2, 2
        };

        SoftmaxRegression model =
                new SoftmaxRegression(
                        3,      // class 개수
                        2,      // feature 개수
                        0.01    // learning rate
                );

        model.train(x, y, 1000);

        double[] test = {8.5, 8.5};

        System.out.println("예측 클래스 : "
                + model.predict(test));

        System.out.println("확률 : "
                + Arrays.toString(
                model.predictProbability(test)
        ));

        System.out.println("CrossEntropy : "
                + model.crossEntropy(test, 1));
    }
}