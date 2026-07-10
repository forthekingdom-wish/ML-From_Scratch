package ml.model;

import ml.optimizer.GradientDescent;
import java.util.Random;

public class SoftmaxRegression {
    private int classSize;
    private int featureSize;
    private double weights[][];
    private double biases[];
    private double learningRate;

    private GradientDescent optimizer = new GradientDescent();

    public  SoftmaxRegression(int classSize, int featureSize, double learningRate) {
        this.classSize = classSize;
        this.featureSize = featureSize;
        this.learningRate = learningRate;
        this.weights = new double[classSize][featureSize];
        this.biases = new double[classSize];

        Random random = new Random();

        for (int i = 0; i < classSize; i++) {
            for (int j = 0; j < featureSize; j++) {
                weights[i][j] = random.nextGaussian() * 0.01;
            }
        }
    }

    public double[] z(double[] x) {
        double[] z = new double[classSize];

        for (int i = 0; i < classSize; i++) {
            z[i] = biases[i];
            for (int j = 0; j < featureSize; j++) {
                z[i] += weights[i][j]*x[j];
            }
        }
        return z;
        // class의 점수 z를 feature들의 합으로 계산한 것.
    }

    public double[] softmax(double[] z) {
        double sum = 0;

        for (int i = 0; i < classSize; i++) {
            z[i] = Math.exp(z[i]);
            sum += z[i];
        }

        for (int i = 0; i < classSize; i++) {
            z[i] /= sum;
        }

        return z;
    }

    public double[] predictProbability(double[] x) {
        return softmax(z(x));
    }

    public int predict(double[] x) {
        double[] probabilities = predictProbability(x);
        int maxIndex = 0;

        for (int i = 1; i < classSize; i++) {
            if (probabilities[i] > probabilities[maxIndex]) {
                maxIndex = i;
            }
        }

        return maxIndex;
    }
    public double crossEntropy(double[] x, int y) {
        double[] probability = predictProbability(x);
        double p = Math.max(probability[y], 1e-15);
        return -Math.log(p);
    }

    public void train(double[][] x, int[] y, int epochs) {

        for (int epoch = 0; epoch < epochs; epoch++) {

            for (int sample = 0; sample < x.length; sample++) {

                // 예측 확률
                double[] probability = predictProbability(x[sample]);

                // 모든 클래스 업데이트
                for (int c = 0; c < classSize; c++) {

                    // One-Hot 정답
                    double target = (y[sample] == c) ? 1.0 : 0.0;

                    // 오차 (Softmax + CrossEntropy의 미분 결과)
                    double error = probability[c] - target;

                    // Weight 업데이트
                    for (int f = 0; f < featureSize; f++) {

                        double gradient = error * x[sample][f];

                        weights[c][f] =
                                optimizer.update(
                                        weights[c][f],
                                        gradient,
                                        learningRate
                                );
                    }

                    // Bias 업데이트
                    biases[c] =
                            optimizer.update(
                                    biases[c],
                                    error,
                                    learningRate
                            );
                }
            }
        }
    }
}
