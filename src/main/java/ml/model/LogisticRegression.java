package ml.model;

import ml.optimizer.GradientDescent;

public class LogisticRegression {
    private double weight;
    private double bias;
    private double learningRate;
    private double threshold;

    private GradientDescent optimizer = new GradientDescent();

    public LogisticRegression(double weight, double bias, double learningRate, double threshold) {
        this.weight = weight;
        this.bias = bias;
        this.learningRate = learningRate;
        this.threshold = threshold;
    }

    public double getWeight() {
        return weight;
    }

    //가설 z
    public double z(double x) {
        return weight * x + bias;
    }

    //sigmoid함수 -> 1일 확률.
    public double sigmoid(double z) {
        return 1.0/(1.0 + Math.exp(-z));
    }

    //x일 때 h가 1일 확률
    public double probability(double x) {
        return sigmoid(z(x));
    }

    public double predict(double x) {
        return probability(x) >= threshold ? 1 : 0;
    }

    //cost는 어떻게 구했나. 정답 y[]와 가설의 값 yHat을 비교한다.
    public double computeCost(double[] x, double[] y) {

        double sum = 0;

        for(int i=0;i<x.length;i++){

            double prediction = probability(x[i]);
            prediction = Math.max(1e-15, Math.min(1-1e-15, prediction));

            double error = (-1.0) * y[i] * Math.log(prediction) - (1.0-y[i]) * Math.log(1.0 - prediction);

            sum += error;

        }

        return sum/x.length;

    }

    public double computeWeightGradient(double[] x,double[] y){

        double gradient = 0;

        for(int i=0;i<x.length;i++){

            gradient += (probability(x[i])-y[i])*x[i];

        }

        return gradient/x.length;

    }

    public double computeBiasGradient(double[] x, double[] y) {

        double gradient = 0;

        for(int i=0;i<x.length;i++){
            gradient += (probability(x[i])-y[i]);
        }
        return gradient / x.length;
    }

    public void train(double[] x,double[] y,int epochs){

        for(int i=0;i<epochs;i++){

            double weightGradient = computeWeightGradient(x,y);
            double biasGradient = computeBiasGradient(x,y);
            weight = optimizer.update(
                    weight,
                    weightGradient,
                    learningRate
            );

            bias = optimizer.update(
                    bias,
                    biasGradient,
                    learningRate
            );

            if(i%100==0){

                System.out.println(
                        "Epoch : "+i+
                                " Weight : "+weight+
                                " Bias : " + bias +
                                " Cost : "+computeCost(x,y)
                );

            }

        }

    }
}
