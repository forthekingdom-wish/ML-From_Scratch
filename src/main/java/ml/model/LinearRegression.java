package ml.model;

import ml.optimizer.GradientDescent;

public class LinearRegression {

    private double weight;
    private double bias;
    private double learningRate;

    private GradientDescent optimizer = new GradientDescent();

    public void setWeight(double weight) {
        this.weight = weight;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public double getWeight() {
        return weight;
    }

    public double predict(double x) {
        return weight * x + bias;
    }

    public double computeCost(double[] x, double[] y) {

        double sum = 0;

        for(int i=0;i<x.length;i++){

            double prediction = predict(x[i]);

            double error = prediction - y[i];

            sum += error * error;

        }

        return sum/(2*x.length);

    }

    public double computeWeightGradient(double[] x,double[] y){

        double gradient = 0;

        for(int i=0;i<x.length;i++){

            gradient += (predict(x[i])-y[i])*x[i];

        }

        return gradient/x.length;

    }

    public double computeBiasGradient(double[] x, double[] y) {

        double gradient = 0;

        for(int i=0;i<x.length;i++){
            gradient += (predict(x[i])-y[i]);
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