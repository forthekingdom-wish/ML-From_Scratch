package ml.optimizer;

public class GradientDescent {

    public double update(double weight,
                         double gradient,
                         double learningRate){

        return weight - learningRate * gradient;

    }

}