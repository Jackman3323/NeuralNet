import java.util.List;

public class Network {
    //INSTANCE-DATA

    private HughesMatrix ihWeights; //Input and hidden layer weights
    private HughesMatrix hoWeights; //Output and hidden layer weights
    private HughesMatrix hiddenBias; //Hidden layer bias
    private HughesMatrix outputBias; //Output layer bias
    private double learningRate = 0.01; //Learning rate

    //CONSTRUCTORS
    //Simple constructor: input the size of the input layer, the hidden layer and the output layer
    public Network(int i, int h, int o){
        this.ihWeights = new HughesMatrix(h,i);
        this.hoWeights = new HughesMatrix(o,h);
        this.hiddenBias = new HughesMatrix(h,1);
        this.outputBias = new HughesMatrix(o,1);
    }

    //METHODS
    //Forward-propogation--input a long column, AI-ify it
    public List<Double> predict (double[] x){
        //Input the data into the neural net!!
        HughesMatrix input = HughesMatrix.fromArray(x);
        //Multiply the input hughesMatrix by the input-hidden weights and create the hidden layer HughesMatrix
        HughesMatrix hidden = HughesMatrix.multiply(ihWeights, input);
        //Add the biases
        hidden.add(hiddenBias);
        //Sigmoid-curve-ify it
        hidden.initSigmoid();
        //Now create the output layer by multiplying the hidden --> output weights and then adding the biases
        HughesMatrix output = HughesMatrix.multiply(hoWeights,hidden);
        output.add(outputBias);
        //Sigmoid-curve-ify it
        output.initSigmoid();
        return output.toList();
    }

    public void train(double [] in, double [] answer){
        //Input the data into the neural net!!
        HughesMatrix input = HughesMatrix.fromArray(in);
        //Multiply the input hughesMatrix by the input-hidden weights and create the hidden layer HughesMatrix
        HughesMatrix hidden = HughesMatrix.multiply(ihWeights, input);
        //Add the biases
        hidden.add(hiddenBias);
        //Sigmoid-curve-ify it
        hidden.initSigmoid();
        //Now create the output layer by multiplying the hidden --> output weights and then adding the biases
        HughesMatrix output = HughesMatrix.multiply(hoWeights,hidden);
        output.add(outputBias);
        //Sigmoid-curve-ify it
        output.initSigmoid();
        //"Correct" output for given input
        HughesMatrix target = HughesMatrix.fromArray(answer);
        //Determine error between input and correct output
        HughesMatrix error = HughesMatrix.subtract(target, output);
        HughesMatrix gradient = output.deriveSigmoid();
        gradient.multiply(error);
        gradient.multiply(learningRate);
        HughesMatrix transposeHidden = HughesMatrix.clone(hidden);
        HughesMatrix delta = HughesMatrix.multiply(gradient,transposeHidden);
        hoWeights.add(delta);
        outputBias.add(gradient);
        HughesMatrix tempOutputWeight = HughesMatrix.clone(hoWeights);
        HughesMatrix hiddenError = HughesMatrix.multiply(tempOutputWeight,error);
        HughesMatrix hiddenGradient = hidden.deriveSigmoid();
        hiddenGradient.multiply(hiddenError);
        hiddenGradient.multiply(learningRate);
        HughesMatrix tempInput = HughesMatrix.clone(input);
        HughesMatrix hiddenDelta = HughesMatrix.multiply(hiddenGradient,tempInput);
        ihWeights.add(hiddenDelta);
        hiddenBias.add(hiddenGradient);
    }

    public void fit(double[][]x, double[][]y, int epochs){
        for(int i = 0; i < epochs; i++){
            int sampleNum = (int)(Math.random() * x.length);
            this.train(x[sampleNum],y[sampleNum]);
        }
    }

    public static void main(String[] args) {
        double [][] X= {
                {0,0},
                {1,0},
                {0,1},
                {1,1}
        };
        double [][] Y= {
                {0},{1},{1},{0}
        };
        Network nn = new Network(2,10,1);
        List<Double>output;
        nn.fit(X, Y, 50000);
        double [][] input = {
                {0,0},{0,1},{1,0},{1,1}
        };
        for(double[] d :input)
        {
            output = nn.predict(d);
            System.out.println(output.toString());
        }
    }
}